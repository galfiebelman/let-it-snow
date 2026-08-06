import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline

logger = logging.getLogger(__name__)


class CogVideoXDirectGuidance:
    """
    CogVideoX guidance module for video score distillation.

    Supports multiple SDS variants:
    - 'sds': Standard SDS (baseline) - assumes model outputs v-prediction
    - 'nfsd': Noise-Free Score Distillation - removes undesired noise term
    - 'cfgpp': CFG++ with manifold projection via VAE decode/encode
    - 'nfsd+cfgpp': Combined NFSD and CFG++ for best of both methods
    - 'bridge': Bridge Score Distillation with source prompt conditioning
    - 'bridge+nfsd': Combined Bridge and NFSD (classifier-only direction)

    Note: CogVideoX transformer outputs v-prediction directly, not noise prediction.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dtype = (
            torch.float16
            if getattr(cfg.guidance, "half_precision_weights", True)
            else torch.float32
        )

        if cfg.guidance.gpu_size == "big" and self.weights_dtype == torch.float16:
            self.weights_dtype = torch.bfloat16

        # Load CogVideoX Video-to-Video pipeline
        logger.info("Loading CogVideoX Video-to-Video pipeline...")
        self.pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
            getattr(
                cfg.guidance,
                "pretrained_model_name_or_path",
                "THUDM/CogVideoX-2B",
            ),
            torch_dtype=self.weights_dtype,
        ).to(self.device)
        # Use DPM scheduler for v-prediction
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Enable memory optimizations
        self.pipe.enable_attention_slicing()

        enable_model_cpu_offload = getattr(
            cfg.guidance, "enable_model_cpu_offload", False
        )
        enable_sequential_cpu_offload = getattr(
            cfg.guidance, "enable_sequential_cpu_offload", False
        )
        enable_channels_last_format = getattr(
            cfg.guidance, "enable_channels_last_format", False
        )
        enable_vae_tiling = getattr(cfg.guidance, "enable_vae_tiling", False)

        if enable_model_cpu_offload:
            self.pipe.enable_model_cpu_offload()

        if enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if enable_channels_last_format:
            self.pipe.to(memory_format=torch.channels_last)

        if enable_vae_tiling:
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()

        # Enable VAE gradient checkpointing for memory efficiency
        enable_vae_gradient_checkpointing = getattr(
            cfg.guidance, "enable_vae_gradient_checkpointing", True
        )
        if enable_vae_gradient_checkpointing and hasattr(
            self.pipe.vae, "enable_gradient_checkpointing"
        ):
            self.pipe.vae.enable_gradient_checkpointing()

        self.pipe.vae.requires_grad_(False)
        self.pipe.transformer.requires_grad_(False)

        # Get scheduler configuration
        num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        min_step_percent = 0.02
        max_step_percent = 0.98
        self.min_step = getattr(
            cfg.guidance, "min_step", int(num_train_timesteps * min_step_percent)
        )
        self.max_step = getattr(
            cfg.guidance, "max_step", int(num_train_timesteps * max_step_percent)
        )

        # Get alphas for weighting
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)

        # SDS configuration
        self.guidance_scale = getattr(cfg.guidance, "guidance_scale", 100.0)
        self.grad_clip_val = getattr(cfg.guidance, "grad_clip", None)
        self.spatial_size = getattr(cfg.guidance, "spatial_size", (480, 720))

        # SDS variant selection
        self.sds_variant = getattr(cfg.guidance, "sds_variant", "sds")
        valid_variants = ["sds", "nfsd", "cfgpp", "nfsd+cfgpp", "bridge", "bridge+nfsd"]
        if self.sds_variant not in valid_variants:
            raise ValueError(
                f"Invalid sds_variant '{self.sds_variant}'. "
                f"Must be one of {valid_variants}"
            )
        logger.info("Using SDS variant: %s", self.sds_variant)

        # CFG++ specific configuration
        self.cfgpp_manifold_projection = getattr(
            cfg.guidance, "cfgpp_manifold_projection", True
        )

        # Max step schedule configuration
        self.max_step_schedule = getattr(cfg.guidance, "max_step_schedule", None)
        if self.max_step_schedule is not None:
            self.initial_max_step = getattr(
                cfg.guidance.max_step_schedule, "initial_max_step", self.max_step
            )
            self.final_max_step = getattr(
                cfg.guidance.max_step_schedule, "final_max_step", 500
            )
            self.transition_step = getattr(
                cfg.guidance.max_step_schedule, "transition_step", 5000
            )
        else:
            self.initial_max_step = self.max_step
            self.final_max_step = self.max_step
            self.transition_step = float("inf")

        # Default prompts
        self.default_text_prompt = getattr(
            cfg.guidance,
            "default_text_prompt",
            "A woman stands upright wearing a purple jacket, dark pants, and "
            "brown boots. She raises both arms forward until they are fully "
            "extended in front of her at shoulder height.",
        )
        self.default_negative_prompt = getattr(
            cfg.guidance,
            "default_negative_prompt",
            "blurry, low quality, distorted, artifacts, bad anatomy",
        )
        self.default_source_prompt = getattr(
            cfg.guidance,
            "default_source_prompt",
            "blurry, oversaturated, smooth, low detail, plastic, 3D render "
            "artifacts, unrealistic colors, washed out, flat lighting, "
            "no fine details",
        )

        # Number of frames configuration (for frame duplication)
        self.num_frames = getattr(cfg.guidance, "num_frames", 49)

        # NFSD: cap guidance scale at nominal CFG scale per the paper
        if "nfsd" in self.sds_variant:
            original_guidance_scale = self.guidance_scale
            self.guidance_scale = min(self.guidance_scale, 7.5)
            if original_guidance_scale != self.guidance_scale:
                logger.info(
                    "NFSD: Reduced guidance scale from %.1f to %.1f (nominal CFG scale)",
                    original_guidance_scale,
                    self.guidance_scale,
                )

        logger.info("Final guidance scale: %.1f", self.guidance_scale)

    def _set_vae_gradient_checkpointing(self, enabled: bool):
        """Set VAE gradient checkpointing using standard diffusers method."""
        try:
            vae = self.pipe.vae
            if enabled:
                vae.enable_gradient_checkpointing()
            else:
                vae.disable_gradient_checkpointing()
        except Exception as e:
            logger.warning("Could not set VAE gradient checkpointing: %s", e)

    # ----- Max Step Schedule -----

    def get_current_max_step(self, current_step: int) -> int:
        """
        Compute the current max_step based on the training step and schedule.

        Implements linear decrease from initial_max_step to final_max_step
        over the first transition_step optimization iterations.

        Args:
            current_step: Current training step.

        Returns:
            Current max noise step to use.
        """
        if self.max_step_schedule is None:
            return self.max_step

        if current_step >= self.transition_step:
            return self.final_max_step

        # Linear interpolation
        progress = current_step / self.transition_step
        current_max_step = self.initial_max_step + progress * (
            self.final_max_step - self.initial_max_step
        )

        return int(current_max_step)

    def encode_prompt_with_pipe(self, prompt, negative_prompt):
        """Encode prompt using the pipeline's method."""
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=self.device,
            dtype=self.weights_dtype,
        )
        # Concatenate for CFG
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        return text_embeddings

    def add_noise_v_prediction(self, original_samples, noise, timesteps):
        """
        Add noise for v-prediction following the CogVideoX approach.
        """
        alphas_cumprod = self.alphas[timesteps]
        sqrt_alpha_prod = alphas_cumprod**0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def compute_grad_sds(self, latents, text_embeddings, t):
        """
        Compute SDS gradient with support for multiple variants:
        - 'sds': Standard SDS with v-prediction
        - 'nfsd': Noise-Free Score Distillation
        - 'cfgpp': CFG++ with manifold projection
        - 'nfsd+cfgpp': Combined NFSD and CFG++
        - 'bridge': Bridge Score Distillation
        - 'bridge+nfsd': Combined Bridge and NFSD
        """
        noise = torch.randn_like(latents)

        # Add noise using scheduler's method
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t)

        with torch.no_grad():
            noisy_latents_unet = noisy_latents.to(self.pipe.transformer.dtype)
            latent_model_input = torch.cat([noisy_latents_unet] * 2)
            t_input = t.expand(latent_model_input.shape[0])

            # Get rotary embeddings if needed
            image_rotary_emb = None
            if self.pipe.transformer.config.use_rotary_positional_embeddings:
                image_rotary_emb = (
                    self.pipe._prepare_rotary_positional_embeddings(
                        self.spatial_size[0],
                        self.spatial_size[1],
                        latents.size(1),
                        self.device,
                    )
                )

            # Predict v-prediction with CogVideoX transformer
            v_pred = self.pipe.transformer(
                hidden_states=latent_model_input.to(self.pipe.transformer.dtype),
                encoder_hidden_states=text_embeddings.to(
                    self.pipe.transformer.dtype
                ),
                timestep=t_input,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            v_pred = v_pred.to(torch.float32)

        # Apply classifier-free guidance
        v_pred_uncond, v_pred_text = v_pred.chunk(2)

        # Branch based on SDS variant
        if self.sds_variant == "sds":
            v_pred_cfg = v_pred_text + self.guidance_scale * (
                v_pred_text - v_pred_uncond
            )
            grad = self._compute_sds_grad(latents, v_pred_cfg, noise, t)

        elif self.sds_variant == "nfsd":
            grad = self._compute_nfsd_grad(
                latents, v_pred_text, v_pred_uncond, noise, t
            )

        elif self.sds_variant == "cfgpp":
            v_pred_cfg = v_pred_text + self.guidance_scale * (
                v_pred_text - v_pred_uncond
            )
            grad = self._compute_sds_grad(latents, v_pred_cfg, noise, t)
            if self.cfgpp_manifold_projection:
                grad = self._apply_cfgpp_manifold_projection(grad, latents)

        elif self.sds_variant == "nfsd+cfgpp":
            grad = self._compute_nfsd_grad(
                latents, v_pred_text, v_pred_uncond, noise, t
            )
            if self.cfgpp_manifold_projection:
                grad = self._apply_cfgpp_manifold_projection(grad, latents)

        elif self.sds_variant == "bridge":
            grad = self._compute_bridge_grad(latents, noisy_latents, t)

        elif self.sds_variant == "bridge+nfsd":
            grad = self._compute_bridge_nfsd_grad(latents, noisy_latents, t)

        else:
            raise ValueError(f"Unknown SDS variant: {self.sds_variant}")

        grad = torch.nan_to_num(grad)
        grad = grad.to(latents.dtype)

        return grad

    def _compute_sds_grad(self, latents, v_pred, noise, t):
        """
        Compute standard SDS gradient using v-prediction.

        Args:
            latents: Clean latents x_0.
            v_pred: V-prediction from the model.
            noise: Random noise used for noising.
            t: Timestep.
        """
        alphas_cumprod = self.alphas
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

        # Compute noisy latents
        zt = sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise

        # Predict x_0 from v-prediction
        z_hat = sqrt_alpha_t * zt - sqrt_one_minus_alpha_t * v_pred

        # SDS gradient
        grad = latents - z_hat

        # Apply weighting factor
        w = (1 - alphas_cumprod[t]).view(-1, 1, 1, 1, 1)
        grad = w * grad

        return grad

    def _compute_nfsd_grad(self, latents, v_pred_text, v_pred_uncond, noise, t):
        """
        Compute Noise-Free Score Distillation (NFSD) gradient.

        Based on "Noise-Free Score Distillation" (https://arxiv.org/abs/2310.17590).
        NFSD removes the undesired noise term from the score distillation process.

        Args:
            latents: Clean latents x_0.
            v_pred_text: V-prediction with text conditioning.
            v_pred_uncond: V-prediction without conditioning.
            noise: Random noise used for noising.
            t: Timestep.
        """
        alphas_cumprod = self.alphas
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

        # NFSD: difference between conditional and unconditional predictions
        # removes the noise term that causes over-smoothing in standard SDS
        v_diff = v_pred_text - v_pred_uncond

        # NFSD gradient
        grad = self.guidance_scale * sqrt_one_minus_alpha_t * v_diff

        # Apply weighting factor
        w = (1 - alphas_cumprod[t]).view(-1, 1, 1, 1, 1)
        grad = w * grad

        return grad

    def _compute_bridge_grad(self, latents, noisy_latents, t):
        """
        Bridge Score Distillation.

        Key insight: SDS artifacts come from poor source distribution estimation.
        Instead of using null/negative text as "unconditional", we use a text
        description of what SDS artifacts look like (the source distribution).

        Reference: "Rethinking Score Distillation as a Bridge Between Image
        Distributions" (NeurIPS 2024).
        """
        self.bridge_embeddings = self.bridge_embeddings.cuda()

        with torch.no_grad():
            noisy_latents_input = noisy_latents.to(self.pipe.transformer.dtype)
            latent_model_input = torch.cat([noisy_latents_input] * 2)
            t_input = t.expand(latent_model_input.shape[0])

            image_rotary_emb = None
            if self.pipe.transformer.config.use_rotary_positional_embeddings:
                image_rotary_emb = (
                    self.pipe._prepare_rotary_positional_embeddings(
                        self.spatial_size[0],
                        self.spatial_size[1],
                        latents.size(1),
                        self.device,
                    )
                )

            # Predict with [source, target] conditioning
            v_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=self.bridge_embeddings.to(
                    self.pipe.transformer.dtype
                ),
                timestep=t_input,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0].to(torch.float32)

            v_pred_source, v_pred_target = v_pred.chunk(2)

        self.bridge_embeddings = self.bridge_embeddings.cpu()

        # Convert v-predictions to x_hat_0
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

        zt = noisy_latents.to(torch.float32)
        x_hat_target = sqrt_alpha_t * zt - sqrt_one_minus_alpha_t * v_pred_target
        x_hat_source = sqrt_alpha_t * zt - sqrt_one_minus_alpha_t * v_pred_source

        # Bridge gradient: transport from source to target
        classifier_direction = x_hat_target - x_hat_source
        reconstruction_term = latents - x_hat_target

        grad = reconstruction_term - self.guidance_scale * classifier_direction

        w = 1 - alpha_t
        grad = w * grad

        return grad

    def _compute_bridge_nfsd_grad(self, latents, noisy_latents, t):
        """
        Bridge + NFSD: Combines Bridge's better source distribution with
        NFSD's removal of the reconstruction term.

        This is the "classifier-only" version of Bridge - no reconstruction
        term, just the direction from source distribution to target.
        """
        self.bridge_embeddings = self.bridge_embeddings.cuda()

        with torch.no_grad():
            noisy_latents_input = noisy_latents.to(self.pipe.transformer.dtype)
            latent_model_input = torch.cat([noisy_latents_input] * 2)
            t_input = t.expand(latent_model_input.shape[0])

            image_rotary_emb = None
            if self.pipe.transformer.config.use_rotary_positional_embeddings:
                image_rotary_emb = (
                    self.pipe._prepare_rotary_positional_embeddings(
                        self.spatial_size[0],
                        self.spatial_size[1],
                        latents.size(1),
                        self.device,
                    )
                )

            v_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=self.bridge_embeddings.to(
                    self.pipe.transformer.dtype
                ),
                timestep=t_input,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0].to(torch.float32)

            v_pred_source, v_pred_target = v_pred.chunk(2)

        self.bridge_embeddings = self.bridge_embeddings.cpu()

        # Convert v-predictions to x_hat_0
        alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

        zt = noisy_latents.to(torch.float32)
        x_hat_target = sqrt_alpha_t * zt - sqrt_one_minus_alpha_t * v_pred_target
        x_hat_source = sqrt_alpha_t * zt - sqrt_one_minus_alpha_t * v_pred_source

        # Bridge+NFSD: ONLY the classifier direction, NO reconstruction term
        classifier_direction = x_hat_target - x_hat_source

        grad = -self.guidance_scale * classifier_direction

        w = 1 - alpha_t
        grad = w * grad

        return grad

    def _apply_cfgpp_manifold_projection(self, grad, latents):
        """
        Apply CFG++ manifold projection by decoding and re-encoding with VAE.

        Based on "CFG++: Manifold-constrained Classifier Free Guidance"
        (https://arxiv.org/abs/2406.08070).

        This addresses off-manifold issues in CFG by projecting back to the
        data manifold.

        Args:
            grad: Current gradient.
            latents: Original latents for reference.

        Returns:
            Manifold-projected gradient.
        """
        with torch.no_grad():
            updated_latents = latents - grad

            # Convert from transformer format (B, T, C, H, W) to VAE format (B, C, T, H, W)
            updated_latents_vae = updated_latents.permute(0, 2, 1, 3, 4)

            try:
                vae_dtype = self.pipe.vae.dtype
                updated_latents_vae = updated_latents_vae.to(dtype=vae_dtype)

                scaled_latents = (
                    updated_latents_vae / self.pipe.vae.config.scaling_factor
                )

                decoded = self.pipe.vae.decode(scaled_latents).sample
                decoded = torch.clamp(decoded, -1.0, 1.0)

                re_encoded = self.pipe.vae.encode(decoded).latent_dist.mode()
                re_encoded = re_encoded * self.pipe.vae.config.scaling_factor
                re_encoded = re_encoded.to(dtype=latents.dtype)

                # Convert back to transformer format (B, T, C, H, W)
                re_encoded = re_encoded.permute(0, 2, 1, 3, 4)

                manifold_grad = latents - re_encoded
                return manifold_grad

            except Exception as e:
                logger.warning(
                    "CFG++ manifold projection failed, using original gradient: %s", e
                )
                return grad

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel-space frames."""
        latents = latents.permute(
            0, 2, 1, 3, 4
        )  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.pipe.vae_scaling_factor_image * latents
        frames = self.pipe.vae.decode(latents).sample
        return frames

    def calc_text_embeddings(self, prompt, negative_prompt, source_prompt=None):
        """Pre-compute and cache text embeddings for the given prompts."""
        with torch.no_grad():
            self.text_embeddings = self.encode_prompt_with_pipe(
                prompt, negative_prompt
            )
            self.text_embeddings.cpu()

            if "bridge" in self.sds_variant:
                source = (
                    source_prompt
                    if source_prompt is not None
                    else self.default_source_prompt
                )
                self.bridge_embeddings = self.encode_prompt_with_pipe(
                    prompt, source
                )
                self.bridge_embeddings = self.bridge_embeddings.cpu()

    def __call__(
        self,
        rgb_BCHW: torch.Tensor = None,
        num_frames: int = 25,
        generator: Optional[torch.Generator] = None,
        current_step: Optional[int] = None,
        use_sigmoid: bool = False,
        **kwargs,
    ):
        """
        Compute SDS loss for the given video tensor.

        Args:
            rgb_BCHW: Input video in (B, C, F, H, W) format, values in [0, 1].
            num_frames: Number of frames in the video sequence.
            generator: Random generator for reproducible results.
            current_step: Current training step for max_step scheduling.
            use_sigmoid: Whether to apply sigmoid to input.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing the SDS loss under key ``loss_sds_video``.
        """
        batch_size = 1

        # Normalize video to [-1, 1]
        video_vae = rgb_BCHW * 2.0 - 1.0

        # Encode video to latents using VAE
        # VAE expects (B, C, T, H, W) format
        video_for_vae = video_vae.to(self.pipe.vae.dtype)
        video_for_vae = video_for_vae.permute(0, 2, 1, 3, 4)
        latents_dist = self.pipe.vae.encode(video_for_vae).latent_dist
        latents = latents_dist.sample() * self.pipe.vae.config.scaling_factor
        # Convert to transformer format: (B, C, T, H, W) -> (B, T, C, H, W)
        latents = latents.permute(0, 2, 1, 3, 4)

        # Duplicate final frame latents (no gradients) if fewer frames than expected
        if num_frames < self.num_frames:
            last_latent = latents[:, -1:, ...]
            with torch.no_grad():
                duplicate_latents = last_latent.detach().repeat(
                    1, self.num_frames - num_frames, 1, 1, 1
                )
            latents = torch.cat([latents, duplicate_latents], dim=1)
            num_frames = self.num_frames

        # Get current max step based on training schedule
        current_max_step = (
            self.get_current_max_step(current_step)
            if current_step is not None
            else self.max_step
        )

        # Sample timestep
        t = torch.randint(
            self.min_step,
            current_max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # Compute gradient
        self.text_embeddings = self.text_embeddings.cuda()
        grad = self.compute_grad_sds(latents, self.text_embeddings, t)
        self.text_embeddings = self.text_embeddings.cpu()

        # Use reparameterization trick for loss computation
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / rgb_BCHW.shape[1]

        loss_sds = loss_sds.float()

        return {"loss_sds_video": loss_sds}

    @torch.no_grad()
    def generate_video(
        self,
        num_frames: int = 48,
        text_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ):
        """
        Run text-to-video generation (inference, not SDS).
        """
        prompt = (
            text_prompt if text_prompt is not None else self.default_text_prompt
        )
        neg_prompt = (
            negative_prompt
            if negative_prompt is not None
            else self.default_negative_prompt
        )

        # For video-to-video, create a random input video
        input_video = torch.randn(
            1,
            3,
            num_frames,
            height or self.spatial_size[0],
            width or self.spatial_size[1],
        ).to(self.device)

        return self.pipe(
            video=input_video,
            prompt=prompt,
            negative_prompt=neg_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            **kwargs,
        )
