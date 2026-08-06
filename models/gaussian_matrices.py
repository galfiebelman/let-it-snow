import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.embeddings import XYZEmbedding, TimeEmbedding


def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-9) + 1e-9)


def integrate_rotation(q, w, dt=1.0):
    """
    Integrates a quaternion rotation given an axis-angle angular velocity.
    :param q: (N, 4) tensor of quaternions (w, x, y, z)
    :param w: (N, 3) tensor of angular velocities (axis-angle)
    :param dt: Time step
    :return: (N, 4) tensor of new quaternions
    """
    angle = torch.norm(w, p=2, dim=-1, keepdim=True) * dt
    axis = F.normalize(w + 1e-9, p=2, dim=-1)

    angle_half = angle / 2.0
    sin_half = torch.sin(angle_half)
    cos_half = torch.cos(angle_half)

    dq_w = cos_half
    dq_xyz = axis * sin_half
    dq = torch.cat([dq_w, dq_xyz], dim=-1)
    dq = F.normalize(dq + 1e-9, p=2, dim=-1)

    w_d, x_d, y_d, z_d = dq.split(1, dim=-1)
    w_o, x_o, y_o, z_o = q.split(1, dim=-1)

    w_new = w_d * w_o - x_d * x_o - y_d * y_o - z_d * z_o
    x_new = w_d * x_o + x_d * w_o + y_d * z_o - z_d * y_o
    y_new = w_d * y_o - x_d * z_o + y_d * w_o + z_d * x_o
    z_new = w_d * z_o + x_d * y_o - y_d * x_o + z_d * w_o

    q_new = torch.cat([w_new, x_new, y_new, z_new], dim=-1)
    return F.normalize(q_new + 1e-9, p=2, dim=-1)


class GaussianMatrices(nn.Module):
    """
    Recurrent neural dynamics model for Gaussian parameters.
    Predicts velocity corrections, angular velocities, and appearance deltas.
    """

    def __init__(self, init_opacities, init_scales, init_colors, init_rotations, hidden_dim=96,
                 device="cuda", mode='full'):
        super().__init__()
        self.device = device
        self.return_orig = False
        self.mode = mode

        op = torch.tensor(init_opacities, dtype=torch.float32, device=device).view(-1, 1)
        op_clamped = torch.clamp(op, 0.01, 0.99)
        self.register_buffer('original_opacities_logit', inverse_sigmoid(op_clamped))

        sc = torch.tensor(init_scales, dtype=torch.float32, device=device).view(-1, 3)
        self.register_buffer('original_scales_log', torch.log(sc + 1e-9))

        col = torch.tensor(init_colors, dtype=torch.float32, device=device).view(-1, 3)
        self.register_buffer('original_colors', col)

        rot = torch.tensor(init_rotations, dtype=torch.float32, device=device).view(-1, 4)
        self.register_buffer('original_rotations', rot)

        self.time_embed_dim = 12
        self.time_embedder = TimeEmbedding(self.time_embed_dim).to(device)

        self.xyz_embed_dim = 12
        self.xyz_embedder = XYZEmbedding(self.xyz_embed_dim).to(device)

        self.velocity_embed_dim = 12
        self.velocity_embedder = XYZEmbedding(self.velocity_embed_dim).to(device)

        self.param_dim = 1 + 3 + 3 + 4
        self.physics_dim = 3 + 3

        self.mlp_input_dim = (
                self.param_dim +
                self.time_embed_dim +
                self.xyz_embed_dim
        )
        if self.mode == 'full':
            self.mlp_input_dim += self.velocity_embed_dim

        self.mlp_backbone = nn.Sequential(
            nn.Linear(self.mlp_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        ).to(device)

        self.app_head = nn.Linear(hidden_dim, 7).to(device)
        self.phys_head = nn.Linear(hidden_dim, 6).to(device)

        nn.init.zeros_(self.app_head.weight)
        nn.init.zeros_(self.app_head.bias)
        nn.init.zeros_(self.phys_head.weight)
        nn.init.zeros_(self.phys_head.bias)

    def get_initial_state(self):
        return {
            "opacity": self.original_opacities_logit.clone(),
            "scale": self.original_scales_log.clone(),
            "color": self.original_colors.clone(),
            "rotation": self.original_rotations.clone()
        }

    def step(self, mask, t,
             prev_app_state,
             prev_x_rendered,
             prev_v_corrected,
             original_xyz_t_sim,
             original_velocity_t_sim):
        """
        Recurrent step: predicts appearance deltas and physics corrections.
        MLP input is the previous rendered state.
        Physics sim data is used for regularization losses.
        """

        if self.return_orig:
            scale = torch.exp(self.original_scales_log[mask])
            opacity = torch.sigmoid(self.original_opacities_logit[mask])
            color = self.original_colors[mask]
            rot = self.original_rotations[mask]
            xyz_render = original_xyz_t_sim[mask]

            new_app_state = self.get_initial_state()
            new_x_rendered = self.original_colors.clone()
            new_v_corrected = self.original_colors.clone()
            new_x_rendered[mask] = original_xyz_t_sim[mask]
            new_v_corrected[mask] = original_velocity_t_sim[mask]

            return (new_app_state, new_x_rendered, new_v_corrected,
                    (scale, opacity, color, xyz_render, rot, 0.0, 0.0, 0.0))

        prev_op = prev_app_state["opacity"][mask]
        prev_sc = prev_app_state["scale"][mask]
        prev_col = prev_app_state["color"][mask]
        prev_rot = prev_app_state["rotation"][mask]

        prev_x = prev_x_rendered[mask]
        prev_v = prev_v_corrected[mask]

        num_particles_in_mask = prev_op.shape[0]

        t_tensor = torch.full((num_particles_in_mask,), t, device=self.device, dtype=torch.long)
        t_embed = self.time_embedder(t_tensor)
        if self.mode == 'full':
            xyz_embed = self.xyz_embedder(prev_x)
            vel_embed = self.velocity_embedder(prev_v)
            mlp_input = torch.cat([
                prev_op, prev_sc, prev_col, prev_rot,
                t_embed, xyz_embed, vel_embed
            ], dim=1)
        else:
            sim_x = original_xyz_t_sim[mask]
            xyz_embed = self.xyz_embedder(sim_x)
            mlp_input = torch.cat([
                prev_op, prev_sc, prev_col, prev_rot,
                t_embed, xyz_embed
            ], dim=1)

        features = checkpoint(self.mlp_backbone, mlp_input, use_reentrant=False)

        app_deltas = self.app_head(features)
        delta_op = app_deltas[:, 0:1]
        delta_sc = app_deltas[:, 1:4]
        delta_col = app_deltas[:, 4:7]

        op_reg_loss = torch.mean(delta_op ** 2)
        sc_reg_loss = torch.mean(delta_sc ** 2)
        col_reg_loss = torch.mean(delta_col ** 2)

        new_op_logit_masked = prev_op + delta_op
        new_sc_log_masked = prev_sc + delta_sc
        new_col_masked = prev_col + delta_col

        new_app_state = {}
        new_app_state["opacity"] = prev_app_state["opacity"].clone()
        new_app_state["opacity"][mask] = new_op_logit_masked
        new_app_state["scale"] = prev_app_state["scale"].clone()
        new_app_state["scale"][mask] = new_sc_log_masked
        new_app_state["color"] = prev_app_state["color"].clone()
        new_app_state["color"][mask] = new_col_masked

        opacity = torch.sigmoid(new_op_logit_masked)
        scale = torch.exp(new_sc_log_masked)
        color = torch.clamp(new_col_masked, 0.0, 1.0)

        new_x_rendered = prev_x_rendered.clone()
        new_v_corrected = prev_v_corrected.clone()

        if self.mode == 'full':
            phys_deltas = self.phys_head(features)
            delta_velocity = phys_deltas[:, 0:3]
            angular_velocity = phys_deltas[:, 3:6]

            new_v_corrected_masked = original_velocity_t_sim[mask] + delta_velocity
            new_x_rendered_masked = prev_x + new_v_corrected_masked

            new_rot_masked = integrate_rotation(prev_rot, angular_velocity)
            rot_for_render = new_rot_masked

            pos_error = original_xyz_t_sim[mask] - new_x_rendered_masked
            xyz_reg_loss = torch.mean(pos_error ** 2)

            vel_error = original_velocity_t_sim[mask] - new_v_corrected_masked
            vel_reg_loss = torch.mean(vel_error ** 2)

            rot_init = self.original_rotations[mask]
            rot_error = rot_init - new_rot_masked
            rot_reg_loss = torch.mean(rot_error ** 2)

            diag_delta_v = torch.mean(torch.abs(delta_velocity))
            diag_angular_v = torch.mean(torch.abs(angular_velocity))

            new_x_rendered[mask] = new_x_rendered_masked
            new_v_corrected[mask] = new_v_corrected_masked

            new_app_state["rotation"] = prev_app_state["rotation"].clone()
            new_app_state["rotation"][mask] = new_rot_masked

        else:
            new_x_rendered_masked = original_xyz_t_sim[mask]
            new_v_corrected_masked = prev_v
            new_rot_masked = prev_rot
            rot_for_render = new_rot_masked

            new_x_rendered = prev_x_rendered
            new_v_corrected = prev_v_corrected
            xyz_reg_loss = 0.0
            rot_init = self.original_rotations[mask]
            rot_error = rot_init - new_rot_masked
            rot_reg_loss = torch.mean(rot_error ** 2) if rot_error.numel() > 0 else 0.0
            vel_reg_loss = 0.0

            diag_delta_v = torch.tensor(0.0, device=self.device)
            diag_angular_v = torch.tensor(0.0, device=self.device)
            new_app_state["rotation"] = prev_app_state["rotation"].clone()
            new_app_state["rotation"][mask] = new_rot_masked

        diag_delta_op = torch.mean(torch.abs(delta_op)) if delta_op.numel() > 0 else torch.tensor(0.0, device=self.device)
        diag_delta_sc = torch.mean(torch.abs(delta_sc)) if delta_sc.numel() > 0 else torch.tensor(0.0, device=self.device)
        diag_delta_col = torch.mean(torch.abs(delta_col)) if delta_col.numel() > 0 else torch.tensor(0.0, device=self.device)

        render_tuple = (scale, opacity, color, new_x_rendered_masked, rot_for_render, xyz_reg_loss, rot_reg_loss,
                        vel_reg_loss,
                        op_reg_loss, sc_reg_loss, col_reg_loss,
                        diag_delta_v, diag_angular_v, diag_delta_op, diag_delta_sc, diag_delta_col)

        return (new_app_state, new_x_rendered, new_v_corrected), render_tuple
