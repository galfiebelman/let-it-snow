from effects.fog_effect import FogEffectRenderer
from effects.rain_effect import RainEffectRenderer
from effects.sandstorm_effect import SandstormEffectRenderer
from effects.snow_effect import SnowEffectRenderer


class EffectRenderer:
    def __init__(self, effect_type, mesh_path, simulation_dir, config_path, bg_path=""):
        self.effect_type = effect_type
        if self.effect_type == "snow":
            self.renderer = SnowEffectRenderer(mesh_path, simulation_dir, config_path, bg_path=bg_path)
        elif self.effect_type == "rain":
            self.renderer = RainEffectRenderer(mesh_path, simulation_dir, config_path)
        elif self.effect_type == "fog":
            self.renderer = FogEffectRenderer(mesh_path, simulation_dir, config_path)
        elif self.effect_type == "sand":
            self.renderer = SandstormEffectRenderer(mesh_path, simulation_dir, config_path)
        else:
            raise ValueError(f"{effect_type} effect not implemented")
