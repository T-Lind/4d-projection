import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os

@dataclass(frozen=True)
class DisplaySettings:
    pixels_per_unit: float
    background_color: Tuple[int, int, int]
    origin_color: Tuple[int, int, int]
    user_color: Tuple[int, int, int]
    default_shape_color: List[int]
    window_size: Tuple[int, int]
    
@dataclass(frozen=True)
class MovementSettings:
    acceleration: float = 0.02
    max_velocity: float = 5.0
    friction: float = 0.95
    rotate_speed: float = 0.06545  # pi/48
    user_width_pixels: int = 20
    user_height_pixels: int = 30
    bounce_factor: float = 0.5
    gravity: float = 0.01
    jump_velocity: float = 0.3;

    def get_collision_dimensions(self, pixels_per_unit: float) -> Tuple[float, float]:
        """Convert pixel dimensions to world units for collision detection"""
        return (
            self.user_width_pixels / pixels_per_unit,
            self.user_height_pixels / pixels_per_unit
        )
    
@dataclass(frozen=True)
class GameplaySettings:
    fall_threshold: float = -10.0  # Y position that triggers reset
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_color: Tuple[int, int, int] = (0, 255, 0)
    target_pulse_rate: float = 0.5  # Seconds per pulse
    target_pulse_magnitude: float = 0.3  # Color intensity variation

class Settings:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join('v5', 'shapes_config.json')
        self.config_data = self._load_config()
        self.display = self._init_display_settings()
        self.gameplay = GameplaySettings()
        self.movement = MovementSettings()
        self.shapes = self.config_data.get('shapes', [])

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            return self._get_default_config()

    def _init_display_settings(self) -> DisplaySettings:
        settings = self.config_data.get('settings', {})
        return DisplaySettings(
            pixels_per_unit=float(settings.get('pixels_per_unit', 100.0)),
            background_color=tuple(settings.get('background_color', [30, 30, 30])),
            origin_color=tuple(settings.get('origin_color', [255, 0, 0])),
            user_color=tuple(settings.get('user_color', [255, 255, 255])),
            default_shape_color=settings.get('default_shape_color', [100, 200, 255]),
            window_size=(800, 600)
        )

    def _get_default_config(self) -> dict:
        return {
            'settings': {
                'pixels_per_unit': 100.0,
                'background_color': [30, 30, 30],
                'origin_color': [255, 0, 0],
                'default_shape_color': [100, 200, 255]
            },
            'shapes': []
        }

    def get_shape_color(self, shape: dict) -> Tuple[int, int, int]:
        color_dict = shape.get('color', {})
        default = self.display.default_shape_color
        return (
            int(color_dict.get('r', default[0] / 255.0) * 255),
            int(color_dict.get('g', default[1] / 255.0) * 255),
            int(color_dict.get('b', default[2] / 255.0) * 255)
        )