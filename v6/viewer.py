import numpy as np
import pygame
from pygame.locals import *
from typing import List, Tuple, Dict
from settings import Settings
from geometry import GeometryHelper
from renderer import Renderer
from level_manager import LevelManager, GameState
# from menu_manager import MenuManager
from asset_manager import AssetManager

class PlaneSliceViewer:
    def __init__(self, settings: Settings, level_manager: LevelManager):
        self.settings = settings
        self.level_manager = level_manager
        self.renderer = Renderer(settings)
        self.geometry = GeometryHelper()
        self.assets = AssetManager()
        self.assets.play_music()
        
        # State
        self.running = True
        self.user_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.plane_angle = 0.0
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # Movement tracking
        self.keys_pressed = {
            K_w: False,
            K_s: False,
            K_a: False,
            K_d: False,
            K_SPACE: False,
        }
        self.ground_contact = False
        
        # Clock for consistent framerate
        self.clock = pygame.time.Clock()
        
        # Compute initial intersections
        self._compute_all_intersections()

        self.level_complete = False
        self.spawn_position = np.array(settings.gameplay.spawn_position, dtype=float)
        self.user_pos = self.spawn_position.copy()
        self.target_pulse_time = 0

        self.points = 10000
        self.points_decrease_rate = 1
        self.jump_penalty = 100
        self.death_penalty = 1000
                
    def _check_fall_condition(self):
        """Check if player has fallen below threshold"""
        if self.user_pos[2] < self.settings.gameplay.fall_threshold:
            self._reset_player()
            self.assets.play_sound('death')
            self.points -= self.death_penalty
            
    def _reset_player(self):
        """Reset player to spawn position"""
        self.user_pos = self.spawn_position.copy()
        self.velocity = np.zeros(3, dtype=float)
        self._compute_all_intersections()
        
    def _update_target_pulse(self):
        """Update target pulsing animation"""
        self.target_pulse_time += 1/60  # Assuming 60 FPS
        pulse_factor = (np.sin(self.target_pulse_time * 2 * np.pi * 
                        self.settings.gameplay.target_pulse_rate) + 1) / 2
        self.current_pulse_factor = pulse_factor
        return pulse_factor
    
    def _adjust_user_position_after_rotation(self):
        """Move the user up if colliding after rotation."""
        max_adjustment = 10.0  # Maximum height to adjust
        adjustment_step = 0.1  # Step size for adjustment
        total_adjustment = 0.0

        while total_adjustment < max_adjustment:
            collision = False
            user_shape = self.geometry.get_user_convex_hull(self.user_pos, self.plane_angle, self.settings)
            for shape in self.settings.shapes:
                shape_hull = self.geometry.get_convex_hull(shape, self.user_pos, self.plane_angle)
                if self.geometry.check_collision(user_shape, shape_hull):
                    collision = True
                    break
            if collision:
                self.user_pos[2] += adjustment_step
                total_adjustment += adjustment_step
            else:
                break

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.running = False
                return

            if event.type == KEYDOWN and event.key in self.keys_pressed:
                self.keys_pressed[event.key] = True
                if event.key == K_SPACE or event.key == K_w:
                    self.points -= self.jump_penalty
            elif event.type == KEYUP and event.key in self.keys_pressed:
                self.keys_pressed[event.key] = False
            
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.plane_angle = (self.plane_angle + self.settings.movement.rotate_speed) % (2 * np.pi)
                    self._compute_all_intersections()
                    self._adjust_user_position_after_rotation()
                elif event.button == 5:  # Scroll down
                    self.plane_angle = (self.plane_angle - self.settings.movement.rotate_speed) % (2 * np.pi)
                    self._compute_all_intersections()
                    self._adjust_user_position_after_rotation()

    def _handle_level_completion(self):
        self.renderer.render_win_message()
        self.renderer.update_display()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_RETURN:
                    waiting = False
        self.running = False
        

    def _update_physics(self):
        movement_acceleration = np.array([0.0, 0.0, -self.settings.movement.gravity], dtype=float)
        
        # Handle jumping
        if (self.keys_pressed[K_SPACE] or self.keys_pressed[K_w]) and self.ground_contact:
            self.velocity[2] = self.settings.movement.jump_velocity
            self.ground_contact = False
            self.assets.play_sound('jump')

        if self.keys_pressed[K_s]:
            movement_acceleration[2] -= self.settings.movement.acceleration
        if self.keys_pressed[K_a]:
            p_x = np.array([-np.sin(self.plane_angle), np.cos(self.plane_angle), 0.0], dtype=float)
            movement_acceleration += -self.settings.movement.acceleration * p_x
        if self.keys_pressed[K_d]:
            p_x = np.array([-np.sin(self.plane_angle), np.cos(self.plane_angle), 0.0], dtype=float)
            movement_acceleration += self.settings.movement.acceleration * p_x

        # Update velocity
        self.velocity += movement_acceleration
        
        # Limit velocity
        speed = np.linalg.norm(self.velocity)
        if speed > self.settings.movement.max_velocity:
            self.velocity = (self.velocity / speed) * self.settings.movement.max_velocity
            
        # Apply friction
        self.velocity *= self.settings.movement.friction

        # Store previous position
        previous_pos = self.user_pos.copy()
        
        # Update position if moving
        if np.linalg.norm(self.velocity) > 0.01:
            self.user_pos += self.velocity
            self._compute_all_intersections()
        
        # Collision Detection
        user_shape = self.geometry.get_user_convex_hull(self.user_pos, self.plane_angle, self.settings)
        collision = False
        for shape in self.settings.shapes:
            shape_hull = self.geometry.get_convex_hull(shape, self.user_pos, self.plane_angle)
            if self.geometry.check_collision(user_shape, shape_hull):
                if shape.get('is_target', False):
                    self.level_complete = True

                # Get collision normal
                normal = self.geometry.get_collision_normal(user_shape, shape_hull, 
                                                         self.user_pos, np.zeros(3))
                if normal is not None:
                    # Reflect velocity off normal with bounce factor
                    normal_2d = np.array([normal[0], normal[1]])
                    velocity_2d = np.array([self.velocity[0], self.velocity[1]])
                    
                    # Reflection formula: v' = v - 2(v·n)n
                    reflected = velocity_2d - 2 * np.dot(velocity_2d, normal_2d) * normal_2d
                    
                    # Apply bounce factor
                    self.velocity[0] = reflected[0] * self.settings.movement.bounce_factor
                    self.velocity[1] = reflected[1] * self.settings.movement.bounce_factor
                    self.velocity[2] *= self.settings.movement.bounce_factor
                
                collision = True
                self.ground_contact = True
                break
                
        if collision:
            self.user_pos = previous_pos

    def _compute_all_intersections(self):
        self.intersection_coords_2D = []
        self.intersection_edges = []
        
        for shape in self.settings.shapes:
            points_2d, edges = self.geometry.compute_intersections(
                shape, self.user_pos, self.plane_angle)
            self.intersection_coords_2D.append(points_2d)
            self.intersection_edges.append(edges)

    def _update(self):
        dt = 1/60  # Fixed timestep
        if not self.level_complete:
            self._update_physics()
            self._check_fall_condition()
            self._update_target_pulse()
            self.points -= self.points_decrease_rate
            if self.points < 0:
                self.points = 0

    def _render(self):
        self.renderer.clear_screen()
        self.renderer.draw_shapes(self.settings.shapes,
                                self.intersection_coords_2D,
                                self.intersection_edges)
        
        # Target shapes with pulsing border
        target_shapes = [s for s in self.settings.shapes if s.get('is_target')]
        target_idx = [i for i, s in enumerate(self.settings.shapes) if s.get('is_target')]
        if target_shapes:
            pulse_factor = self.current_pulse_factor if hasattr(self, 'current_pulse_factor') else 1.0
            for i in target_idx:
                coords = self.intersection_coords_2D[i]
                edges = self.intersection_edges[i]
                self.renderer.draw_pulsing_target(coords, edges, pulse_factor)
        
    

        self.renderer.draw_origin_marker()
        self.renderer.draw_user()
        # Debug visualization
        user_hull = self.geometry.get_user_convex_hull(self.user_pos, self.plane_angle, self.settings)
        shape_hulls = [self.geometry.get_convex_hull(shape, self.user_pos, self.plane_angle) for shape in self.settings.shapes]
        self.renderer.draw_debug_hulls(user_hull, shape_hulls)

        if self.level_complete:
            self.renderer._render_win_message()
        
        self.renderer.draw_status_text(self.user_pos, self.plane_angle, self.points)
        self.renderer.update_display()

    def run(self):
        while self.running:
            self.clock.tick(60)
            self._handle_events()
            self._update()
            self._render()
        
        self.renderer.quit()