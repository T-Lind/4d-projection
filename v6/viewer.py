import numpy as np
import pygame
from pygame.locals import *
from typing import List, Tuple, Dict
from settings import Settings
from geometry import GeometryHelper
from renderer import Renderer

class PlaneSliceViewer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.renderer = Renderer(settings)
        self.geometry = GeometryHelper()
        
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
            K_d: False
        }
        
        # Clock for consistent framerate
        self.clock = pygame.time.Clock()
        
        # Compute initial intersections
        self._compute_all_intersections()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.running = False
                return

            if event.type == KEYDOWN and event.key in self.keys_pressed:
                self.keys_pressed[event.key] = True
            elif event.type == KEYUP and event.key in self.keys_pressed:
                self.keys_pressed[event.key] = False
            
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.plane_angle = (self.plane_angle + self.settings.movement.rotate_speed) % (2 * np.pi)
                    self._compute_all_intersections()
                elif event.button == 5:  # Scroll down
                    self.plane_angle = (self.plane_angle - self.settings.movement.rotate_speed) % (2 * np.pi)
                    self._compute_all_intersections()

    def _update_physics(self):
        movement_acceleration = np.array([0.0, 0.0, 0.0], dtype=float)
        
        if self.keys_pressed[K_w]:
            movement_acceleration[2] += self.settings.movement.acceleration
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

        # Update position if moving
        if np.linalg.norm(self.velocity) > 0.01:
            self.user_pos += self.velocity
            self._compute_all_intersections()
        else:
            self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)

    def _compute_all_intersections(self):
        self.intersection_coords_2D = []
        self.intersection_edges = []
        
        for shape in self.settings.shapes:
            points_2d, edges = self.geometry.compute_intersections(
                shape, self.user_pos, self.plane_angle)
            self.intersection_coords_2D.append(points_2d)
            self.intersection_edges.append(edges)

    def _render(self):
        self.renderer.clear_screen()
        self.renderer.draw_shapes(self.settings.shapes, 
                                self.intersection_coords_2D, 
                                self.intersection_edges)
        self.renderer.draw_origin_marker()
        self.renderer.draw_status_text(self.user_pos, self.plane_angle)
        self.renderer.update_display()

    def run(self):
        while self.running:
            self.clock.tick(60)
            self._handle_events()
            self._update_physics()
            self._render()
        
        self.renderer.quit()