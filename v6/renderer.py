import pygame
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull
from settings import Settings

class Renderer:
    def __init__(self, settings: Settings):
        pygame.init()
        self.settings = settings
        self.screen = pygame.display.set_mode(settings.display.window_size)
        pygame.display.set_caption("Vertical Plane Slice Viewer")
        self.font = pygame.font.SysFont(None, 24)
        self.center_2D = (settings.display.window_size[0] // 2, 
                         settings.display.window_size[1] // 2)

    def clear_screen(self):
        self.screen.fill(self.settings.display.background_color)

    def draw_shapes(self, shapes: List[dict], 
                   intersection_coords_2D: List[List[Tuple[float, float]]], 
                   intersection_edges: List[List[Tuple[int, int]]]):
        for i, shape in enumerate(shapes):
            color = self.settings.get_shape_color(shape)
            coords_2d = intersection_coords_2D[i]
            edges_2d = intersection_edges[i]

            if len(coords_2d) >= 3:
                self._draw_polygon(coords_2d, color)
            elif len(coords_2d) == 2:
                self._draw_line_segment(coords_2d[0], coords_2d[1], color)

    def _draw_polygon(self, coords_2d: List[Tuple[float, float]], color: Tuple[int, int, int]):
        try:
            hull = ConvexHull(coords_2d)
            hull_indices = hull.vertices
            polygon = [coords_2d[j] for j in hull_indices]
            
            # Convert to screen coordinates
            polygon_screen = [self._to_screen_coords(pt) for pt in polygon]
            
            pygame.draw.polygon(self.screen, color, polygon_screen)
            pygame.draw.polygon(self.screen, (0, 0, 0), polygon_screen, 1)
        except:
            pass

    def _draw_line_segment(self, pt1: Tuple[float, float], pt2: Tuple[float, float], 
                          color: Tuple[int, int, int]):
        screen_pt1 = self._to_screen_coords(pt1)
        screen_pt2 = self._to_screen_coords(pt2)
        pygame.draw.line(self.screen, color, screen_pt1, screen_pt2, 2)

    def draw_origin_marker(self):
        pygame.draw.circle(self.screen, self.settings.display.origin_color, 
                         self.center_2D, 5)

    def draw_status_text(self, user_pos: np.ndarray, plane_angle: float):
        coord_text = f"User Position: (X: {user_pos[0]:.2f}, Y: {user_pos[1]:.2f}, Z: {user_pos[2]:.2f})"
        angle_degrees = np.degrees(plane_angle) % 360
        angle_text = f"Plane Angle: {angle_degrees:.1f}°"
        
        text_surface1 = self.font.render(coord_text, True, (255, 255, 255))
        text_surface2 = self.font.render(angle_text, True, (255, 255, 255))
        
        self.screen.blit(text_surface1, (10, 10))
        self.screen.blit(text_surface2, (10, 30))

    def _to_screen_coords(self, point: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(self.center_2D[0] + point[0] * self.settings.display.pixels_per_unit),
            int(self.center_2D[1] - point[1] * self.settings.display.pixels_per_unit)
        )

    def update_display(self):
        pygame.display.flip()

    def quit(self):
        pygame.quit()