import pygame
import numpy as np
from math import cos, sin
import sys
from shapes import Shape4D, load_4ds, ShapeLoader

class Tesseract(Shape4D):
    def __init__(self):
        vertices = np.array([
            [x, y, z, w] for x in [-1, 1] 
                        for y in [-1, 1]
                        for z in [-1, 1]
                        for w in [-1, 1]
        ], dtype=float)
        
        edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                if sum(abs(vertices[i] - vertices[j]) < 0.1) == 3]
        
        faces = [
            # Front cube (w = -1)
            (0, 1, 3, 2), (0, 2, 6, 4), (0, 1, 5, 4),
            (1, 3, 7, 5), (2, 3, 7, 6), (4, 5, 7, 6),
            # Back cube (w = 1)
            (8, 9, 11, 10), (8, 10, 14, 12), (8, 9, 13, 12),
            (9, 11, 15, 13), (10, 11, 15, 14), (12, 13, 15, 14),
            # Connecting faces
            (0, 1, 9, 8), (2, 3, 11, 10), (4, 5, 13, 12),
            (6, 7, 15, 14), (0, 2, 10, 8), (1, 3, 11, 9),
            (4, 6, 14, 12), (5, 7, 15, 13)
        ]
        
        # Colors for faces
        face_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (255, 0, 255), (0, 255, 255)] * 4

        super().__init__(vertices, edges, faces, face_colors)

class LightSource:
    def __init__(self, position, color, intensity, falloff=1.0):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float) / 255.0
        self.intensity = float(intensity)
        self.falloff = float(falloff)

class Material:
    def __init__(self, ambient=0.2, diffuse=0.7, specular=0.5, shininess=32):
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.specular = float(specular)
        self.shininess = float(shininess)

class FourDRenderer:
    def __init__(self, shapes=None, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("4D Shape Visualization")
        
        self.clock = pygame.time.Clock()
        if not shapes:
            self.shape = Tesseract()  # Default shape
        else:
            self.shape = shapes[0]  # Only can load one shape for the moment.
        
        self.scale = 100
        self.angles_3d = [0, 0, 0]  # XY, YZ, XZ rotations
        self.angles_4d = [0, 0, 0, 0]  # XW, YW, ZW rotations
        self.position = [width//2, height//2, 0]

        self.lights = [
            LightSource(
                position=[500, -300, 1000],
                color=[255, 255, 255],
                intensity=5.0
            )
        ]
        self.default_material = Material()

    def calculate_lighting(self, point, normal, material, view_pos, debug=False):
        """Calculate lighting for a point using Phong illumination"""
        final_color = np.zeros(3)
        
        # Debug prints
        if debug:
            print(f"Point: {point}")
            print(f"Normal: {normal}")
        
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude == 0:
            return np.ones(3)
        normal = normal / normal_magnitude
        
        view_dir = view_pos - point
        view_magnitude = np.linalg.norm(view_dir)
        if view_magnitude == 0:
            view_dir = np.array([0, 0, 1])
        else:
            view_dir = view_dir / view_magnitude

        for light in self.lights:
            # Vec to light
            light_dir = light.position - point
            distance = np.linalg.norm(light_dir)
            if distance == 0:
                continue
            light_dir = light_dir / distance
            
            # Decr falloff effect
            attenuation = 1.0 / (1.0 + light.falloff * distance * 0.001)
            
            # Incr ambient light
            ambient = material.ambient * light.color * light.intensity * 0.5
            
            # Calculate diffuse
            diff = max(np.dot(normal, light_dir), 0.0)
            diffuse = material.diffuse * diff * light.color * light.intensity
            
            # Calculate specular
            reflect_dir = 2.0 * np.dot(normal, light_dir) * normal - light_dir
            spec = max(np.dot(view_dir, reflect_dir), 0.0)
            if spec > 0:
                spec = pow(spec, material.shininess)
            specular = material.specular * spec * light.color * light.intensity
            
            if debug:
                # Print debug info
                print(f"Light contribution: ambient={ambient}, diffuse={diffuse}, specular={specular}")
            
            final_color += (ambient + diffuse + specular) * attenuation

        # Min brightness defined here
        final_color = np.maximum(final_color, 0.2)
        return np.clip(final_color, 0, 1)
        
    def load_shape(self, filename: str):
        self.shape = load_4ds(filename)
        
    def rotate_4d(self, points, angles_4d):
        rotations = []
        
        # XW rot
        rot_xw = np.eye(4)
        c, s = cos(angles_4d[0]), sin(angles_4d[0])
        rot_xw[[0, 3], [0, 3]] = c
        rot_xw[[0, 3], [3, 0]] = [-s, s]
        rotations.append(rot_xw)
        
        # YW rot
        rot_yw = np.eye(4)
        c, s = cos(angles_4d[1]), sin(angles_4d[1])
        rot_yw[[1, 3], [1, 3]] = c
        rot_yw[[1, 3], [3, 1]] = [-s, s]
        rotations.append(rot_yw)
        
        # ZW rot
        rot_zw = np.eye(4)
        c, s = cos(angles_4d[2]), sin(angles_4d[2])
        rot_zw[[2, 3], [2, 3]] = c
        rot_zw[[2, 3], [3, 2]] = [-s, s]
        rotations.append(rot_zw)
        
        result = points.copy()
        for rotation in rotations:
            result = np.dot(result, rotation)
        return result

    def project_4d_to_3d(self, points_4d):
        scale = 1 / (4 - points_4d[:, 3])
        points_3d = points_4d[:, :3] * scale[:, np.newaxis]
        return points_3d

    def rotate_3d(self, points, angles):
        for axis in range(3):
            c, s = cos(angles[axis]), sin(angles[axis])
            if axis == 0:  # XY
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            elif axis == 1:  # YZ
                rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            else:  # XZ
                rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
            points = np.dot(points, rotation)
        return points

    def project_3d_to_2d(self, points_3d):
        scale = 1000 / (5 - points_3d[:, 2])
        points_2d = points_3d[:, :2] * scale[:, np.newaxis]
        return points_2d

    def calculate_face_normal(self, points_3d, face):
        # Calculate normal vector for face culling
        v1 = points_3d[face[1]] - points_3d[face[0]]
        v2 = points_3d[face[2]] - points_3d[face[0]]
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)

    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # Apply 4D rotations
        rotated_4d = self.rotate_4d(self.shape.vertices, self.angles_4d)
        
        # Project to 3D
        points_3d = self.project_4d_to_3d(rotated_4d)
        
        # Apply 3D rotations
        points_3d = self.rotate_3d(points_3d, self.angles_3d)
        
        # Sort faces by depth for painter's algorithm
        face_depths = [(face, np.mean(points_3d[list(face)][:, 2])) 
                      for face, _ in zip(self.shape.faces, self.shape.face_colors)]
        face_depths.sort(key=lambda x: x[1])
        
        # Project to 2D
        points_2d = self.project_3d_to_2d(points_3d)

        face_depths = [(face, np.mean(points_3d[list(face)][:, 2])) 
                    for face, _ in zip(self.shape.faces, self.shape.face_colors)]
        # Sort faces back-to-front for corret overlay
        face_depths.sort(key=lambda x: -x[1])  # Neg sign for reverse sort

        for face, depth in face_depths:
            normal = self.calculate_face_normal(points_3d, face)
            if normal[2] < -0.1:
                continue
                
            # Center point of face for lighting
            face_center = np.mean(points_3d[list(face)], axis=0)
            view_pos = np.array([0, 0, 1000])  # Camera position
            
            # Get base color and convert to float RGB
            base_color = np.array(self.shape.face_colors[self.shape.faces.index(face)][:3]) / 255.0
            
            # Calc lighting
            light_intensity = self.calculate_lighting(
                face_center, 
                normal, 
                self.default_material,
                view_pos
            )
            
            # Apply lighting to base color
            final_color = np.clip(base_color * light_intensity, 0, 1)
            
            # Convert back to 8-bit RGB
            render_color = tuple(int(c * 255) for c in final_color)
            
            # Draw the face
            face_points = [(int(points_2d[i][0] + self.position[0]),
                        int(points_2d[i][1] + self.position[1]))
                        for i in face]
            pygame.draw.polygon(self.screen, render_color, face_points)
            # anti-aliased edges
            for i in range(len(face_points)):
                start = face_points[i]
                end = face_points[(i + 1) % len(face_points)]
                pygame.draw.aaline(self.screen, render_color, start, end)


        edges_with_depth = []
        for edge in self.shape.edges:
            start_idx, end_idx = edge
            start_3d = points_3d[start_idx]
            end_3d = points_3d[end_idx]
            edge_depth = (start_3d[2] + end_3d[2]) / 2
            start_2d = points_2d[start_idx]
            end_2d = points_2d[end_idx]
            edges_with_depth.append((edge_depth, start_2d, end_2d))

        # Sort faces and edges by depth
        face_depths = self.shape.get_projected_face_depths(points_3d)
        sorted_faces = sorted(zip(face_depths, self.shape.faces), key=lambda x: x[0])
        edges_with_depth.sort(key=lambda x: x[0])

        # Draw visible edges
        for edge_depth, start_2d, end_2d in edges_with_depth:
            # See if edge is behind any face
            is_visible = True
            edge_start = np.array([start_2d[0], start_2d[1]])
            edge_end = np.array([end_2d[0], end_2d[1]])
            
            for face_depth, face in sorted_faces:
                if face_depth < edge_depth:  # Face is in front of edge
                    # Project face points to 2D
                    face_points_2d = [np.array([points_2d[i][0], points_2d[i][1]]) 
                                    for i in face]
                    
                    if self.shape.edge_intersects_face(edge_start, edge_end, face_points_2d):
                        is_visible = False
                        break
            
            if is_visible:
                start_pos = (int(start_2d[0] + self.position[0]), 
                            int(start_2d[1] + self.position[1]))
                end_pos = (int(end_2d[0] + self.position[0]), 
                        int(end_2d[1] + self.position[1]))
                pygame.draw.aaline(self.screen, (255, 255, 255), start_pos, end_pos)


        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            keys = pygame.key.get_pressed()
            
            # 4D rots
            if keys[pygame.K_q]: self.angles_4d[0] += 0.02
            if keys[pygame.K_w]: self.angles_4d[0] -= 0.02
            if keys[pygame.K_a]: self.angles_4d[1] += 0.02
            if keys[pygame.K_s]: self.angles_4d[1] -= 0.02
            if keys[pygame.K_z]: self.angles_4d[2] += 0.02
            if keys[pygame.K_x]: self.angles_4d[2] -= 0.02
            
            # 3D rots
            if keys[pygame.K_r]: self.angles_3d[0] += 0.02
            if keys[pygame.K_t]: self.angles_3d[0] -= 0.02
            if keys[pygame.K_f]: self.angles_3d[1] += 0.02
            if keys[pygame.K_g]: self.angles_3d[1] -= 0.02
            if keys[pygame.K_v]: self.angles_3d[2] += 0.02
            if keys[pygame.K_b]: self.angles_3d[2] -= 0.02
            
            # Translation
            if keys[pygame.K_LEFT]: self.position[0] -= 5
            if keys[pygame.K_RIGHT]: self.position[0] += 5
            if keys[pygame.K_PAGEUP]: self.position[1] -= 5
            if keys[pygame.K_PAGEDOWN]: self.position[1] += 5
            
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    renderer = FourDRenderer()
    renderer.run()