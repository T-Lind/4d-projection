import pygame
import numpy as np
from math import cos, sin
import sys

class Tesseract:
    def __init__(self):
        # Define vertices of a tesseract (16 points in 4D space)
        self.vertices_4d = np.array([
            [x, y, z, w] for x in [-1, 1] 
                        for y in [-1, 1]
                        for z in [-1, 1]
                        for w in [-1, 1]
        ], dtype=float)
        
        # Define edges - pairs of vertex indices that form lines
        self.edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                     if sum(abs(self.vertices_4d[i] - self.vertices_4d[j]) < 0.1) == 3]

class FourDRenderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("4D Tesseract Visualization")
        
        self.clock = pygame.time.Clock()
        self.tesseract = Tesseract()
        
        # Camera/view parameters
        self.scale = 100
        self.w_slice = 0
        self.angles_3d = [0, 0, 0]  # XY, YZ, XZ rotations
        self.angles_4d = [0, 0, 0, 0]  # XW, YW, ZW rotations
        self.position = [width//2, height//2, 0]
        
    def rotate_4d(self, points, angles_4d):
        # Create rotation matrices for 4D rotations
        rotations = []
        
        # XW rotation
        rot_xw = np.eye(4)
        c, s = cos(angles_4d[0]), sin(angles_4d[0])
        rot_xw[[0, 3], [0, 3]] = c
        rot_xw[[0, 3], [3, 0]] = [-s, s]
        rotations.append(rot_xw)
        
        # YW rotation
        rot_yw = np.eye(4)
        c, s = cos(angles_4d[1]), sin(angles_4d[1])
        rot_yw[[1, 3], [1, 3]] = c
        rot_yw[[1, 3], [3, 1]] = [-s, s]
        rotations.append(rot_yw)
        
        # ZW rotation
        rot_zw = np.eye(4)
        c, s = cos(angles_4d[2]), sin(angles_4d[2])
        rot_zw[[2, 3], [2, 3]] = c
        rot_zw[[2, 3], [3, 2]] = [-s, s]
        rotations.append(rot_zw)
        
        result = points.copy()
        for rotation in rotations:
            result = np.dot(result, rotation)
        return result

    def project_4d_to_3d(self, points_4d, w_slice):
        # Project 4D points to 3D by intersecting with w=w_slice hyperplane
        scale = 1 / (4 - points_4d[:, 3])
        points_3d = points_4d[:, :3] * scale[:, np.newaxis]
        return points_3d

    def rotate_3d(self, points, angles):
        # Apply 3D rotations
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
        # Simple perspective projection
        scale = 1000 / (5 - points_3d[:, 2])
        points_2d = points_3d[:, :2] * scale[:, np.newaxis]
        return points_2d

    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # Apply 4D rotations
        rotated_4d = self.rotate_4d(self.tesseract.vertices_4d, self.angles_4d)
        
        # Project to 3D
        points_3d = self.project_4d_to_3d(rotated_4d, self.w_slice)
        
        # Apply 3D rotations
        points_3d = self.rotate_3d(points_3d, self.angles_3d)
        
        # Project to 2D
        points_2d = self.project_3d_to_2d(points_3d)
        
        # Draw edges
        for edge in self.tesseract.edges:
            start = points_2d[edge[0]]
            end = points_2d[edge[1]]
            start_pos = (int(start[0] + self.position[0]), 
                        int(start[1] + self.position[1]))
            end_pos = (int(end[0] + self.position[0]), 
                      int(end[1] + self.position[1]))
            pygame.draw.line(self.screen, (0, 255, 0), start_pos, end_pos, 1)
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            keys = pygame.key.get_pressed()
            
            # 4D rotations
            if keys[pygame.K_q]: self.angles_4d[0] += 0.02
            if keys[pygame.K_w]: self.angles_4d[0] -= 0.02
            if keys[pygame.K_a]: self.angles_4d[1] += 0.02
            if keys[pygame.K_s]: self.angles_4d[1] -= 0.02
            if keys[pygame.K_z]: self.angles_4d[2] += 0.02
            if keys[pygame.K_x]: self.angles_4d[2] -= 0.02
            
            # 3D rotations
            if keys[pygame.K_r]: self.angles_3d[0] += 0.02
            if keys[pygame.K_t]: self.angles_3d[0] -= 0.02
            if keys[pygame.K_f]: self.angles_3d[1] += 0.02
            if keys[pygame.K_g]: self.angles_3d[1] -= 0.02
            if keys[pygame.K_v]: self.angles_3d[2] += 0.02
            if keys[pygame.K_b]: self.angles_3d[2] -= 0.02
            
            # W-slice control
            if keys[pygame.K_UP]: self.w_slice += 0.1
            if keys[pygame.K_DOWN]: self.w_slice -= 0.1
            
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