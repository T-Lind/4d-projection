import pygame
import math
import sys

################################################################################
# 3D "slice" of a 4D tesseract, shown as a wireframe.
#
# - We directly define the slice as the hyperplane w=0 in 4D (a 2x2x2 cube).
# - Then rotate that slice in 4D by angle alpha in the x-w plane.
# - Render the resulting 3D shape in pygame.
#
# CONTROLS:
#   W/S: move forward/back
#   A/D: strafe left/right
#   Q/E: move down/up
#   Arrow Keys: rotate camera yaw/pitch
#   O/P: rotate hyperplane in 4D (x-w plane)
################################################################################

pygame.init()
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Slice of a 4D Tesseract (Fixed)")

clock = pygame.time.Clock()

# -----------------------------------------------------------------------------
# 1) Define the default 3D slice in 4D: a cube at w=0
# -----------------------------------------------------------------------------
# The 3D slice is a standard cube of side length 2 (corners at ±1).
# Each vertex is (±1, ±1, ±1, 0) in 4D.
slice_vertices_4d = []
for x in [-1, 1]:
    for y in [-1, 1]:
        for z in [-1, 1]:
            slice_vertices_4d.append((x, y, z, 0.0))

# Cube edges in 3D (each pair of vertices differs in exactly one coordinate):
slice_edges = []
index_map = {}
for i, v in enumerate(slice_vertices_4d):
    index_map[v] = i

# Build edges for the 3D cube
for i, v1 in enumerate(slice_vertices_4d):
    for j, v2 in enumerate(slice_vertices_4d):
        if j <= i:
            continue
        diff_count = sum(abs(a - b) for a, b in zip(v1, v2))
        # For a standard cube edge in 3D, exactly one coordinate differs by 2
        if diff_count == 2:
            slice_edges.append((i, j))

# -----------------------------------------------------------------------------
# 2) 4D rotation in x-w plane
# -----------------------------------------------------------------------------
def rotate_4d_xw(vertex, alpha):
    """
    Rotate a 4D vertex in the x-w plane by angle alpha.
    (x', y', z', w') = ( x*cos(alpha) - w*sin(alpha),
                         y,
                         z,
                         x*sin(alpha) + w*cos(alpha) )
    """
    x, y, z, w = vertex
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    x_prime = x*cos_a - w*sin_a
    w_prime = x*sin_a + w*cos_a
    return (x_prime, y, z, w_prime)

# -----------------------------------------------------------------------------
# 3) 3D camera transform and perspective projection
# -----------------------------------------------------------------------------
def rotate_3d_x(vec, angle):
    """Rotate vector in 3D around X-axis by 'angle'."""
    x, y, z = vec
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    y2 = y*cos_a - z*sin_a
    z2 = y*sin_a + z*cos_a
    return (x, y2, z2)

def rotate_3d_y(vec, angle):
    """Rotate vector in 3D around Y-axis by 'angle'."""
    x, y, z = vec
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    z2 = z*cos_a - x*sin_a
    x2 = z*sin_a + x*cos_a
    return (x2, y, z2)

def perspective_project(point_3d, fov, aspect_ratio, near, far):
    """
    Simple perspective projection of a 3D point.
    Return (x_screen, y_screen, behind_camera).
    """
    x, y, z = point_3d
    # We assume camera looks down the -Z axis (so points with z < 0 are "in front").
    if z > -0.1:  # behind or too close to camera
        return None, None, True
    
    f = 1.0 / math.tan(fov / 2.0)
    # project
    X_proj = (f / aspect_ratio) * (x / -z)
    Y_proj = f * (y / -z)
    
    x_screen = int((X_proj + 1) * 0.5 * WIDTH)
    y_screen = int((1 - Y_proj) * 0.5 * HEIGHT)
    
    return x_screen, y_screen, False

# -----------------------------------------------------------------------------
# 4) Main loop
# -----------------------------------------------------------------------------
def main():
    running = True
    
    # 4D rotation angle (in x-w plane)
    alpha = 0.0
    
    # 3D camera
    cam_x, cam_y, cam_z = 0.0, 0.0, 5.0
    yaw = 0.0
    pitch = 0.0
    
    move_speed = 0.05
    rotate_speed = 0.02
    four_d_rotate_speed = 0.02
    
    fov = math.radians(70.0)
    aspect_ratio = WIDTH / HEIGHT
    near_plane = 0.1
    far_plane = 100.0
    
    font = pygame.font.SysFont(None, 24)
    
    while running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # 4D rotation
        if keys[pygame.K_o]:
            alpha -= four_d_rotate_speed
        if keys[pygame.K_p]:
            alpha += four_d_rotate_speed
        
        # camera rotation (yaw/pitch)
        if keys[pygame.K_UP]:
            pitch += rotate_speed
        if keys[pygame.K_DOWN]:
            pitch -= rotate_speed
        if keys[pygame.K_LEFT]:
            yaw -= rotate_speed
        if keys[pygame.K_RIGHT]:
            yaw += rotate_speed
        
        # camera movement
        forward_dir = rotate_3d_y((0, 0, -1), yaw)
        right_dir   = rotate_3d_y((1, 0, 0), yaw)
        up_dir      = (0, 1, 0)
        
        if keys[pygame.K_w]:
            cam_x += forward_dir[0] * move_speed
            cam_y += forward_dir[1] * move_speed
            cam_z += forward_dir[2] * move_speed
        if keys[pygame.K_s]:
            cam_x -= forward_dir[0] * move_speed
            cam_y -= forward_dir[1] * move_speed
            cam_z -= forward_dir[2] * move_speed
        
        if keys[pygame.K_a]:
            cam_x -= right_dir[0] * move_speed
            cam_y -= right_dir[1] * move_speed
            cam_z -= right_dir[2] * move_speed
        if keys[pygame.K_d]:
            cam_x += right_dir[0] * move_speed
            cam_y += right_dir[1] * move_speed
            cam_z += right_dir[2] * move_speed
        
        if keys[pygame.K_q]:
            cam_y -= move_speed  # "down" in local vertical
        if keys[pygame.K_e]:
            cam_y += move_speed  # "up"
        
        # -----------------------------------------------------------------------------
        # Generate the 3D shape of the "rotated slice"
        # -----------------------------------------------------------------------------
        rotated_vertices_3d = []
        for v4 in slice_vertices_4d:
            # 4D rotate in x-w plane by alpha
            v4_rot = rotate_4d_xw(v4, alpha)
            # drop the w' coordinate to get a 3D point (x',y',z')
            x3, y3, z3, w3 = v4_rot
            rotated_vertices_3d.append((x3, y3, z3))
        
        # -----------------------------------------------------------------------------
        # Render
        # -----------------------------------------------------------------------------
        screen.fill((0,0,0))
        
        # Transform all points into camera space, then project
        transformed_points = []
        for (x, y, z) in rotated_vertices_3d:
            # translate
            px = x - cam_x
            py = y - cam_y
            pz = z - cam_z
            
            # rotate by camera yaw/pitch
            px, py, pz = rotate_3d_y((px, py, pz), -yaw)
            px, py, pz = rotate_3d_x((px, py, pz), -pitch)
            
            transformed_points.append((px, py, pz))
        
        projected_points = []
        for pt3d in transformed_points:
            sx, sy, behind = perspective_project(pt3d, fov, aspect_ratio, near_plane, far_plane)
            if behind:
                projected_points.append(None)
            else:
                projected_points.append((sx, sy))
        
        # Draw edges of the rotated slice
        for (i1, i2) in slice_edges:
            p1 = projected_points[i1]
            p2 = projected_points[i2]
            if p1 is not None and p2 is not None:
                pygame.draw.line(screen, (255,255,255), p1, p2, 2)
        
        # HUD text
        lines = [
            "Controls:",
            "  W/A/S/D, Q/E = move in 3D",
            "  Arrow keys   = rotate camera",
            "  O/P          = rotate slice in 4D (x-w plane)",
            f"alpha (deg) = {math.degrees(alpha):.2f}",
        ]
        for i, txt in enumerate(lines):
            surf = font.render(txt, True, (200,200,200))
            screen.blit(surf, (10, 10 + 20*i))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
