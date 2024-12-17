import argparse
import json
import sys
import numpy as np
import pygame
from pygame.locals import *
from scipy.spatial import ConvexHull

#
# ------------------------------ Math / Geometry Helpers ------------------------------
#

def intersect_edge_with_plane(A, B, user_pos, plane_angle):
    """
    Intersect line segment AB with the vertical plane that passes through user_pos
    and has normal n(theta) = (cos(theta), sin(theta), 0).

    :param A: (xA, yA, zA)
    :param B: (xB, yB, zB)
    :param user_pos: (ux, uy, uz)
    :param plane_angle: float, angle around +Z axis
    :return: (xI, yI, zI) if intersection in [0,1], else None
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    U = np.array(user_pos, dtype=float)

    # Plane normal
    n = np.array([np.cos(plane_angle), np.sin(plane_angle), 0.0], dtype=float)

    AB = B - A
    AU = A - U

    denom = np.dot(n, AB)
    nom = -np.dot(n, AU)  # we want n dot ((A + t*AB) - U) = 0

    # If denom is 0, line is parallel to plane or no intersection
    if abs(denom) < 1e-12:
        return None

    t = nom / denom
    if 0.0 <= t <= 1.0:
        I = A + t * AB
        return tuple(I)
    else:
        return None

def project_point_onto_plane_2D(point_3d, user_pos, plane_angle):
    """
    Convert a 3D intersection point to 2D plane coordinates (X,Z).
    The plane is spanned by p_x(theta) = (-sin theta, cos theta, 0) (horizontal)
    and p_z = (0,0,1) (vertical).
    """
    P = np.array(point_3d, dtype=float)
    U = np.array(user_pos, dtype=float)
    relative = P - U  # vector from user to intersection

    p_x = np.array([-np.sin(plane_angle), np.cos(plane_angle), 0.0], dtype=float)  # horizontal
    p_z = np.array([0.0, 0.0, 1.0], dtype=float)                                   # vertical

    X = np.dot(relative, p_x)
    Z = np.dot(relative, p_z)
    return (X, Z)

#
# ------------------------------ Main Program ------------------------------
#

def main():
    parser = argparse.ArgumentParser(description="Vertical Plane Slice Viewer")
    parser.add_argument("file", nargs='?', default=None, help="Path to config JSON file")
    args = parser.parse_args()

    # Load config JSON
    try:
        if args.file is not None:
            with open(args.file, 'r') as f:
                config = json.load(f)
        else:
            # Fallback if no file is provided
            with open('v3/3d_demo_file.json', 'r') as f:
                config = json.load(f)
    except Exception as e:
        print("Error loading config JSON:", e)
        sys.exit(1)

    # Pygame initialization
    pygame.init()
    display_size = (800, 600)
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption("Vertical Plane Slice Viewer")

    # Define colors
    BACKGROUND_COLOR = (30, 30, 30)
    ORIGIN_COLOR = (255, 0, 0)  # Red
    DEFAULT_SHAPE_COLOR = (100, 200, 255)  # Light Blue

    # We'll draw in 2D: The plane's coordinate system is (X axis horizontal, Z axis vertical).
    # We'll define a scale factor to visually spread out the geometry on screen:
    pixels_per_unit = 100.0
    center_2D = (display_size[0] // 2, display_size[1] // 2)

    # User position in 3D
    user_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    # Plane angle around Z (0 = plane normal pointing +X, pi/2 = plane normal pointing +Y, etc.)
    plane_angle = 0.0

    # Movement parameters
    move_speed = 0.2  # Units per frame
    rotate_speed = np.pi / 48  # Radians per scroll event

    # Clock for consistent framerate
    clock = pygame.time.Clock()
    running = True

    # A function to get intersection polygons for each shape
    def compute_all_intersections():
        """
        For each shape, compute the intersection points. Then form edges by either:
        - computing the convex hull, or
        - storing the connectivity directly from the shape edges that produce valid intersections.
        """
        intersection_coords_2D = []   # list of lists, each shape's intersection in plane coords
        intersection_edges = []       # matching list of edges or hull edges

        shapes = config.get('shapes', [])
        for shape in shapes:
            pts_3d = shape['points']
            edges = shape['edges']

            # Gather intersection points
            intersection_points_3d = []
            for edge in edges:
                A_idx, B_idx = edge
                A_3d = pts_3d[A_idx]
                B_3d = pts_3d[B_idx]
                I_3d = intersect_edge_with_plane(A_3d, B_3d, user_pos, plane_angle)
                if I_3d is not None:
                    intersection_points_3d.append(I_3d)

            # Convert to 2D plane coords
            intersection_points_2d = [project_point_onto_plane_2D(pt, user_pos, plane_angle)
                                      for pt in intersection_points_3d]

            # Attempt to form a polygon via ConvexHull (if 3 or more points)
            edges_2d = []
            if len(intersection_points_2d) >= 3:
                try:
                    hull = ConvexHull(intersection_points_2d)
                    hull_indices = hull.vertices
                    # edges in hull order
                    edges_2d = [(hull_indices[i], hull_indices[(i+1) % len(hull_indices)])
                                for i in range(len(hull_indices))]
                except:
                    pass
            elif len(intersection_points_2d) == 2:
                # With exactly two intersection points, it's just a single line segment
                edges_2d = [(0,1)]

            intersection_coords_2D.append(intersection_points_2d)
            intersection_edges.append(edges_2d)

        return intersection_coords_2D, intersection_edges

    # Precompute
    intersection_coords_2D, intersection_edges = compute_all_intersections()

    # Key state tracking for smooth movement
    keys_pressed = {
        K_w: False,
        K_s: False,
        K_a: False,
        K_d: False
    }

    # Main loop
    while running:
        dt = clock.tick(60)  # Limit to 60 FPS

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

                elif event.key in keys_pressed:
                    keys_pressed[event.key] = True

            elif event.type == KEYUP:
                if event.key in keys_pressed:
                    keys_pressed[event.key] = False

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    plane_angle += rotate_speed
                    plane_angle %= 2 * np.pi
                    intersection_coords_2D, intersection_edges = compute_all_intersections()
                elif event.button == 5:  # Scroll down
                    plane_angle -= rotate_speed
                    plane_angle %= 2 * np.pi
                    intersection_coords_2D, intersection_edges = compute_all_intersections()

        # Handle continuous key presses for smooth movement
        movement_vector = np.array([0.0, 0.0, 0.0], dtype=float)

        if keys_pressed[K_w]:
            movement_vector[2] += move_speed  # Move up along Z
        if keys_pressed[K_s]:
            movement_vector[2] -= move_speed  # Move down along Z
        if keys_pressed[K_a]:
            # Move left in plane
            p_x = np.array([-np.sin(plane_angle), np.cos(plane_angle), 0.0], dtype=float)
            movement_vector += -move_speed * p_x
        if keys_pressed[K_d]:
            # Move right in plane
            p_x = np.array([-np.sin(plane_angle), np.cos(plane_angle), 0.0], dtype=float)
            movement_vector += move_speed * p_x

        if np.linalg.norm(movement_vector) > 0:
            user_pos += movement_vector
            intersection_coords_2D, intersection_edges = compute_all_intersections()

        # --- Draw 2D slice ---
        screen.fill(BACKGROUND_COLOR)  # Background color

        shapes = config.get('shapes', [])
        for i, shape in enumerate(shapes):
            color = shape.get('color', None)
            if color is None:
                # Default color
                color = DEFAULT_SHAPE_COLOR
            elif isinstance(color, list) or isinstance(color, tuple):
                if all(0.0 <= c <= 1.0 for c in color):
                    # Assuming color components are floats in [0,1]
                    color = tuple(int(255 * c) for c in color)
                elif all(0 <= c <= 255 for c in color):
                    # Assuming color components are already in [0,255]
                    color = tuple(color)
                else:
                    # Fallback to default if color format is unrecognized
                    color = DEFAULT_SHAPE_COLOR
            else:
                # Fallback to default if color format is unrecognized
                color = DEFAULT_SHAPE_COLOR

            coords_2d = intersection_coords_2D[i]
            edges_2d = intersection_edges[i]

            # Draw the edges
            for e in edges_2d:
                i1, i2 = e
                if i1 >= len(coords_2d) or i2 >= len(coords_2d):
                    continue  # Skip invalid indices
                pt1 = coords_2d[i1]
                pt2 = coords_2d[i2]

                # Convert plane coords (X,Z) to screen coords
                x1_screen = int(center_2D[0] + pt1[0] * pixels_per_unit)
                y1_screen = int(center_2D[1] - pt1[1] * pixels_per_unit)  # Y inversion for screen
                x2_screen = int(center_2D[0] + pt2[0] * pixels_per_unit)
                y2_screen = int(center_2D[1] - pt2[1] * pixels_per_unit)

                pygame.draw.line(screen, color, (x1_screen, y1_screen), (x2_screen, y2_screen), 2)

        # Draw a small marker for the user's origin in plane-coords
        pygame.draw.circle(screen, ORIGIN_COLOR, center_2D, 5)

        # Optionally, display user coordinates and plane angle
        font = pygame.font.SysFont(None, 24)
        coord_text = f"User Position: (X: {user_pos[0]:.2f}, Y: {user_pos[1]:.2f}, Z: {user_pos[2]:.2f})"
        angle_degrees = np.degrees(plane_angle) % 360
        angle_text = f"Plane Angle: {angle_degrees:.1f}°"
        text_surface1 = font.render(coord_text, True, (255, 255, 255))
        text_surface2 = font.render(angle_text, True, (255, 255, 255))
        screen.blit(text_surface1, (10, 10))
        screen.blit(text_surface2, (10, 30))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
