import argparse
import json
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial import ConvexHull

################################################################################
# 4D → 3D Hyperplane Slice Demo (Rotating about camera position)
################################################################################

"""
JSON CONFIG EXAMPLE (4d_demo_file.json):
{
    "hyperplane_offset": 0.0,
    "hyperplane_rotation_xy": 0.0,
    "hyperplane_rotation_xz": 0.0,
    "hyperplane_rotation_xw": 0.0,
    "hyperplane_normal": [0.0, 0.0, 0.0, 1.0],
    "intersection_point_color": [1.0, 1.0, 0.0],
    "shapes": [
        {
            "points": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1]
            ],
            "edges": [
                [0,1],[1,2],[2,3],[3,0],
                [4,5],[6,7],
                [0,4],[1,4],[2,4],[3,4],
                [0,5],[1,5],[2,5],[3,5],
                [0,6],[1,6],[2,6],[3,6],
                [0,7],[1,7],[2,7],[3,7]
            ],
            "color": [0.0, 1.0, 1.0]
        }
    ]
}
"""

###############################################################################
# Helper math functions
###############################################################################

def normalize(v):
    """Normalize a 4D vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return v / norm

def line_hyperplane_intersection_4d(A, B, normal, offset):
    """
    Given a line segment from A to B in 4D and a hyperplane defined by
    dot(normal, X) = offset,
    find intersection I if it lies within [A,B].
    Return None if there's no intersection within that segment.
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    n = np.array(normal, dtype=np.float64)

    denom = np.dot(n, (B - A))
    if abs(denom) < 1e-9:
        return None  # line is parallel to hyperplane

    t = (offset - np.dot(n, A)) / denom
    if 0.0 <= t <= 1.0:
        return A + t*(B - A)
    return None

def build_orthonormal_basis(normal_4d):
    """
    Build an orthonormal basis [u1, u2, u3] spanning the hyperplane
    orthogonal to 'normal_4d'.
    """
    n = normalize(normal_4d)
    basis = []
    candidate_vectors = np.eye(4)
    for c in candidate_vectors:
        # project out the component along n
        c_orth = c - np.dot(c, n)*n
        norm_c = np.linalg.norm(c_orth)
        if norm_c > 1e-9:
            c_orth /= norm_c
            # Orthogonalize w.r.t. previously found basis vectors
            for b in basis:
                dotp = np.dot(c_orth, b)
                c_orth -= dotp*b
            norm_c2 = np.linalg.norm(c_orth)
            if norm_c2 > 1e-9:
                c_orth /= norm_c2
                basis.append(c_orth)
            if len(basis) == 3:
                break
    return basis  # [u1, u2, u3]

def project_point_onto_3d_hyperplane(pt_4d, normal_4d, basis_3):
    """
    For a 4D point pt_4d that is already on the hyperplane,
    express it in the 3D basis basis_3 = [u1, u2, u3].
    Return [X, Y, Z].
    """
    return np.array([np.dot(pt_4d, b) for b in basis_3], dtype=np.float64)

def rotate_4d(vec_4d, angle, plane_indices=(0,1)):
    """
    Rotate the 4D vector 'vec_4d' in the plane given by 'plane_indices', e.g. (0,1).
    A simplistic approach for controlling orientation in 4D.
    """
    rot = np.array(vec_4d, dtype=np.float64)
    i, j = plane_indices
    c = np.cos(angle)
    s = np.sin(angle)
    x_i = rot[i]
    x_j = rot[j]
    rot[i] = c*x_i - s*x_j
    rot[j] = s*x_i + c*x_j
    return rot

###############################################################################
# Rendering helpers
###############################################################################

def draw_axes(length=1.0):
    """Draw X (red), Y (green), Z (blue) axes in 3D."""
    glBegin(GL_LINES)
    # X axis
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)
    # Y axis
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)
    # Z axis
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)
    glEnd()

def draw_sphere_3d(position, radius=0.02, slices=8, stacks=8, color=(1,1,0)):
    """Draw a small sphere at 'position' = (x,y,z) with the given color."""
    glPushMatrix()
    glTranslatef(*position)
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, slices, stacks)
    glPopMatrix()

def draw_shape_3d(points_3d, edges, color=(1,1,1)):
    """
    Draw edges of the shape in 3D.
    points_3d: Nx3
    edges: list of (i,j)
    """
    glColor3f(*color)
    glBegin(GL_LINES)
    for (i, j) in edges:
        glVertex3f(*points_3d[i])
        glVertex3f(*points_3d[j])
    glEnd()

def compute_convex_edges_3d(points_3d):
    """
    For a set of 3D points, compute the edges of the 3D convex hull.
    Return a list of (i, j) edges. This helps visualize the shape by connecting hull edges.
    """
    edges = []
    if len(points_3d) < 3:
        return edges
    try:
        hull = ConvexHull(points_3d)
        edge_set = set()
        # Each hull simplex is a triangular facet (3 vertices).
        for simplex in hull.simplices:
            for k in range(len(simplex)):
                v1 = simplex[k]
                v2 = simplex[(k+1) % len(simplex)]
                edge = tuple(sorted([v1, v2]))
                edge_set.add(edge)
        edges = list(edge_set)
    except:
        pass
    return edges

###############################################################################
# 2D text overlay
###############################################################################

def draw_text_overlay(display, font, lines, color=(255,255,255)):
    """
    Renders a list of text lines on top of the OpenGL scene using pygame.
    """
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)

    screen = pygame.display.get_surface()
    line_height = font.get_linesize()
    x, y = 10, 10
    for line in lines:
        text_surface = font.render(line, True, color)
        screen.blit(text_surface, (x, y))
        y += line_height

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

###############################################################################
# Main code
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs='?', default="4d_demo_file.json", help="4D config JSON file")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.file, 'r') as f:
            config = json.load(f)
    except:
        print("Failed to load config. Using defaults.")
        config = {
            "hyperplane_offset": 0.0,
            "hyperplane_rotation_xy": 0.0,
            "hyperplane_rotation_xz": 0.0,
            "hyperplane_rotation_xw": 0.0,
            "hyperplane_normal": [0, 0, 0, 1],
            "intersection_point_color": [1.0, 1.0, 0.0],
            "shapes": []
        }

    # Extract config
    hyperplane_offset = float(config.get("hyperplane_offset", 0.0))
    rot_xy = float(config.get("hyperplane_rotation_xy", 0.0))
    rot_xz = float(config.get("hyperplane_rotation_xz", 0.0))
    rot_xw = float(config.get("hyperplane_rotation_xw", 0.0))
    base_hyperplane_normal = np.array(config.get("hyperplane_normal", [0, 0, 0, 1]), dtype=float)
    intersection_point_color = config.get("intersection_point_color", [1.0, 1.0, 0.0])
    shapes_4d = config.get("shapes", [])

    # Initialize pygame/OpenGL
    pygame.init()
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("4D->3D slicing demo (Rotating hyperplane about camera)")
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    # Font for text overlay
    pygame.font.init()
    overlay_font = pygame.font.SysFont('Arial', 16, bold=False)

    # Camera variables
    camera_pos = np.array([0.0, 0.0, 5.0], dtype=float)
    camera_rot_x = 0.0
    camera_rot_y = 0.0
    mouse_down = False
    last_mouse_pos = (0, 0)

    # If shape has no color, default to neutral gray
    for shape in shapes_4d:
        if "color" not in shape:
            shape["color"] = [0.7, 0.7, 0.7]

    def update_hyperplane_normal():
        """
        Build the hyperplane normal by applying 4D rotations to base_hyperplane_normal
        according to rot_xy, rot_xz, rot_xw.
        """
        n = np.copy(base_hyperplane_normal)
        # XY-plane rotation
        if abs(rot_xy) > 1e-9:
            n = rotate_4d(n, rot_xy, (0, 1))
        # XZ-plane rotation
        if abs(rot_xz) > 1e-9:
            n = rotate_4d(n, rot_xz, (0, 2))
        # XW-plane rotation
        if abs(rot_xw) > 1e-9:
            n = rotate_4d(n, rot_xw, (0, 3))
        return normalize(n)

    def slice_4d_shapes():
        """
        For each shape in 4D, compute intersection points with the 3D hyperplane
        rotating about the camera's 4D position. Return list of (points_3d, edges_3d, shape_color).
        """
        normal_4d = update_hyperplane_normal()
        
        # Convert camera_pos to 4D by appending w=0
        camera_4d = np.array([camera_pos[0], camera_pos[1], camera_pos[2], 0.0], dtype=float)
        # This offset ensures the hyperplane rotates about the camera position
        dynamic_offset = np.dot(normal_4d, camera_4d) + hyperplane_offset

        basis_3 = build_orthonormal_basis(normal_4d)
        shape_slices = []

        for shape in shapes_4d:
            pts_4d = shape["points"]
            edges = shape["edges"]
            shape_color = shape["color"]

            intersects_4d = []
            for e in edges:
                A_4d = pts_4d[e[0]]
                B_4d = pts_4d[e[1]]
                I_4d = line_hyperplane_intersection_4d(A_4d, B_4d, normal_4d, dynamic_offset)
                if I_4d is not None:
                    intersects_4d.append(I_4d)

            if len(intersects_4d) == 0:
                shape_slices.append(([], [], shape_color))
                continue

            # Project intersection points onto 3D
            pts_3d = [project_point_onto_3d_hyperplane(pt, normal_4d, basis_3) for pt in intersects_4d]
            pts_3d = np.array(pts_3d, dtype=np.float64)

            # Build edges from 3D convex hull
            edges_3d = compute_convex_edges_3d(pts_3d)
            shape_slices.append((pts_3d, edges_3d, shape_color))

        return shape_slices

    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(60)  # ~60 fps
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    last_mouse_pos = event.pos
                elif event.button == 4:  # scroll up
                    hyperplane_offset += 0.1
                elif event.button == 5:  # scroll down
                    hyperplane_offset -= 0.1
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == MOUSEMOTION:
                if mouse_down:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    camera_rot_x += dy * 0.2
                    camera_rot_y += dx * 0.2
                    last_mouse_pos = event.pos
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        # Continuous key handling
        keys = pygame.key.get_pressed()
        move_speed = 0.05
        rotate_speed = 0.03

        # Camera movement
        if keys[K_w]:
            camera_pos[2] -= move_speed
        if keys[K_s]:
            camera_pos[2] += move_speed
        if keys[K_a]:
            camera_pos[0] -= move_speed
        if keys[K_d]:
            camera_pos[0] += move_speed
        if keys[K_q]:
            camera_pos[1] += move_speed
        if keys[K_e]:
            camera_pos[1] -= move_speed

        # Hyperplane rotations (about the camera pos in 4D)
        if keys[K_i]:
            rot_xy += rotate_speed
        if keys[K_k]:
            rot_xy -= rotate_speed
        if keys[K_j]:
            rot_xz += rotate_speed
        if keys[K_l]:
            rot_xz -= rotate_speed
        if keys[K_u]:
            rot_xw += rotate_speed
        if keys[K_o]:
            rot_xw -= rotate_speed

        shape_slices = slice_4d_shapes()

        # Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera transform
        glTranslatef(-camera_pos[0], -camera_pos[1], -camera_pos[2])
        glRotatef(camera_rot_x, 1, 0, 0)
        glRotatef(camera_rot_y, 0, 1, 0)

        # Draw axes
        draw_axes(length=1.0)

        # Draw intersection geometry
        for (pts_3d, edges_3d, shape_color) in shape_slices:
            if len(pts_3d) == 0:
                continue
            draw_shape_3d(pts_3d, edges_3d, color=shape_color)
            for p3 in pts_3d:
                draw_sphere_3d(p3, 0.03, 8, 8, color=intersection_point_color)

        # Visualize the hyperplane normal arrow ignoring w component
        normal_4d = update_hyperplane_normal()
        glBegin(GL_LINES)
        glColor3f(1, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(normal_4d[0], normal_4d[1], normal_4d[2])  # ignoring w
        glEnd()

        # On-screen text
        controls_text = [
            "Controls:",
            "  W/S: Move camera forward/back",
            "  A/D: Move camera left/right",
            "  Q/E: Move camera up/down",
            "  Mouse drag: Rotate camera",
            "  I/K: Rotate hyperplane in XY plane (about camera)",
            "  J/L: Rotate hyperplane in XZ plane (about camera)",
            "  U/O: Rotate hyperplane in XW plane (about camera)",
            "  Scroll: Move hyperplane offset near/far",
            "  ESC: Quit",
            "",
            f"Hyperplane offset: {hyperplane_offset:.2f}",
            f"Camera pos: {camera_pos}"
        ]
        draw_text_overlay(display, overlay_font, controls_text, color=(255,255,255))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
