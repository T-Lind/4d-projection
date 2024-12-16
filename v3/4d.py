import argparse
import json
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial import ConvexHull
from random import random as rand

################################################################################
# 4D → 3D Hyperplane Slice Demo
################################################################################

"""
JSON CONFIG EXAMPLE (4d_demo_file.json):
{
    "hyperplane_offset": 0.0,
    "hyperplane_rotation_xy": 0.0,
    "hyperplane_rotation_xz": 0.0,
    "hyperplane_rotation_xw": 0.0,
    "hyperplane_normal": [0.0, 0.0, 0.0, 1.0],
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
                [0,1],[1,2],[2,3],[3,0],  // XY square in w=0 plane
                [4,5],[6,7],            // Z-line, W-line
                [0,4],[1,4],[2,4],[3,4],  // Connect square to z=+1
                [0,5],[1,5],[2,5],[3,5],  // Connect square to z=-1
                [0,6],[1,6],[2,6],[3,6],  // Connect square to w=+1
                [0,7],[1,7],[2,7],[3,7]   // Connect square to w=-1
            ]
        }
    ]
}
"""

###############################################################################
# Helper math functions
###############################################################################

def normalize(v):
    """Normalize a 4D vector."""
    return v / np.linalg.norm(v)

def line_hyperplane_intersection_4d(A, B, normal, offset):
    """
    Given a line segment A->B in 4D and a hyperplane defined by
    dot(normal, X) = offset,
    find intersection I if it lies within the segment.
    Returns None if there's no intersection within [A, B].
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    n = np.array(normal, dtype=np.float64)

    denom = np.dot(n, B - A)
    if abs(denom) < 1e-9:
        return None  # line is parallel or nearly parallel to hyperplane

    t = (offset - np.dot(n, A)) / denom
    if 0.0 <= t <= 1.0:  # intersection within the segment
        return A + t * (B - A)
    return None

def build_orthonormal_basis(normal_4d):
    """
    Build an orthonormal basis {u1, u2, u3} spanning the hyperplane
    orthogonal to 'normal_4d'. We'll use Gram-Schmidt or a simple approach:
    
    We want vectors in R^4 that are orthonormal and each orthogonal to 'normal_4d'.
    """
    n = normalize(normal_4d)
    # We'll attempt to find 3 linearly independent vectors orthonormal to n.
    # A simple approach: start with standard basis vectors e0,e1,e2,e3
    # and pick the first three that remain linearly independent after we remove their
    # component along n.
    basis = []
    candidate_vectors = np.eye(4)
    for c in candidate_vectors:
        # project out the component along n
        c_orth = c - np.dot(c, n)*n
        norm_c = np.linalg.norm(c_orth)
        if norm_c > 1e-9:
            c_orth = c_orth / norm_c
            # Make sure it's orthonormal w.r.t. previously found basis vectors
            for b in basis:
                # project out any existing component
                dotp = np.dot(c_orth, b)
                c_orth = c_orth - dotp*b
            norm_c2 = np.linalg.norm(c_orth)
            if norm_c2 > 1e-9:
                c_orth = c_orth / norm_c2
                basis.append(c_orth)
            if len(basis) == 3:
                break
    return basis  # a list [u1, u2, u3]

def project_point_onto_3d_hyperplane(pt_4d, normal_4d, offset, basis_3):
    """
    Take a 4D point that is on the hyperplane dot(normal_4d, x) = offset,
    and express it in the 3D coordinate system spanned by basis_3 = [u1, u2, u3].
    Return [X, Y, Z] in that local coordinate system.
    """
    # If the point is truly in the hyperplane, dot(normal, pt_4d) ~ offset
    # We'll just compute: coords = (pt dot u1, pt dot u2, pt dot u3)
    return np.array([np.dot(pt_4d, b) for b in basis_3], dtype=np.float64)

def rotate_4d(normal_4d, angle, plane_indices=(0,1)):
    """
    Rotate the 4D normal vector in the plane spanned by the given plane_indices,
    e.g. (0,1) = rotation in X-Y plane of the 4D vector, etc.
    This is a simplistic approach for controlling the hyperplane orientation.
    """
    rot = np.array(normal_4d)
    i, j = plane_indices
    c = np.cos(angle)
    s = np.sin(angle)
    # rotation in the (i,j) subplane:
    x_i = rot[i]
    x_j = rot[j]
    rot[i] = c*x_i - s*x_j
    rot[j] = s*x_i + c*x_j
    return rot

###############################################################################
# Rendering helpers
###############################################################################

def draw_axes(length=1.0):
    """Draw X, Y, Z axes in 3D."""
    glBegin(GL_LINES)
    # X axis in red
    glColor3f(1,0,0)
    glVertex3f(0,0,0)
    glVertex3f(length,0,0)
    # Y axis in green
    glColor3f(0,1,0)
    glVertex3f(0,0,0)
    glVertex3f(0,length,0)
    # Z axis in blue
    glColor3f(0,0,1)
    glVertex3f(0,0,0)
    glVertex3f(0,0,length)
    glEnd()

def draw_sphere_3d(position, radius=0.02, slices=8, stacks=8, color=(1,1,1)):
    """Draw a small sphere at 'position' = (x,y,z)."""
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
    For a set of 3D points, compute the edges of their 3D convex hull.
    Return a list of (i, j) edges.
    """
    edges = []
    if len(points_3d) < 4:
        # For fewer than 4 points, we can just connect them in a loop or something simpler
        # But let's try a 3D hull anyway
        pass
    try:
        hull = ConvexHull(points_3d)
        # The hull simplices are triangular facets. 
        # We build edges from each facet's vertices.
        edge_set = set()
        for simplex in hull.simplices:
            # simplex is a triple of indices
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
            "hyperplane_normal": [0.0, 0.0, 0.0, 1.0],
            "shapes": []
        }

    hyperplane_normal = np.array(config.get("hyperplane_normal", [0,0,0,1]), dtype=float)
    hyperplane_offset = float(config.get("hyperplane_offset", 0.0))
    rot_xy = float(config.get("hyperplane_rotation_xy", 0.0))
    rot_xz = float(config.get("hyperplane_rotation_xz", 0.0))
    rot_xw = float(config.get("hyperplane_rotation_xw", 0.0))

    shapes_4d = config.get("shapes", [])

    # Initialize pygame/OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("4D->3D slicing demo")
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    # Camera vars
    camera_pos = np.array([0.0, 0.0, 5.0], dtype=float)
    camera_rot_x = 0.0
    camera_rot_y = 0.0
    mouse_down = False
    last_mouse_pos = (0,0)

    auto_rotate = False

    # Precompute shape edges in 4D for convenience
    # shapes_4d is a list of { "points":[...], "edges":[(p1,p2)...], "color":[r,g,b] }
    # We don't do intersection yet. That happens each frame so we can re-slice.
    
    def update_hyperplane_normal():
        """
        Update the hyperplane normal based on the rotation angles in XY, XZ, XW planes.
        This is a simplistic approach to 4D rotation. 
        """
        base_normal = np.array([0,0,0,1], dtype=float)
        # Start from base normal (0,0,0,1), apply XY rotation, then XZ rotation, then XW rotation
        # so that the user can manipulate the orientation in 4D:
        n = np.copy(base_normal)
        if rot_xy != 0.0:
            n = rotate_4d(n, rot_xy, plane_indices=(0,1))
        if rot_xz != 0.0:
            n = rotate_4d(n, rot_xz, plane_indices=(0,2))
        if rot_xw != 0.0:
            n = rotate_4d(n, rot_xw, plane_indices=(0,3))
        return normalize(n)

    def slice_4d_shapes():
        """
        For each shape in 4D, compute intersection points with the 3D hyperplane,
        and project them to 3D for rendering. Then produce edges (via hull or connectivity).
        """
        normal_4d = update_hyperplane_normal()
        intersection_points_4d = []
        for shape in shapes_4d:
            pts = shape["points"]
            edges = shape["edges"]
            intersects = []
            for e in edges:
                A_4d = pts[e[0]]
                B_4d = pts[e[1]]
                I_4d = line_hyperplane_intersection_4d(A_4d, B_4d, normal_4d, hyperplane_offset)
                if I_4d is not None:
                    intersects.append(I_4d)
            intersection_points_4d.append(np.array(intersects, dtype=float))
        # Now for each set of intersection points, build a local 3D representation
        shapes_3d = []
        shapes_3d_edges = []

        normal_4d = update_hyperplane_normal()
        basis_3 = build_orthonormal_basis(normal_4d)
        for i_pts_4d in intersection_points_4d:
            if len(i_pts_4d) == 0:
                shapes_3d.append([])
                shapes_3d_edges.append([])
                continue
            # Project each intersection point to 3D coords
            pts_3d = [project_point_onto_3d_hyperplane(pt, normal_4d, hyperplane_offset, basis_3)
                      for pt in i_pts_4d]
            pts_3d = np.array(pts_3d)
            # compute edges via 3D convex hull:
            edges_3d = compute_convex_edges_3d(pts_3d)
            shapes_3d.append(pts_3d)
            shapes_3d_edges.append(edges_3d)
        return shapes_3d, shapes_3d_edges

    # Main loop
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    last_mouse_pos = event.pos
                elif event.button == 4:  # scroll up
                    # Let scroll up rotate hyperplane in the xw-plane for demonstration
                    # or you could make scroll move offset
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
                elif event.key == K_w:
                    camera_pos[2] -= 0.2
                elif event.key == K_s:
                    camera_pos[2] += 0.2
                elif event.key == K_a:
                    camera_pos[0] -= 0.2
                elif event.key == K_d:
                    camera_pos[0] += 0.2
                elif event.key == K_q:
                    camera_pos[1] += 0.2
                elif event.key == K_e:
                    camera_pos[1] -= 0.2
                elif event.key == K_i:
                    rot_xy += 0.05
                elif event.key == K_k:
                    rot_xy -= 0.05
                elif event.key == K_j:
                    rot_xz += 0.05
                elif event.key == K_l:
                    rot_xz -= 0.05
                elif event.key == K_u:
                    rot_xw += 0.05
                elif event.key == K_o:
                    rot_xw -= 0.05
                elif event.key == K_SPACE:
                    auto_rotate = not auto_rotate

        if auto_rotate:
            rot_xy += 0.01  # auto-rotate in the XY plane

        # Slicing step
        shapes_3d_list, shapes_3d_edges_list = slice_4d_shapes()

        # Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera transform
        glTranslatef(-camera_pos[0], -camera_pos[1], -camera_pos[2])
        glRotatef(camera_rot_x, 1, 0, 0)
        glRotatef(camera_rot_y, 0, 1, 0)

        # Draw coordinate axes
        draw_axes(length=1.0)

        # Draw intersection geometry
        for idx, shape in enumerate(shapes_3d_list):
            color = (rand(), rand(), rand())
            draw_shape_3d(shape, shapes_3d_edges_list[idx], color)
            # Optionally draw spheres at intersection points
            for p3 in shape:
                draw_sphere_3d(p3, 0.03, 8, 8, color)

        # Also visualize the hyperplane normal arrow in 3D if desired
        normal_4d = update_hyperplane_normal()
        # The normal_4d itself doesn't map directly to 3D, but we can project it onto the subspace spanned by {x,y,z,w=0?}
        # For a quick visual: let's just pretend the (x,y,z) components of normal_4d are used as an arrow
        glBegin(GL_LINES)
        glColor3f(1, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(normal_4d[0], normal_4d[1], normal_4d[2])  # ignoring w component for visualization
        glEnd()

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
