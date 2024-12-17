import pygame
import sys
import math
from math import sin, cos, radians
import itertools

pygame.init()

SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 600
BG_COLOR = (30, 30, 30)
SLICE_COLOR = (100, 200, 100)  # Color for the filled slice

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# ---------------------------
# 3D VECTOR / MATH UTILITIES
# ---------------------------

def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vec_cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def vec_scale(a, s):
    return (a[0]*s, a[1]*s, a[2]*s)

def vec_length(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def vec_normalize(a):
    l = vec_length(a)
    if l < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0]/l, a[1]/l, a[2]/l)

# ---------------------------
# GEOMETRY: INTERSECTION WITH PLANE
# ---------------------------

def plane_triangle_intersection(plane_point, plane_normal, triangle):
    """
    Compute intersection (if any) between a plane and a single triangular face.
    :param plane_point: A point on the plane (camera position).
    :param plane_normal: Normal vector of the plane (normalized).
    :param triangle: A tuple of three vertices [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)].
    :return: A list of intersection points. Typically 0, 1, or 2 points.
    """
    # The plane equation is plane_normal . (X - plane_point) = 0
    # We'll check each edge of the triangle for intersection with the plane.
    
    # Intersection points found:
    intersection_points = []
    
    # Edges: (v0->v1), (v1->v2), (v2->v0)
    vertices = [triangle[0], triangle[1], triangle[2]]
    edges = [(0,1), (1,2), (2,0)]
    
    for e0, e1 in edges:
        v0 = vertices[e0]
        v1 = vertices[e1]
        
        # Parametric form: V(t) = v0 + t*(v1 - v0)
        # We want t s.t. plane_normal . (V(t) - plane_point) = 0
        # => plane_normal . (v0 - plane_point + t*(v1 - v0)) = 0
        # => plane_normal . (v0 - plane_point) + t * plane_normal . (v1 - v0) = 0
        denom = vec_dot(plane_normal, vec_sub(v1, v0))
        num   = vec_dot(plane_normal, vec_sub(v0, plane_point))
        
        if abs(denom) > 1e-12:
            t = -num / denom
            if 0.0 <= t <= 1.0:
                # Intersection is within the segment
                intersect = vec_add(v0, vec_scale(vec_sub(v1, v0), t))
                # Store if not a duplicate
                if not any(vec_length(vec_sub(intersect, ip)) < 1e-9 for ip in intersection_points):
                    intersection_points.append(intersect)
    
    # intersection_points may have 0, 1 or 2 points for a single triangle.
    return intersection_points

def reconstruct_polygons(intersection_segments):
    """
    intersection_segments is a list of line segments, each is a tuple ((x1,y1), (x2,y2)) in 2D plane coords.
    We want to find closed loops (polygons). We'll do a naive approach:
      1. Build a graph of endpoints => connected segments
      2. Trace out cycles
    This is a simplistic method which assumes well-formed geometry.
    
    Returns a list of polygons, each polygon is a list of 2D vertices in order.
    """
    # Build adjacency: endpoint -> list of other endpoints
    adjacency = {}
    
    def add_edge(p1, p2):
        adjacency.setdefault(p1, []).append(p2)
        adjacency.setdefault(p2, []).append(p1)
    
    for seg in intersection_segments:
        p1, p2 = seg
        # Round coords a bit to avoid floating mis-matches as keys
        rp1 = (round(p1[0], 7), round(p1[1], 7))
        rp2 = (round(p2[0], 7), round(p2[1], 7))
        add_edge(rp1, rp2)
    
    visited = set()
    polygons = []
    
    def find_loop(start):
        # Depth-first or walk until closed. We attempt to form a cycle.
        polygon = [start]
        current = start
        visited.add(start)
        
        while True:
            neighbors = adjacency.get(current, [])
            # Choose the next neighbor that doesn't immediately form a dead-end
            next_p = None
            for nb in neighbors:
                # We allow a revisit only if it closes the loop (i.e. nb == start)
                if nb == start and len(polygon) > 2:
                    # We closed a loop
                    return polygon
                if nb not in visited:
                    next_p = nb
                    break
            
            if next_p is None:
                # no unvisited neighbor -> done
                return None
            
            polygon.append(next_p)
            visited.add(next_p)
            current = next_p
    
    for pt in adjacency.keys():
        if pt not in visited:
            loop = find_loop(pt)
            if loop and len(loop) >= 3:
                polygons.append(loop)
    
    return polygons

# ---------------------------
# PROJECTION TO THE PLANE & DRAWING
# ---------------------------

def compute_slice_polygons(mesh_faces, camera_pos, plane_normal):
    """
    For each face in mesh_faces (list of triangles), compute intersection with the plane.
    Then transform intersection points to the plane's 2D coordinate system.
    Finally, reconstruct polygons from the union of line segments.
    """
    # First gather intersection line segments in 3D
    line_segments_3d = []
    for tri in mesh_faces:
        pts = plane_triangle_intersection(camera_pos, plane_normal, tri)
        if len(pts) == 2:
            line_segments_3d.append((pts[0], pts[1]))
    
    if not line_segments_3d:
        return []
    
    # Define a local 2D coordinate system for the plane
    # plane_normal is the Z' axis for the plane. We need two orthonormal vectors X' and Y'.
    # A simple approach: we want plane_xaxis to be perpendicular to plane_normal and also perpendicular to the global Z if possible.
    
    # But a simpler approach: pick an arbitrary vector that isn't parallel to plane_normal, cross to get xaxis.
    up = (0,0,1)
    # If plane_normal is nearly parallel to up, pick a different 'up' to avoid degeneracy
    if abs(vec_dot(plane_normal, up)) > 0.99:
        up = (0,1,0)  # fallback
    plane_xaxis = vec_normalize(vec_cross(up, plane_normal))
    plane_yaxis = vec_normalize(vec_cross(plane_normal, plane_xaxis))
    
    # Convert 3D points to plane 2D coords:
    def to_plane_2d(p3):
        # Vector from camera to p3
        cp = vec_sub(p3, camera_pos)
        x_val = vec_dot(cp, plane_xaxis)
        y_val = vec_dot(cp, plane_yaxis)
        return (x_val, y_val)
    
    line_segments_2d = []
    for seg3d in line_segments_3d:
        p3d1, p3d2 = seg3d
        p2d1 = to_plane_2d(p3d1)
        p2d2 = to_plane_2d(p3d2)
        line_segments_2d.append((p2d1, p2d2))
    
    # Reconstruct polygons from line segments in 2D
    polygons_2d = reconstruct_polygons(line_segments_2d)
    return polygons_2d

def draw_polygons_2d(polygons, surface, color, scale=100.0, center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)):
    """
    Draw filled polygons given in 2D plane coords onto the pygame surface.
    scale: a factor to scale the slice for visualization.
    center: the pixel center of the slicing plane on screen.
    """
    for poly in polygons:
        # poly is a list of (x,y) in plane coordinates
        transformed = []
        for (px, py) in poly:
            sx = center[0] + px * scale
            sy = center[1] - py * scale  # invert Y for screen
            transformed.append((sx, sy))
        
        # Fill the polygon
        if len(transformed) >= 3:
            pygame.draw.polygon(surface, color, transformed)

# ---------------------------
# DEMO MESH: A Unit Cube
# ---------------------------
# This is just a simple example. You can load your own 3D model from file.

def create_unit_cube_mesh():
    """
    Returns a list of triangular faces. Each face is ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3)).
    A unit cube centered at origin, corners from -0.5..+0.5 in x,y,z.
    """
    # 8 corners
    v = [
        (-0.5, -0.5, -0.5),
        ( 0.5, -0.5, -0.5),
        ( 0.5,  0.5, -0.5),
        (-0.5,  0.5, -0.5),
        (-0.5, -0.5,  0.5),
        ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5,  0.5),
        (-0.5,  0.5,  0.5),
    ]
    # 12 triangular faces
    faces = []
    # Each face is two triangles:
    # bottom (-z)
    faces.append((v[0], v[1], v[2]))
    faces.append((v[0], v[2], v[3]))
    # top (+z)
    faces.append((v[4], v[6], v[5]))
    faces.append((v[4], v[7], v[6]))
    # front (-y)
    faces.append((v[0], v[4], v[5]))
    faces.append((v[0], v[5], v[1]))
    # back (+y)
    faces.append((v[3], v[2], v[6]))
    faces.append((v[3], v[6], v[7]))
    # left (-x)
    faces.append((v[0], v[3], v[7]))
    faces.append((v[0], v[7], v[4]))
    # right (+x)
    faces.append((v[1], v[5], v[6]))
    faces.append((v[1], v[6], v[2]))
    
    return faces

def load_obj(filename, scale=1.0, translate=(0, 0, 0)):
    """
    Load a .obj file and return a list of faces (triangles).
    :param filename: Path to the .obj file.
    :param scale: Uniform scaling factor.
    :param translate: (x, y, z) translation to apply to all vertices.
    :return: List of triangles [(v1, v2, v3), ...] where each vertex vi is (x, y, z).
    """
    vertices = []
    faces = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue  # Skip empty lines or comments
            if parts[0] == 'v':  # Vertex
                x, y, z = map(float, parts[1:4])
                x, y, z = x * scale + translate[0], y * scale + translate[1], z * scale + translate[2]
                vertices.append((x, y, z))
            elif parts[0] == 'f':  # Face (1-based index)
                indices = [int(idx.split('/')[0]) - 1 for idx in parts[1:]]
                if len(indices) == 3:  # Triangular face
                    faces.append((vertices[indices[0]], vertices[indices[1]], vertices[indices[2]]))
                elif len(indices) > 3:  # Convert polygon to triangles
                    for i in range(1, len(indices) - 1):
                        faces.append((
                            vertices[indices[0]],
                            vertices[indices[i]],
                            vertices[indices[i + 1]]
                        ))
    
    return faces


# ---------------------------
# MAIN LOOP
# ---------------------------

def main():
    pygame.display.set_caption("3D Object Slicing with PyGame")

    # Camera state
    camera_x = 0.0
    camera_y = 0.0
    camera_z = 0.0
    
    # Plane rotation angle around +Z, in degrees
    plane_angle = 0.0
    
    # Load or define your mesh
    mesh_faces = create_unit_cube_mesh()
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Scroll wheel up/down
                if event.button == 4:   # wheel up
                    plane_angle += 5.0
                elif event.button == 5: # wheel down
                    plane_angle -= 5.0
        
        # Handle keyboard input
        keys = pygame.key.get_pressed()
        move_speed = 1.0 * dt
        
        if keys[pygame.K_w]:
            camera_y += move_speed
        if keys[pygame.K_s]:
            camera_y -= move_speed
        if keys[pygame.K_a]:
            camera_x -= move_speed
        if keys[pygame.K_d]:
            camera_x += move_speed
        
        if keys[pygame.K_ESCAPE]:
            running = False
        
        # Define plane normal by rotating (1,0,0) around +Z by plane_angle
        # So at angle=0, plane is yz-plane. As angle changes, it rotates around Z.
        rad = math.radians(plane_angle)
        nx = math.cos(rad)
        ny = math.sin(rad)
        nz = 0.0
        plane_normal = (nx, ny, nz)
        
        camera_pos = (camera_x, camera_y, camera_z)
        
        # Compute the intersection polygons for the slicing plane
        polygons_2d = compute_slice_polygons(mesh_faces, camera_pos, plane_normal)
        
        screen.fill(BG_COLOR)
        # Draw the cross-section polygons
        draw_polygons_2d(polygons_2d, screen, SLICE_COLOR, scale=200.0)
        
        # Simple on-screen instructions:
        info_text = [
            "W/A/S/D to move camera (XY plane)",
            "Mouse Scroll Wheel to rotate plane around Z-axis",
            f"Camera = ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f})",
            f"Plane Angle = {plane_angle:.1f} deg",
        ]
        font = pygame.font.SysFont(None, 24)
        yoff = 10
        for line in info_text:
            txt_surf = font.render(line, True, (200,200,200))
            screen.blit(txt_surf, (10, yoff))
            yoff += 22
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()



# if __name__ == "__main__":
    # main()

def main_with_multiple_objects():
    pygame.display.set_caption("3D Object Slicing with Multiple Objects")
    
    # Camera state
    camera_x = 0.0
    camera_y = 0.0
    camera_z = 0.0
    
    # Plane rotation angle around +Z, in degrees
    plane_angle = 0.0
    
    # Load objects into the scene
    scene_objects = [
        load_obj("v4/cube.obj", scale=1.0, translate=(-2, -2, 0)),
        load_obj("v4/cube.obj", scale=1.0, translate=(2, 2, 0)),
        load_obj("v4/pyramid.obj", scale=1.0, translate=(0, 0, 0)),
    ]
    
    # Flatten into a single list of triangles
    scene_faces = list(itertools.chain.from_iterable(scene_objects))
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Scroll wheel up/down
                if event.button == 4:   # wheel up
                    plane_angle += 5.0
                elif event.button == 5: # wheel down
                    plane_angle -= 5.0
        
        # Handle keyboard input
        keys = pygame.key.get_pressed()
        move_speed = 1.0 * dt
        
        if keys[pygame.K_w]:
            camera_y += move_speed
        if keys[pygame.K_s]:
            camera_y -= move_speed
        if keys[pygame.K_a]:
            camera_x -= move_speed
        if keys[pygame.K_d]:
            camera_x += move_speed
        
        if keys[pygame.K_ESCAPE]:
            running = False
        
        # Define plane normal by rotating (1,0,0) around +Z by plane_angle
        rad = math.radians(plane_angle)
        nx = math.cos(rad)
        ny = math.sin(rad)
        nz = 0.0
        plane_normal = (nx, ny, nz)
        
        camera_pos = (camera_x, camera_y, camera_z)
        
        # Compute the intersection polygons for the slicing plane
        polygons_2d = compute_slice_polygons(scene_faces, camera_pos, plane_normal)
        
        screen.fill(BG_COLOR)
        # Draw the cross-section polygons
        draw_polygons_2d(polygons_2d, screen, SLICE_COLOR, scale=200.0)
        
        # Simple on-screen instructions:
        info_text = [
            "W/A/S/D to move camera (XY plane)",
            "Mouse Scroll Wheel to rotate plane around Z-axis",
            f"Camera = ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f})",
            f"Plane Angle = {plane_angle:.1f} deg",
        ]
        font = pygame.font.SysFont(None, 24)
        yoff = 10
        for line in info_text:
            txt_surf = font.render(line, True, (200,200,200))
            screen.blit(txt_surf, (10, yoff))
            yoff += 22
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_with_multiple_objects()
