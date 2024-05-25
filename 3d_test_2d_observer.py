import numpy as np
import pygame
from OpenGL.GL import *
from pygame.locals import *
from scipy.spatial import ConvexHull

from v1.helper import calculate_intersection, convert_to_plane_coordinates

print("Controls: w/a/s/d: control x/y of plane, scroll wheel: rotate plane, space: toggle auto rotate plane")

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("2D Observer from planar slice")

# Set up OpenGL perspective
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, display[0], 0, display[1], -1, 1)
glMatrixMode(GL_MODELVIEW)

shape_points = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (2, 1, 1),

]
# Convex shapes only
shape_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2),
    (1, 3),
    (4, 0),
    (4, 1),
    (4, 2),
]
plane_size = 2
plane_x = 1
plane_y = 0.5
plane_angle = 0.95

plane_points = [
    (plane_x + plane_size * np.cos(plane_angle), plane_y + plane_size * np.sin(plane_angle), -1),
    (plane_x - plane_size * np.cos(plane_angle), plane_y - plane_size * np.sin(plane_angle), -1),
    (plane_x - plane_size * np.cos(plane_angle), plane_y - plane_size * np.sin(plane_angle), 1),
    (plane_x + plane_size * np.cos(plane_angle), plane_y + plane_size * np.sin(plane_angle), 1),
]
plane_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
]

intersections = []
for edge in shape_edges:
    a = shape_points[edge[0]]
    b = shape_points[edge[1]]
    intersection = calculate_intersection(a, b, {"x": plane_x, "y": plane_y, "angle": plane_angle})
    if intersection:
        intersections.append(intersection)

# Mouse motion variables
last_x, last_y = 0, 0
mouse_down = False

converted_coords = []
for intersection in intersections:
    coord = convert_to_plane_coordinates(intersection, (plane_x, plane_y, 0), plane_angle)
    converted_coords.append(coord)
converted_edges = []


def drawLine(start, end, color):
    glBegin(GL_LINES)
    glColor3fv(color)
    glVertex2fv(start)
    glVertex2fv(end)
    glEnd()


center = (display[0] // 2, display[1] // 2)
pixels_per_unit = 100


def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Draw the convex hull (2D)
    for edge in converted_edges:
        drawLine((converted_coords[edge[0]][0] * pixels_per_unit + center[0],
                  converted_coords[edge[0]][1] * pixels_per_unit + center[1]),
                 (converted_coords[edge[1]][0] * pixels_per_unit + center[0],
                  converted_coords[edge[1]][1] * pixels_per_unit + center[1]), (0, 1, 0))

    pygame.display.flip()


def handle_mouse_motion(x, y):
    global last_x, last_y, camera_rotation_x, camera_rotation_y, mouse_down

    if mouse_down:
        dx = x - last_x
        dy = y - last_y

        camera_rotation_x += dy * 0.2
        camera_rotation_y += dx * 0.2

    last_x, last_y = x, y


def update():
    global plane_x, plane_y, plane_angle, plane_points, intersections, converted_coords, converted_edges
    # Update the plane points based on the new angle
    plane_points = [
        (plane_x + plane_size * np.cos(plane_angle), plane_y + plane_size * np.sin(plane_angle), -1),
        (plane_x - plane_size * np.cos(plane_angle), plane_y - plane_size * np.sin(plane_angle), -1),
        (plane_x - plane_size * np.cos(plane_angle), plane_y - plane_size * np.sin(plane_angle), 1),
        (plane_x + plane_size * np.cos(plane_angle), plane_y + plane_size * np.sin(plane_angle), 1),
    ]

    # Recalculate intersections and converted coordinates
    intersections = []
    for edge in shape_edges:
        a = shape_points[edge[0]]
        b = shape_points[edge[1]]
        intersection = calculate_intersection(a, b, {"x": plane_x, "y": plane_y, "angle": plane_angle})
        if intersection:
            intersections.append(intersection)

    converted_coords = []
    for intersection in intersections:
        coord = convert_to_plane_coordinates(intersection, (plane_x, plane_y, 0), plane_angle)
        converted_coords.append(coord)

    converted_edges = []
    if len(converted_coords) > 2:
        try:
            hull = ConvexHull(converted_coords)
        except:
            return

        for i in range(len(hull.vertices) - 1):
            converted_edges.append((hull.vertices[i], hull.vertices[i + 1]))

        converted_edges.append((hull.vertices[-1], hull.vertices[0]))


angle_step = np.pi / 48

# Main loop
update()
running = True
auto_bouncing_angle = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Move the camera with the WASD keys
            if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_SPACE]:
                if event.key == pygame.K_w:
                    plane_y -= 0.25
                if event.key == pygame.K_s:
                    plane_y += 0.25
                if event.key == pygame.K_a:
                    plane_x -= 0.25
                if event.key == pygame.K_d:
                    plane_x += 0.25
                if event.key == pygame.K_SPACE:
                    auto_bouncing_angle = not auto_bouncing_angle
                update()
        elif event.type == pygame.MOUSEWHEEL:
            # Update the plane angle based on the mouse wheel scroll
            plane_angle += event.y * angle_step
            plane_angle %= 2 * np.pi  # Keep the angle within 0 to 2π
            update()

    if auto_bouncing_angle:
        plane_angle += angle_step
        plane_angle %= 2 * np.pi
        update()

    draw()

pygame.quit()
