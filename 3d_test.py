# Displays the object/points of intersection and plane in 3D, then plots the 2D coordinates of the intersection points
# on the plane and the convex hull of the points.

from random import random as rand

import matplotlib.pyplot as plt
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from scipy.spatial import ConvexHull
from v1.helper import calculate_intersection, convert_to_plane_coordinates, draw_sphere, draw_shape

print("Controls: w/a/s/d and mouse: control camera, i/j/k/l: control x/y of plane, scroll wheel: rotate plane, space: toggle auto rotate plane")

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Observer with planar slice")
# Set up OpenGL perspective
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
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

# Camera position and rotation
camera_x, camera_y, camera_z = 0.5, 0.5, 5.0
camera_rotation_x, camera_rotation_y = 45, -45

# Mouse motion variables
last_x, last_y = 0, 0
mouse_down = False

converted_coords = []
for intersection in intersections:
    coord = convert_to_plane_coordinates(intersection, (plane_x, plane_y, 0), plane_angle)
    converted_coords.append(coord)


def draw():
    global camera_x, camera_y, camera_z, camera_rotation_x, camera_rotation_y
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Set camera position and rotation
    glTranslatef(-camera_x, -camera_y, -camera_z)
    glRotatef(camera_rotation_x, 1, 0, 0)
    glRotatef(camera_rotation_y, 0, 1, 0)

    draw_shape(shape_points, shape_edges)
    draw_shape(plane_points, plane_edges, (1.0, 0.0, 0.0))

    for intersection in intersections:
        glColor3f(rand(), rand(), rand())
        draw_sphere(intersection)

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
    global plane_x, plane_y, plane_angle, plane_points, intersections, converted_coords
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


# Main loop
running = True
auto_rotate = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_down = True
                last_x, last_y = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                mouse_down = False
        elif event.type == pygame.MOUSEMOTION:
            handle_mouse_motion(event.pos[0], event.pos[1])
        elif event.type == pygame.KEYDOWN:
            # Move the camera with the WASD keys
            if event.key == pygame.K_w:
                camera_z -= 0.25
            if event.key == pygame.K_s:
                camera_z += 0.25
            if event.key == pygame.K_a:
                camera_x -= 0.25
            if event.key == pygame.K_d:
                camera_x += 0.25

            # i/j/k/l controls x/y of plane
            if event.key == pygame.K_i:
                plane_y += 0.25
                update()
            if event.key == pygame.K_k:
                plane_y -= 0.25
                update()
            if event.key == pygame.K_j:
                plane_x -= 0.25
                update()
            if event.key == pygame.K_l:
                plane_x += 0.25
                update()
            if event.key == pygame.K_SPACE:
                auto_rotate = not auto_rotate

        elif event.type == pygame.MOUSEWHEEL:
            # Update the plane angle based on the mouse wheel scroll
            plane_angle += event.y * np.pi / 48
            plane_angle %= 2 * np.pi  # Keep the angle within 0 to 2π
            update()

    if auto_rotate:
        plane_angle += np.pi / 48
        plane_angle %= 2 * np.pi
        update()

    draw()

pygame.quit()

# Plot projected 2D coordinates
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)

for coord in converted_coords:
    ax.scatter(*coord)

hull = ConvexHull(converted_coords)

edges = []
for i in range(len(hull.vertices) - 1):
    edges.append((hull.vertices[i], hull.vertices[i + 1]))

edges.append((hull.vertices[-1], hull.vertices[0]))

for edge in edges:
    from_vertex = converted_coords[edge[0]]
    to_vertex = converted_coords[edge[1]]
    ax.plot([from_vertex[0], to_vertex[0]], [from_vertex[1], to_vertex[1]], 'r')

plt.show()
