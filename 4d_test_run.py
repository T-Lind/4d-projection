# shapes = helper_4d.transform_shapes_4d_to_3d(helper_4d.load_4d_shapes('4d_demo_file.json'), [0, 0, 0, 1], [0, 0, 0, 0])
import argparse
import json
from random import random as rand

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from scipy.spatial import ConvexHull

from v1.helper import calculate_intersection, convert_to_plane_coordinates, draw_sphere, draw_shape, drawLine

print("Controls: w/a/s/d and mouse: control camera, i/j/k/l: control x/y of plane, scroll wheel: rotate plane, space: toggle auto rotate plane")

# get first arg
parser = argparse.ArgumentParser()
parser.add_argument("file", help="file to read")
try:
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        config = json.loads(f.read())
except:
    with open('3d_demo_file.json', 'r') as f:
        config = json.loads(f.read())

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Observer with planar slice")
# Set up OpenGL perspective
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

plane_length = config['plane_length']
plane_width = config['plane_width']
plane_x = config['plane_x']
plane_y = config['plane_y']
plane_angle = config['plane_angle']

plane_points = [
    (plane_x + plane_length * np.cos(plane_angle), plane_y + plane_length * np.sin(plane_angle), -plane_width),
    (plane_x - plane_length * np.cos(plane_angle), plane_y - plane_length * np.sin(plane_angle), -plane_width),
    (plane_x - plane_length * np.cos(plane_angle), plane_y - plane_length * np.sin(plane_angle), plane_width),
    (plane_x + plane_length * np.cos(plane_angle), plane_y + plane_length * np.sin(plane_angle), plane_width),
]
plane_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
]
shape_intersections_total = []
for shape in config['shapes']:
    intersections = []
    shape_points = shape['points']
    shape_edges = shape['edges']
    for edge in shape_edges:
        a = shape_points[edge[0]]
        b = shape_points[edge[1]]
        intersection = calculate_intersection(a, b, {"x": plane_x, "y": plane_y, "angle": plane_angle})
        if intersection:
            intersections.append(intersection)
    shape_intersections_total.append(intersections)

# Camera position and rotation
camera_x, camera_y, camera_z = 0.5, 0.5, 5.0
camera_rotation_x, camera_rotation_y = 45, -45

# Mouse motion variables
last_x, last_y = 0, 0
mouse_down = False

converted_coords_total = []
for intersections in shape_intersections_total:
    converted_coords = []
    for intersection in intersections:
        coord = convert_to_plane_coordinates(intersection, (plane_x, plane_y, 0), plane_angle)
        converted_coords.append(coord)
    converted_coords_total.append(converted_coords)
converted_edges_total = []

center = (display[0] // 2, display[1] // 2)
pixels_per_unit = 100


def draw():
    global camera_x, camera_y, camera_z, camera_rotation_x, camera_rotation_y, shape_intersections_total, converted_edges_total
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glPushMatrix()


    # Restore the previous model-view matrix
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glDisable(GL_BLEND)

    # Set camera position and rotation
    glTranslatef(-camera_x, -camera_y, -camera_z)
    glRotatef(camera_rotation_x, 1, 0, 0)
    glRotatef(camera_rotation_y, 0, 1, 0)


    # draw_shape(shape_points, shape_edges)
    for shape in config['shapes']:
        draw_shape(shape['points'], shape['edges'], shape.get('color', (1.0, 1.0, 1.0)))

    draw_shape(plane_points, plane_edges, (1.0, 0.0, 0.0))

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
    global plane_x, plane_y, plane_angle, plane_points, intersections, converted_coords, shape_intersections_total, converted_coords_total, converted_edges_total
    # Update the plane points based on the new angle
    plane_points = [
        (plane_x + plane_length * np.cos(plane_angle), plane_y + plane_length * np.sin(plane_angle), -plane_width),
        (plane_x - plane_length * np.cos(plane_angle), plane_y - plane_length * np.sin(plane_angle), -plane_width),
        (plane_x - plane_length * np.cos(plane_angle), plane_y - plane_length * np.sin(plane_angle), plane_width),
        (plane_x + plane_length * np.cos(plane_angle), plane_y + plane_length * np.sin(plane_angle), plane_width),
    ]

    # Recalculate intersections and converted coordinates
    shape_intersections_total = []
    for shape in config['shapes']:
        intersections = []
        shape_points = shape['points']
        shape_edges = shape['edges']
        for edge in shape_edges:
            a = shape_points[edge[0]]
            b = shape_points[edge[1]]
            intersection = calculate_intersection(a, b, {"x": plane_x, "y": plane_y, "angle": plane_angle})
            if intersection:
                intersections.append(intersection)
        shape_intersections_total.append(intersections)

    converted_coords_total = []
    for intersections in shape_intersections_total:
        converted_coords = []
        for intersection in intersections:
            coord = convert_to_plane_coordinates(intersection, (plane_x, plane_y, 0), plane_angle)
            converted_coords.append(coord)
        converted_coords_total.append(converted_coords)

    converted_edges_total = []
    for converted_coords in converted_coords_total:
        converted_edges = []
        if len(converted_coords) > 2:
            try:
                hull = ConvexHull(converted_coords)
            except:
                return

            for i in range(len(hull.vertices) - 1):
                converted_edges.append((hull.vertices[i], hull.vertices[i + 1]))

            converted_edges.append((hull.vertices[-1], hull.vertices[0]))
        converted_edges_total.append(converted_edges)


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
