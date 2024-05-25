# Displays the object/points of intersection and plane in 3D, then plots the 2D coordinates of the intersection points
# on the plane and the convex hull of the points.

import argparse
import json

import cdd as pcdd
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

from v2.helper import calculate_intersection, convert_to_3d_coordinates, draw_shape

print("Controls: w/a/s/d and mouse: control camera, scroll wheel: rotate plane, space: toggle auto rotate plane")

# get first arg
parser = argparse.ArgumentParser()
parser.add_argument("file", help="file to read")
try:
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        config = json.loads(f.read())
except:
    with open('4d_demo_file.json', 'r') as f:
        config = json.loads(f.read())

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("4D Observer with hyperplanar slice")
# Set up OpenGL perspective
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

# Camera position and rotation
camera_x, camera_y, camera_z = 0, 0.5, 5.0
camera_rotation_x, camera_rotation_y = 45, -45

# Mouse motion variables
last_x, last_y = 0, 0
mouse_down = False

hplane_angle = config['hyperplane_angle']
hplane_point = config['hyperplane_position']

shape_intersections_total = []
converted_coords_total = []
converted_edges_total = []


def handle_mouse_motion(x, y):
    global last_x, last_y, camera_rotation_x, camera_rotation_y, mouse_down

    if mouse_down:
        dx = x - last_x
        dy = y - last_y

        camera_rotation_x += dy * 0.2
        camera_rotation_y += dx * 0.2

    last_x, last_y = x, y


def compute_shapes():
    global shape_intersections_total, converted_edges_total, converted_coords_total
    shape_intersections_total = []
    for shape in config['shapes']:
        intersections = []
        for i in range(len(shape['points']) - 1):
            a = shape['points'][i]
            b = shape['points'][i + 1]
            intersection = calculate_intersection(a, b, {'point': hplane_point, 'angle': hplane_angle})
            if intersection:
                intersections.append(intersection)
        shape_intersections_total.append(intersections)

    converted_coords_total = []
    for intersections in shape_intersections_total:
        converted_coords = []
        for intersection in intersections:
            converted_coords.append(
                convert_to_3d_coordinates(intersection, (hplane_point[0], hplane_point[1], hplane_point[2], 0),
                                          hplane_angle))
        converted_coords_total.append(converted_coords)

    converted_edges_total = []
    for converted_coords in converted_coords_total:
        converted_edges = []
        if len(converted_coords) > 3:
            np_converted_coords = np.array(converted_coords)
            vertices = np.hstack((np.ones((np_converted_coords.shape[0], 1)), np_converted_coords))

            mat = pcdd.Matrix(vertices.tolist(), linear=False, number_type="fraction")
            mat.rep_type = pcdd.RepType.GENERATOR
            poly = pcdd.Polyhedron(mat)

            adjacencies = [list(x) for x in poly.get_input_adjacency()]

            for i, indices in enumerate(adjacencies[:-1]):
                indices = list(filter(lambda x: x > i, indices))
                col1 = np.full((len(indices), 1), i)
                indices = np.reshape(indices, (len(indices), 1))
                if len(indices) > 0:
                    converted_edges.append(np.hstack((col1, indices)))
            converted_edges = np.vstack(converted_edges)

        converted_edges_total.append(converted_edges)


def draw():
    global camera_x, camera_y, camera_z, camera_rotation_x, camera_rotation_y, shape_intersections_total, converted_edges_total
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Set camera position and rotation
    glTranslatef(-camera_x, -camera_y, -camera_z)
    glRotatef(camera_rotation_x, 1, 0, 0)
    glRotatef(camera_rotation_y, 0, 1, 0)

    for i in range(min(len(converted_coords_total), len(converted_edges_total))):
        draw_shape(converted_coords_total[i], converted_edges_total[i], color=config['shapes'][i]['color'])

    pygame.display.flip()


compute_shapes()
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
            if event.key == pygame.K_SPACE:
                auto_rotate = not auto_rotate
        elif event.type == pygame.MOUSEWHEEL:
            # Update the plane angle based on the mouse wheel scroll
            hplane_angle += event.y * np.pi / 48
            hplane_angle %= 2 * np.pi  # Keep the angle within 0 to 2π
            compute_shapes()
    if auto_rotate:
        hplane_angle += np.pi / 48
        hplane_angle %= 2 * np.pi
        compute_shapes()

    draw()
