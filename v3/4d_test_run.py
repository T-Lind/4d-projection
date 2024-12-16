import argparse
import json
import time

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

from v2.helper import transform_shapes_4d_to_3d, load_4d_shapes, draw_shape

print(
    "Controls: w/a/s/d and mouse: control camera, i/j/k/l: control x/y of plane, scroll wheel: rotate plane, space: toggle auto rotate plane")

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
pygame.display.set_caption("3D Observer with hyperplanar slice")
# Set up OpenGL perspective
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

shapes = transform_shapes_4d_to_3d(load_4d_shapes('4d_demo_file.json'), [0, 0, 0, 1], [0, 0, 0, 0])

# Camera position and rotation
camera_x, camera_y, camera_z = 0.5, 0.5, 5.0
camera_rotation_x, camera_rotation_y = 45, -45

# Mouse motion variables
last_x, last_y = 0, 0
mouse_down = False

center = (display[0] // 2, display[1] // 2)
pixels_per_unit = 100

plane_angle = 0
plane_point = [0, 0, 0, 0]


def draw():
    global camera_x, camera_y, camera_z, camera_rotation_x, camera_rotation_y
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glPushMatrix()

    for shape in shapes:
        draw_shape(shape['points'], shape['edges'], shape.get('color', (1.0, 1.0, 1.0)))

    # Restore the previous model-view matrix
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glDisable(GL_BLEND)

    # Set camera position and rotation
    glTranslatef(-camera_x, -camera_y, -camera_z)
    glRotatef(camera_rotation_x, 1, 0, 0)
    glRotatef(camera_rotation_y, 0, 1, 0)

    # draw_shape(shape_points, shape_edges)
    for shape in shapes:
        draw_shape(shape['points'], shape['edges'], shape.get('color', (1.0, 1.0, 1.0)))

    pygame.display.flip()


def handle_mouse_motion(x, y):
    global last_x, last_y, camera_rotation_x, camera_rotation_y, mouse_down

    if mouse_down:
        dx = x - last_x
        dy = y - last_y

        camera_rotation_x += dy * 0.2
        camera_rotation_y += dx * 0.2

    last_x, last_y = x, y


def get_plane_normal(angle):

    # Calculate the components of the normal vector
    x = np.cos(angle)
    y = np.sin(angle)
    z = 0
    w = 1  # The w component is 1 because we're rotating about the fourth dimensional axis

    # Return the normal vector
    return np.array([x, y, z, w])


def update():
    global shapes
    shapes = transform_shapes_4d_to_3d(load_4d_shapes('4d_demo_file.json'), get_plane_normal(plane_angle), plane_point)


# Main loop
running = True
auto_rotate = False
start_time = time.time()
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
            if event.key == pygame.K_i:
                camera_y += 0.25
            if event.key == pygame.K_k:
                camera_y -= 0.25

            if event.key == pygame.K_SPACE:
                auto_rotate = not auto_rotate

        elif event.type == pygame.MOUSEWHEEL:
            # Update the plane angle based on the mouse wheel scroll
            plane_angle += event.y * np.pi / 48
            plane_angle %= 2 * np.pi  # Keep the angle within 0 to 2π
            update()

    if auto_rotate:
        plane_angle = (time.time() - start_time) % (2 * np.pi)
        update()

    draw()

pygame.quit()
