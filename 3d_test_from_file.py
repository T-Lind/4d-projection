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

# Configuration loading
parser = argparse.ArgumentParser()
parser.add_argument("file", help="file to read")
try:
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        config = json.loads(f.read())
except:
    with open('3d_demo_file.json', 'r') as f:
        config = json.loads(f.read())

# Initialize Pygame and OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Observer with planar slice")
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

# Plane configuration
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
plane_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

# Calculate initial intersections
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

# Camera setup
camera_x, camera_y, camera_z = 0.5, 0.5, 5.0
camera_rotation_x, camera_rotation_y = 45, -45
last_x, last_y = 0, 0
mouse_down = False

# Convert coordinates
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
    global camera_x, camera_y, camera_z, camera_rotation_x, camera_rotation_y
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # 2D rendering
    glPushMatrix()
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    for i in range(min(len(converted_coords_total), len(converted_edges_total))):
        converted_coords = converted_coords_total[i]
        converted_edges = converted_edges_total[i]
        color = config['shapes'][i].get('color', (rand(), rand(), rand()))

        for edge in converted_edges:
            drawLine(
                (converted_coords[edge[0]][0] * pixels_per_unit + center[0],
                 converted_coords[edge[0]][1] * pixels_per_unit + center[1]),
                (converted_coords[edge[1]][0] * pixels_per_unit + center[0],
                 converted_coords[edge[1]][1] * pixels_per_unit + center[1]),
                color
            )

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glDisable(GL_BLEND)

    # 3D rendering
    glTranslatef(-camera_x, -camera_y, -camera_z)
    glRotatef(camera_rotation_x, 1, 0, 0)
    glRotatef(camera_rotation_y, 0, 1, 0)

    for intersections in shape_intersections_total:
        for intersection in intersections:
            color = config.get('intersection_point_color', (rand(), rand(), rand()))
            glColor3f(*color)
            draw_sphere(intersection)

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
    global plane_x, plane_y, plane_angle, plane_points, shape_intersections_total
    global converted_coords_total, converted_edges_total

    # Update plane geometry
    plane_points = [
        (plane_x + plane_length * np.cos(plane_angle), plane_y + plane_length * np.sin(plane_angle), -plane_width),
        (plane_x - plane_length * np.cos(plane_angle), plane_y - plane_length * np.sin(plane_angle), -plane_width),
        (plane_x - plane_length * np.cos(plane_angle), plane_y - plane_length * np.sin(plane_angle), plane_width),
        (plane_x + plane_length * np.cos(plane_angle), plane_y + plane_length * np.sin(plane_angle), plane_width),
    ]

    # Recalculate intersections
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

    # Update converted coordinates
    converted_coords_total = []
    for intersections in shape_intersections_total:
        converted_coords = []
        for intersection in intersections:
            coord = convert_to_plane_coordinates(intersection, (plane_x, plane_y, 0), plane_angle)
            converted_coords.append(coord)
        converted_coords_total.append(converted_coords)

    # Calculate convex hulls
    converted_edges_total = []
    for converted_coords in converted_coords_total:
        converted_edges = []
        if len(converted_coords) > 2:
            try:
                hull = ConvexHull(converted_coords)
                for i in range(len(hull.vertices) - 1):
                    converted_edges.append((hull.vertices[i], hull.vertices[i + 1]))
                converted_edges.append((hull.vertices[-1], hull.vertices[0]))
            except:
                pass
        converted_edges_total.append(converted_edges)

# Main loop
running = True
auto_rotate = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                last_x, last_y = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_down = False
        elif event.type == pygame.MOUSEMOTION:
            handle_mouse_motion(event.pos[0], event.pos[1])
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                camera_z -= 0.25
            if event.key == pygame.K_s:
                camera_z += 0.25
            if event.key == pygame.K_a:
                camera_x -= 0.25
            if event.key == pygame.K_d:
                camera_x += 0.25
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
            plane_angle += event.y * np.pi / 48
            plane_angle %= 2 * np.pi
            update()

    if auto_rotate:
        plane_angle += np.pi / 48
        plane_angle %= 2 * np.pi
        update()

    draw()

pygame.quit()