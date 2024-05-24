import math
from random import random as rand
import pygame
import sympy as sp
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Set up OpenGL perspective
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

shape_points = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (2, 1, -1)
]

shape_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2),
    (1, 3),
    (4, 0),
    (4, 1),
    (4, 3),
]
plane_size = 2
plane_x = 1
plane_y = 0.5
plane_angle = 0.65

plane_points = [
    (plane_x + plane_size * math.cos(plane_angle), plane_y + plane_size * math.sin(plane_angle), -1),
    (plane_x - plane_size * math.cos(plane_angle), plane_y - plane_size * math.sin(plane_angle), -1),
    (plane_x - plane_size * math.cos(plane_angle), plane_y - plane_size * math.sin(plane_angle), 1),
    (plane_x + plane_size * math.cos(plane_angle), plane_y + plane_size * math.sin(plane_angle), 1),
]
plane_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
]


def calculate_intersection(a, b, plane):
    x1, y1, z1 = a
    x2, y2, z2 = b
    px, py = plane['x'], plane['y']
    angle = plane['angle']

    # Define symbolic variable
    t = sp.symbols('t')

    # Parametric equations of the line segment
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)

    # Plane equation parameters
    tan_theta = sp.tan(angle)

    # Plane equation
    plane_eq = y - (tan_theta * (x - px) + py)

    # Solve for t
    t_sol = sp.solve(plane_eq, t)

    # Check if the solution lies within the segment
    if t_sol:
        t_val = t_sol[0]
        if 0 <= t_val <= 1:
            # Calculate intersection point
            ix = x1 + t_val * (x2 - x1)
            iy = y1 + t_val * (y2 - y1)
            iz = z1 + t_val * (z2 - z1)
            return float(ix), float(iy), float(iz)

    return None

intersections = []
for edge in shape_edges:
    a = shape_points[edge[0]]
    b = shape_points[edge[1]]
    intersection = calculate_intersection(a, b, {"x": plane_x, "y": plane_y, "angle": plane_angle})
    if intersection:
        intersections.append(intersection)

print("Found intersections: ", intersections)

# Camera position and rotation
camera_x, camera_y, camera_z = 0.5, 0.5, 5.0
camera_rotation_x, camera_rotation_y = 45, -45

# Mouse motion variables
last_x, last_y = 0, 0
mouse_down = False


def draw_sphere(center, radius=0.025):
    slices = 30
    stacks = 30

    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluQuadricTexture(quadric, GL_TRUE)
    gluSphere(quadric, radius, slices, stacks)
    gluDeleteQuadric(quadric)
    glPopMatrix()

def draw_shape(points, edges, color=(1.0, 1.0, 1.0)):
    glBegin(GL_LINES)
    glColor3f(*color)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(points[vertex])
    glEnd()

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


# Main loop
running = True
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
            elif event.key == pygame.K_s:
                camera_z += 0.25
            elif event.key == pygame.K_a:
                camera_x -= 0.25
            elif event.key == pygame.K_d:
                camera_x += 0.25

    draw()

pygame.quit()

# print(calculate_intersection((1, 2, 2), (-1, 0, 0), {"x": -1.4, "y": 0.3, "angle": 0.65}))
# find points of intersection between the plane and the shape


