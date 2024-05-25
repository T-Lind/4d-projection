import numpy as np
import sympy as sp
from OpenGL.GL import *
from OpenGL.GLU import *


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


def convert_to_plane_coordinates(intersection_point, plane_origin, angle):
    p_prime = np.array(intersection_point) - np.array(plane_origin)

    u_prime = np.dot(p_prime, (0, 0, 1))
    v_prime = np.dot(p_prime, (np.cos(angle), np.sin(angle), 0))

    return u_prime, v_prime


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
    glColor3f(*color[:3])
    for edge in edges:
        for vertex in edge:
            glVertex3fv(points[vertex])
    glEnd()

def drawLine(start, end, color):
    glBegin(GL_LINES)
    if len(color) == 3:
        glColor3fv(color)
    elif len(color) == 4:
        glColor4fv(color)
    else:
        raise ValueError("Color must be a 3-tuple or 4-tuple")
    glVertex2fv(start)
    glVertex2fv(end)
    glEnd()

