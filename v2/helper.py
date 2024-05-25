import numpy as np
import sympy as sp
from OpenGL.GL import *
from OpenGL.GLU import *


def calculate_intersection(a, b, hyperplane):
    # Coordinates of the line segment endpoints in 4D
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    px, py, pz = hyperplane['point']
    angle = hyperplane['angle']

    # Define symbolic variable
    t = sp.symbols('t')

    # Parametric equations of the line segment
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)
    w = w1 + t * (w2 - w1)

    # Hyperplane equation parameters
    tan_theta = sp.tan(angle)

    # Hyperplane equation
    hyperplane_eq = y - (tan_theta * (x - px) + py)

    # Solve for t
    t_sol = sp.solve(hyperplane_eq, t)

    # Check if the solution lies within the segment
    if t_sol:
        t_val = t_sol[0]
        if 0 <= t_val <= 1:
            # Calculate intersection point
            ix = x1 + t_val * (x2 - x1)
            iy = y1 + t_val * (y2 - y1)
            iz = z1 + t_val * (z2 - z1)
            iw = w1 + t_val * (w2 - w1)
            return float(ix), float(iy), float(iz), float(iw)

    return None


def convert_to_3d_coordinates(intersection_point, hyperplane_origin, hyperplane_angle):
    # Convert points to numpy arrays
    p_prime = np.array(intersection_point) - np.array(hyperplane_origin)

    # Define the rotation matrix about the w-axis
    cos_theta = np.cos(hyperplane_angle)
    sin_theta = np.sin(hyperplane_angle)
    rotation_matrix = np.array([
        [cos_theta, sin_theta, 0, 0],
        [-sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation to p_prime
    rotated_p_prime = np.dot(rotation_matrix, p_prime)

    # Extract the coordinates in the hyperplane
    u_prime = rotated_p_prime[2]  # The component along the z-axis (analogous to the plane's normal direction)
    v_prime = rotated_p_prime[0]  # The component along the x-axis (after rotation)
    w_prime = rotated_p_prime[1]  # The component along the y-axis (after rotation)

    return u_prime, v_prime, w_prime


def draw_shape(points, edges, color=(1.0, 1.0, 1.0)):
    glBegin(GL_LINES)
    glColor3f(*color[:3])
    for edge in edges:
        for vertex in edge:
            glVertex3fv(points[vertex])
    glEnd()
