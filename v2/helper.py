import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

def load_json_4d(file_path):
    import json
    with open(file_path) as f:
        data = json.load(f)
    return data


def rotate_4d_vector(vector, angle, axis1, axis2):
    """
    Rotate a 4D vector around the plane defined by axis1 and axis2.
    """
    rotation_matrix = np.eye(4)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix[axis1, axis1] = cos_angle
    rotation_matrix[axis1, axis2] = -sin_angle
    rotation_matrix[axis2, axis2] = cos_angle
    rotation_matrix[axis2, axis1] = sin_angle
    return np.dot(rotation_matrix, vector)


def project_4d_to_3d(point_4d, plane_normal, plane_point):
    """
    Project a 4D point onto a 3D hyperplane.
    """
    point_4d = np.array(point_4d)
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)

    d = np.dot(plane_point, plane_normal)
    t = (d - np.dot(plane_normal, point_4d)) / np.dot(plane_normal, plane_normal)
    intersection_point = point_4d + t * plane_normal
    return intersection_point[:3]


def load_4d_shapes(file_path):
    data = load_json_4d(file_path)
    shapes = data["shapes"]
    return shapes


def transform_shapes_4d_to_3d(shapes, plane_normal, plane_point):
    """
    Transform 4D shapes to 3D using the projection method.
    """
    transformed_shapes = []
    for shape in shapes:
        transformed_shape = {}
        transformed_shape["color"] = shape["color"]
        transformed_shape["points"] = [project_4d_to_3d(point, plane_normal, plane_point) for point in shape["points"]]
        transformed_shape["edges"] = shape["edges"]
        transformed_shapes.append(transformed_shape)
    # print("Transformed 4D shapes to 3D", transformed_shapes)
    return transformed_shapes


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
            pt = points[vertex]
            if len(pt) != 3:
                raise ValueError("Each vertex must have 3 coordinates, fail: ", pt)
            glVertex3fv(pt)
    glEnd()
