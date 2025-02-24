import sys
import json
import numpy as np
from scipy.spatial import ConvexHull
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import MeshData
from OpenGL.GL import glEnable, glLightfv, GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_AMBIENT, GL_DIFFUSE, GL_POSITION

# --------------------------
# JSON Loading (with optional position)
# --------------------------
def load_shapes_json(filename):
    """
    Load multiple shapes from a JSON file.
    The JSON file is expected to contain a list of shape definitions.
    Each shape should have:
      - "vertices": a list of 4D vertices.
      - "color": a list [R, G, B, A] for the face color.
      - (Optional) "position": a four-element list to offset each vertex.
    Returns a list of tuples: (vertices, color)
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    shapes = []
    for shape in data:
        vertices = [np.array(v, dtype=float) for v in shape['vertices']]
        if 'position' in shape:
            pos_offset = np.array(shape['position'], dtype=float)
            vertices = [v + pos_offset for v in vertices]
        color = tuple(shape.get('color', [1, 0, 0, 1]))
        shapes.append((vertices, color))
    return shapes

# --------------------------
# Geometry Functions
# --------------------------
def get_tesseract_edges(vertices):
    """
    Given a list of 4D vertices, return a list of edges.
    Two vertices share an edge if they differ in exactly one coordinate.
    """
    edges = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.abs(vertices[i] - vertices[j])
            if np.count_nonzero(diff > 1e-6) == 1:
                edges.append((vertices[i], vertices[j]))
    return edges

def hyperplane_intersection(P, Q, a, b, c, d, e_val):
    """
    Compute the intersection (if any) between the edge from P to Q and the hyperplane
    defined by: a*x + b*y + c*z + d*w = e_val.
    """
    fP = a * P[0] + b * P[1] + c * P[2] + d * P[3]
    fQ = a * Q[0] + b * Q[1] + c * Q[2] + d * Q[3]
    tol = 1e-6
    if abs(fP - e_val) < tol and abs(fQ - e_val) < tol:
        return [P, Q]
    if (fP - e_val) * (fQ - e_val) > tol:
        return None
    if abs(fQ - fP) < tol:
        return None
    t = (e_val - fP) / (fQ - fP)
    if t < -tol or t > 1 + tol:
        return None
    I = P + t * (Q - P)
    return I

def find_intersection_points(vertices, a, b, c, d, e_val):
    """
    For a given set of 4D vertices and a hyperplane,
    compute the unique intersection points.
    """
    edges = get_tesseract_edges(vertices)
    intersection_points = []
    for edge in edges:
        P, Q = edge
        res = hyperplane_intersection(P, Q, a, b, c, d, e_val)
        if res is None:
            continue
        if isinstance(res, list):
            for pt in res:
                intersection_points.append(pt)
        else:
            intersection_points.append(res)
    unique_points = []
    for pt in intersection_points:
        if not any(np.linalg.norm(pt - up) < 1e-6 for up in unique_points):
            unique_points.append(pt)
    return unique_points

def project_to_3d(points, a, b, c, d, e_val):
    """
    Given 4D points (which lie on the hyperplane), project them onto a 3D coordinate system.
    A particular point on the hyperplane and three orthonormal basis vectors (via Gram–Schmidt)
    are used to express the 4D points in 3D.
    """
    tol = 1e-6
    if abs(a) > tol:
        x0 = np.array([e_val / a, 0, 0, 0], dtype=float)
    elif abs(b) > tol:
        x0 = np.array([0, e_val / b, 0, 0], dtype=float)
    elif abs(c) > tol:
        x0 = np.array([0, 0, e_val / c, 0], dtype=float)
    elif abs(d) > tol:
        x0 = np.array([0, 0, 0, e_val / d], dtype=float)
    else:
        raise ValueError("Invalid hyperplane coefficients: all zero.")
    n = np.array([a, b, c, d], dtype=float)
    n = n / np.linalg.norm(n)
    basis = []
    candidate_vectors = [
        np.array([1, 0, 0, 0], dtype=float),
        np.array([0, 1, 0, 0], dtype=float),
        np.array([0, 0, 1, 0], dtype=float),
        np.array([0, 0, 0, 1], dtype=float),
    ]
    for v in candidate_vectors:
        proj = np.dot(v, n) * n
        v_perp = v - proj
        if np.linalg.norm(v_perp) > tol:
            v_perp = v_perp / np.linalg.norm(v_perp)
            basis.append(v_perp)
        if len(basis) == 3:
            break
    projected_points = []
    for pt in points:
        diff = pt - x0
        coords = [np.dot(diff, bvec) for bvec in basis]
        projected_points.append(coords)
    return np.array(projected_points), basis, x0

def create_convex_hull_mesh(projected_points):
    """
    Compute the convex hull of the projected 3D points.
    Returns:
      - vertices: the projected 3D points.
      - faces: an array (n_faces x 3) of triangle indices.
    """
    hull = ConvexHull(projected_points)
    faces = hull.simplices
    return projected_points, faces

# --------------------------
# Custom OpenGL Widget with FPS-style controls, momentum, improved edge rendering,
# plus floor, gravity, jumping with surface detection, and mouse look.
# --------------------------
class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, shapes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_items = []
        for vertices, color in shapes:
            self.shape_items.append({
                "vertices": vertices,
                "color": color,
                "mesh_item": None,
                "edge_item": None,
                "convex_vertices": None
            })
        self.theta = 3  # hyperplane rotation about the W-Z plane
        self.moveSpeed = 50.0
        self.rotateSpeed = 50.0
        self.zoomSpeed = 0.5  # zoom keys are disabled
        self.opts['fov'] = 120  # increased FOV

        self.camera_pos = np.array([10.0, 0.0, 10.0], dtype=float)
        self.azimuth = 0.0
        self.elevation = 45.0
        self.camera_distance = 1.0

        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.angular_velocity = np.array([0.0, 0.0], dtype=float)
        self.keysDown = {}
        self.setWindowTitle('3D Intersection of Multiple 4D Shapes and Hyperplane')
        # Floor at z = -1.
        floor_color = (0.2, 0.2, 0.2, 1.0)
        floor_vertices = np.array([
            [-100, -100, -1],
            [ 100, -100, -1],
            [ 100,  100, -1],
            [-100,  100, -1]
        ], dtype=float)
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        floor_meshdata = MeshData(vertexes=floor_vertices, faces=floor_faces)
        self.floor_item = gl.GLMeshItem(meshdata=floor_meshdata,
                                        shader='shaded',
                                        smooth=True,
                                        glOptions='opaque',
                                        faceColor=floor_color)
        self.addItem(self.floor_item)

        # For jump logic.
        self.can_jump = False

        self.debug_label = QtWidgets.QLabel(self)
        self.debug_label.setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 150); color: white; }")
        self.debug_label.setGeometry(10, 10, 350, 120)
        self.debug_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)  # ~60 FPS

        self.last_mouse_pos = None
        self.setMouseTracking(True)

        self.update_geometry()
        self.update_camera()

    def initializeGL(self):
        super().initializeGL()
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.3, 0.3, 0.3, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.7, 0.7, 0.7, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 10, 10, 1))

    def update_geometry(self):
        a = 1; b = 1
        c = np.cos(self.theta) - np.sin(self.theta)
        d = np.sin(self.theta) + np.cos(self.theta)
        player_4d = np.array([self.camera_pos[0], self.camera_pos[1], self.camera_pos[2], 0])
        e_val = a * player_4d[0] + b * player_4d[1] + c * player_4d[2] + d * player_4d[3]
        for shape in self.shape_items:
            vertices = shape["vertices"]
            intersection_points = find_intersection_points(vertices, a, b, c, d, e_val)
            if len(intersection_points) < 4:
                if shape["mesh_item"] is not None:
                    self.removeItem(shape["mesh_item"])
                    shape["mesh_item"] = None
                if shape["edge_item"] is not None:
                    self.removeItem(shape["edge_item"])
                    shape["edge_item"] = None
                shape["convex_vertices"] = None
                continue
            projected_points, basis, x0 = project_to_3d(intersection_points, a, b, c, d, e_val)
            vertices_3d, faces = create_convex_hull_mesh(projected_points)
            shape["convex_vertices"] = vertices_3d
            meshdata = MeshData(vertexes=vertices_3d, faces=faces)
            if shape["mesh_item"] is None:
                shape["mesh_item"] = gl.GLMeshItem(meshdata=meshdata,
                                                    shader='shaded',
                                                    smooth=True,
                                                    glOptions='opaque',
                                                    faceColor=shape["color"],
                                                    color=shape["color"])
                self.addItem(shape["mesh_item"])
            else:
                shape["mesh_item"].setMeshData(meshdata=meshdata)
            face_normals = {}
            for i, face in enumerate(faces):
                v0 = vertices_3d[face[0]]
                v1 = vertices_3d[face[1]]
                v2 = vertices_3d[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                norm_val = np.linalg.norm(normal)
                if norm_val > 1e-6:
                    normal = normal / norm_val
                face_normals[i] = normal
            edge_faces = {}
            for i, face in enumerate(faces):
                for j in range(3):
                    edge = tuple(sorted((face[j], face[(j + 1) % 3])))
                    edge_faces.setdefault(edge, []).append(i)
            filtered_edges = []
            angle_threshold = np.cos(np.radians(5))
            for edge, face_list in edge_faces.items():
                if len(face_list) == 1:
                    filtered_edges.append(edge)
                elif len(face_list) == 2:
                    n1 = face_normals[face_list[0]]
                    n2 = face_normals[face_list[1]]
                    if np.dot(n1, n2) < angle_threshold:
                        filtered_edges.append(edge)
                else:
                    filtered_edges.append(edge)
            line_points = []
            for edge in filtered_edges:
                p1 = vertices_3d[edge[0]]
                p2 = vertices_3d[edge[1]]
                line_points.append(p1)
                line_points.append(p2)
            if len(line_points) > 0:
                line_points = np.array(line_points)
            else:
                line_points = np.empty((0, 3))
            if shape["edge_item"] is None:
                shape["edge_item"] = gl.GLLinePlotItem(pos=line_points, color=(1,1,1,1), width=2, mode='lines')
                shape["edge_item"].setGLOptions('opaque')
                self.addItem(shape["edge_item"])
            else:
                shape["edge_item"].setData(pos=line_points)
                shape["edge_item"].setGLOptions('opaque')

    def update_camera(self):
        yaw = np.radians(self.azimuth)
        pitch = np.radians(self.elevation)
        forward = np.array([np.cos(pitch)*np.cos(yaw),
                            np.cos(pitch)*np.sin(yaw),
                            np.sin(pitch)])
        target = self.camera_pos + forward
        self.setCameraPosition(pos=pg.Vector(self.camera_pos),
                               distance=self.camera_distance,
                               azimuth=self.azimuth,
                               elevation=self.elevation)

    def animate(self):
        dt = 0.016
        acceleration = np.array([0.0, 0.0, 0.0])
        angular_acc = np.array([0.0, 0.0])
        yaw = np.radians(self.azimuth)
        pitch = np.radians(self.elevation)
        forward = np.array([np.cos(pitch)*np.cos(yaw),
                            np.cos(pitch)*np.sin(yaw),
                            np.sin(pitch)])
        right = np.cross(forward, np.array([0,0,1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1,0,0])
        right /= np.linalg.norm(right)
        up = np.array([0,0,1], dtype=float)
        if self.keysDown.get(Qt.Key_W, False):
            acceleration -= forward
        if self.keysDown.get(Qt.Key_S, False):
            acceleration += forward
        if self.keysDown.get(Qt.Key_A, False):
            acceleration += right
        if self.keysDown.get(Qt.Key_D, False):
            acceleration -= right
        self.velocity += acceleration * self.moveSpeed * dt
        self.angular_velocity += angular_acc * self.rotateSpeed * dt
        self.velocity *= 0.9
        self.angular_velocity *= 0.8
        # Apply stronger gravity.
        gravity = -20.0
        self.velocity[2] += gravity * dt
        self.camera_pos += self.velocity * dt

        # --- Collision Resolution with Floor ---
        floor_height = -1.0
        ground_level = floor_height + 1.0  # i.e. 0.0
        if self.camera_pos[2] < ground_level:
            self.camera_pos[2] = ground_level
            if self.velocity[2] < 0:
                self.velocity[2] = 0

        if self.camera_pos[2] < -10:
            self.camera_pos = np.array([10.0, 0.0, 10.0], dtype=float)
            self.velocity = np.zeros(3)

        user_radius = 0.5
        for shape in self.shape_items:
            if shape["convex_vertices"] is not None:
                vertices_3d = shape["convex_vertices"]
                center = np.mean(vertices_3d, axis=0)
                radii = np.linalg.norm(vertices_3d - center, axis=1)
                shape_radius = np.max(radii)
                diff = self.camera_pos - center
                dist = np.linalg.norm(diff)
                min_dist = user_radius + shape_radius
                if dist < min_dist:
                    if dist > 1e-6:
                        penetration = min_dist - dist
                        correction = (diff / dist) * penetration
                        self.camera_pos += correction
                    else:
                        self.camera_pos[0] += min_dist

        # --- Surface Detection for Jumping ---
        # We'll set self.can_jump if either on floor or on a shape with a sufficiently vertical contact normal.
        self.can_jump = False
        # Check floor.
        if abs(self.camera_pos[2] - ground_level) < 0.05:
            self.can_jump = True
        else:
            up_vector = np.array([0,0,1], dtype=float)
            normal_threshold = 0.7
            for shape in self.shape_items:
                if shape["convex_vertices"] is not None:
                    vertices_3d = shape["convex_vertices"]
                    center = np.mean(vertices_3d, axis=0)
                    radii = np.linalg.norm(vertices_3d - center, axis=1)
                    shape_radius = np.max(radii)
                    diff = self.camera_pos - center
                    dist = np.linalg.norm(diff)
                    min_dist = user_radius + shape_radius
                    if dist < min_dist + 0.05:  # small tolerance
                        contact_normal = diff / (dist + 1e-6)
                        if np.dot(contact_normal, up_vector) > normal_threshold:
                            self.can_jump = True
                            break

        self.update_camera()
        # --- Update Debug Overlay ---
        debug_text = (f"Plane angle: {np.degrees(self.theta):.2f} deg\n"
                      f"Camera pos: {self.camera_pos}\n"
                      f"Velocity: {self.velocity}\n"
                      f"Azimuth: {self.azimuth:.2f}, Elevation: {self.elevation:.2f}\n"
                      f"Can Jump: {self.can_jump}")
        self.debug_label.setText(debug_text)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.theta += delta * 0.05
        self.update_geometry()
        event.accept()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            self.last_mouse_pos = event.pos()
        delta = event.pos() - self.last_mouse_pos
        sensitivity = 0.2
        # Inverted mouse look.
        self.azimuth -= delta.x() * sensitivity
        self.elevation += delta.y() * sensitivity
        self.elevation = np.clip(self.elevation, -89, 89)
        self.last_mouse_pos = event.pos()
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            # Jump only if surface contact is detected.
            if self.can_jump:
                self.velocity[2] = 35.0  # increased jump impulse
                self.can_jump = False  # disable further jumping until contact is re-established
        else:
            self.keysDown[key] = True
        event.accept()

    def keyReleaseEvent(self, event):
        self.keysDown[event.key()] = False
        event.accept()

# --------------------------
# Main Execution
# --------------------------
def main():
    shapes_file = "shapes.json"
    shapes = load_shapes_json(shapes_file)
    app = QtWidgets.QApplication(sys.argv)
    view = CustomGLViewWidget(shapes)
    view.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
