import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

@dataclass
class Shape4D:
    vertices: np.ndarray
    edges: List[Tuple[int, int]]
    faces: List[Tuple[int, ...]]
    face_colors: List[Tuple[int, int, int, int]]  # RGBA
    
    def save(self, filename: str):
        data = {
            'vertices': self.vertices.tolist(),
            'edges': self.edges,
            'faces': self.faces,
            'face_colors': self.face_colors
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def get_projected_face_depths(self, projected_vertices: np.ndarray) -> List[float]:
        """Calculate average Z depth of each face after projection"""
        depths = []
        for face in self.faces:
            face_verts = projected_vertices[list(face)]
            avg_depth = np.mean(face_verts[:, 2])  # Z coordinate
            depths.append(avg_depth)
        return depths
    
    def edge_intersects_face(self, edge_p1: np.ndarray, edge_p2: np.ndarray, 
                            face_points: List[np.ndarray]) -> bool:
        """Check if a 2D edge intersects with a 2D polygon face"""
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

        def segments_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            if o1 != o2 and o3 != o4: return True
            if o1 == 0 and on_segment(p1, p2, q1): return True
            if o2 == 0 and on_segment(p1, q2, q1): return True
            if o3 == 0 and on_segment(p2, p1, q2): return True
            if o4 == 0 and on_segment(p2, q1, q2): return True
            return False

        # Check if edge intersects any face edge
        for i in range(len(face_points)):
            face_p1 = face_points[i]
            face_p2 = face_points[(i + 1) % len(face_points)]
            if segments_intersect(edge_p1, edge_p2, face_p1, face_p2):
                return True
                
        return False

class ShapeLoader:
    @staticmethod
    def load_shape(filename: str) -> Shape4D:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        vertices = np.array(data['vertices'], dtype=float)
        edges = [tuple(e) for e in data['edges']]
        faces = [tuple(f) for f in data['faces']]
        face_colors = [tuple(c) for c in data.get('face_colors', 
                      [(0, 255, 0, 128)] * len(faces))]
        
        return Shape4D(vertices, edges, faces, face_colors)

def load_4ds(directory: str = "shapes") -> Dict[str, Shape4D]:
    """Load all .4ds files from the specified directory"""
    shapes = {}
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for filename in os.listdir(directory):
        if filename.endswith('.4ds'):
            filepath = os.path.join(directory, filename)
            shape_name = os.path.splitext(filename)[0]
            shapes[shape_name] = ShapeLoader.load_shape(filepath)
    
    return shapes