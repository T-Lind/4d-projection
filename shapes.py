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