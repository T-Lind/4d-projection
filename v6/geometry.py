import numpy as np
from scipy.spatial import ConvexHull
from typing import Tuple, List, Optional, Union
from settings import Settings

class GeometryHelper:
    @staticmethod
    def intersect_edge_with_plane(A: Tuple[float, float, float], 
                                 B: Tuple[float, float, float],
                                 user_pos: Tuple[float, float, float], 
                                 plane_angle: float) -> Optional[Tuple[float, float, float]]:
        """
        Intersect line segment AB with the vertical plane that passes through user_pos
        and has normal n(theta) = (cos(theta), sin(theta), 0).
        """
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        U = np.array(user_pos, dtype=float)

        # Plane normal
        n = np.array([np.cos(plane_angle), np.sin(plane_angle), 0.0], dtype=float)

        AB = B - A
        AU = A - U

        denom = np.dot(n, AB)
        nom = -np.dot(n, AU)  # we want n dot ((A + t*AB) - U) = 0

        # If denom is 0, line is parallel to plane or no intersection
        if abs(denom) < 1e-12:
            return None

        t = nom / denom
        if 0.0 <= t <= 1.0:
            I = A + t * AB
            return tuple(I)
        else:
            return None

    @staticmethod
    def project_point_onto_plane_2D(point_3d: Tuple[float, float, float],
                                   user_pos: Tuple[float, float, float],
                                   plane_angle: float) -> Tuple[float, float]:
        """
        Convert a 3D intersection point to 2D plane coordinates (X,Z).
        """
        P = np.array(point_3d, dtype=float)
        U = np.array(user_pos, dtype=float)
        relative = P - U  # vector from user to intersection

        p_x = np.array([-np.sin(plane_angle), np.cos(plane_angle), 0.0], dtype=float)
        p_z = np.array([0.0, 0.0, 1.0], dtype=float)

        X = np.dot(relative, p_x)
        Z = np.dot(relative, p_z)
        return (X, Z)

    @staticmethod
    def compute_intersections(shape: dict, 
                            user_pos: np.ndarray, 
                            plane_angle: float) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """
        Compute intersection points and edges for a single shape.
        """
        pts_3d = shape['points']
        edges = shape['edges']
        
        # Gather intersection points
        intersection_points_3d = []
        for edge in edges:
            A_idx, B_idx = edge
            A_3d = pts_3d[A_idx]
            B_3d = pts_3d[B_idx]
            I_3d = GeometryHelper.intersect_edge_with_plane(A_3d, B_3d, user_pos, plane_angle)
            if I_3d is not None:
                intersection_points_3d.append(I_3d)

        # Convert to 2D plane coords
        intersection_points_2d = [GeometryHelper.project_point_onto_plane_2D(pt, user_pos, plane_angle)
                                for pt in intersection_points_3d]

        # Compute edges based on number of points
        edges_2d = []
        if len(intersection_points_2d) >= 3:
            try:
                hull = ConvexHull(intersection_points_2d)
                hull_indices = hull.vertices
                edges_2d = [(hull_indices[i], hull_indices[(i+1) % len(hull_indices)])
                           for i in range(len(hull_indices))]
            except:
                pass
        elif len(intersection_points_2d) == 2:
            edges_2d = [(0, 1)]

        return intersection_points_2d, edges_2d
    

    @staticmethod
    def get_user_convex_hull(user_pos: np.ndarray, plane_angle: float, settings: Settings) -> List[Tuple[float, float]]:
        """Generate the user's convex hull in 2D plane coordinates"""
        width, height = settings.movement.get_collision_dimensions(settings.display.pixels_per_unit)
        w, h = width/2, height/2  # half dimensions
        
        user_shape = [
            (-w, -h),  # Bottom left
            (w, -h),   # Bottom right
            (w, h),    # Top right
            (-w, h)    # Top left
        ]
        return user_shape

    @staticmethod
    def get_convex_hull(shape: dict, user_pos: np.ndarray, plane_angle: float) -> List[Tuple[float, float]]:
        """Get shape's convex hull in current intersection plane"""
        points_2d, _ = GeometryHelper.compute_intersections(shape, user_pos, plane_angle)
        if len(points_2d) < 3:
            return points_2d
        try:
            hull = ConvexHull(points_2d)
            return [points_2d[i] for i in hull.vertices]
        except:
            return points_2d
        
    @staticmethod
    def get_collision_normal(hull1: List[Tuple[float, float]], hull2: List[Tuple[float, float]], 
                            pos1: np.ndarray, pos2: np.ndarray) -> Optional[np.ndarray]:
        """Calculate normal vector pointing from hull2 to hull1"""
        if not hull1 or not hull2:
            return None
        
        # Get centers
        center1 = np.mean(hull1, axis=0)
        center2 = np.mean(hull2, axis=0)
        
        # Direction from shape to user
        normal = center1 - center2
        if np.linalg.norm(normal) < 1e-6:
            return np.array([1.0, 0.0])  # Default if centers are too close
            
        return normal / np.linalg.norm(normal)

    @staticmethod
    def check_collision(hull1: List[Tuple[float, float]], hull2: List[Tuple[float, float]], margin: int = 0) -> bool:
        """
        Check collision between two convex hulls using the Separating Axis Theorem.
        """

        if not hull1 or not hull2:
            return False
        
        
        for hull in [hull1, hull2]:
            for i in range(len(hull)):
                # Get the edge
                p1 = np.array(hull[i])
                p2 = np.array(hull[(i + 1) % len(hull)])
                edge = p2 - p1
                # Get the perpendicular axis
                axis = np.array([-edge[1], edge[0]])
                axis = axis / np.linalg.norm(axis)
                
                # Project both hulls onto the axis
                projections1 = [np.dot(np.array(point), axis) for point in hull1]
                projections2 = [np.dot(np.array(point), axis) for point in hull2]
                
                min1, max1 = min(projections1), max(projections1)
                min2, max2 = min(projections2), max(projections2)
                
                # Check for separation
                if max1 < min2 or max2 < min1:
                    return False  # No collision
        return True  # Collision detected