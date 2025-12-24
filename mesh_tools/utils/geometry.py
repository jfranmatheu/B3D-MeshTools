from math import dist, sqrt
import math
from typing import Tuple

from mathutils import Vector
from mathutils.geometry import convex_hull_2d, delaunay_2d_cdt, area_tri, intersect_point_line

from .math import _as_vec3


def shortest_distance_between_strokes(stroke1, stroke2):
    """ Get the shortest distance between two strokes. """
    min_dist = float('inf')
    min_points = None
    indices = None
    
    for i1, p1 in enumerate(stroke1):
        for i2, p2 in enumerate(stroke2):
            d = dist(p1, p2)
            if d < min_dist:
                min_dist = d
                min_points = (p1, p2)
                indices = (i1, i2)
    
    return indices, min_points, min_dist


def sample_points_along_line(start: Vector, end: Vector, num_points: int) -> list[Vector]:
    """Returns list of evenly spaced points along line from start to end (inclusive)."""
    if num_points < 2:
        return [start.copy()]
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        points.append(start.lerp(end, t))
    return points


def calc_tri_area(A: Vector, B: Vector, C: Vector) -> float:
    """Calculate the area of a triangle (supports 2D or 3D vectors)."""
    a = _as_vec3(A)
    b = _as_vec3(B)
    c = _as_vec3(C)
    area_vec: Vector = (b - a).cross(c - a)  # type: ignore[assignment]
    return 0.5 * area_vec.length


def is_point_in_tri(point: Vector, tri: Tuple[Vector, Vector, Vector], eps: float = 1e-9) -> bool:
    """
    Check if a point is inside a triangle using barycentric coordinates.

    Works for 2D or 3D vectors (3D case is treated in 3D space).
    """
    p = _as_vec3(point)
    a = _as_vec3(tri[0])
    b = _as_vec3(tri[1])
    c = _as_vec3(tri[2])

    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= eps:
        # Degenerate triangle
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Inside if u,v are non-negative and u+v <= 1 (with tolerance)
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps

def is_point_inside_circle(point: Vector, center: Vector, radius: float) -> bool:
    """Check if a point is inside a circle."""
    return (point - center).length_squared <= radius * radius

def dist_to_segment(p: Vector, a: Vector, b: Vector) -> float:
    closest, percent = intersect_point_line(p, a, b)
    if percent < 0:
        return (p - a).length
    elif percent > 1:
        return (p - b).length
    else:
        return (p - closest).length


def closest_point_to_segment(segment_start: Vector, segment_end: Vector, points: list[Vector]) -> Tuple[int, float]:
    """ Get the index and distance of the closest point to the segment. """
    min_distance = float('inf')
    index = -1
    for i, point in enumerate(points):
        d = dist_to_segment(point, segment_start, segment_end)
        if d < min_distance:
            min_distance = d
            index = i
    return index, min_distance

def closest_segment_to_point(point: Vector, segments: list[Tuple[Vector, Vector]]) -> Tuple[int, float]:
    """ Get the index and distance of the closest segment to the point. """
    min_distance = float('inf')
    index = -1
    for i, segment in enumerate(segments):
        d = dist_to_segment(point, segment[0], segment[1])
        if d < min_distance:
            min_distance = d
            index = i
    return index, min_distance

def check_line_circle_intersect(line: list[Vector], center: Vector, radius: float) -> bool:
    """Check if a line intersects with a circle."""
    for i in range(len(line) - 1):
        if is_point_inside_circle(line[i], center, radius) or is_point_inside_circle(line[i + 1], center, radius):
            return True
        if dist_to_segment(center, line[i], line[i + 1]) <= radius:
            return True
    return False

def line_segment_inside_or_intersecting_circle(p1: Vector, p2: Vector, center: Vector, radius: float) -> bool:
    """
    Check if the line segment from p1 to p2 is fully inside the circle
    or intersects/touches its boundary.
    
    Args:
        p1 (Vector): Start point of the line segment (2D or 3D, but circle is in XY plane)
        p2 (Vector): End point of the line segment
        center (Vector): Center of the circle (2D or 3D)
        radius (float): Radius of the circle
    
    Returns:
        bool: True if segment is fully inside OR intersects/touches the circle
              False if entirely outside with no contact
    """
    # Work in 2D (XY plane) - ignore Z if present
    A = Vector((p1.x, p1.y))
    B = Vector((p2.x, p2.y))
    C = Vector((center.x, center.y))
    
    # Vector from A to B
    AB = B - A
    segment_length_sq = AB.length_squared
    
    # Vector from A to circle center C
    AC = C - A
    
    # If segment is degenerate (point)
    if segment_length_sq < 1e-8:
        return AC.length_squared <= radius * radius
    
    # Check if both endpoints are inside the circle → whole segment is inside
    if AC.length_squared <= radius * radius and (C - B).length_squared <= radius * radius:
        return True
    
    # Project circle center onto the line and find closest point
    # t = projection parameter (clamped between 0 and 1 later)
    t = AC.dot(AB) / segment_length_sq
    
    # Closest point on the infinite line
    closest = A + AB * t
    
    # Distance from circle center to the line
    dist_to_line_sq = (C - closest).length_squared
    
    # If the closest distance is greater than radius → no intersection possible
    if dist_to_line_sq > radius * radius:
        return False
    
    # Now check if the closest point is on the segment (t in [0,1])
    # Or if we already know endpoints are inside (handled above)
    if 0 <= t <= 1:
        return True
    
    # Otherwise: the perpendicular foot is outside the segment,
    # but we already checked both endpoints aren't both inside.
    # Still possible to intersect if one or both ends are inside — but we caught that earlier.
    # So if we reach here and dist_to_line <= radius, it means intersection at endpoint or crossing.
    # But safest: fall back to endpoint check (already done) and intersection logic.
    
    # More precise: compute discriminant for exact intersection points
    a = segment_length_sq
    b = 2 * AC.dot(AB)
    c = AC.length_squared - radius * radius
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False
    
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    
    # If any intersection point lies on the segment
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)
