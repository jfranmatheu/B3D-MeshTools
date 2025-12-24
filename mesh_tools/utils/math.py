from typing import Tuple
from math import sqrt, acos, degrees, dist, pi, cos, sin
import numpy as np

from mathutils import Vector


def clamp(value: float | int, min_value: float | int = 0.0, max_value: float | int = 1.0) -> float | int:
    """ Clamp the value between the min and max values. """
    return min(max(value, min_value), max_value)

def map_value(val: float, src: Tuple[float, float] | Vector | list[float], dst: Tuple[float, float] | Vector | list[float] = (0.0, 1.0)) -> float:
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

def direction(_p1: Vector, _p2: Vector, _norm=True):
    """ Get the direction of the vector defined by the two given points. """
    if _norm:
        return (_p1 - _p2).normalized()
    else:
        return _p1 - _p2

def dotproduct(v1: np.ndarray | tuple[float, float], v2: np.ndarray | tuple[float, float]) -> float:
    """ Get the dot product of the two vectors. """
    return sum((a*b) for a, b in zip(v1, v2))

def length(v: np.ndarray | tuple[float, float]) -> float:
    """ Get the length of the vector. """
    return sqrt(dotproduct(v, v))

def angle_between(v1: np.ndarray | tuple[float, float], v2: np.ndarray | tuple[float, float], as_degrees: bool = True) -> float:
    """ Get the angle between the two vectors. """
    a = acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    return degrees(a) if as_degrees else a

def unit_vector(vector: np.ndarray) -> np.ndarray:
    """ Returns the unit vector of the vector"""
    return vector / np.linalg.norm(vector)

def angle_signed(vector1: np.ndarray, vector2: np.ndarray, as_degrees: bool = False) -> float:
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        return 0
        raise NotImplementedError('Too odd vectors =(')
    ang_rad = np.sign(minor) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return degrees(ang_rad) if as_degrees else ang_rad

def distance_between(_p1, _p2):
    """ Get the distance between the two points. """
    return dist(_p1, _p2) # hypot(_p1[0] - _p2[0], _p1[1] - _p2[1])
    #return math.sqrt((_p1[1] - _p1[0])**2 + (_p2[1] - _p2[0])**2)

def lineseg_dist(p, a, b):
    """ Get the distance between the point and the line segment defined by the two points. """
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))


def is_close(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    """ Check if two values are close to each other within specified tolerances. """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def is_close_vector(first, second, rel_tol=1e-09, abs_tol=0.0):
    """
    Check if two vectors are close to each other within specified tolerances.
    
    Parameters:
    -----------
    first : array-like
        First vector to compare
    second : array-like
        Second vector to compare
    rel_tol : float, optional
        Relative tolerance (default is 1e-09)
    abs_tol : float, optional
        Absolute tolerance (default is 0.0)
    
    Returns:
    --------
    bool
        True if vectors are close, False otherwise
    
    Raises:
    -------
    ValueError
        If vectors have different lengths
    """
    # Convert inputs to numpy arrays to ensure consistent handling
    first = np.asarray(first)
    second = np.asarray(second)
    
    # Check if vectors have the same length
    if first.shape != second.shape:
        raise ValueError("Vectors must have the same shape")
    
    # Calculate the absolute difference between corresponding elements
    abs_diff = np.abs(first - second)
    
    # Calculate the tolerance for each element
    tolerances = np.maximum(
        rel_tol * np.maximum(np.abs(first), np.abs(second)), 
        abs_tol
    )
    
    # Check if all absolute differences are within their respective tolerances
    return np.all(abs_diff <= tolerances)


def points_around(center: Vector | tuple[float, float] | list[float], r: float, N: int) -> list[Vector]:
    # Keep the ring generation in the same dimensionality as the input to avoid
    # Vector dimension mismatches (mouse coords are 2D for view2D raycasts).
    center_vec = Vector(center) if isinstance(center, (tuple, list)) else center.copy()
    dims = len(center_vec)
    if dims not in (2, 3):
        raise ValueError(f"points_around only supports 2D or 3D vectors, got {dims}D")
    points = []
    for i in range(N):
        angle = i * 2 * pi / N
        offset = Vector((cos(angle) * r, sin(angle) * r)) if dims == 2 else Vector((cos(angle) * r, sin(angle) * r, 0))
        points.append(center_vec + offset)
    return points


def _as_vec3(v: Vector) -> Vector:
    """Return a 3D Vector from 2D/3D input without mutating the original."""
    if len(v) == 3:
        return v
    return Vector((v.x, v.y, 0.0))
