"""
Utility functions for transformations in the parameter space.
"""

import numpy as np


def planar_tilt(x, y, a, b, c):
    """ Calculates a planar tilt along the third axis. """
    z = a - b * x - c * y

    return z


def periodic_wiggle(x, y, q, r):
    if q or r:
        raise NotImplementedError("Can't add ripples yet, sorry!")

    return np.zeros_like(x)


def radial_offset(x, y, x0, y0, a, b):
    if a or b:
        raise NotImplementedError("WIP, sorry!")

    return np.zeros_like(x)


def radially_decreasing(x, y, a=1):
    """
    Basically, a hill in 3D. Does this have its own name?
    
    Probably an okay choice for spatial intensity distribution.
    """
    z = a * (np.sin(x**2 + y**2) / (x**2 + y**2))**2

    return z
