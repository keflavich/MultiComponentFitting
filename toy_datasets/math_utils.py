"""
Utility functions for transformations in the parameter space.
"""

import numpy as np


def intensity_from_density(x, y, nbins=300, gauss_width=None):
    """
    Makes a 2d histogram of the data, then smoothes it to
    get mock intensities

    TODO: normalize it or set a scaling parameter?
    """

    gauss_width = nbins / 30
    from scipy.stats import binned_statistic_2d
    density_grid = binned_statistic_2d(x, y, np.ones_like(x), 'count',
            bins=nbins, expand_binnumbers=True)
    import scipy.ndimage as ndimage
    smooth_density = ndimage.gaussian_filter(density_grid.statistic,
            sigma=gauss_width, order=0)
    # remap the resultant array back into the cloud point form
    xy_ind = np.ravel_multi_index(density_grid.binnumber-1,
                                  density_grid.statistic.shape)
    # flatten always returns a copy, ravel tries to give a view
    intensities = smooth_density.ravel()[xy_ind]

    return intensities, smooth_density


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


def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def himmelblau_grid(shape, xlims=(-6, 6), ylims=(-6, 6)):
    """
    Himmelblau's funciton computed over a given x/y range.
    """
    x = np.linspace(*xlims, num=shape[0])
    y = np.linspace(*ylims, num=shape[1])
    xx, yy = np.meshgrid(x, y)

    return himmelblau(xx, yy)


def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2


def rosenbrock_grid(shape, xlims=(-1.5, 2), ylims=(-0.5, 3), **kwargs):
    """
    Rosenbrock's funciton computed over a given x/y range.
    """
    x = np.linspace(*xlims, num=shape[0])
    y = np.linspace(*ylims, num=shape[1])
    xx, yy = np.meshgrid(x, y)

    return rosenbrock(xx, yy, **kwargs)
