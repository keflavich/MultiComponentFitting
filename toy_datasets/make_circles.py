"""
Functions to generate toy datasets for the multiple component hacking week.

Adapted from `sklearn.datasets.make_moons`, with generalized circle parameters
and extra dimensionality added.
"""

# The preliminary philosophy for this module is as follows.
#
# I see the generation of toy data as a tast that can be implemented in four
# sequential steps:
#   1. Generate one or more clouds of points in 2d
#   2. Transform the scatter density for each into ra/dec intensity, add noise
#   3. Add more parameter dimensions through implisit transforms - e.g. impose
#      velocity gradients, line broadenings, etc.
#   4. Grid to taste, make fits files or the true parameters and ideal / noisy
#      toy spectral cubes to test on.
#
# a) individual functions generating points in pp space [make_cloud.py]
# b) conversion function that converts these points into fits files [TODO]
#    - take the z-axis values, convolve them with a gaussian kernel, treat
#      the resultant grid as Vlsr / any other parameter
#    - once the grid is read, can use make both parcubes and toy spec. cubes
# c) a routine to export the ppv cloud in leodis-friendly format [TODO]
# d) a minimal class definition that handles multi-dimensionality and
#    remembers which dimensions are spatial, radial, or neither [TODO]
# e) a few `math_unils.py` functions that transform spatial coordinates into
#    other types - for the joy of toying around and generating interesting
#    testing cases. [math_units.py]

import numpy as np
from sklearn import utils

try:
    #from .math_utils import planar_tilt, periodic_wiggle,
    from . import math_utils
except SystemError:
    # forgive my non-pythonic blasphemy, but I like to %run my scripts
    import math_utils
    #from math_utils import planar_tilt, periodic_wiggle, intensity_from_density

def two_circles(n_samples=100, i_xy0=(0, 0), j_xy0=(1, .5), i_range=[0, np.pi],
                i_tilt=[0, 0, 0], i_wiggle=[0, 0], j_range=[0, np.pi],
                j_tilt=[0, 0, 0], j_wiggle=[0, 0], **kwargs):
    """
    Generates a 3D cloud of points sampled from two overlaping circles.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.

    i_tilt / j_tilt : sequence of [a, b, c] shape
        Adds a planar tilt in the z-axis given by the following
        implicit equation: z = a - b*x - c*y.

    i_wiggle / j_wiggle : sequence of [q, r] shape
        Adds a periodic perturbation wiggle to each circle (WIP)

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    TODO: this doesn"t belong here, time to move out

    Returns
    -------
    D : array of shape [n_samples, 3]
        The generated samples.

    l : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    n_samples_i = n_samples // 2
    n_samples_j = n_samples - n_samples_i + 20

    # generating the first circle
    i_circ_x, i_circ_y = make_circle(n_samples_i, i_xy0,
            phase_range=i_range, **kwargs)
    # vlsr and intensity values for the first circe
    i_velocity = math_utils.planar_tilt(i_circ_x, i_circ_y, *i_tilt)
    i_velocity += math_utils.periodic_wiggle(i_circ_x, i_circ_y, *i_wiggle)
    i_intensity, _ = math_utils.intensity_from_density(i_circ_x, i_circ_y)

    # generating the second circle
    j_circ_x, j_circ_y = make_circle(n_samples_j, j_xy0,
            phase_range=j_range, **kwargs)
    # vlsr and intensity values for the second circe
    j_velocity = math_utils.planar_tilt(j_circ_x, j_circ_y, *j_tilt)
    j_velocity += math_utils.periodic_wiggle(j_circ_x, j_circ_y, *j_wiggle)
    j_intensity, _ = math_utils.intensity_from_density(j_circ_x, j_circ_y)

    #xspace = np.concatenate([i_circ_x, j_circ_x])
    #yspace = np.concatenate([i_circ_y, j_circ_y])

    # pos_x, pos_y, velocity, intensity, line width
    i_data = [i_circ_x, i_circ_y, i_velocity, i_intensity]
    j_data = [j_circ_x, j_circ_y, j_velocity, j_intensity]

    D, l = assemble_components(i_data, j_data, **kwargs)

    return D, l

def make_circle(n_samples=100, origin=(0.0, 0.0), r=1, phase_range=(0, np.pi),
                random_state=None, noise=None, **kwargs):
    """
    Generates a 3D cloud of points a circle on the xy-grid.

    Adapted from `sklearn.datasets.make_moons`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.

    origin : sequence of two floats
        Sets the center of the circle

    radius : float
        Radius of the circle

    phase_range : phase length of a circle segment in radians

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    Returns
    -------
    XY : array of shape [n_samples, 2]
         The generated samples.
    """
    x0, y0 = origin
    circ_x = x0 + r * np.cos(np.linspace(*phase_range, num=n_samples))
    circ_y = y0 + r * np.sin(np.linspace(*phase_range, num=n_samples))

    generator = utils.check_random_state(random_state)

    if noise is not None:
        circ_x += generator.normal(scale=noise, size=circ_x.shape)
        circ_y += generator.normal(scale=noise, size=circ_x.shape)

    return np.array([circ_x, circ_y])

# whaaat since when does it break for python 2?
#def assemble_circles(*components, shuffle=True, noise=None, random_state=None):
def assemble_components(*components, **kwargs):
    """
    Assembly of multiple components.

    Parameters
    ----------

    components : an iterable of arrays of [n_dimensions, n_points] shape

    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.

    Returns
    -------
    D : array of shape [n_samples, 3]
        The generated samples.

    l : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.

    """

    shuffle = kwargs.pop("shuffle", True)
    random_state = kwargs.pop("random_state", None)

    generator = utils.check_random_state(random_state)

    # stacking the 3D point cloud in a single array
    D = np.concatenate(components, axis=1).T

    # storing the true allegiance of the datapoints
    n_samples = [len(c[0]) for c in components]
    l = np.concatenate(
        [np.zeros(n, dtype=int) + i for i, n in enumerate(n_samples)])

    if shuffle:
        D, l = utils.shuffle(D, l, random_state=generator)

    return D, l
