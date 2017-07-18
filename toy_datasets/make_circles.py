"""
Functions to generate toy datasets for the multiple component hacking week.
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

if __name__ != "__main__":
    from .math_utils import planar_tilt, periodic_wiggle
else: # forgive my non-pythonic blasphemy, but I like to %run my scripts
    from math_utils import planar_tilt, periodic_wiggle

# TODO: SCRAP FOR PARTS!!! a much better approach is the have a similar
#       function for just one velocity component that generates a scatter
#       in ra/dec. Then all the subsequent tinkering can be done independently
#       of the initial number of components / dimensions
def two_circles(n_samples=100, shuffle=True, noise=None, random_state=None,
                i_range=[0, np.pi], i_tilt=[0, 0, 0], i_wiggle=[0, 0],
                j_range=[0, np.pi], j_tilt=[0, 0, 0], j_wiggle=[0, 0]):
    """
    Generates a 3D cloud of points sampled from two overlaping circles.

    Taken from `sklearn.datasets.make_moons` and sprinkled
    with extra dimensionality and additional parametrizations.

    ++ TODO #0: pp --> ppv
    ++ TODO #1: make the components overlap in velocity.
    -- TODO #2: add intensity / line widths structures? Nah do it elsewhere...

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

    Returns
    -------
    D : array of shape [n_samples, 3]
        The generated samples.

    l : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    n_samples_i = n_samples // 2
    n_samples_j = n_samples - n_samples_i

    generator = utils.check_random_state(random_state)

    #if i_wiggle != [0, 0] or j_wiggle != [0, 0]:
    #    raise NotImplementedError("WIP, sorry!")

    # generating the first circle
    i_circ_x = np.cos(np.linspace(*i_range, num=n_samples_i))
    i_circ_y = np.sin(np.linspace(*i_range, num=n_samples_i))
    i_circ_z = planar_tilt(i_circ_x, i_circ_y, *i_tilt)
    i_circ_z += periodic_wiggle(i_circ_x, i_circ_y, *i_wiggle)

    # generating the second circle
    j_circ_x = 1 - np.cos(np.linspace(*j_range, num=n_samples_j))
    j_circ_y = 1 - np.sin(np.linspace(*j_range, num=n_samples_j)) - .5
    j_circ_z = planar_tilt(j_circ_x, j_circ_y, *j_tilt)
    j_circ_z += periodic_wiggle(j_circ_x, j_circ_y, *j_wiggle)

    # stacking the 3D point cloud in a single array
    D = np.vstack((np.append(i_circ_x, j_circ_x),
                   np.append(i_circ_y, j_circ_y),
                   np.append(i_circ_z, j_circ_z))).T
    # storing the true allegiance of the datapoints
    l = np.hstack([np.zeros(n_samples_j, dtype=np.intp),
                   np.ones(n_samples_i, dtype=np.intp)])

    if shuffle:
        D, l = utils.shuffle(D, l, random_state=generator)

    if noise is not None:
        D += generator.normal(scale=noise, size=D.shape)

    return D, l

if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure('z-axis errors')
    ax = fig.add_subplot(111, projection='3d')

    D, l = two_circles(1000, i_range=[-np.pi/2, np.pi/2], i_tilt=[1, .2, -.5],
                       j_tilt=[0.5, -.4, .3], noise=0.1)

    ax.scatter(*D[l==0, :].T)
    ax.scatter(*D[l==1, :].T)

    plt.show()
