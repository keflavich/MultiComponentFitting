
'''
Tests for writing a spatially-aware spectral fitting routine

* hierarchical approach

    1. Data level - each spectra to be fit
    2. Individual spectrum parameters - how to handle varying # of params?
    3. Group parameter level - assume that each parameter for individual
        spectra belongs to a family drawn from the same distribution
        * Code in the spatial covariance here? Assume known correlation
         on a "beam" scale (or equivalent)
    4. Constraints on the group parameters - here is where the parameters
        for gradients would live. These would need a prior guess on what
        are acceptable ranges (a known large-scale gradient?).
'''

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel,
                                              ConstantKernel as C)

import emcee


def parameter_distrib(pos, value, grad):

    return value + grad * pos


def model_function(var, lscale, noise_level,
                   var_value_bounds=(1e-4, 1e-2),
                   length_scale_bounds=(3, 100),
                   noise_level_bounds=(1e-6, 1e-4)):

    kernel = C(var, constant_value_bounds=var_value_bounds) * \
        RBF(length_scale=lscale,
            length_scale_bounds=length_scale_bounds) \
        + WhiteKernel(noise_level=noise_level,
                      noise_level_bounds=noise_level_bounds)

    return kernel


def gp_model_function(X, y, params, **params_kwargs):

    kernel = model_function(*params, **params_kwargs)
    gp = GaussianProcessRegressor(kernel=kernel)

    gp.y_train_ = y
    gp.X_train_ = X

    return gp


def log_likelihood():
    pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from spectral_cube import SpectralCube

    cube = SpectralCube.read("gauss_cube_x2.fits")

    # Pick a 3x3 region to start
    subcube = cube[:, 50:53, 50:53]

    # Naive fitting of GP w/ all spectra and w/o spatial information
    # Use pixel channel labels (for now)
    X = np.tile(np.arange(subcube.shape[0])[:, None], (1, 9))
    y = np.vstack([subcube[:, i, j].value for i in range(3)
                   for j in range(3)]).T
    gp = gp_model_function(X, y, (1e-3, 30, 1e-5))

    gp.fit(X, y)

    y_mean, y_cov = gp.predict(X, return_cov=True)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for n, ax in enumerate(axes.ravel()):

        ax.plot(y[:, n], alpha=0.6, drawstyle='steps-mid')
        ax.plot(X, y_mean[:, n], 'k', lw=3, zorder=9)
        ax.fill_between(X[:, 0], y_mean[:, n] - np.sqrt(np.diag(y_cov)),
                        y_mean[:, n] + np.sqrt(np.diag(y_cov)),
                        alpha=0.3, color='k')
