
'''
Smoothing w/ GPs + adding in spatial information
'''

import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube

from scipy.optimize import differential_evolution

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic

from gp_extras.kernels import LocalLengthScalesKernel

from paths import fourteenB_wGBT_HI_file_dict

cube = SpectralCube.read(fourteenB_wGBT_HI_file_dict['Cube'])

spec_axis = cube.spectral_axis
chan_width = np.abs(np.diff(spec_axis[:2])[0])


# Signal is modelled by the first RBF. The second term is the noise,
# accounting for correlation w/ another RBF component
# kernel = 0.003**2 * RBF(length_scale=50, length_scale_bounds=(5, 1e3)) \
#     + RBF(length_scale=2, length_scale_bounds=(1, 5)) \
#     * WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-4))

kernel = 0.003**2 * RBF(length_scale=30, length_scale_bounds=(3, 100)) \
    + WhiteKernel(noise_level=2.2e-6, noise_level_bounds=(1e-6, 1e-4))

# kernel = 1 * RationalQuadratic(length_scale=30, length_scale_bounds=(3, 100),
#                                alpha=1e5, alpha_bounds=(100, 1e7)) \
#     + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-7, 1e-4))

# kernel = 1 * Matern(length_scale=30, length_scale_bounds=(3, 100),
#                     nu=1.5) \
#     + 1 * RBF(length_scale=2, length_scale_bounds=(1, 100)) \
#     + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-7, 1e-4))


# Check on a subset of the cube
for i in range(604, 800):
    for j in range(634, 700):

        spec = cube[:, i, j]
        spec2 = cube[:, i, j + 1]

        X = np.tile(np.arange(spec.size)[:, None], (1, 2))
        y = np.vstack([spec.value, spec2.value]).T

        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
        gp.fit(X, y)

        y_mean, y_cov = gp.predict(X, return_cov=True)

        plt.plot(y, alpha=0.6, drawstyle='steps-mid')
        plt.plot(X, y_mean, 'k', lw=3, zorder=9)
        for z in range(2):
            plt.fill_between(X[:, 0], y_mean[:, z] - np.sqrt(np.diag(y_cov)),
                             y_mean[:, z] + np.sqrt(np.diag(y_cov)),
                             alpha=0.3, color='k')
        # plt.plot(X, y_mean_ll, 'g', lw=3, zorder=9)
        plt.axhline(0.0)
        plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
                  % (kernel, gp.kernel_,
                     gp.log_marginal_likelihood(gp.kernel_.theta)), fontsize=10)
        plt.draw()
        raw_input("{0}, {1}?".format(i, j))
        plt.clf()
