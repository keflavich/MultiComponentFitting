# Example run for minicube_fit with three components

import numpy as np
import pymc3 as pm
import pylab as pl

from multicomponentfitting.spatially_aware_fitting.minicube_fit import minicube_model
from multicomponentfitting.spatially_aware_fitting.minicube_pymc import minicube_pymc_fit, spatial_covariance_structure

num_pts = 200
npix = 5
ncomps = 3

model = minicube_model(np.arange(num_pts),
                       0.5, -0.3, 0.2,
                       80, 1, -2,
                       10, 0.0, 0.00, npix=npix)

# simulate a "real sky" - no negative emission.
model[model < 0] = 0

model2 = minicube_model(np.arange(num_pts),
                        0.3, 0.3, -0.2,
                        100, -2, 1,
                        10, -0, 0.0, npix=npix)

model2[model2 < 0] = 0

model3 = minicube_model(np.arange(num_pts),
                        0.2, 0.02, -0.02,
                        100, 1, 1,
                        25, 0, 0.0, npix=npix)

model3[model3 < 0] = 0

model_with_noise = np.random.randn(*model.shape) * 0.1 + \
    model + model2 + model3

guess = {'amp0': 0.3, 'ampdx0': 0, 'ampdy0': 0,
         'center0': 40, 'centerdx0': 0, 'centerdy0': 0,
         'sigma0': 10, 'sigmadx0': 0, 'sigmady0': 0,
         'amp1': 0.3, 'ampdx1': 0, 'ampdy1': 0,
         'center1': 120, 'centerdx1': 0, 'centerdy1': 0,
         'sigma1': 5, 'sigmadx1': 0, 'sigmady1': 0,
         'amp2': 0.3, 'ampdx2': 0, 'ampdy2': 0,
         'center2': 120, 'centerdx2': 0, 'centerdy2': 0,
         'sigma2': 5, 'sigmadx2': 0, 'sigmady2': 0}
# Impose a spatial covariance structure to guide the Bernoulli parameters
kern_width = 1
# First step: give it the right answer
guess['on0'] = model.sum(0) > 0
# Wipe out a whole extra row.
guess['on0'][:, -1] = False
guess['p0'] = spatial_covariance_structure(guess['on0'], stddev=kern_width)
guess['on1'] = model2.sum(0) > 0
# Wipe out a whole extra row.
guess['on1'][-1] = False
guess['p1'] = spatial_covariance_structure(guess['on1'], stddev=kern_width)

guess['on2'] = model.sum(0) > 0
guess['p2'] = spatial_covariance_structure(guess['on2'], stddev=kern_width)

# MCMC fit w/ pymc3
pymc_medians, pymc_stddevs, trace, pymc_model = \
    minicube_pymc_fit(np.arange(num_pts), model_with_noise, guess,
                      ncomps=ncomps,
                      tune=500, draws=500, fmin=None)
print("pymc Parameters:")
for par in pymc_medians:
    if "__" in par:
        continue
    print("{0}: {1}+/-{2}".format(par, pymc_medians[par],
                                  pymc_stddevs[par]))

# plot
# Make individual fit cubes so we can see how well each component was fit

fitcube_pymc = pymc_medians['on0'] * \
    minicube_model(np.arange(num_pts), pymc_medians['amp0'],
                   pymc_medians['ampdx0'], pymc_medians['ampdy0'],
                   pymc_medians['center0'], pymc_medians['centerdx0'],
                   pymc_medians['centerdy0'], pymc_medians['sigma0'],
                   pymc_medians['sigmadx0'], pymc_medians['sigmady0'],
                   npix=npix, force_positive=False)

fitcube_pymc1 = pymc_medians['on1'] * \
    minicube_model(np.arange(num_pts), pymc_medians['amp1'],
                   pymc_medians['ampdx1'], pymc_medians['ampdy1'],
                   pymc_medians['center1'], pymc_medians['centerdx1'],
                   pymc_medians['centerdy1'], pymc_medians['sigma1'],
                   pymc_medians['sigmadx1'], pymc_medians['sigmady1'],
                   npix=npix, force_positive=False)

fitcube_pymc2 = pymc_medians['on2'] * \
    minicube_model(np.arange(num_pts), pymc_medians['amp2'],
                   pymc_medians['ampdx2'], pymc_medians['ampdy2'],
                   pymc_medians['center2'], pymc_medians['centerdx2'],
                   pymc_medians['centerdy2'], pymc_medians['sigma2'],
                   pymc_medians['sigmadx2'], pymc_medians['sigmady2'],
                   npix=npix, force_positive=False)

pl.figure(1).clf()
fig, axes = pl.subplots(npix, npix, sharex=True, sharey=True, num=1)

for ii, ((yy, xx), ax) in enumerate(zip(np.ndindex((npix, npix)),
                                        axes.ravel())):
    ax.plot(model[:, yy, xx], 'k-', alpha=0.25, zorder=-10, linewidth=3,
            drawstyle='steps-mid')
    ax.plot(model2[:, yy, xx], 'k-.', alpha=0.25, zorder=-10, linewidth=3,
            drawstyle='steps-mid')
    ax.plot(model3[:, yy, xx], 'k-.', alpha=0.25, zorder=-10, linewidth=3,
            drawstyle='steps-mid')
    ax.plot(model_with_noise[:, yy, xx], 'k-', zorder=-5, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(fitcube_pymc[:, yy, xx], 'r-.', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(fitcube_pymc1[:, yy, xx], 'm-.', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(fitcube_pymc2[:, yy, xx], 'c-.', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(model_with_noise[:, yy, xx] - fitcube_pymc[:, yy, xx] -
            fitcube_pymc1[:, yy, xx] - fitcube_pymc2[:, yy, xx],
            'r:', zorder=-1, linewidth=1,
            drawstyle='steps-mid')

pl.tight_layout()
pl.subplots_adjust(hspace=0, wspace=0)


# Plot the posteriors
for i in range(ncomps):
    pm.plot_posterior(trace, varnames=[var for var in trace.varnames
                                       if str(i) in var and
                                       "__" not in var and
                                       'on' not in var])
