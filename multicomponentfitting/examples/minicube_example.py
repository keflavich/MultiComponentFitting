# Example run for minicube_fit

import numpy as np

from multicomponentfitting.spatially_aware_fitting.minicube_fit import minicube_model, unconstrained_fitter
from multicomponentfitting.spatially_aware_fitting.minicube_pymc import minicube_pymc_fit, spatial_covariance_structure

num_pts = 100
npix = 5

model = minicube_model(np.arange(num_pts),
                       0.2, 0.5, -0.1,
                       25, 1, -2,
                       10, 0.1, 0.01, npix=npix)

# simulate a "real sky" - no negative emission.
model[model<0] = 0

model_with_noise = np.random.randn(*model.shape)*0.1 + model

guess = {'amp0': 0.3, 'ampdx0': 0, 'ampdy0': 0,
         'center0': 25, 'centerdx0': 0, 'centerdy0': 0,
         'sigma0': 10, 'sigmadx0': 0, 'sigmady0': 0,}
# Impose a spatial covariance structure to guide the Bernoulli parameters
kern_width = 1
# First step: give it the right answer
guess['on0'] = model.sum(0) > 0
# Wipe out a whole extra row.
# guess['on0'][:, -1] = False
guess['p0'] = spatial_covariance_structure(guess['on0'], stddev=kern_width)

result = unconstrained_fitter(model_with_noise, np.arange(num_pts), guess,
                              npix=npix)
print("LSQ Parameters:")
for par in result.params:
    print("{0}: {1}+/-{2}".format(par, result.params[par].value,
                                  result.params[par].stderr))

# Now use emcee to fit
result_mcmc = result.emcee(steps=1000, burn=500, thin=2)
print("MCMC Parameters:")
for par in result_mcmc.params:
    print("{0}: {1}+/-{2}".format(par, result_mcmc.params[par].value,
                                  result_mcmc.params[par].stderr))

# MCMC fit w/ pymc3
pymc_medians, pymc_stddevs, trace, pymc_model = \
    minicube_pymc_fit(np.arange(num_pts), model_with_noise, guess)
print("pymc Parameters:")
for par in pymc_medians:
    print("{0}: {1}+/-{2}".format(par, pymc_medians[par],
                                  pymc_stddevs[par]))

# plot
import pylab as pl

fitcube = minicube_model(np.arange(num_pts),
                         *[x.value for x in result.params.values()],
                         npix=npix)
fitcube_mcmc = minicube_model(np.arange(num_pts),
                              *[x.value for x in result_mcmc.params.values()],
                              npix=npix)

fitcube_pymc = minicube_model(np.arange(num_pts), pymc_medians['amp0'],
                              pymc_medians['ampdx0'], pymc_medians['ampdy0'],
                              pymc_medians['center0'], pymc_medians['centerdx0'],
                              pymc_medians['centerdy0'], pymc_medians['sigma0'],
                              pymc_medians['sigmadx0'], pymc_medians['sigmady0'],
                              npix=npix, force_positive=False)

fitcube_pymc = pymc_medians['on0'] * fitcube_pymc


pl.figure(1).clf()
fig, axes = pl.subplots(npix, npix, sharex=True, sharey=True, num=1)

for ii,((yy,xx), ax) in enumerate(zip(np.ndindex((npix,npix)), axes.ravel())):
    ax.plot(model[:,yy,xx], 'k-', alpha=0.25, zorder=-10, linewidth=3,
            drawstyle='steps-mid')
    ax.plot(model_with_noise[:,yy,xx], 'k-', zorder=-5, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(fitcube[:,yy,xx], 'b--', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(fitcube_mcmc[:,yy,xx], 'g--', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(fitcube_pymc[:,yy,xx], 'r-.', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(model_with_noise[:,yy,xx] - fitcube[:,yy,xx], 'r:', zorder=-1, linewidth=1,
            drawstyle='steps-mid')

pl.tight_layout()
