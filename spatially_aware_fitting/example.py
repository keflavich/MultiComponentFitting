# Example run for minicube_fit

import numpy as np

# HACK
import imp
import minicube_fit
imp.reload(minicube_fit)

from minicube_fit import minicube_model, unconstrained_fitter
from minicube_pymc import minicube_pymc_fit

num_pts = 100
npix = 5

model = minicube_model(np.arange(num_pts),
                       0.2, 0.5, -0.1,
                       25, 1, -2,
                       10, 0.1, 0.01, npix=npix)

# simulate a "real sky" - no negative emission.
model[model<0] = 0

model_with_noise = np.random.randn(*model.shape)*0.1 + model

guess = {'amp': 0.3, 'ampdx': 0, 'ampdy': 0,
         'center': 25, 'centerdx': 0, 'centerdy': 0,
         'sigma': 10, 'sigmadx': 0, 'sigmady': 0,}

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

fitcube_pymc = minicube_model(np.arange(num_pts), pymc_medians['amp'],
                              pymc_medians['ampdx'], pymc_medians['ampdy'],
                              pymc_medians['center'], pymc_medians['centerdx'],
                              pymc_medians['centerdy'], pymc_medians['sigma'],
                              pymc_medians['sigmadx'], pymc_medians['sigmady'],
                              npix=npix, force_positive=False)

fitcube_pymc = pymc_medians['on'] * fitcube_pymc


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
