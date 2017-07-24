# Example run for minicube_fit

import numpy as np

# HACK
import imp
import minicube_fit
imp.reload(minicube_fit)

from minicube_fit import minicube_model, unconstrained_fitter

model = minicube_model(np.arange(10),
                       1, 0.5, -0.1,
                       5, 1, -2,
                       1, 0.1, 0.01,)

model_with_noise = np.random.randn(*model.shape)*0.1 + model

guess = {'amp': 0.8, 'ampdx': 0, 'ampdy': 0,
         'center': 5, 'centerdx': 0, 'centerdy': 0,
         'sigma': 1.5, 'sigmadx': 0, 'sigmady': 0,}

result = unconstrained_fitter(model_with_noise, np.arange(10), guess)
print("Result = {0}".format(result.params))

# plot
import pylab as pl

fitcube = minicube_model(np.arange(10), *[x.value for x in result.params.values()])

pl.figure(1).clf()
for ii,(yy,xx) in enumerate(np.ndindex((3,3))):
    ax = pl.subplot(3, 3, ii+1)
    ax.plot(model[:,yy,xx], 'k-', alpha=0.25, zorder=-10, linewidth=3,
            drawstyle='steps-mid')
    ax.plot(model_with_noise[:,yy,xx], 'k-', zorder=-5, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(fitcube[:,yy,xx], 'b--', zorder=0, linewidth=1,
            drawstyle='steps-mid')
    ax.plot(model_with_noise[:,yy,xx] - fitcube[:,yy,xx], 'r:', zorder=-1, linewidth=1,
            drawstyle='steps-mid')
