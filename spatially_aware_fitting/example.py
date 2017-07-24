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

model_with_noise = np.random.randn(*model.shape) + model

guess = {'amp': 0.8, 'ampdx': 0, 'ampdy': 0,
         'center': 5, 'centerdx': 0, 'centerdy': 0,
         'sigma': 1.5, 'sigmadx': 0, 'sigmady': 0,}

result = unconstrained_fitter(model_with_noise, np.arange(10), guess)
print("Result = {0}".format(result))
