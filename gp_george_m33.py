'''
Smoothing w/ GPs + adding in spatial information
'''

import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
import george
from george import kernels
from scipy.optimize import minimize
import emcee

from paths import fourteenB_wGBT_HI_file_dict

cube = SpectralCube.read(fourteenB_wGBT_HI_file_dict['Cube'])

spec_axis = cube.spectral_axis
chan_width = np.abs(np.diff(spec_axis[:2])[0])

for i in range(604, 800):
    for j in range(634, 700):

        spec = cube[:, i, j]

        yn = spec.value
        x = np.arange(spec.size)

        kernel = np.var(yn) * kernels.ExpSquaredKernel(30)
        # kernel = np.var(y) * kernels.LocalGaussianKernel(location=0, log_width=np.log(0.25)) + \
        #     np.var(y) * kernels.LocalGaussianKernel(location=1.5, log_width=np.log(0.25))
        gp = george.GP(kernel, white_noise=0.002, fit_white_noise=True, fit_mean=False)
        gp.compute(x, 0.005)


        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(yn)


        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(yn)


        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        print(result)

        gp.set_parameter_vector(result.x)
        print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(yn)))

        x_pred = x

        pred, pred_var = gp.predict(yn, x_pred, return_var=True)

        # def lnprob2(p):
        #     gp.set_parameter_vector(p)
        #     return gp.log_likelihood(yn, quiet=True) + gp.log_prior()

        # initial = gp.get_parameter_vector()
        # ndim, nwalkers = len(initial), 32
        # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)

        # print("Running first burn-in...")
        # p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        # p0, lp, _ = sampler.run_mcmc(p0, 2000)

        # print("Running second burn-in...")
        # p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
        # sampler.reset()
        # p0, _, _ = sampler.run_mcmc(p0, 1000)
        # sampler.reset()

        # print("Running production...")
        # sampler.run_mcmc(p0, 1000)

        plt.plot(x, yn, 'r--', alpha=0.6, drawstyle='steps-mid')

        # samples = sampler.flatchain
        # for s in samples[np.random.randint(len(samples), size=100)]:
        #     gp.set_parameter_vector(s)
        #     mu = gp.sample_conditional(yn, x)
        #     plt.plot(x, mu, color="#4682b4", alpha=0.3)

        plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                         color="b", alpha=0.2)
        plt.plot(x_pred, pred, "b", lw=1.5, alpha=0.4)
        plt.axhline(0.0)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.draw()
        raw_input("{0}, {1}".format(i, j))
        plt.clf()

# corner.corner(sampler.flatchain)
