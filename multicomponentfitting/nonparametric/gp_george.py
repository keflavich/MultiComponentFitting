
'''
Try out George for gp
'''

import numpy as np
import george
from george import kernels
from scipy.optimize import minimize
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 500)
y = np.exp(-0.5 * (x / 0.25)**2) * 1.0
yn = y + np.random.randn(len(x)) * 0.2

y2 = y + np.roll(y, -100)
yn2 = y2 + np.random.randn(len(x)) * 0.1

# y3 = y + np.roll(y, -30) + np.roll(y, 80) * 0.2
# yn3 = y3 + np.random.randn(len(x)) * 0.2

y2diff = np.exp(-(0.5 * x**2 / 0.3**2)) * 0.5 + np.exp(-(0.5 * (x - 1)**2 / 1.5**2)) * 0.5
y2diffn = y2diff + np.random.randn(len(x)) * 0.1

y = y2diff
yn = y2diffn

kernel = np.var(yn) * kernels.ExpSquaredKernel(30)
# kernel = np.var(y) * kernels.LocalGaussianKernel(location=0, log_width=np.log(0.25)) + \
#     np.var(y) * kernels.LocalGaussianKernel(location=1.5, log_width=np.log(0.25))
gp = george.GP(kernel, white_noise=0.1, fit_white_noise=True, fit_mean=False)
gp.compute(x, 0.1)


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

plt.errorbar(x, yn, yerr=0.1, fmt=".k", capsize=0, alpha=0.2)
plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                 color="k", alpha=0.2)
plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
plt.plot(x_pred, y, "--g")
# plt.xlim(0, 10)
# plt.ylim(-1.45, 1.45)
plt.xlabel("x")
plt.ylabel("y")

