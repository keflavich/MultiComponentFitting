import imp
import numpy as np
import mcmcmc
imp.reload(mcmcmc)
import matplotlib.pyplot as plt

nPts = 401
Xdata = np.linspace(-5, 5, nPts)
Ydata = 5.2 * np.exp(-(Xdata - 0.3)**2 / (2*0.4**2)) + np.random.randn(nPts)
Ydata += 1.3 * np.exp(-(Xdata +0.4)**2 / (2*1.0**2))
sigma = 1.0
plt.plot(Xdata, Ydata)
trace, model = mcmcmc.BayesianMultiComponentFit(Xdata, Ydata, max_ncomp=3, y_error = 1.0)
Ymodels = mcmcmc.RealizedFit(trace, Xdata, model)
plt.plot(Xdata, Ymodels, color='red')
