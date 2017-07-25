import imp
import numpy as np
import mcmcmc
imp.reload(mcmcmc)
import matplotlib.pyplot as plt

nPts = 401
Xdata = np.linspace(-5, 5, nPts)
Ydata_comp1 = 5.2 * np.exp(-(Xdata - 0.3)**2 / (2*0.4**2))
Ydata_comp2 = 2.0 * np.exp(-(Xdata + 0.4)**2 / (2*1.0**2))
Ydata_orig = Ydata_comp1 + Ydata_comp2
sigma = 1.0
Ydata = Ydata_orig + sigma * np.random.randn(nPts)

plt.plot(Xdata, Ydata)
plt.plot(Xdata, Ydata_comp1, 'k:', linewidth=3, alpha=0.5)
plt.plot(Xdata, Ydata_comp2, 'k:', linewidth=3, alpha=0.5)
plt.plot(Xdata, Ydata_orig, 'k-.', linewidth=5, alpha=0.5)
trace, model = mcmcmc.BayesianMultiComponentFit(Xdata, Ydata, max_ncomp=3,
                                                y_error=sigma)
Ymodels = mcmcmc.RealizedFit(trace, Xdata, model)
plt.plot(Xdata, Ymodels, color='red', alpha=0.5)
