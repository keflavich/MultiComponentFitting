
import numpy as np
import matplotlib.pyplot as plt
import astropy.stats as astats
import scipy.signal as signal

from adaptive_savgol import adaptive_savgol_filter
from wiener_filter import wiener_filter
from myAG_decomposer import gaussian, func, vals_vec_from_lmfit, errs_vec_from_lmfit, paramvec_to_lmfit
from lmfit import minimize as lmfit_minimize
from lmfit import Parameters

np.random.seed(3456734978)

x = np.linspace(-5, 5, 500)
y = np.exp(-0.5 * (x / 0.25)**2) * 1.0
yn = y + np.random.randn(len(x)) * 2.0

y2 = y + np.roll(y, -30)
yn2 = y2 + np.random.randn(len(x)) * 0.1

y3 = y + np.roll(y, -30) + np.roll(y, 80) * 0.2
yn3 = y3 + np.random.randn(len(x)) * 0.2

y2diff = np.exp(-(0.5 * x**2 / 0.3**2)) * 0.5 + np.exp(-(0.5 * (x - 1)**2 / 1.5**2)) * 0.5
y2diffn = y2diff + np.random.randn(len(x)) * 0.1


data = y3
ndata = yn3

wien_filt = wiener_filter(x, ndata, width_factor=3, return_PSDs=True)


def extrema_find(y, mode='peak'):
    '''
    Adapted from: https://stackoverflow.com/questions/24656367/find-peaks-location-in-a-spectrum-numpy
    '''

    if mode not in ['peak', 'dip', 'all']:
        raise TypeError("mode must be 'peak', 'dip' or 'all'.")

    kernel = [1, 0, -1]
    dY = signal.convolve(y, kernel, 'valid')

    # Checking for sign-flipping
    S = np.sign(dY)
    ddS = signal.convolve(S, kernel, 'valid')

    if mode == 'peak' or mode == 'all':
        candidates = np.where(dY > 0)[0] + (len(kernel) - 1)

        peaks = \
            sorted(set(candidates).intersection(np.where(ddS == -2)[0] + 1))

    if mode == 'dip' or mode == 'all':
        candidates = np.where(dY < 0)[0] + (len(kernel) - 1)
        dips = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))

    if mode == 'all':
        return peaks.extend(dips)
    elif mode == 'peak':
        return peaks
    else:
        return dips


# Check the noise level in the noise filtered spectrum
noise_std = astats.mad_std(wien_filt[-2])

# We want to find peaks in the filtered spectrum. Start fitting with the one
# defined at the max and add components until they don't improve the fit
# anymore.
# peaks_idx = signal.find_peaks_cwt(wien_filt[0], np.linspace(0.1, 2.0) / np.diff(x)[0]).astype(int)
peaks_idx = extrema_find(wien_filt[0])
peaks = x[peaks_idx]
peak_vals = wien_filt[0][peaks_idx]

peaks = peaks[np.argsort(peak_vals)[::-1]]
peak_vals = peak_vals[np.argsort(peak_vals)[::-1]]

init_width = 0.25

aics = []

for i in range(len(peaks)):
    if i == 0:
        v0 = (peak_vals[i], init_width, peaks[i])
    else:
        v0 += (peak_vals[i], init_width, peaks[i])

    v0_lmfit = paramvec_to_lmfit(v0)

    def objective_leastsq(paramslm):
        if not isinstance(paramslm, tuple):
            params = vals_vec_from_lmfit(paramslm)
        else:
            params = paramslm
        resids = (func(x, *params).ravel() - ndata) / noise_std
        return resids

    # Final fit using unconstrained parameters
    result2 = lmfit_minimize(objective_leastsq, v0_lmfit, method='leastsq')
    params_fit = vals_vec_from_lmfit(result2.params)
    params_errs = errs_vec_from_lmfit(result2.params)
    ncomps_fit = len(params_fit) / 3

    if i == 0:
        aics.append(result2.aic)
        result_old = result2
        params_fit_old = params_fit
        params_errs_old = params_errs
        ncomps_fit_old = ncomps_fit
        continue
    elif result2.aic > aics[i - 1]:
        break
    else:
        aics.append(result2.aic)
        result_old = result2
        params_fit_old = params_fit
        params_errs_old = params_errs
        ncomps_fit_old = ncomps_fit

plt.plot(x, ndata, alpha=0.4)
plt.plot(x, data)
plt.plot(x, wien_filt[0])
plt.plot(x, func(x, *params_fit_old))
for i in range(ncomps_fit_old):
    plt.plot(x, func(x, *params_fit_old[i::ncomps_fit_old]))

# test_smoothed = adaptive_savgol_filter(yn, 2, min_window=11, mode='wrap',
#                                        crit_val=0.3, max_iter=300)
# test_smoothed = adaptive_savgol_filter(yn2, 2, min_window=11, mode='wrap',
#                                        crit_val=0.3, max_iter=300)

# # test_smoothed = adaptive_savgol_filter(y, 2, min_window=11, mode='wrap',
# #                                        crit_val=0.3, max_iter=300)

# plt.plot(x, yn, alpha=0.5)
# plt.plot(x, y, alpha=0.7)
# plt.plot(x, test_smoothed[0])

# new_posns = []

# for i in range(501):
#     yvals = test_smoothed[3][i]
#     if max(yvals) == 0:
#         new_posns.append(0)
#         continue
#     new_posns.append(grad_peak(yvals))
