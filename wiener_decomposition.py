
from lmfit import Parameters
from lmfit import minimize as lmfit_minimize
import numpy as np
import astropy.stats as astats
from scipy.interpolate import UnivariateSpline

from wiener_filter import wiener_filter
from extrema_find import extrema_find
# from myAG_decomposer import (func, vals_vec_from_lmfit,
#                              errs_vec_from_lmfit, paramvec_to_lmfit)


def decompose(x, y, plot_fit=True, diagnostic_plots=False, min_peak=0.0,
              **wiener_kwargs):
    '''
    Use a Wiener filter to create a low-pass filter of `y`, find the peaks,
    and, starting with the largest peak, fit gaussian components until the
    AIC stops decreasing.
    '''

    yfilt = wiener_filter(x, y, **wiener_kwargs)

    # Catch where the PSD outputs have been passed
    y_lp = yfilt[0]
    y_hp = yfilt[1]

    if len(yfilt) > 2:
        PSD_out = yfilt[2:]

    # Estimate the noise level from the high-pass filtered data
    noise_std = astats.mad_std(y_hp)

    # Get peaks in the low-pass spectrum using the minima in the 2nd
    # derivative
    # y_spl = UnivariateSpline(x, y_lp, s=0, k=4)
    # y_spl_2der = y_spl.derivative(n=2)

    # import matplotlib.pyplot as plt
    # plt.plot(x, y_lp)
    # plt.plot(x, y_spl(x))

    peaks_idx = extrema_find(y_lp, mode='peak')
    # peaks_idx = extrema_find(y_spl_2der(x), mode='dip')
    peaks = x[peaks_idx]
    peak_vals = y_lp[peaks_idx]

    # Remove peaks below min_peak
    peaks = peaks[peak_vals >= min_peak]
    peak_vals = peak_vals[peak_vals >= min_peak]

    peaks = peaks[np.argsort(peak_vals)[::-1]]
    peak_vals = peak_vals[np.argsort(peak_vals)[::-1]]

    # NOTE: This needs to be set using the Wiener filter width! Or some other
    # estimator.
    # init_width = 0.25
    init_width = 10000
    # init_width = 100

    # u2 = np.diff(np.diff(y_lp))  # / np.diff(x[:2])[0]
    # Find points of inflection
    # inflection = np.abs(np.diff(np.sign(u2)))

    # Find Relative widths, then measure
    # peak-to-inflection distance for sharpest peak
    # widths = np.sqrt(np.abs(y_lp / y_spl_2der(x))[peaks_idx])

    # import matplotlib.pyplot as plt
    # plt.plot(x, np.sqrt(np.abs(y_lp / y_spl_2der(x))))
    # for peak in peaks:
    #     plt.axvline(peak)

    # print(widths)
    # print(argh)

    # FWHMs = widths * 2.355
    aics = []
    bics = []

    for i in range(len(peaks)):
        if i == 0:
            v0 = (peak_vals[i], init_width, peaks[i])
        else:
            v0 += (peak_vals[i], init_width, peaks[i])

        v0_lmfit = gaussian_params_lmfit(v0)

        def objective_leastsq(paramslm):
            resids = (func(x, paramslm).ravel() - y) / noise_std
            return resids

        # Final fit using unconstrained parameters
        result2 = lmfit_minimize(objective_leastsq, v0_lmfit, method='leastsq')
        ncomps_fit = i + 1
        params_fit = vals_vec_from_lmfit(result2.params)

        if diagnostic_plots:
            import matplotlib.pyplot as plt
            plt.plot(x, y, alpha=0.7)
            plt.plot(x, y_lp)
            for peak in peaks:
                plt.axvline(peak, color='g', linestyle='-.', alpha=0.5)
            plt.plot(x, func(x, result2.params))

            for j in range(1, ncomps_fit + 1):
                pars = params_fit[j]
                num_str = "{}".format(j)
                plt.plot(x, gaussian(pars['amp' + num_str],
                                     pars['fwhm' + num_str],
                                     pars['cent' + num_str])(x),
                         label=num_str)
            plt.legend()
            plt.draw()
            raw_input("?")
            plt.clf()

        if i == 0:
            aics.append(result2.aic)
            bics.append(result2.bic)
            result_prev = result2
            params_fit_prev = params_fit
            ncomps_fit_prev = ncomps_fit
            continue
        elif result2.aic > aics[i - 1] or result2.bic > bics[i - 1]:
            aics.append(result2.aic)
            bics.append(result2.bic)
            break
        else:
            aics.append(result2.aic)
            bics.append(result2.bic)
            result_prev = result2
            params_fit_prev = params_fit
            ncomps_fit_prev = ncomps_fit

    print("AICS: {}".format(aics))
    print("BICS: {}".format(bics))

    if plot_fit:
        import matplotlib.pyplot as plt
        plt.plot(x, y, alpha=0.7)
        plt.plot(x, y_lp)
        plt.axhline(noise_std)
        for peak in peaks:
            plt.axvline(peak, color='g', linestyle='-.', alpha=0.5)
        for i in range(1, ncomps_fit_prev + 1):
            pars = params_fit_prev[i]
            num_str = "{}".format(i)
            plt.plot(x, gaussian(pars['amp' + num_str],
                                 pars['fwhm' + num_str],
                                 pars['cent' + num_str])(x),
                     label=num_str)
        plt.plot(x, func(x, result_prev.params))

        plt.legend(frameon=True)

    return result_prev, params_fit_prev


def vals_vec_from_lmfit(lmfit_params):
    """ Return Python list of parameter values from LMFIT Parameters object"""
    ncomps = len(lmfit_params) / 3

    comps = {}
    for i in range(1, ncomps + 1):
        num_str = "{}".format(i)
        comps[i] = Parameters()
        comps[i].add(lmfit_params["amp" + num_str])
        comps[i].add(lmfit_params["fwhm" + num_str])
        comps[i].add(lmfit_params["cent" + num_str])

    return comps


def gaussian_params_lmfit(values):
    '''
    params will have an order of (amp, width, center)
    '''
    ncomps = len(values) / 3

    params = Parameters()

    for i in range(ncomps):
        params.add('amp{}'.format(i + 1), value=values[3 * i], min=0.0)
        params.add('fwhm{}'.format(i + 1), value=values[3 * i + 1], min=5 * 2.35 * 200.)
        # params.add('fwhm{}'.format(i + 1), value=values[3 * i + 1], min=0.0)
        params.add('cent{}'.format(i + 1), value=values[3 * i + 2])

    return params


def gaussian(peak, FWHM, mean):
    """Return a Gaussian function
    """
    sigma = FWHM / 2.354820045  # (2 * sqrt( 2 * ln(2)))
    return lambda x: peak * np.exp(-(x - mean)**2 / 2. / sigma**2)


def func(x, params):
    """ Return multi-component Gaussian model F(x).

    Parameter vector from gaussian_params_lmfit,
    and therefore has len(args) = 3 x N_components.
    """
    ncomps = len(params) / 3
    yout = x * 0.
    for i in range(1, ncomps + 1):
        num_str = "{}".format(i)
        yout = yout + gaussian(params["amp" + num_str].value,
                               params["fwhm" + num_str].value,
                               params["cent" + num_str].value)(x)
    return yout
