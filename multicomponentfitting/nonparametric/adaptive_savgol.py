
'''
Create an adaptive Savitzky-Golay filter.

Follows the algorithm in Browne, Mayer & Cutmore, 2007,
Digital Signal Processing, 17(1), 69-75.

'''

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal._savitzky_golay import savgol_coeffs, _fit_edges_polyfit
import matplotlib.pyplot as plt
from astropy.utils.console import ProgressBar
from itertools import groupby, chain
from operator import itemgetter
from scipy import ndimage as nd


def adaptive_savgol_filter(x, polyorder, min_window=11, deriv=0, delta=1.0,
                           axis=-1, mode='interp', cval=0.0, crit_val=0.3,
                           max_iter=500, smooth_adjust=5):
    """ Apply a Savitzky-Golay filter to an array.
    This is a 1-d filter.  If `x`  has dimension greater than 1, `axis`
    determines the axis along which the filter is applied.
    Parameters
    ----------
    x : array_like
        The data to be filtered.  If `x` is not a single or double precision
        floating point array, it will be converted to type `numpy.float64`
        before filtering.
    window_length : int
        The length of the filter window (i.e. the number of coefficients).
        `window_length` must be a positive odd integer.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute.  This must be a
        nonnegative integer.  The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.  Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.
    """
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    abs_min_window = polyorder + 1 if polyorder % 2 == 0 else polyorder + 2
    if min_window < abs_min_window:
        min_window = abs_min_window
    window_lengths = np.ones_like(x, dtype=int) * min_window
    # Array for the final adaptively smooth values.
    # y_hat = np.empty_like(x)
    y_hat_final = np.zeros_like(x)
    final_posn = np.zeros_like(x, dtype=int)
    # TODO: Change to a better edge estimate. But for now
    # y_hat[:polyorder] = 0
    # y_hat[-polyorder:] = 0

    # Max iter limited by the length of x
    if min_window + max_iter * 2 > x.shape[axis]:
        max_iter = int(np.floor((x.shape[axis] - min_window) / 2.))

    print(max_iter)
    # print(argh)

    # Now we'll track all of these
    corrs = np.zeros((len(x), max_iter + 1))
    resids = np.zeros((len(x), max_iter + 1))
    y_hat = np.zeros((len(x), max_iter + 1))

    print(corrs.shape)

    if len(x) % 2 == 0:
        hidx = (len(x) + 1) / 2
    else:
        hidx = len(x) / 2

    for i in ProgressBar(x.size):

        xroll = np.roll(x, i - hidx)

        j = 0
        while True:

            # k is half the window size  (window = 2k + 1)
            k = (window_lengths[i] - 1) / 2
            x_loc = xroll[max(hidx - k, 0): min(hidx + k + 1, y_hat.size)]

            coeffs = savgol_coeffs(window_lengths[i], polyorder, deriv=deriv,
                                   delta=delta)

            if mode == "interp":
                # Do not pad.  Instead, for the elements within `window_length // 2`
                # of the ends of the sequence, use the polynomial that is fitted to
                # the last `window_length` elements.
                y = convolve1d(x_loc, coeffs, axis=axis, mode="constant")
                _fit_edges_polyfit(x, window_lengths[i], polyorder, deriv, delta, axis, y)
            else:
                # Any mode other than 'interp' is passed on to ndimage.convolve1d.
                y = convolve1d(x_loc, coeffs, axis=axis, mode=mode, cval=cval)

            # Find the residuals and calculate the stop statistic

            resid = y - x_loc

            # plt.subplot(121)
            # plt.plot(x_loc, 'rD')
            # plt.plot(y, 'b--')
            # plt.subplot(122)
            # plt.plot(resid, 'bD')
            # plt.axhline(0)
            # plt.draw()
            # raw_input("{0} {1}".format(i, j))
            # plt.clf()

            crit_pass, corr = stop_statistic(resid, crit_val)

            corrs[i, j] = corr
            resids[i, j] = resid[k]
            y_hat[i, j] = y[k]

                # break
            # else:
                # Increase the window size and recompute
            window_lengths[i] += 2

            if j == max_iter:
                # if y_hat[i] != 0.0:
                    # y_hat[i] = y[k]
                break
            j += 1
        # We want to find the places where each below-to-above crit value occurs

        above_crit = np.where(corrs[i] > crit_val)[0]

        # Take the largest smoothed value available if none are above the
        # critical value, or none of the last 10% of smoothing windows are.
        # tenth_frac = int(corrs[i].size * 0.1)
        tenth_frac = 20
        if above_crit.size == 0 or (corrs[i, -tenth_frac:] < crit_val).all():
            y_hat_final[i] = y_hat[i, -1]
            final_posn[i] = len(y_hat[i]) - 1
            continue

        if [0, 1, 2] in above_crit:
            y_hat_final[i] = y_hat[i, 0]
            final_posn[i] = 0
            continue

        start_edges = []
        for k, g in groupby(enumerate(above_crit), lambda (i, x): i - x):
            sequences = map(itemgetter(1), g)
            start_edges.append(sequences[0])

        y_hat_final[i] = y_hat[i, start_edges[-1]]
        final_posn[i] = start_edges[-1]

    return y_hat_final, window_lengths, y_hat, corrs, resids, final_posn


def stop_statistic(resid, crit_val=0.3):
    '''
    Calculate Pearson's correlation (1st order).
    '''

    r0 = resid[:-1]
    r1 = resid[1:]

    stat = np.corrcoef(r0, r1)

    # print(stat[0, 1])

    if np.abs(stat[0, 1]) <= crit_val:
        return False, np.abs(stat[0, 1])
        # return False
    else:
        return True, np.abs(stat[0, 1])
        # return True


def grad_peak(y, gauss_width=2, grad_thresh=0.015):

    y_sm = nd.gaussian_filter1d(y, 2, axis=0)

    yderiv = np.gradient(y_sm)

    above_crit = np.where(yderiv >= grad_thresh)[0]

    if above_crit.size == 0:
        return y.size - 1

    peaks = []
    for k, g in groupby(enumerate(above_crit), lambda (i, x): i - x):
        sequence = np.array(map(itemgetter(1), g))

        peaks.append(sequence[np.argmax(yderiv[sequence])])

    peak_vals = yderiv[np.array(peaks)]

    return peaks[np.argmax(peak_vals)]


def corr_from_resid(resids, crit_val=0.3, min_window=11):


    corrs = np.zeros_like(resids)

    if resids.shape[0] % 2 == 0:
        hidx = (resids.shape[0] + 1) / 2
    else:
        hidx = resids.shape[0] / 2

    posns = np.empty(resids.shape[0], dtype=int)
    for i in range(resids.shape[0]):
        # Roll along zeroth axis
        roll_resid = np.roll(resids, i - hidx, axis=0)

        # Continue along second axis until the critical val is reached
        window = min_window
        for j in range(resids.shape[1]):
            k = (window - 1) / 2

            corr = stop_statistic(roll_resid[hidx - k:hidx + k + 1, j],
                                  crit_val=crit_val)[1]

            corrs[i, j] = corr

            # if corr >= crit_val:
            #     posns[i] = j
            #     break

            window += 2
        else:
            posns[i] = resids.shape[1] - 1

    return posns, corrs
