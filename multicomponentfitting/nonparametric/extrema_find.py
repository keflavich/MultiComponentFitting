
import scipy.signal as signal
import numpy as np


def extrema_find(y, mode='peak'):
    '''
    Adapted from: https://stackoverflow.com/questions/24656367/find-peaks-location-in-a-spectrum-numpy

    Parameters
    ----------
    y : np.ndarray
        1D Data.
    mode : {"peak", "dip", "all"}, optional
        Type of extrema to find.

    Returns
    -------
    peaks/dips : list
        List of the extrema found.
    '''

    if mode not in ['peak', 'dip', 'all']:
        raise TypeError("mode must be 'peak', 'dip' or 'all'.")

    kernel = [1, 0, -1]
    dY = signal.convolve(y, kernel, 'valid')

    # Checking for sign-flipping
    S = np.sign(dY)
    ddS = signal.convolve(S, [1, -1], 'valid')

    if mode == 'peak' or mode == 'all':
        candidates = np.where(dY > 0)[0] + (len(kernel) - 1)
        peaks = \
            sorted(set(candidates).intersection(np.where(ddS == -2)[0] + 2))

    if mode == 'dip' or mode == 'all':
        candidates = np.where(dY < 0)[0] + (len(kernel) - 1)
        dips = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))

    if mode == 'all':
        return peaks.extend(dips)
    elif mode == 'peak':
        return peaks
    else:
        return dips
