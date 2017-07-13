import numpy as np
from scipy import optimize, fftpack, signal


def wiener_filter(t, h, signal='gaussian', noise='flat', return_PSDs=False,
                  signal_params=None, noise_params=None, width_factor=1):
    """Compute a Wiener-filtered time-series

    Parameters
    ----------
    t : array_like
        evenly-sampled time series, length N
    h : array_like
        observations at each t
    signal : str (optional)
        currently only 'gaussian' is supported
    noise : str (optional)
        currently only 'flat' is supported
    return_PSDs : bool (optional)
        if True, then return (PSD, P_S, P_N)
    signal_guess : tuple (optional)
        A starting guess at the parameters for the signal.  If not specified,
        a suitable guess will be estimated from the data itself. (see Notes
        below)
    noise_guess : tuple (optional)
        A starting guess at the parameters for the noise.  If not specified,
        a suitable guess will be estimated from the data itself. (see Notes
        below)

    Returns
    -------
    h_smooth : ndarray
        a smoothed version of h, length N

    Notes
    -----
    The Wiener filter operates by fitting a functional form to the PSD::

       PSD = P_S + P_N

    The resulting frequency-space filter is given by::

       Phi = P_S / (P_S + P_N)

    This entire operation is equivalent to a kernel smoothing by a
    kernel whose Fourier transform is Phi.

    Choosing Signal/Noise Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    the arguments ``signal_guess`` and ``noise_guess`` specify the initial
    guess for the characteristics of signal and noise used in the minimization.
    They are generally expected to be tuples, and the meaning varies depending
    on the form of signal and noise used.  For ``gaussian``, the params are
    (amplitude, width).  For ``flat``, the params are (amplitude,).

    See Also
    --------
    scipy.signal.wiener : a static (non-adaptive) wiener filter
    """
    # Validate signal
    if signal != 'gaussian':
        raise ValueError("only signal='gaussian' is supported")
    if signal_params is not None and len(signal_params) != 2:
        raise ValueError("signal_params should be length 2")

    # Validate noise
    if noise != 'flat':
        raise ValueError("only noise='flat' is supported")
    if noise_params is not None and len(noise_params) != 1:
        raise ValueError("noise_params should be length 1")

    # Validate t and hd
    t = np.asarray(t)
    h = np.asarray(h)

    if (t.ndim != 1) or (t.shape != h.shape):
        raise ValueError('t and h must be equal-length 1-dimensional arrays')

    # compute the PSD of the input
    N = len(t)
    Df = np.diff(t)[0]
    f = np.fft.rfftfreq(N, 1)

    H = np.fft.rfft(h)
    PSD = abs(H) ** 2

    # fit signal/noise params if necessary
    if signal_params is None:
        amp_guess = np.max(PSD[1:])
        width_guess = np.min(np.abs(f[PSD[1:] < np.mean(PSD[1:])]))
        signal_params = (amp_guess, width_guess)
    if noise_params is None:
        noise_params = (np.mean(PSD[1:]),)
        # noise_params = (np.mean(PSD[1:]), 4 * width_guess, 1)
        # noise_params = (np.mean(PSD[1:]), -1)

    # Set up the Wiener filter:
    #  fit a model to the PSD: sum of signal form and noise form

    def signal(x, A, width):
        width = abs(width) + 1E-99  # prevent divide-by-zero errors
        return A * np.exp(-0.5 * (x / width) ** 2)

    def noise(x, n):
        return n * np.ones(x.shape)

    # use [1:] here to remove the zero-frequency term: we don't want to
    # fit to this for data with an offset.
    min_func = lambda v: np.sum((PSD[1:] - signal(f[1:], v[0], v[1])
                                 - noise(f[1:], v[2])) ** 2)
    v0 = tuple(signal_params) + tuple(noise_params)
    v = optimize.fmin(min_func, v0)

    P_S = signal(f, v[0], v[1] * width_factor)
    P_N = noise(f, v[-1])
    Phi = P_S / (P_S + P_N)
    Phi_N = P_N / (P_S + P_N)
    Phi[0] = 1  # correct for DC offset

    print(v0)
    print(v)

    # Use Phi to filter and smooth the values
    h_smooth = np.fft.irfft(Phi * H)
    h_noise = np.fft.irfft(Phi_N * H)

    if not np.iscomplexobj(h):
        h_smooth = h_smooth.real
        h_noise = h_noise.real

    if return_PSDs:
        return h_smooth, f, PSD, P_S, P_N, Phi, h_noise, Phi_N
    else:
        return h_smooth


def corr_noise_model(f, N0, f0, a):

    return N0 / (1 + np.abs(f / f0)**a)


def plaw_noise_model(f, N0, a):

    return N0 * np.abs(f)**a


def sinc_PSD(f, N0):

    f_norm = f / f.max()

    return N0 * (np.sin(np.pi * f_norm) / (np.pi * f_norm))**4
