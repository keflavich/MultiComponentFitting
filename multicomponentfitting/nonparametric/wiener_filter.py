import numpy as np
from scipy import optimize, fftpack, signal

'''
Copyright (c) 2012-2013, Jacob Vanderplas
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

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
    f = np.fft.fftfreq(N, Df)

    H = np.fft.fft(h)
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
    N_half = N / 2
    min_func = lambda v: np.sum((PSD[1:N_half] -
                                 signal(f[1:N_half], v[0], v[1]) -
                                 noise(f[1:N_half], v[2])) ** 2)
    v0 = tuple(signal_params) + tuple(noise_params)
    opt_out = optimize.fmin(min_func, v0, disp=0, full_output=1)

    v = opt_out[0]
    warns = opt_out[-1]
    if warns != 0:
        print("Warning from optimize.fmin: {}".format(warns))

    P_S = signal(f, v[0], v[1] * width_factor)
    P_N = noise(f, v[-1])
    Phi = P_S / (P_S + P_N)
    Phi_N = P_N / (P_S + P_N)
    Phi[0] = 1  # correct for DC offset

    # print(v0)
    # print(v)

    # Use Phi to filter and smooth the values
    h_smooth = np.fft.ifft(Phi * H)
    h_noise = np.fft.ifft(Phi_N * H)

    if not np.iscomplexobj(h):
        h_smooth = h_smooth.real
        h_noise = h_noise.real

    if return_PSDs:
        return h_smooth, h_noise, f, PSD, P_S, P_N, Phi, Phi_N
    else:
        return h_smooth, h_noise


def corr_noise_model(f, N0, f0, a):

    return N0 / (1 + np.abs(f / f0)**a)


def plaw_noise_model(f, N0, a):

    return N0 * np.abs(f)**a


def sinc_PSD(f, N0):

    f_norm = f / f.max()

    return N0 * (np.sin(np.pi * f_norm) / (np.pi * f_norm))**4
