import numpy as np
import lmfit

def plane(xx, yy, value, dx, dy, xcen=1, ycen=1):
    z = ((xx-xcen) * dx + (yy-ycen) * dy) + value
    return z

def gaussian(xax, amp, cen, wid):
    return np.exp(-(xax-cen)**2/(2*wid**2)) * amp

def minicube_model(xax,
                   amp, ampdx, ampdy,
                   center, centerdx, centerdy,
                   sigma, sigmadx, sigmady,
                   npix=3,
                   func=gaussian,
                  ):

    for par in (amp, ampdx, ampdy, center, centerdx, centerdy,
                sigma, sigmadx, sigmady):
        assert np.isfinite(par)

    yy,xx = np.indices([npix, npix], dtype='float')

    amps = plane(xx, yy, amp, ampdx, ampdy, xcen=npix//2, ycen=npix//2)
    centers = plane(xx, yy, center, centerdx, centerdy, xcen=npix//2, ycen=npix//2)
    sigmas = plane(xx, yy, sigma, sigmadx, sigmady, xcen=npix//2, ycen=npix//2)

    model = gaussian(xax[:,None,None], amps, centers, sigmas)

    return model

def minicube_model_generator(npix=3, func=gaussian,):

    def minicube_modelfunc(xax, amp, ampdx, ampdy, center, centerdx, centerdy,
                           sigma, sigmadx, sigmady,):
        return minicube_model(xax, amp, ampdx, ampdy, center, centerdx,
                              centerdy, sigma, sigmadx, sigmady, npix=npix,
                              func=func)

    return minicube_modelfunc


def unconstrained_fitter(minicube, xax, input_parameters, **model_kwargs):
    """
    input_parameters should be a dict
    """

    model = lmfit.Model(minicube_model_generator(**model_kwargs),
                        independent_vars=['xax'])

    params = model.make_params()

    for par in params:
        params[par].value = input_parameters[par]

    result = model.fit(minicube, xax=xax,
                       params=params)

    return result
