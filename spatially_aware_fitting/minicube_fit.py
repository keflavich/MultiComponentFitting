def plane(xx, yy, value, dx, dy, xcen=1, ycen=1):
    z = ((xx-xcen) * dx + (yy-ycen) * dy) * value + value
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

    yy,xx = np.indices([npix, npix])

    amps = plane(xx, yy, amp, ampdx, ampdy, xcen=npix//2, ycen=npix//2)
    centers = plane(xx, yy, center, centerdx, centerdy, xcen=npix//2, ycen=npix//2)
    sigmas = plane(xx, yy, sigma, sigmadx, sigmady, xcen=npix//2, ycen=npix//2)

    model = gaussian(xax, amps, centers, sigmas)

    return model
