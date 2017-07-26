import numpy as np
from astropy.utils.console import ProgressBar

def make_model_cube(xarr, parcube):

    modelcube = np.empty(shape=(xarr.size, ) + parcube.shape[1:])

    # okay this is getting too hacky, but just I want to get a toy cube fast...
    # TODO refactor into pyspeckit spectral models / proper xarr...
    def gauss(x, a, xoff, sig):
        return a*np.exp(-(x - xoff)**2 / sig**2 / 2)

    def gauss_x2(x, pars):
        p1, p2 = pars[:3], pars[3:]
        return gauss(x, *p1) + gauss(x, *p2)

    def model_a_pixel(xy):
        x, y = int(xy[0]), int(xy[1])
        modelcube[:, y, x] = gauss_x2(xarr, pars=parcube[:, y, x])

    for x, y in ProgressBar(list(np.ndindex(parcube.shape[1:]))):
        model_a_pixel([x, y])

    return modelcube
