""" TODO's / WIP
- wrap the guesses back to the full cube...
- run pyspeckit on the whole thing
"""
from scipy.optimize import differential_evolution
from multicube import SubCube
from pyspeckit.specwarnings import PyspeckitWarning
from astropy.utils.console import ProgressBar
from astropy import log
import warnings
import numpy as np
import os

def opencube(fname, errmap=None, errorcube=None, data_dir="."):
    """
    Opens the cube and calculates all the pre-fitting attributes of interest.

    Remember to change the data_dir to a proper one. Not sure how to properly
    make it machine-agnostic, so feel free to edit!
    """
    data_dir = os.path.expanduser(data_dir)
    cube_file = os.path.join(data_dir, fname)
    spc = SubCube(cube_file)

    spc.xarr.velocity_convention = "radio"
    spc.xarr.convert_to_unit("km/s")
    spc.unit = "K"

    if errorcube is not None:
        spc.errorcube = errorcube
    elif errmap is not None:
        spc.errorcube = np.repeat([errmap], spc.xarr.size, axis=0)
    else:
        log.warn("errmap not specified, assuming uniform weighting.")
        spc.errorcube = np.ones_like(spc.cube)

    #spc.snrmap = (spc.cube / spc.errorcube).max(axis=0)

    return spc

def multicube_best_guesses(spc, priors, finesse, fixed=None, model_grid=None,
                           **kwargs):
    """
    Parameters
    ----------
    spc : a SubCube instance

    finesse : list(ndim) of int's
              Grid size in a parameter space for each free variable.
              this WILL blow up your memory if np.product(finesse) is large
              enough!

    model_grid : filename, np.ndarray, or `None`
                 A grid of models to run against the cube.
                 By default will be generated on the fly, if a filename is
                 specified, will check if such a file exists first and will
                 then read from it.
    Returns
    -------
    best_guesses : (ndims, x, y)-shaped np.ndarray
                   An array of best guesses.

    fiteach_kwargs : convenience parameters to pass to pyspeckit.Cube.fiteach
    """
    if fixed is None:
        fixed = np.zeros_like(finesse, dtype=bool)

    minpars = np.array(priors)[:, 0]
    maxpars = np.array(priors)[:, 1]
    spc.make_guess_grid(minpars, maxpars, finesse, fixed=fixed)

    # all the number crunching happens here: model grids generated,
    # best models searched based on the squared residual sum
    redo = kwargs.pop("redo", False)
    spc.generate_model(redo=redo, **kwargs)
    spc.best_guess(**kwargs)

    return spc

def differential_evolution_best_guesses(spc, priors, fixed=None, **kwargs):
    """
    Could also use DE as a starting point, or any other global regression
    method that is fast enough...
    """
    spc.parcube = np.empty(shape=(len(priors),) + spc.cube.shape[1:])
    for y, x in ProgressBar(list(np.ndindex(spc.cube.shape[1:]))):
        # we want our progressbars intact
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PyspeckitWarning)
            warnings.filterwarnings("ignore", "(under|divide|over)")
            sp = spc.get_spectrum(x, y)
        if not np.isfinite(sp.data.data).any():
            continue
        modelfunc = sp.specfit.get_full_model
        # masked arrays are a bit slower
        if type(sp.data) is np.ma.masked_array:
            specdata = sp.data.data
        else:
            specdata = sp.data
        # FIXME: uniform weighting is forced here, not always the case!
        resid = lambda pars: ((modelfunc(pars=pars) - specdata)**2).sum()
        res = differential_evolution(resid, bounds=priors, **kwargs)
        spc.parcube[:, y, x] = res.x

    return spc.parcube

def best_guesses_saa(fits_flist, priors, fittype="gaussian",
                     method="multicube", npeaks=1, npars=3, data_dir=".",
                     **kwargs):
    """
    Wrapper function that runs multicube_best_guesses on a list of
    pre-averaged fits files with a given model and number of velocity
    components.

    method : str; either "multicube" or "diffevolution"
    """

    # allow giving npars=1 form for priors / finesse for npars>1:
    if len(priors) == npars and npeaks > 1:
        log.debug("Expanding priors from {} to"
                  " {}".format(priors, priors * npeaks))
        # don't want to accidentally multiply the values
        priors = list(priors) * npeaks

    for key in ["fixed", "finesse"]: # do the same for kwargs
        if key in kwargs:
            if len(kwargs[key]) == npars and npeaks > 1:
                log.debug("Expanding {} from {} to"
                          " {}".format(key, priors, priors * npeaks))
                # don't want to accidentally multiply the values
                kwargs[key] = list(kwargs[key]) * npeaks

    spc_list = []
    for fits_file in fits_flist:
        spc = opencube(fits_file, data_dir=data_dir)

        # TODO: for more sophisticated spectral models the arglist would have
        #       to be expanded... here's a template of what cold ammonia setup
        #       would have to look like:
        #fitmodel = cold_ammonia_model
        #line_names = ["oneone", "twotwo"]
        #spc.specfit.Registry.add_fitter(fittype, npars=npars, function=
        #                                fitmodel(line_names=line_names))
        spc.update_model(fittype)
        spc.specfit.fitter.npeaks = npeaks

        # causes spc.best_guesses to be generated
        if method == "multicube":
            multicube_best_guesses(spc, priors, fixed=None, **kwargs)
        if method == "diffevolution":
            differential_evolution_best_guesses(spc, priors, fixed=None, **kwargs)

        spc_list.append(spc)

    return spc_list
