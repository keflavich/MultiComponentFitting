from multicube.subcube import SubCube
from astropy import log
import warnings
import numpy as np
import os

def make_cube(npars=6, npeaks=2, data_dir="~/Data/multiple-components/"):
    """
    Opens the cube and caclulates all the pre-fitting attributes of interest.

    Remember to change the data_dir to a proper one. Not sure how to properly
    make it machine-agnostic, so feel free to edit!
    """
    data_dir = os.path.expanduser(data_dir)
    cube_file = os.path.join(data_dir, "NGC1333_NH3_11_sub.fits")
    spc = SubCube(cube_file)

    #spc.xarr.refX = n2hp.freq_dict["123-012"]*u.Hz
    spc.xarr.velocity_convention = "radio"
    spc.xarr.convert_to_unit("km/s")
    #spc.Registry.add_fitter("n2hp_vs_vtau", n2hp_vs.n2hp_vs_vtau_fitter, 4)
    spc.unit = "K"

    rms = np.vstack([spc.slice(-20., -14., "km/s").cube,
                     spc.slice(-9.5, -2.5, "km/s").cube,
                     spc.slice(2.50, 5.50, "km/s").cube,
                     spc.slice(10.0, 13.0, "km/s").cube,
                     spc.slice(17.0, 25.0, "km/s").cube,
                     spc.slice(29.0, 36.0, "km/s").cube]).std(axis=0)

    snr = spc.cube.max(axis=0) / rms

    #do_fit, regen = True, True
    do_fit, regen = False, False
    spc.update_model("gaussian")
    spc.specfit.fitter.npeaks = npeaks

    spc.errmap = rms
    spc.snrmap = snr
    # easier to handle everything get_spectrum-related
    spc.errorcube = np.repeat([rms], spc.xarr.size, axis=0)

    return spc

def make_cube_shh(**kwargs):
    """ Shush! Opens the cube without triggering a wall of warnings. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_log = log.level
        log.setLevel("ERROR")
        spc = make_cube(**kwargs)
        log.setLevel(old_log)

    return spc
