import numpy as np
import os
from astropy import wcs
import warnings
warnings.simplefilter('ignore', wcs.FITSFixedWarning)
from spectral_cube import SpectralCube
from astropy import units as u

try:
    from . import stage_1
    from . import stage_2
except SystemError:
    import stage_1
    import stage_2

def run_scousepy():
    # Input values for core SCOUSE stages
    # (just put it in the directory, it will get ignored by git)
    datadirectory    =  '.'
    # The data cube to be analysed
    filename         =  'CMZ_3mm_HNCO_60'
    # Fits extension
    fitsfile         =  os.path.join(datadirectory, filename+'.fits')
    # The range in velocity, x, and y over which to fit
    ppv_vol          =  [0.0,0.0,0.0,0.0,0.0,0.0] # NOTE: not used?
    # Radius for the spectral averaging areas. Map units.
    rsaa             =  [2.0,5.0,8.0]
    # Enter an approximate rms value for the data.
    rms_approx       =  0.05
    # Threshold below which all channel values set to 0.0
    sigma_cut        =  3.0

    cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s)

    momzero = cube.with_mask(cube > u.Quantity(
        rms_approx * sigma_cut, cube.unit)).moment0(axis=0).value

    # get the coverage / average the subcube spectra
    coverage_coordinates, saa_spectra = [], []
    for r in rsaa:
        cc, ss = stage_1.define_coverage(cube, momzero, r)
        coverage_coordinates.append(cc)
        saa_spectra.append(ss)

    # write fits files for all the averaged spectra
    stage_1.write_averaged_spectra(cube.header, saa_spectra, rsaa)

    # plot multiple coverage areas
    stage_1.plot_rsaa(coverage_coordinates, momzero, rsaa)

    # TODO: PARALLELISE MULTICUBE!!! (after broadcasting MemoryError's are caught)
    npeaks = 1
    npeaks2finesse = {1: [20, 10, 10]}
    multicube_kwargs = dict(
        fits_flist=['saa_cube_r{}.fits'.format(r) for r in rsaa],
        fittype="gaussian",
        # [amlitude_range, velocity_range, sigma_range]
        priors=[[0, 2], [-110, 110], [10, 50]],
        finesse=npeaks2finesse[npeaks],
        npeaks=npeaks,  # priors and finesse can be expanded if need be
        npars=3,
        clip_edges=False,
        model_grid=None, # we can directly pass an array of spectral models
        # to avoid regenerating the spectral models (`redo=True` forces it anyway)
        model_file="model_grid_x{}.npy".format(npeaks),
        redo=False,
        data_dir=".")

    # remove the spectral model file
    try:
        os.remove(multicube_kwargs["model_file"])
    except FileNotFoundError:
        pass

    spc_list = stage_2.best_guesses_saa(**multicube_kwargs)

    # inspect the guesses suggested:
    for spc in spc_list:
        spc.parcube = spc.best_guesses
        # HACK to allow the guess inspection (no errors on guesses):
        spc.errcube = np.full_like(spc.parcube, 0.1)
        # uuuuh not sure why this is needed
        spc.specfit.fitter._make_parinfo(npeaks=multicube_kwargs['npeaks'])
        spc.specfit.parinfo = spc.specfit.fitter.parinfo

        spc.mapplot()

if __name__ == "__main__":
    run_scousepy()

def run_scousepy_de():
    datadirectory    =  '.'
    filename         =  'CMZ_3mm_HNCO_60'
    fitsfile         =  os.path.join(datadirectory, filename+'.fits')
    rsaa             =  [2.0,5.0,8.0]
    rms_approx       =  0.05
    sigma_cut        =  3.0

    cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s)

    momzero = cube.with_mask(cube > u.Quantity(
        rms_approx * sigma_cut, cube.unit)).moment0(axis=0).value

    # get the coverage / average the subcube spectra
    coverage_coordinates, saa_spectra = [], []
    for r in rsaa:
        cc, ss = stage_1.define_coverage(cube, momzero, r)
        coverage_coordinates.append(cc)
        saa_spectra.append(ss)

    # write fits files for all the averaged spectra
    stage_1.write_averaged_spectra(cube.header, saa_spectra, rsaa)

    # plot multiple coverage areas
    stage_1.plot_rsaa(coverage_coordinates, momzero, rsaa)

    npeaks = 1
    diffevolution_kwargs = dict(
        fits_flist=['saa_cube_r{}.fits'.format(r) for r in rsaa],
        fittype="gaussian",
        # [amlitude_range, velocity_range, sigma_range]
        priors=[[0, 2], [-110, 110], [10, 50]],
        npeaks=npeaks,  # priors and finesse can be expanded if need be
        npars=3,
        data_dir=".",
        polish=False,
        )

    spc_list = stage_2.best_guesses_saa(method="diffevolution",
                                        **diffevolution_kwargs)

    # inspect the guesses suggested:
    for spc in spc_list:
        # HACK to allow the guess inspection (no errors on guesses):
        spc.errcube = np.full_like(spc.parcube, 0.1)
        # uuuuh not sure why this is needed
        spc.specfit.fitter._make_parinfo(npeaks=diffevolution_kwargs['npeaks'])
        spc.specfit.parinfo = spc.specfit.fitter.parinfo

        spc.mapplot()
