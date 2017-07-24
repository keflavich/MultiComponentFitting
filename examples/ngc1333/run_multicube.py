import sys
import numpy as np
from astropy import log
from opencube import make_cube_shh
from pyspeckit.spectrum.models.ammonia import cold_ammonia_model

line_names = ['oneone']
npars = 6 # for an ammonia model

try:
    npeaks = int(sys.argv[1])
except IndexError:
    npeaks = 2
    log.info("npeaks not specified, setting to {}".format(npeaks))

spc = make_cube_shh(npeaks=npeaks, npars=npars)
err = spc.errmap

fittype_fmt = 'cold_ammonia_x{}'
fitmodel = cold_ammonia_model
spc.specfit.Registry.add_fitter(fittype_fmt.format(npeaks), npars=npars,
                                function=fitmodel(line_names=line_names))
spc.update_model(fittype_fmt.format(npeaks))

# might be needed on old pyspeckit versions because of this:
# https://github.com/pyspeckit/pyspeckit/issues/179
spc.specfit.fitter.npeaks = npeaks

# shamelessly copying from some Bayesian inference script I have...
sig2fwhm = 2*(2*np.log(2))**0.5
min_sigma = 0.2 / sig2fwhm # from velocty resolution unit, can also go for
# a thermal line wight for some Tk...
# Some justification for the other prior ranges:
#   - Total ammonia column: ranging from dex 14 to 15, for a typical
#     NH3 abundance of 1e-8 we're tracing H2 densities of 1e22 to 1e23
#   - Vlsr - from prior knowledge of cloud kinematics
priors = [[10, 10], [5, 15], [14.0, 15.0],
          [min_sigma, 1.0], [5, 10], [0.5, 0.5]] * npeaks

# this is important! we have to set the proper weights
assert spc.errorcube is not None

# okay let's crunch it - following the Jupyter notebook example over at
# multinest repo from here onwards
minpars = np.array(priors)[:, 0]
maxpars = np.array(priors)[:, 1]
fixed = [True, False, False, False, False, True]

# NOTE: this is the most crucial parameter in the whole script!
#       a performace bottleneck and the human interaction head-scratcher,
#       all in one line of code
finesse = [1, 2, 2, 2, 20, 1] * npeaks

spc.make_guess_grid(minpars, maxpars, finesse, fixed=fixed)

# all the number crunching happens here: model grids generated,
# best models searched based on the squared residual sum
spc.generate_model()
spc.get_snr_map()
spc.best_guess()

# run pyspeckit with the guesses we got!
from multiprocessing import cpu_count
spc.fiteach(fittype=spc.fittype,
            guesses=spc.best_guesses,
            multicore=cpu_count()-1,
            #errmap=spc._rms_map, # already supplied as spc.errorcube
            **spc.fiteach_args)

# uuuuh not sure why this is needed
spc.specfit.fitter._make_parinfo(npeaks=2)
spc.specfit.parinfo = spc.specfit.fitter.parinfo

spc.mapplot()

# inspect a fit @ (x, y)
sp = spc.get_spectrum(12, 6)
sp.plotter(errstyle='fill')
sp.specfit.plot_fit()
sp.plotter.savefig('multicube-pyspekit-x12y6.png')
