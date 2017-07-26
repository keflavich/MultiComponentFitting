
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Box1DKernel

from wiener_decomposition import decompose

from paths import fourteenB_wGBT_HI_file_dict


cube = SpectralCube.read(fourteenB_wGBT_HI_file_dict['Cube'])

spec_axis = cube.spectral_axis
chan_width = np.abs(np.diff(spec_axis[:2])[0])

# Check on a subset of the cube
for i in range(603, 800):
    for j in range(450, 700):

        spec = cube[:, i, j]
        # new_spec = np.arange(spec_axis.min().value,
        #                      spec_axis.max().value + 4 * chan_width.value,
        #                      4 * chan_width.value) * spec_axis.unit
        # degraded_spec = spec.spectral_smooth(Box1DKernel(4))
        # degraded_spec = degraded_spec.spectral_interpolate(new_spec)
        degraded_spec = spec
        new_spec = spec_axis

        tester = decompose(new_spec.value, degraded_spec.value, width_factor=3)
        plt.draw()
        raw_input("{0}, {1}?".format(i, j))
        plt.clf()
