import numpy as np
import itertools
from astropy.io import fits

def get_coverage(momzero, spacing):
    """
    Returns locations of SAAss
    """
    cols, rows = np.where(momzero != 0.0)

    rangex = [np.min(rows), np.max(rows)]
    sizex = np.abs(np.min(rows)-np.max(rows))
    rangey = [np.min(cols), np.max(cols)]
    sizey = np.abs(np.min(cols)-np.max(cols))

    nposx = int((sizex/(spacing*2.))+1.0)
    nposy = int((sizey/(spacing*2.))+1.0)

    cov_x = np.max(rangex)-(spacing*2.)*np.arange(nposx)
    cov_y = np.min(rangey)+(spacing*2.)*np.arange(nposy)

    return cov_x, cov_y

def define_coverage(cube, momzero, rsaa):
    """
    Returns locations of SAAs which contain significant information and computes
    a spatially-averaged spectrum.
    """

    spacing = rsaa/2.
    cov_x, cov_y = get_coverage(momzero, spacing)

    coverage = np.full([len(cov_y)*len(cov_x),2], np.nan)
    spec = np.full([cube.shape[0], len(cov_y), len(cov_x)], np.nan)

    for cx,cy in itertools.product(cov_x, cov_y):

        idx = int((cov_x[0]-cx)/rsaa)
        idy = int((cy-cov_y[0])/rsaa)

        momzero_cutout = momzero[int(cy-spacing):int(cy+spacing),
                                 int(cx-spacing):int(cx+spacing)]

        finite = np.isfinite(momzero_cutout)
        nmask = np.count_nonzero(finite)
        if nmask > 0:
            tot_non_zero = np.count_nonzero(np.isfinite(momzero_cutout) & (momzero_cutout!=0))
            fraction = tot_non_zero / nmask
            if fraction > 0.5:
                coverage[idy+(idx*len(cov_y)),:] = cx,cy
                spec[:, idy, idx] = cube[:,
                                         int(cy-spacing*2.):int(cy+spacing*2.),
                                         int(cx-spacing*2.):int(cx+spacing*2.)].mean(axis=(1,2))


    return coverage, spec

def write_averaged_spectra(cube_header, saa_spectra, rsaa,
                           fits_fmatter='saa_cube_r{}.fits'):
    """
    Writes spectra averaged on multiple scales into fits files.

    Parameters
    ----------
    cube_header : FITS header of the original spectral cube

    saa_spectra : len(N) list
                  Contains spectra averaged over N scales

    rsaa : len(N) list
           List of averaging radii

    fits_fmatter : a string formatter for output files to be written to
    """
    for r, aver_cube in zip(rsaa, saa_spectra):
        hdu = fits.PrimaryHDU(data=aver_cube, header=cube_header)
        hdu.header['RSAA'] = r
        hdu.writeto('saa_cube_r{}.fits'.format(r), overwrite=True)

def plot_rsaa(coverage_coordinates, momzero, rsaa):
    """
    Plot the SAA boxes
    """
    # TODO: make an HDU instead and plot with astropy maybe?
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig = plt.figure(1, figsize=(10.0, 4.0))
    fig.clf()
    ax = fig.add_subplot(111)
    plt.imshow(momzero, cmap='Greys', origin='lower',
               interpolation='nearest', vmax=100)
    cols = ['black', 'red', 'blue']
    size = [0.5, 1, 2]
    alpha = [1, 0.8, 0.5]
    for i, (r, covcoords) in enumerate(zip(rsaa, coverage_coordinates)):
        for j in range(covcoords.shape[0]):
            if np.all(np.isfinite(covcoords[j, :])):
                ax.add_patch(
                    patches.Rectangle(
                        (covcoords[j, 0] - r, covcoords[j, 1] - r),
                        r * 2., r * 2., facecolor='none',
                        edgecolor=cols[i], lw=size[i], alpha=alpha[i]))
    plt.draw()
    plt.show()
