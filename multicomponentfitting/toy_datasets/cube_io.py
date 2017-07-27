def get_dummy_header(cdelt1=-(4e-3 + 1e-8), cdelt2=4e-3 + 1e-8, cdelt3=1,
                     update_header_dict={}):
    # the strange cdelt values are a workaround
    # for what seems to be a bug in wcslib:
    # https://github.com/astropy/astropy/issues/4555
    keylist = {"CTYPE1": "RA---GLS", "CTYPE2": "DEC--GLS", "CTYPE3": "PARS",
               "CDELT1": cdelt1, "CDELT2": cdelt2, "CDELT3": cdelt3,
               "CRVAL1": 0, "CRVAL2": 0, "CRVAL3": 1,
               "CRPIX1": 9, "CRPIX2": 0, "CRPIX3": 0,
               "CUNIT1": "deg", "CUNIT2": "deg", "CUNIT3": "VARIOUS",
               "BMAJ": cdelt2 * 3, "BMIN": cdelt2 * 3, "BPA": 0.0,
               "BUNIT" : "VARIOUS", "EQUINOX": 2000.0}
    keylist.update(update_header_dict)

    return keylist
