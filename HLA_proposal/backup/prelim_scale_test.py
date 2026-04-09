# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:56:55 2022

@author: sunlu
"""

import matplotlib.pyplot as plt
import numpy
import astropy
from astropy.cosmology import WMAP9 as cosmo

print('SDSS / HST resolution comparison')

redshift = numpy.linspace(0, 1., 1000)

HST_PSF = 0.1 # Approximate, but typical for ACS / WFC3
# https://archive.stsci.edu/pub/hlsp/candels/goods-s/gsd01/v0.5/hlsp_candels_hst_acs-wfc_gsd01_readme_v0.5.pdf

SDSS_PSF = 1.0 # Also approximate
# https://skyserver.sdss.org/dr1/en/proj/advanced/color/sdssfilters.asp

HST_pixscale = 0.05
# Should be ~0.05 for ACS,

