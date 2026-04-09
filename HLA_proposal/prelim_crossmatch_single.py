# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:56:55 2022

@author: sunlu
"""

import matplotlib.pyplot as plt
import numpy
import astropy
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky

def crossmatch(samplefile, filename, ra_index, dec_index, mode, delim=None):
    
    print(filename)
    
    main_subsample = numpy.genfromtxt('data/' + samplefile, names=True)
    main_subsample_int = numpy.genfromtxt('data/' + samplefile, names=True, dtype=numpy.uint64)
    
    # Load SDSS params
    try:
        SDSS_objID = main_subsample_int['objid']
    except:
        SDSS_objID = main_subsample_int['objID']
    SDSS_ra = main_subsample['ra']
    SDSS_dec = main_subsample['dec']
    
    # Load cat params
    if mode == 'loadtxt':
        match_cat = numpy.loadtxt('data/{}'.format(filename))
        match_ra = match_cat[:,ra_index]
        match_dec = match_cat[:,dec_index]
    if mode == 'genfromtxt':
        match_cat = numpy.genfromtxt('data/{}'.format(filename), delimiter=delim, names=True)
        match_ra = match_cat[ra_index]
        match_dec = match_cat[dec_index]
    
    base_coord = SkyCoord(ra=SDSS_ra, dec=SDSS_dec, unit="deg")
    match_coord = SkyCoord(ra=match_ra, dec=match_dec, unit="deg")
    
    # Match to GSWLC
    idmatch, d2d, d3d = match_coordinates_sky(base_coord, match_coord)
    
    sep = d2d.to(u.arcsec)/u.arcsec
    
    objID = SDSS_objID[sep < 1]
    idmatch = idmatch[sep < 1]
    sep = sep[sep < 1]
    
    used_IDs = []
    
    for i in range(0,len(objID)):
        if objID[i] not in used_IDs:
            used_IDs.append(objID[i])
    
    print(len(used_IDs), ' objects in match')
    
def main():
    
    #crossmatch('GSWLC-X2.dat', 5, 6, 'loadtxt')
    #crossmatch('zoo2MainSpecz.csv', 'ra', 'dec', 'genfromtxt', delim=',')
    """crossmatch('crossmatch_GSW_Simard.dat', 'DR13_extracted_psat.dat', 'ra', 'dec', 'genfromtxt', delim='  ')
    crossmatch('subsample.dat', 'NA2010_visual_morphologies.csv', '_RA', '_DE', 'genfromtxt', delim=',')
    crossmatch('crossmatch_GSW.dat', 'NA2010_visual_morphologies.csv', '_RA', '_DE', 'genfromtxt', delim=',')
    crossmatch('crossmatch_GSW_Simard.dat', 'NA2010_visual_morphologies.csv', '_RA', '_DE', 'genfromtxt', delim=',')"""
    
    #crossmatch('crossmatch_GSW.dat', 'DR13_extracted_psat.dat', 'ra', 'dec', 'genfromtxt', delim='  ')
    crossmatch('crossmatch_GSW.dat','zoo2MainSpecz.csv', 'ra', 'dec', 'genfromtxt', delim=',')
    
main()