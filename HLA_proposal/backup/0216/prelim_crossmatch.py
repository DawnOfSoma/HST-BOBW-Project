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

def main():
    
    print('Sample crossmatch')
    
    main_subsample = numpy.genfromtxt('data/subsample.dat', names=True)
    main_subsample_int = numpy.genfromtxt('data/subsample.dat', names=True, dtype=numpy.uint64)
    GSWLC = numpy.loadtxt('data/GSWLC-X2.dat')
    
    # Load SDSS params
    SDSS_objID = main_subsample_int['objID']
    matchid = main_subsample_int['matchid']
    SDSS_z = main_subsample['z']
    SDSS_ra = main_subsample['ra']
    SDSS_dec = main_subsample['dec']
    
    # Load GSW params
    GSW_ra = GSWLC[:,5]
    GSW_dec = GSWLC[:,6]
    GSW_SFR = GSWLC[:,11]
    GSW_Mstar = GSWLC[:,9]
    GSW_sed_flag = GSWLC[:,19]
    
    GSW_ra = GSW_ra[GSW_sed_flag == 0]
    GSW_dec = GSW_dec[GSW_sed_flag == 0]
    GSW_SFR = GSW_SFR[GSW_sed_flag == 0]
    GSW_Mstar = GSW_Mstar[GSW_sed_flag == 0]
    GSW_sSFR = GSW_SFR - GSW_Mstar
    
    base_coord = SkyCoord(ra=SDSS_ra, dec=SDSS_dec, unit="deg")
    gsw_coord = SkyCoord(ra=GSW_ra, dec=GSW_dec, unit="deg")
    
    # Match to GSWLC
    idmatch, d2d, d3d = match_coordinates_sky(base_coord, gsw_coord)
    print(len(idmatch), len(base_coord), len(gsw_coord))
    
    sep = d2d.to(u.arcsec)/u.arcsec
    
    objID = SDSS_objID[sep < 1]
    z = SDSS_z[sep < 1]
    matchid = matchid[sep < 1]
    ra = SDSS_ra[sep < 1]
    dec = SDSS_dec[sep < 1]
    idmatch = idmatch[sep < 1]
    sep = sep[sep < 1]
    
    Mstar = numpy.asarray([GSW_Mstar[index] for index in idmatch])
    sSFR = numpy.asarray([GSW_sSFR[index] for index in idmatch])
    
    print(len(sep))
    
    with open('data/crossmatch_GSW.dat', 'w') as outfile:
        outfile.write('#objid  matchid  z  ra  dec  idmatch  sSFR  Mstar  sep\n')
        for i in range(0, len(sep)):
            outfile.write('{}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(objID[i], matchid[i], z[i], ra[i], dec[i], idmatch[i], sSFR[i], Mstar[i], sep[i]))
    
    # Crossmatch to simard
    base_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    
    simard_catalog = fits.open('data/Simard2011_vizier_n4d1.fit')
    simard_data = simard_catalog[1].data
    
    simard_ra = simard_data.field(63)
    simard_dec = simard_data.field(64)
    simard_coord = SkyCoord(ra=simard_ra, dec=simard_dec, unit="deg")
    
    sidmatch, sd2d, sd3d = match_coordinates_sky(base_coord, simard_coord)
    print(len(sidmatch), len(base_coord), len(simard_coord))
    
    simsep = sd2d.to(u.arcsec)/u.arcsec
    
    objID = objID[simsep < 1]
    z = z[simsep < 1]
    matchid = matchid[simsep < 1]
    ra = ra[simsep < 1]
    dec = dec[simsep < 1]
    idmatch = idmatch[simsep < 1]
    sep = sep[simsep < 1]
    Mstar = Mstar[simsep < 1]
    sSFR = sSFR[simsep < 1]
    sidmatch = sidmatch[simsep < 1]
    simsep = simsep[simsep < 1]
    
    print(len(sSFR))
    
    with open('data/crossmatch_GSW_Simard.dat', 'w') as outfile:
        outfile.write('#objid  matchid  z  ra  dec  idmatch_GSW  sSFR  Mstar  sep_GSW  idmatch_Simard  sep_Simard\n')
        for i in range(0, len(sep)):
            outfile.write('{}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(objID[i], matchid[i], z[i], ra[i], dec[i], idmatch[i], sSFR[i], Mstar[i], sep[i], sidmatch[i], simsep[i]))
    
    with open('data/crossmatch_GSW_Simard_radec.csv', 'w') as outfile:
        #outfile.write('RA, DEC\n')
        for i in range(0, len(sep)):
            outfile.write('{}, {}\n'.format(ra[i], dec[i]))
    
    
main()