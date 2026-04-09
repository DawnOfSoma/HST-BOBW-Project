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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def main():
    
    print('GSW - simard crossmatch')
   
    GSWLC = numpy.loadtxt('data/GSWLC-X2.dat')
    
    # Load GSW params
    GSW_ra = GSWLC[:,5]
    GSW_dec = GSWLC[:,6]
    GSW_SFR = GSWLC[:,11]
    GSW_Mstar = GSWLC[:,9]
    GSW_sed_flag = GSWLC[:,19]
    
    ra = GSW_ra[GSW_sed_flag == 0]
    dec = GSW_dec[GSW_sed_flag == 0]
    SFR = GSW_SFR[GSW_sed_flag == 0]
    Mstar = GSW_Mstar[GSW_sed_flag == 0]
    sSFR = SFR - Mstar
    
    base_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    
    simard_catalog = fits.open('data/Simard2011_vizier_nfree.fit')
    simard_data = simard_catalog[1].data
    
    simard_catalog_n4d1 = fits.open('data/Simard2011_vizier_n4d1.fit')
    simard_data_n4d1 = simard_catalog_n4d1[1].data
    
    simard_catalog_nfd1 = fits.open('data/Simard2011_freebulge_plusdisk.fit')
    simard_data_nfd1 = simard_catalog_nfd1[1].data
    
    simard_n = simard_data.field(35)
    simard_ra = simard_data.field(40)
    simard_dec = simard_data.field(41)
    simard_coord = SkyCoord(ra=simard_ra, dec=simard_dec, unit="deg")
    
    simard_BT_n4d1 = simard_data_n4d1.field(15)
    simard_n4bulgeRe = simard_data_n4d1.field(23) # kpc
    simard_n4rd = simard_data_n4d1.field(29)
    simard_scale = simard_data_n4d1.field(4) # kpc/"
    simard_BT_nfd1 = simard_data_nfd1.field(15)
    simard_nb = simard_data_nfd1.field(57)
    
    sidmatch, sd2d, sd3d = match_coordinates_sky(base_coord, simard_coord)
    print(len(sidmatch), len(base_coord), len(simard_coord))
    
    simsep = sd2d.to(u.arcsec)/u.arcsec
    
    ra = ra[simsep < 1]
    dec = dec[simsep < 1]
    Mstar = Mstar[simsep < 1]
    SFR = SFR[simsep < 1]
    sSFR = sSFR[simsep < 1]
    sidmatch = sidmatch[simsep < 1]
    simsep = simsep[simsep < 1]
    
    n = numpy.asarray([simard_n[index] for index in sidmatch])
    nb = numpy.asarray([simard_nb[index] for index in sidmatch])
    BTnf = numpy.asarray([simard_BT_nfd1[index] for index in sidmatch])
    BTn4 = numpy.asarray([simard_BT_n4d1[index] for index in sidmatch])
    n4bulgeRe = numpy.asarray([simard_n4bulgeRe[index]/simard_scale[index] for index in sidmatch])
    n4rd = numpy.asarray([simard_n4rd[index]/simard_scale[index] for index in sidmatch])
    
    print(len(sSFR))
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar, SFR, c=n, s=18)
    plt.xlabel('log $M_{*}$')
    plt.xlim(8, 12.5)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(direction='in', size=16, top=True)
    ax.yaxis.set_tick_params(direction='in', size=10, right=True)
    ax.xaxis.set_tick_params(direction='in', size=8, top=True, bottom=True, which='minor')
    ax.yaxis.set_tick_params(direction='in', size=5, right=True, left=True, which='minor')
    #plt.ylim(-14, -7.8)
    #plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    #plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('Log SFR [$M_{\odot}$/year]')
    plt.colorbar(label='Sersic n')
    plt.savefig('Plots/sSFR_v_Mstar_ncolor_GSW_Simard_all.png')
    plt.show()
    plt.close()
    
main()