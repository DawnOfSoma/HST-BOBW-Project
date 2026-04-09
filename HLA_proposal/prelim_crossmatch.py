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
    main_subsample_str = numpy.genfromtxt('data/subsample.dat', names=True, dtype=None, encoding=None)
    main_subsample_int = numpy.genfromtxt('data/subsample.dat', names=True, dtype=numpy.uint64)
    GSWLC = numpy.loadtxt('data/GSWLC-X2.dat')
    
    # Load SDSS params
    SDSS_objID = main_subsample_int['objID']
    matchid = main_subsample_int['matchid']
    
    imname = main_subsample_str['imname']
    instrument = main_subsample_str['instrument']
    detector = main_subsample_str['detector']
    aperture = main_subsample_str['aperture']
    pfilter = main_subsample_str['filter']
    
    SDSS_z = main_subsample['z']
    SDSS_ra = main_subsample['ra']
    SDSS_dec = main_subsample['dec']
    exptime = main_subsample['exptime']
    
    pra = main_subsample['pra']
    pdec = main_subsample['pdec']
    prad = main_subsample['prad']
    pr50 = main_subsample['pr50']
    pr90 = main_subsample['pr90']
    
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
    exptime = exptime[sep < 1]
    pfilter = pfilter[sep < 1]
    idmatch = idmatch[sep < 1]
    imname = imname[sep < 1]
    instrument = instrument[sep < 1]
    detector = detector[sep < 1]
    aperture = aperture[sep < 1]
    pra = pra[sep < 1]
    pdec = pdec[sep < 1]
    prad = prad[sep < 1]
    pr50 = pr50[sep < 1]
    pr90 = pr90[sep < 1]
    sep = sep[sep < 1]
    
    Mstar = numpy.asarray([GSW_Mstar[index] for index in idmatch])
    sSFR = numpy.asarray([GSW_sSFR[index] for index in idmatch])
    
    print(len(sep))
    
    with open('data/crossmatch_GSW.dat', 'w') as outfile:
        outfile.write('#objid  matchid  z  ra  dec  idmatch  sSFR  Mstar  sep  imname  instrument  detector  aperture  exptime  filter  pra  pdec  prad  pr50  pr90\n')
        for i in range(0, len(sep)):
            outfile.write('{}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(objID[i], matchid[i], z[i], ra[i], dec[i], idmatch[i], sSFR[i], Mstar[i], sep[i], imname[i], instrument[i], detector[i], aperture[i], exptime[i], pfilter[i], pra[i], pdec[i], prad[i], pr50[i], pr90[i]))
    
    # Crossmatch to simard
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
    
    objID = objID[simsep < 1]
    z = z[simsep < 1]
    matchid = matchid[simsep < 1]
    ra = ra[simsep < 1]
    dec = dec[simsep < 1]
    exptime = exptime[simsep < 1]
    pfilter = pfilter[simsep < 1]
    idmatch = idmatch[simsep < 1]
    sep = sep[simsep < 1]
    Mstar = Mstar[simsep < 1]
    sSFR = sSFR[simsep < 1]
    imname = imname[simsep < 1]
    instrument = instrument[simsep < 1]
    detector = detector[simsep < 1]
    sidmatch = sidmatch[simsep < 1]
    pra = pra[simsep < 1]
    pdec = pdec[simsep < 1]
    prad = prad[simsep < 1]
    pr50 = pr50[simsep < 1]
    pr90 = pr90[simsep < 1]
    simsep = simsep[simsep < 1]
    
    n = numpy.asarray([simard_n[index] for index in sidmatch])
    nb = numpy.asarray([simard_nb[index] for index in sidmatch])
    BTnf = numpy.asarray([simard_BT_nfd1[index] for index in sidmatch])
    BTn4 = numpy.asarray([simard_BT_n4d1[index] for index in sidmatch])
    n4bulgeRe = numpy.asarray([simard_n4bulgeRe[index]/simard_scale[index] for index in sidmatch])
    n4rd = numpy.asarray([simard_n4rd[index]/simard_scale[index] for index in sidmatch])
    
    print(len(sSFR))
    
    with open('data/crossmatch_GSW_Simard.dat', 'w') as outfile:
        outfile.write('#objid  matchid  z  ra  dec  idmatch_GSW  sSFR  Mstar  sep_GSW  idmatch_Simard  sep_Simard  n  nb  BTnf  BTn4  n4bulgeRe  n4rd  imname  instrument  detector  exptime  filter  pra  pdec  prad  pr50  pr90\n')
        for i in range(0, len(sep)):
            outfile.write('{}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(objID[i], matchid[i], z[i], ra[i], dec[i], idmatch[i], sSFR[i], Mstar[i], sep[i], sidmatch[i], simsep[i], n[i], nb[i], BTnf[i], BTn4[i], n4bulgeRe[i], n4rd[i], imname[i], instrument[i], detector[i], exptime[i], pfilter[i], pra[i], pdec[i], prad[i], pr50[i], pr90[i]))
    
    with open('data/crossmatch_GSW_Simard_radec.csv', 'w') as outfile:
        #outfile.write('RA, DEC\n')
        for i in range(0, len(sep)):
            outfile.write('{}, {}\n'.format(ra[i], dec[i]))
    
    # Crossmatch to environment catalog
    base_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    
    env_catalog = numpy.loadtxt('data/DR13_extracted_psat.dat')
    
    env_ra = env_catalog[:,1]
    env_dec = env_catalog[:,2]
    env_psat = env_catalog[:,3]
    env_coord = SkyCoord(ra=env_ra, dec=env_dec, unit="deg")
    
    eidmatch, ed2d, ed3d = match_coordinates_sky(base_coord, env_coord)
    print(len(eidmatch), len(base_coord), len(env_coord))
    
    esep = ed2d.to(u.arcsec)/u.arcsec
    
    objID = objID[esep < 1]
    z = z[esep < 1]
    matchid = matchid[esep < 1]
    ra = ra[esep < 1]
    dec = dec[esep < 1]
    n = n[esep < 1]
    nb = nb[esep < 1]
    BTnf = BTnf[esep < 1]
    BTn4 = BTn4[esep < 1]
    n4bulgeRe = n4bulgeRe[esep < 1]
    n4rd = n4rd[esep < 1]
    exptime = exptime[esep < 1]
    pfilter = pfilter[esep < 1]
    idmatch = idmatch[esep < 1]
    sep = sep[esep < 1]
    Mstar = Mstar[esep < 1]
    sSFR = sSFR[esep < 1]
    imname = imname[esep < 1]
    instrument = instrument[esep < 1]
    detector = detector[esep < 1]
    sidmatch = sidmatch[esep < 1]
    simsep = simsep[esep < 1]
    eidmatch = eidmatch[esep < 1]
    pra = pra[esep < 1]
    pdec = pdec[esep < 1]
    prad = prad[esep < 1]
    pr50 = pr50[esep < 1]
    pr90 = pr90[esep < 1]
    esep = esep[esep < 1]
    
    psat = numpy.asarray([env_psat[index] for index in eidmatch])
    
    print(len(sSFR))
    
    with open('data/crossmatch_GSW_Simard_Group.dat', 'w') as outfile:
        outfile.write('#objid  matchid  z  ra  dec  idmatch_GSW  sSFR  Mstar  sep_GSW  idmatch_Simard  sep_Simard  n  nb  BTnf  BTn4  n4bulgeRe  n4rd  imname  instrument  detector  exptime  filter  idmatch_env  psat  sep_env  pra  pdec  prad  pr50  pr90\n')
        for i in range(0, len(sep)):
            outfile.write('{}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(objID[i], matchid[i], z[i], ra[i], dec[i], idmatch[i], sSFR[i], Mstar[i], sep[i], sidmatch[i], simsep[i], n[i], nb[i], BTnf[i], BTn4[i], n4bulgeRe[i], n4rd[i], imname[i], instrument[i], detector[i], exptime[i], pfilter[i], eidmatch[i], psat[i], esep[i], pra[i], pdec[i], prad[i], pr50[i], pr90[i]))
    
    # Crossmatch to environment catalog
    base_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    
    gzoo_catalog = numpy.genfromtxt('data/gz2_hart16.csv', delimiter=',', names=True, usecols=(1,2,13,19,37,43,61,121,127))
    
    gzoo_ra = gzoo_catalog['ra']
    gzoo_dec = gzoo_catalog['dec']
    gzoo_fbar = gzoo_catalog['t03_bar_a06_bar_debiased']
    gzoo_fring = gzoo_catalog['t08_odd_feature_a19_ring_debiased']
    gzoo_flens = gzoo_catalog['t08_odd_feature_a20_lens_or_arc_debiased']
    gzoo_nospiral = gzoo_catalog['t04_spiral_a09_no_spiral_debiased']
    gzoo_smooth = gzoo_catalog['t01_smooth_or_features_a01_smooth_debiased']
    gzoo_diskfeat = gzoo_catalog['t01_smooth_or_features_a02_features_or_disk_debiased']
    gzoo_notedgeon = gzoo_catalog['t02_edgeon_a05_no_debiased']
    gzoo_coord = SkyCoord(ra=gzoo_ra, dec=gzoo_dec, unit="deg")
    
    gidmatch, gd2d, gd3d = match_coordinates_sky(base_coord, gzoo_coord)
    print(len(gidmatch), len(base_coord), len(gzoo_coord))
    
    gsep = gd2d.to(u.arcsec)/u.arcsec
    
    objID = objID[gsep < 1]
    z = z[gsep < 1]
    matchid = matchid[gsep < 1]
    ra = ra[gsep < 1]
    dec = dec[gsep < 1]
    n = n[gsep < 1]
    nb = nb[gsep < 1]
    BTnf = BTnf[gsep < 1]
    BTn4 = BTn4[gsep < 1]
    n4bulgeRe = n4bulgeRe[gsep < 1]
    n4rd = n4rd[gsep < 1]
    exptime = exptime[gsep < 1]
    pfilter = pfilter[gsep < 1]
    Mstar = Mstar[gsep < 1]
    sSFR = sSFR[gsep < 1]
    imname = imname[gsep < 1]
    instrument = instrument[gsep < 1]
    detector = detector[gsep < 1]
    psat = psat[gsep < 1]
    sep = sep[gsep < 1]
    pra = pra[gsep < 1]
    pdec = pdec[gsep < 1]
    prad = prad[gsep < 1]
    pr50 = pr50[gsep < 1]
    pr90 = pr90[gsep < 1]
    gidmatch = gidmatch[gsep < 1]
    
    fbar = numpy.asarray([gzoo_fbar[index] for index in gidmatch])
    fring = numpy.asarray([gzoo_fring[index] for index in gidmatch])
    flens = numpy.asarray([gzoo_flens[index] for index in gidmatch])
    fnospiral = numpy.asarray([gzoo_nospiral[index] for index in gidmatch])
    fdiskfeat = numpy.asarray([gzoo_diskfeat[index] for index in gidmatch])
    fnotedgeon = numpy.asarray([gzoo_notedgeon[index] for index in gidmatch])
    fsmooth = numpy.asarray([gzoo_smooth[index] for index in gidmatch])
    
    print(len(sSFR))
    
    with open('data/crossmatch_GSW_Simard_Group_Gzoo.dat', 'w') as outfile:
        outfile.write('#objid  matchid  z  ra  dec  sSFR  Mstar  n  nb  BTnf  BTn4  n4bulgeRe  n4rd  imname  instrument  detector  exptime  filter  psat  fbar  fring  flens  fnospiral  fdiskfeat  fnotedgeon  fsmooth  pra  pdec  prad  pr50  pr90\n')
        for i in range(0, len(sep)):
            outfile.write('{}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(objID[i], matchid[i], z[i], ra[i], dec[i], sSFR[i], Mstar[i], n[i], nb[i], BTnf[i], BTn4[i], n4bulgeRe[i], n4rd[i], imname[i], instrument[i], detector[i], exptime[i], pfilter[i], psat[i], fbar[i], fring[i], flens[i], fnospiral[i], fdiskfeat[i], fnotedgeon[i], fsmooth[i], pra[i], pdec[i], prad[i], pr50[i], pr90[i]))
    
main()