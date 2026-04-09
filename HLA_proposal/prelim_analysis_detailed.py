# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:41:21 2022

@author: sunlu
"""

import numpy
import matplotlib.pyplot as plt

def main():
    
    print('HSC Prelim Analysis')
    
    catalog = numpy.genfromtxt('data/D030122_SDSS_HSCV3_Xmatch_osbornct_0.csv', names=True, delimiter=',', dtype=float)
    catalog_str = numpy.genfromtxt('data/D030122_SDSS_HSCV3_Xmatch_osbornct_0.csv', names=True, delimiter=',', dtype=None, encoding=None)
    catalog_int = numpy.genfromtxt('data/D030122_SDSS_HSCV3_Xmatch_osbornct_0.csv', names=True, delimiter=',', dtype=numpy.uint64)
    
    names = catalog.dtype.names

    ID = catalog_int['objID']
    phot_type = catalog_int['type']
    matchid = catalog_int['matchid']
    
    redshift = catalog['z']
    ra = catalog['ra']
    dec = catalog['dec']
    rpetro = catalog['petroMag_r']
    exptime = catalog['ExposureTime']
    
    pra = catalog['photoRa']
    pdec = catalog['photoDec']
    petroRad = catalog['petroRad_r']
    petroR50 = catalog['petroR50_r']
    petroR90 = catalog['petroR90_r']
    
    specclass = catalog_str['class']
    specsubclass = catalog_str['subclass']
    specsurvey = catalog_str['survey']
    programName = catalog_str['programname']
    imageName = catalog_str['ImageName']
    instrument = catalog_str['Instrument']
    detector = catalog_str['Detector']
    pfilter = catalog_str['Filter']
    aperture = catalog_str['Aperture']
    
    ID_select = []
    phottype_select = []
    matchid_select = []
    
    z_select = []
    ra_select = []
    dec_select = []
    rpetro_select = []
    exptime_select = []
    
    pra_sel = []
    pdec_sel = []
    prad_sel = []
    pr50_sel = []
    pr90_sel = []
    fracdev_sel = []
    
    specclass_select = []
    specsubclass_select = []
    specsurvey_select = []
    programname_select = []
    imname_select = []
    instr_select = []
    det_select = []
    filter_select = []
    aper_select = []
    
    used_IDs = []  # Used to check for dupes
    
    count_nodupes = 0
    
    griz_filters = ['F390W', 'F435W', 'F438W', 'F450W', 'F475W', 'F555W', 'F569W', 'F606W', 'F622W', 'F625W', 'F675W', 
                    'F702W', 'F775W', 'F791W', 'F814W', 'F850LP']
    
    for i in range(0, len(ID)):
        
        # Exclude stars and IR detections
        if specclass[i] != 'STAR' and pfilter[i] in griz_filters: 
                
            #redshift[i] >= 0. and redshift[i] < 0.3:
            z_select.append(redshift[i])
            ra_select.append(ra[i])
            dec_select.append(dec[i])
            rpetro_select.append(rpetro[i])
            exptime_select.append(exptime[i])
            
            ID_select.append(ID[i])
            matchid_select.append(matchid[i])
            phottype_select.append(phot_type[i])
            
            imname_select.append(imageName[i])
            instr_select.append(instrument[i])
            det_select.append(detector[i])
            specclass_select.append(specclass[i])
            specsubclass_select.append(specsubclass[i].replace(' ', ''))
            specsurvey_select.append(specsurvey[i])
            programname_select.append(programName[i])
            filter_select.append(pfilter[i])
            aper_select.append(aperture[i])
            
            pra_sel.append(pra[i])
            pdec_sel.append(pdec[i])
            prad_sel.append(petroRad[i])
            pr50_sel.append(petroR50[i])
            pr90_sel.append(petroR90[i])
            
            if ID[i] not in used_IDs:
                count_nodupes += 1
                used_IDs.append(ID[i])
    
    print(count_nodupes, ' galaxies in selected subsample')
    
    with open('data/subsample.dat', 'w') as outfile:
        outfile.write('#objID  matchid  type  z  ra  dec exptime  imname  instrument  detector  aperture  class  subclass  survey  program  filter  pra  pdec  prad  pr50  pr90  rpetro\n')
        for i in range(0,len(ID_select)):
            outfile.write('{}  {}  {}  '.format( ID_select[i], matchid_select[i], phottype_select[i]))
            outfile.write('{}  {}  {}  {}  '.format(z_select[i], ra_select[i], dec_select[i], exptime_select[i]))
            outfile.write('{}  {}  {}  {}  {}  {}  {}  {}  {}  '.format(imname_select[i], instr_select[i], det_select[i], aper_select[i], specclass_select[i], specsubclass_select[i], specsurvey_select[i], programname_select[i], filter_select[i]))
            outfile.write('{}  {}  {}  {}  {}  {}\n'.format(pra_sel[i], pdec_sel[i], prad_sel[i], pr50_sel[i], pr90_sel[i], rpetro_select[i]))
    
main()