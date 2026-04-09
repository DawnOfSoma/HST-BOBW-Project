# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:41:21 2022

@author: sunlu
"""

import numpy
import matplotlib.pyplot as plt

def main():
    
    print('HSC Prelim Analysis')
    
    catalog = numpy.genfromtxt('data/SDSS_HSCV3_Xmatch_wPhoto_osbornct.csv', names=True, delimiter=',', dtype=float)
    catalog_int = numpy.genfromtxt('data/SDSS_HSCV3_Xmatch_wPhoto_osbornct.csv', names=True, delimiter=',', dtype=numpy.uint64)
    
    names = catalog.dtype.names
    numcols = len(names)

    ID = catalog_int['objID']
    matchid = catalog_int['matchid']
    redshift = catalog['z']
    ra = catalog['ra']
    dec = catalog['dec']
    
    has_wide_photo = []
    ID_select = []
    matchid_select = []
    z_select = []
    ra_select = []
    dec_select = []
    used_IDs = []  # Used to check for dupes
    
    for i in range(0, len(ID)):
        
        sumval = 0
        
        if (ID[i] not in used_IDs):
        
            for j in range(19, 46):
                
                tempcol = catalog[names[j]]
                tempval = tempcol[i]
                
                if (tempval >= 0.): # Not nan
                    sumval += tempval
                    
            if sumval > 0:
                has_wide_photo.append(1)
                #used_IDs.append(ID[i])
                if redshift[i] >= 0.0 and redshift[i] < 0.3:
                    z_select.append(redshift[i])
                    ra_select.append(ra[i])
                    dec_select.append(dec[i])
                    ID_select.append(ID[i])
                    matchid_select.append(matchid[i])
            else:
                has_wide_photo.append(0)
        
    has_wide_photo = numpy.asarray(has_wide_photo)
    print(len(has_wide_photo[has_wide_photo == 1]), ' / ', len(has_wide_photo), ' have wide-band photometry')
    
    print(len(z_select), ' galaxies in selected subsample')
    
    with open('data/subsample.dat', 'w') as outfile:
        outfile.write('#objID  matchid  z  ra  dec\n')
        for i in range(0,len(ID_select)):
            outfile.write('{}  {}  {}  {}  {}\n'.format( ID_select[i], matchid_select[i], z_select[i], ra_select[i], dec_select[i] ))
    
    plt.hist(z_select)
    plt.show()
    
main()