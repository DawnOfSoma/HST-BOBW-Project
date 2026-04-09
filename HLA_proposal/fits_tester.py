# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:32:30 2022

@author: sunlu
"""

from astropy.io import fits
import numpy
import os

def process(imname, ncombine_list, bunit_list, zpt_list):
    
    data = fits.open(imname)
    header = data[0].header
    #print(header)
    
    zeropoint = header['PHOTZPT']
    if zeropoint not in zpt_list:
        zpt_list.append(zeropoint)
    
    try:
        ncombine = header['NCOMBINE']
    except KeyError:
        ncombine = 'N/A'
        #print(header['PHOTMODE'])
        #if 'D7' in header['PHOTMODE']:
        #    print('D7')
        #else:
        #    print('D15')
    if ncombine not in ncombine_list:
        ncombine_list.append(ncombine)
        
    try:
        bunit = header['BUNIT']
    except KeyError:
        bunit = 'N/A'
    if bunit not in bunit_list:
        bunit_list.append(bunit)
        
    #if bunit == 'COUNTS/S':
        #print(header)
        #print(imname, ncombine)
        
    # no gain keywords lol
        
    # PHOTZPT gives 'ST magnitude zero point' whatever that means
    # PHOTFLAM gives inverse sensitivity in ergs/cm2/A/e-
    # PHOTFNU gives inverse sensitivity in Jy*sec/e-
    
    return 0

def main():
    
    data = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True)
    data_int = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=numpy.uint64)
    data_str = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=None, encoding=None)
    
    objid = data_int['objid']
    imageNames = data_str['imname']
    
    objid_unique = numpy.unique(objid)
    
    ncombine_list = []
    bunit_list = []
    zpt_list = []
    
    for i in range(0,len(objid_unique)):
        
        if i % 100 == 0:
            print(i, '/', len(objid_unique))
       
        all_indices = numpy.where(objid == objid_unique[i])
        all_imageNames = imageNames[all_indices[0]]
        
        images_unique = numpy.unique(all_imageNames)
    
        for name in images_unique:
            impath = 'HST_image_repo/' + str(objid_unique[i]) + '/' + name + '.fits'
            if os.path.exists(impath):
                process(impath, ncombine_list, bunit_list, zpt_list)
                
    print(ncombine_list)
    print(bunit_list)
    print(zpt_list)
    
main()