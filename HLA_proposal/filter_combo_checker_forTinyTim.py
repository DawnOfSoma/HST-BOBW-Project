# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:34:18 2022

@author: sunlu
"""

import numpy as np

def main():
    
    which_instrument = 'WFPC2'
    
    data = np.genfromtxt('data/crossmatch_GSW.dat', names=True)
    data_int = np.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=np.uint64)
    data_str = np.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=None, encoding=None)
    
    objid = data_int['objid']
    instrumentNames = data_str['instrument']
    detectorNames = data_str['detector']
    filterNames = data_str['filter']
    aperNames = data_str['aperture']
    
    unique_instr = []
    unique_det = []
    unique_aper = []
    unique_filt = []
    
    for i in range(0,len(objid)):
        
        if which_instrument in instrumentNames[i]:
        
            if instrumentNames[i] not in unique_instr:
                unique_instr.append(instrumentNames[i])
                
            if detectorNames[i] not in unique_det:
                unique_det.append(detectorNames[i])
                
            if filterNames[i] not in unique_filt:
                unique_filt.append(filterNames[i])
                
            if aperNames[i] not in unique_aper:
                unique_aper.append(aperNames[i])
            
    print(unique_instr)
    print(unique_det)
    print(unique_aper)
    print(unique_filt)
    
main()