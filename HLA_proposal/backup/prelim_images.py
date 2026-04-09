# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:57:59 2022

@author: sunlu
"""

import numpy
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView

def main():
    
    print('Image downloader')
    
    data = numpy.genfromtxt('data/crossmatch_GSW_Simard.dat', names=True)
    
    ra = data['ra']
    dec = data['dec']
    
    
    
main()