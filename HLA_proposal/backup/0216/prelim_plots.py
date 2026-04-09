# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:05:09 2022

@author: sunlu
"""

import numpy
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 26})

def main():
    
    print('Prelim plots')
    
    data = numpy.genfromtxt('data/crossmatch_GSW_Simard.dat', names=True)
    data_int = numpy.genfromtxt('data/crossmatch_GSW_Simard.dat', names=True, dtype=numpy.uint64)
    
    objid_all = data_int['objid']
    Mstar_all = data['Mstar']
    sSFR_all = data['sSFR']
    
    objid_used = []
    Mstar = []
    sSFR = []
    for i in range(0,len(objid_all)):
        if objid_all[i] not in objid_used:
            Mstar.append(Mstar_all[i])
            sSFR.append(sSFR_all[i])
            objid_used.append(objid_all[i])
    print(len(sSFR))
    Mstar = numpy.asarray(Mstar)
    sSFR = numpy.asarray(sSFR)
    
    SFG_Mstar = Mstar[sSFR >= -10.8]
    SFG_sSFR = sSFR[sSFR >= -10.8]
    
    RS_Mstar = Mstar[sSFR < -11.8]
    RS_sSFR = sSFR[sSFR < -11.8]
    
    GV_Mstar = Mstar[sSFR >= -11.8]
    GV_sSFR = sSFR[sSFR >= -11.8]
    GV_Mstar = GV_Mstar[GV_sSFR < -10.8]
    GV_sSFR = GV_sSFR[GV_sSFR < -10.8]
    
    plt.figure(figsize=(12,9))
    plt.scatter(SFG_Mstar, SFG_sSFR, c='b', s=18)
    plt.scatter(GV_Mstar, GV_sSFR, c='g', s=18)
    plt.scatter(RS_Mstar, RS_sSFR, c='r', s=18)
    plt.title('SFG: {}, GV: {}, RS: {}'.format(len(SFG_Mstar), len(GV_Mstar), len(RS_Mstar)))
    plt.xlabel('log $M_{*}$')
    plt.ylabel('log sSFR')
    plt.savefig('Plots/sSFR_v_Mstar_GSW_Simard.png')
    plt.show()
    plt.close()
    
main()