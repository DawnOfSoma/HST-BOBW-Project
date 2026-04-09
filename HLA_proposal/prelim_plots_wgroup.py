# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:05:09 2022

@author: sunlu
"""

import numpy
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 26})

def make_plots(matchcat):
    
    print(matchcat)
    
    data = numpy.genfromtxt('data/crossmatch_{}.dat'.format(matchcat), names=True)
    data_int = numpy.genfromtxt('data/crossmatch_{}.dat'.format(matchcat), names=True, dtype=numpy.uint64)
    
    objid_all = data_int['objid']
    Mstar_all = data['Mstar']
    sSFR_all = data['sSFR']
    n_all = data['n']
    psat_all = data['psat']
    
    objid_used = []
    Mstar = []
    sSFR = []
    n = []
    psat = []
    for i in range(0,len(objid_all)):
        if objid_all[i] not in objid_used:
            Mstar.append(Mstar_all[i])
            sSFR.append(sSFR_all[i])
            n.append(n_all[i])
            psat.append(psat_all[i])
            objid_used.append(objid_all[i])
    print(len(sSFR))
    Mstar = numpy.asarray(Mstar)
    sSFR = numpy.asarray(sSFR)
    n = numpy.asarray(n)
    psat = numpy.asarray(psat)
    
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
    plt.xlim(8, 12.5)
    plt.ylim(-14, -7.8)
    plt.ylabel('log sSFR')
    plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.savefig('Plots/sSFR_v_Mstar_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar, sSFR, c=n, s=18)
    plt.title('SFG: {}, GV: {}, RS: {}'.format(len(SFG_Mstar), len(GV_Mstar), len(RS_Mstar)))
    plt.xlabel('log $M_{*}$')
    plt.xlim(8, 12.5)
    plt.ylim(-14, -7.8)
    plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('log sSFR')
    plt.colorbar(label='n')
    plt.savefig('Plots/sSFR_v_Mstar_ncolor_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(sSFR, n, c='b', s=18)
    plt.xlabel('log sSFR')
    plt.xlim(-14, -7.8)
    plt.axvline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axvline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('n')
    plt.savefig('Plots/n_vs_sSFR_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar[psat==1], sSFR[psat == 1], c='b', s=18, label='Sat')
    plt.scatter(Mstar[psat==0], sSFR[psat == 0], c='r', s=18, label='Cent')
    plt.title('SFG: {}, GV: {}, RS: {}'.format(len(SFG_Mstar), len(GV_Mstar), len(RS_Mstar)))
    plt.xlabel('log $M_{*}$')
    plt.xlim(8, 12.5)
    plt.ylim(-14, -7.8)
    plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('log sSFR')
    plt.legend()
    plt.savefig('Plots/sSFR_v_Mstar_ncolor_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(sSFR[psat==1], n[psat == 1], c='b', s=18, label='Sat')
    plt.scatter(sSFR[psat==0], n[psat == 0], c='r', s=18, label='Cent')
    plt.xlabel('log sSFR')
    plt.xlim(-14, -7.8)
    plt.axvline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axvline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('n')
    plt.legend()
    plt.savefig('Plots/n_vs_sSFR_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
def main():
    
    #make_plots('GSW')
    #make_plots('GSW_Simard')
    make_plots('GSW_Simard_Group')
    
main()