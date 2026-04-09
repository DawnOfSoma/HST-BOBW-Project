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
    z_all = data['z']
    Mstar_all = data['Mstar']
    sSFR_all = data['sSFR']
    n_all = data['n']
    psat_all = data['psat']
    fbar_all = data['fbar']
    fring_all = data['fring']
    flens_all = data['flens']
    fnospiral_all = data['fnospiral']
    fdiskfeat_all = data['fdiskfeat']
    fedgeon_all = data['fnotedgeon']
    fsmooth_all = data['fsmooth']
    nb_all = data['nb']
    BTnf_all = data['BTnf']
    BTn4_all = data['BTn4']
    n4bulgeRe_all = data['n4bulgeRe']
    n4rd_all = data['n4rd']
    
    objid_used = []
    z = []
    Mstar = []
    sSFR = []
    n = []
    psat = []
    fbar = []
    fring = []
    flens = []
    fsmooth = []
    nb = []
    BTnf = []
    BTn4 = []
    n4bulgeRe = []
    n4rd = []
    for i in range(0,len(objid_all)):
        if objid_all[i] not in objid_used:
            Mstar.append(Mstar_all[i])
            z.append(z_all[i])
            n4bulgeRe.append(n4bulgeRe_all[i])
            n4rd.append(n4rd_all[i])
            sSFR.append(sSFR_all[i])
            n.append(n_all[i])
            nb.append(nb_all[i])
            BTnf.append(BTnf_all[i])
            BTn4.append(BTn4_all[i])
            psat.append(psat_all[i])
            fring.append(fring_all[i])
            fbar.append(fbar_all[i])
            flens.append(flens_all[i])
            objid_used.append(objid_all[i])
            fsmooth.append(fsmooth_all[i])
            
    print(len(sSFR))
    Mstar = numpy.asarray(Mstar)
    z = numpy.asarray(z)
    sSFR = numpy.asarray(sSFR)
    n = numpy.asarray(n)
    psat = numpy.asarray(psat)
    fbar = numpy.asarray(fbar)
    fring = numpy.asarray(fring)
    flens = numpy.asarray(flens)
    fsmooth = numpy.asarray(fsmooth)
    nb = numpy.asarray(nb)
    BTnf = numpy.asarray(BTnf)
    BTn4 = numpy.asarray(BTn4)
    n4bulgeRe = numpy.asarray(n4bulgeRe)
    n4rd = numpy.asarray(n4rd)
    
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
    plt.savefig('Plots/sSFR_v_Mstar_psatcolor_{}.png'.format(matchcat))
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
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar[n <= 3], sSFR[n <= 3], c='b', s=18, label='n < 3')
    plt.scatter(Mstar[n > 3], sSFR[n > 3], c='r', s=18, label='n > 3')
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
    plt.scatter(sSFR, fbar, c='k', s=18)
    plt.xlabel('log sSFR')
    plt.xlim(-14, -7.8)
    plt.axvline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axvline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('f_bar')
    plt.savefig('Plots/fbar_vs_sSFR_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(sSFR, fring, c='k', s=18)
    plt.xlabel('log sSFR')
    plt.xlim(-14, -7.8)
    plt.axvline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axvline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('f_ring')
    plt.savefig('Plots/fring_vs_sSFR_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(sSFR, flens, c='k', s=18)
    plt.xlabel('log sSFR')
    plt.xlim(-14, -7.8)
    plt.axvline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axvline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('f_lens')
    plt.savefig('Plots/flens_vs_sSFR_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    Mstar = Mstar[fsmooth > 0.7]
    sSFR = sSFR[fsmooth > 0.7]
    n = n[fsmooth > 0.7]
    psat = psat[fsmooth > 0.7]
    nb = nb[fsmooth > 0.7]
    BTnf = BTnf[fsmooth > 0.7]
    BTn4 = BTn4[fsmooth > 0.7]
    z = z[fsmooth > 0.7]
    n4bulgeRe = n4bulgeRe[fsmooth > 0.7]
    n4rd = n4rd[fsmooth > 0.7]
    fsmooth = fsmooth[fsmooth > 0.7]
    
    SFG_Mstar = Mstar[sSFR >= -10.8]
    SFG_sSFR = sSFR[sSFR >= -10.8]
    
    RS_Mstar = Mstar[sSFR < -11.8]
    RS_sSFR = sSFR[sSFR < -11.8]
    
    GV_Mstar = Mstar[sSFR >= -11.8]
    GV_sSFR = sSFR[sSFR >= -11.8]
    GV_Mstar = GV_Mstar[GV_sSFR < -10.8]
    GV_sSFR = GV_sSFR[GV_sSFR < -10.8]
    
    plt.figure(figsize=(12,9))
    plt.hist(BTnf)
    plt.title('B/T, n free + disk')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.hist(z)
    plt.title('z, fsmooth > 0.7')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.hist(BTn4)
    plt.title('B/T, n4 + disk')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.hist(n4rd*1.678, color='b', alpha=0.5)
    plt.hist(n4bulgeRe, color='r', alpha=0.5)
    plt.title('Reff comparison')
    plt.show()
    plt.close()
    
    n4bulgeRe_scaled = 2 * n4bulgeRe / 1.43
    # see Mishra 2019: https://arxiv.org/pdf/1905.10739.pdf
    scalecut = 1.678 * n4rd / n4bulgeRe
    
    plt.figure(figsize=(12,9))
    plt.scatter(n[scalecut >= 1], BTn4[scalecut >= 1], s=18, c='b')
    plt.scatter(n[scalecut < 1], BTn4[scalecut < 1], s=18, c='r')
    plt.xlabel('n')
    plt.ylabel('BTn4')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(n, BTnf, s=18, c=nb)
    plt.xlabel('n')
    plt.ylabel('BTnf')
    plt.colorbar(label='nb')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(numpy.log10(scalecut), BTn4, s=18, c='b')
    plt.xlabel('Reff disk / Reff bulge')
    plt.ylabel('BTn4')
    plt.show()
    plt.close()
    
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
    plt.savefig('Plots/sSFR_v_Mstar_smooth_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar, sSFR, c=n, s=18)
    plt.xlabel('log $M_{*}$')
    plt.xlim(8, 12.5)
    plt.ylim(-14, -7.8)
    plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('log sSFR')
    plt.title('N = {}'.format(len(Mstar)))
    plt.colorbar(label='n', cmap='inferno')
    plt.savefig('Plots/sSFR_v_Mstar_smooth_ncolor_{}.png'.format(matchcat))
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
    plt.savefig('Plots/sSFR_v_Mstar_smooth_psatcolor_{}.png'.format(matchcat))
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
    plt.savefig('Plots/n_vs_sSFR_smooth_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar[n <= 3], sSFR[n <= 3], c='b', s=18, label='n < 3')
    Mstar_temp = Mstar[n > 3]
    sSFR_temp = sSFR[n > 3]
    n_temp = n[n > 3]
    plt.scatter(Mstar_temp[n_temp < 4], sSFR_temp[n_temp < 4], c='g', s=18, label='3 < n < 4')
    plt.scatter(Mstar[n >= 4], sSFR[n >= 4], c='r', s=18, label='n > 4')
    plt.xlabel('log $M_{*}$')
    plt.xlim(8, 12.5)
    plt.ylim(-14, -7.8)
    plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('log sSFR')
    plt.legend()
    plt.savefig('Plots/sSFR_v_Mstar_smooth_ncolor_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar, sSFR, c=BTn4, s=18)
    plt.xlabel('log $M_{*}$')
    plt.xlim(8, 12.5)
    plt.ylim(-14, -7.8)
    plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('log sSFR')
    plt.colorbar(label='BTn4')
    plt.savefig('Plots/sSFR_v_Mstar_smooth_BTn4color_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(Mstar, sSFR, c=BTnf, s=18)
    plt.xlabel('log $M_{*}$')
    plt.xlim(8, 12.5)
    plt.ylim(-14, -7.8)
    plt.axhline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axhline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('log sSFR')
    plt.colorbar(label='BTnf')
    plt.savefig('Plots/sSFR_v_Mstar_smooth_BTnfcolor_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
    z_tot = z
    
    Mstar = Mstar[BTn4 < 0.5]
    sSFR = sSFR[BTn4 < 0.5]
    psat = psat[BTn4 < 0.5]
    BTnf = BTnf[BTn4 < 0.5]
    fsmooth = fsmooth[BTn4 < 0.5]
    nb = nb[BTn4 < 0.5]
    z = z[BTn4 < 0.5]
    n = n[BTn4 < 0.5]
    BTn4 = BTn4[BTn4 < 0.5]
    
    plt.figure(figsize=(12,9))
    plt.hist(z_tot, color='b')
    plt.hist(z, color='r')
    plt.title('z')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(sSFR, nb, c=BTnf, s=18)
    plt.xlabel('log sSFR')
    plt.xlim(-14, -7.8)
    plt.axvline(-10.8, color='k', linestyle='--', linewidth=2)
    plt.axvline(-11.8, color='k', linestyle='--', linewidth=2)
    plt.ylabel('n, bulge')
    plt.colorbar(label='BTnf')
    plt.savefig('Plots/n_vs_sSFR_smooth_S0_{}.png'.format(matchcat))
    plt.show()
    plt.close()
    
def main():
    
    make_plots('GSW_Simard_Group_Gzoo')
    
main()