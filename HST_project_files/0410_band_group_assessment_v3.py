# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:09:23 2023

@author: sunlu
"""

import numpy
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9 as cosmo
import astropy.units as apu

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def get_best_and_2ndbest_diffs(catalog, arcsec_per_kpc, mask):
    
    mags = catalog['s_mag'][mask]
    exptimes = catalog['exptime'][mask] * ( 10**(-0.4 * mags) )
    n = numpy.log10( catalog['s_n'][mask] )
    nb = numpy.log10( catalog['fb_n'][mask] )
    BTn4 = catalog['4_BT'][mask]
    BTnf = catalog['f_BT'][mask]
    pixScale = catalog['pixScale'][mask]
    Reff = numpy.log10( catalog['s_Reff'][mask] * (pixScale / arcsec_per_kpc) )
    
    max_exptime_index = numpy.where( exptimes == numpy.max(exptimes) )[0][0]
    n_1 = n[max_exptime_index]
    nb_1 = nb[max_exptime_index]
    BTn4_1 = BTn4[max_exptime_index]
    BTnf_1 = BTnf[max_exptime_index]
    Reff_1 = Reff[max_exptime_index]
    
    exptimes = numpy.delete( exptimes, max_exptime_index )
    mags = numpy.delete( mags, max_exptime_index )
    n = numpy.delete( n, max_exptime_index )
    nb = numpy.delete( nb, max_exptime_index )
    BTn4 = numpy.delete( BTn4, max_exptime_index )
    BTnf = numpy.delete( BTnf, max_exptime_index )
    Reff = numpy.delete( Reff, max_exptime_index )
    
    max_exptime_index = numpy.where( exptimes == numpy.max(exptimes) )[0][0]
    n_2 = n[max_exptime_index]
    nb_2 = nb[max_exptime_index]
    BTn4_2 = BTn4[max_exptime_index]
    BTnf_2 = BTnf[max_exptime_index]
    Reff_2 = Reff[max_exptime_index]
    
    expt = exptimes[max_exptime_index]
    mag = mags[max_exptime_index]
    delta_n = n_1 - n_2
    delta_nb = nb_1 - nb_2
    delta_BTn4 = BTn4_1 - BTn4_2
    delta_BTnf = BTnf_1 - BTnf_2
    delta_Reff = Reff_1 - Reff_2
    
    return expt, mag, delta_n, delta_Reff, delta_BTn4, delta_BTnf, delta_nb

catalog = numpy.genfromtxt('0327_CO_HST_improcess_infocat.txt', names=True, dtype=numpy.uint64)
ids = catalog['objid']
unique_ids = numpy.unique( ids )
print(len(unique_ids))

catalog = numpy.genfromtxt('0320_final_morphology_catalog_consolidated.txt', names=True)
catalog_str = numpy.genfromtxt('0320_final_morphology_catalog_consolidated.txt', names=True, dtype=None, encoding=None)
catalog_int = numpy.genfromtxt('0320_final_morphology_catalog_consolidated.txt', names=True, dtype=numpy.uint64)
ids = catalog_int['objid']
unique_ids = numpy.unique( ids )
print(len(unique_ids))

used_ids = []

B_count = 0
R_count = 0
I_count = 0

BorR_count = 0
BorI_count = 0
RorI_count = 0

BandR_count = 0
BandI_count = 0
RandI_count = 0

BorRorI_count = 0
BandRandI_count = 0

Bmult_count = 0
Rmult_count = 0
Imult_count = 0

delta_n = []
delta_Reff = []
delta_BTn4 = []
delta_BTnf = []
delta_nb = []

bri_n = []
bri_n_sim = []

expts = []
mags = []

plot_label = 'default'
band_label = 'R*'

gswlc = numpy.genfromtxt('data/crossmatch_GSW_Simard_inclusive.dat', names=True)
gswlc_int = numpy.genfromtxt('data/crossmatch_GSW_Simard_inclusive.dat', names=True, dtype=numpy.uint64)
gswid = gswlc_int['objid']
gsw_z = gswlc['z']
sim_n = gswlc['n']
 
for i in range(0,len(unique_ids)):
    
    hasB = False
    hasR = False
    hasI = False
    
    if unique_ids[i] not in used_ids:
        
        gsw_matchid = numpy.where( gswid == unique_ids[i]  )[0][0]
        redshift = gsw_z[gsw_matchid]
        n = sim_n[gsw_matchid]
        arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift) * (apu.kpc / apu.arcsec)
        
        matchids = numpy.where( catalog_int['objid'] == unique_ids[i] )
        matchids = matchids[0]
        
        temp_filters = catalog_str['filter'][matchids]
        
        temp_mag = catalog['s_mag'][matchids]
        temp_exptimes = catalog['exptime'][matchids]
        temp_n = catalog['s_n'][matchids]
        temp_Reff = catalog['s_Reff'][matchids]
        temp_nb = catalog['fb_n'][matchids]
        temp_BTn4 = catalog['4_BT'][matchids]
        temp_BTnf = catalog['f_BT'][matchids]
        temp_pixScale = catalog['pixScale'][matchids]
        
        temp_dict = {'filter': temp_filters,
                     'exptime': temp_exptimes,
                     's_mag': temp_mag,
                     's_n': temp_n,
                     's_Reff': temp_Reff,
                     'fb_n': temp_nb,
                     '4_BT': temp_BTn4,
                     'f_BT': temp_BTnf,
                     'pixScale': temp_pixScale
            }
        
        if 'F606W' in temp_filters:
        
            mag_filter = (temp_filters == 'F606W') & (temp_exptimes == numpy.max(temp_exptimes[temp_filters == 'F606W']))
            best_mag = temp_mag[mag_filter][0]
            
            #print((temp_filters == 'F390W'))
            Bfilt = (temp_filters == 'F390W') | (temp_filters == 'F435W') | (temp_filters == 'F438W') | (temp_filters == 'F450W') | (temp_filters == 'F475W')
            Rfilt = (temp_filters == 'F606W')
            Ifilt = (temp_filters == 'F814W') | (temp_filters == 'F850LP')
            
            """if len(temp_filters[Bfilt]) > 1:
                Bmult_count += 1
                temp_expt, temp_mag, temp_dn, temp_dReff, temp_dBTn4, temp_dBTnf, temp_dnb = get_best_and_2ndbest_diffs(temp_dict, arcsec_per_kpc, mask=Bfilt)
                delta_n.append(temp_dn)
                delta_Reff.append(temp_dReff)
                delta_BTn4.append(temp_dBTn4)
                delta_BTnf.append(temp_dBTnf)
                delta_nb.append(temp_dnb)
                expts.append(temp_expt)
                mags.append(best_mag)
                plot_label = 'Bmult'
                band_label = 'B*'"""
            """if len(temp_filters[Rfilt]) > 1:
                Rmult_count += 1
                temp_expt, temp_mag, temp_dn, temp_dReff, temp_dBTn4, temp_dBTnf, temp_dnb = get_best_and_2ndbest_diffs(temp_dict, arcsec_per_kpc, mask=Rfilt)
                delta_n.append(temp_dn)
                delta_Reff.append(temp_dReff)
                delta_BTn4.append(temp_dBTn4)
                delta_BTnf.append(temp_dBTnf)
                delta_nb.append(temp_dnb)
                expts.append(temp_expt)
                mags.append(best_mag)
                plot_label = 'Rmult'
                band_label = 'R*'"""
            if len(temp_filters[Ifilt]) > 1:
                Imult_count += 1
                temp_expt, temp_mag, temp_dn, temp_dReff, temp_dBTn4, temp_dBTnf, temp_dnb = get_best_and_2ndbest_diffs(temp_dict, arcsec_per_kpc, mask=Ifilt)
                delta_n.append(temp_dn)
                delta_Reff.append(temp_dReff)
                delta_BTn4.append(temp_dBTn4)
                delta_BTnf.append(temp_dBTnf)
                delta_nb.append(temp_dnb)
                expts.append(temp_expt)
                mags.append(best_mag)
                plot_label = 'Imult'
                band_label = 'I*'
                
            #if len(temp_filters[Bfilt]) > 1 and len(temp_filters[Rfilt]) > 1 and len(temp_filters[Ifilt]) > 1:
            #if len(temp_filters[Rfilt]) > 1 and len(temp_filters[Ifilt]) > 1:
            
        
        # get counts
        
        Bcond = ('F390W' in temp_filters) or ('F435W' in temp_filters) or ('F438W' in temp_filters) or ('F450W' in temp_filters) and ('F475W' in temp_filters)
        Rcond = ('F606W' in temp_filters)
        Icond = ('F814W' in temp_filters) or ('F850LP' in temp_filters)
        
        if Bcond:
            hasB = True
            B_count += 1
        if Rcond:
            hasR = True
            R_count += 1
        if Icond:
            hasI = True
            I_count += 1
            
        if hasB or hasR:
            BorR_count += 1
        if hasB or hasI:
            BorI_count += 1
        if hasR or hasI:
            RorI_count += 1
            
        if hasB and hasR:
            BandR_count += 1
        if hasB and hasI:
            BandI_count += 1
        if hasR and hasI:
            RandI_count += 1
        
        if hasB or hasR or hasI:
            BorRorI_count += 1
        if hasB and hasR and hasI:
            BandRandI_count += 1
            
        if hasB and hasR and hasI:
            bri_n.append( temp_dict['s_n'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0][0] ] )
            bri_n_sim.append( n )
        
        used_ids.append(unique_ids[i])
        
    else:
        continue
    
x1to1 = numpy.linspace(-50, 50, 100)
    
"""plt.scatter(numpy.log10(bri_n), numpy.log10(bri_n_sim), s=16, c='b')
plt.plot(x1to1, x1to1, 'k-', linewidth=1)
plt.xlabel('log $n$ (HST)', size=18)
plt.ylabel('log $n$ (Simard)', size=18)
ax = plt.gca()
major_xticks = [-0.5, 0.0, 0.5, 1.]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
ax.set_yticks(major_xticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(-0.5, 1)
plt.ylim(-0.5, 1)
ax.set_aspect('equal', 'box')
td = numpy.log10(bri_n_sim) - numpy.log10(bri_n)
sigma = 0.5*(np.percentile(td,84)-np.percentile(td,16))
plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(numpy.median(td),2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/kcorr_other/totaln_1to1_simard_vs_hst.pdf', bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()"""
    
delta_Reff = numpy.asarray( delta_Reff )
expts = numpy.log10( numpy.asarray(expts) )#* 10**(-0.4*numpy.asarray(mags)) )
                
plt.scatter(expts, delta_n, s=16, c='b')
plt.xlabel('Log Min. Flux', size=18)
plt.ylabel('$\Delta$ log $n$', size=18)
plt.axhline(y=0, c='k', linewidth=1)
plt.axhline(y=numpy.median(delta_n), c='r', linestyle='--', linewidth=1)
ax = plt.gca()
major_xticks = [-5, -4, -3, -2, -1]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
plt.xlim(-5.8, -0.4)
major_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.ylim(-0.6, 0.6)
sigma = 0.5*(np.percentile(delta_n,84)-np.percentile(delta_n,16))
plt.text(x=0.7, y=0.2, s='$\Delta$ = {}'.format(round(numpy.median(delta_n),2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.7, y=0.1, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.075, y=0.85, s=band_label, fontsize=18, transform=ax.transAxes)
plt.savefig('Plots/analysis/kcorr_other/{}_totaln.pdf'.format(plot_label), bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()


plt.scatter(expts, delta_Reff, s=16, c='b')
plt.xlabel('Log Min. Flux', size=18)
plt.ylabel('$\Delta$ log $R_{eff}$', size=18)
plt.axhline(y=0, c='k', linewidth=1)
plt.axhline(y=numpy.median(delta_Reff), c='r', linestyle='--', linewidth=1)
ax = plt.gca()
major_xticks = [-5, -4, -3, -2, -1]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
plt.xlim(-5.8, -0.4)
major_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.ylim(-0.6, 0.6)
sigma = 0.5*(np.percentile(delta_Reff,84)-np.percentile(delta_Reff,16))
plt.text(x=0.7, y=0.2, s='$\Delta$ = {}'.format(round(numpy.median(delta_Reff),2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.7, y=0.1, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.075, y=0.85, s=band_label, fontsize=18, transform=ax.transAxes)
plt.savefig('Plots/analysis/kcorr_other/{}_R50.pdf'.format(plot_label), bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()


plt.scatter(expts, delta_BTn4, s=16, c='b')
plt.xlabel('Log Min. Flux', size=18)
plt.ylabel('$\Delta$ B/T 4+1', size=18)
plt.axhline(y=0, c='k', linewidth=1)
plt.axhline(y=numpy.median(delta_BTn4), c='r', linestyle='--', linewidth=1)
ax = plt.gca()
major_xticks = [-5, -4, -3, -2, -1]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
plt.xlim(-5.8, -0.4)
major_yticks = [-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.ylim(-1, 1)
sigma = 0.5*(np.percentile(delta_BTn4,84)-np.percentile(delta_BTn4,16))
plt.text(x=0.7, y=0.2, s='$\Delta$ = {}'.format(round(numpy.median(delta_BTn4),2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.7, y=0.1, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.075, y=0.85, s=band_label, fontsize=18, transform=ax.transAxes)
plt.savefig('Plots/analysis/kcorr_other/{}_BTn4.pdf'.format(plot_label), bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()


plt.scatter(expts, delta_BTnf, s=16, c='b')
plt.xlabel('Log Min. Flux', size=18)
plt.ylabel('$\Delta$ B/T n+1', size=18)
plt.axhline(y=0, c='k', linewidth=1)
plt.axhline(y=numpy.median(delta_BTnf), c='r', linestyle='--', linewidth=1)
ax = plt.gca()
major_xticks = [-5, -4, -3, -2, -1]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
plt.xlim(-5.8, -0.4)
major_yticks = [-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.ylim(-1, 1)
sigma = 0.5*(np.percentile(delta_BTnf,84)-np.percentile(delta_BTnf,16))
plt.text(x=0.7, y=0.2, s='$\Delta$ = {}'.format(round(numpy.median(delta_BTnf),2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.7, y=0.1, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.075, y=0.85, s=band_label, fontsize=18, transform=ax.transAxes)
plt.savefig('Plots/analysis/kcorr_other/{}_BTnf.pdf'.format(plot_label), bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()


plt.scatter(expts, delta_nb, s=16, c='b')
plt.xlabel('Log Min. Flux', size=18)
plt.ylabel('$\Delta$ log $n_{bulge}$', size=18)
plt.axhline(y=0, c='k', linewidth=1)
plt.axhline(y=numpy.median(delta_nb), c='r', linestyle='--', linewidth=1)
ax = plt.gca()
major_xticks = [-5, -4, -3, -2, -1]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
plt.xlim(-5.8, -0.4)
major_yticks = [-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.ylim(-1, 1)
sigma = 0.5*(np.percentile(delta_nb,84)-np.percentile(delta_nb,16))
plt.text(x=0.7, y=0.2, s='$\Delta$ = {}'.format(round(numpy.median(delta_nb),2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.7, y=0.1, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=18, transform=ax.transAxes)
plt.text(x=0.075, y=0.85, s=band_label, fontsize=18, transform=ax.transAxes)
plt.savefig('Plots/analysis/kcorr_other/{}_nb.pdf'.format(plot_label), bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()


print('B, R, I')
print(B_count, R_count, I_count)
print('')

print('B or R, B or I, R or I')
print(BorR_count, BorI_count, RorI_count)
print('')

print('B and R, B and I, R and I')
print(BandR_count, BandI_count, RandI_count)
print('')

print('B or R or I, B and R and I')
print(BorRorI_count, BandRandI_count)

print('Multiple of either B, R, or I')
print(Bmult_count, Rmult_count, Imult_count)