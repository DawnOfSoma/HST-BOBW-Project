# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:09:23 2023

@author: sunlu
"""

import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from astropy.cosmology import WMAP9 as cosmo
import astropy.units as apu

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def weighted_mean(x, w):
    #value = numpy.nansum( x*w ) / numpy.nansum( w )
    index = numpy.where( w == numpy.nanmax(w) )[0][0]
    value = x[index]
    return value

def npaa(x):
    return numpy.asarray(x)

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

catalog = numpy.genfromtxt('0320_final_morphology_catalog_consolidated.txt', names=True)
catalog_str = numpy.genfromtxt('0320_final_morphology_catalog_consolidated.txt', names=True, dtype=None, encoding=None)
catalog_int = numpy.genfromtxt('0320_final_morphology_catalog_consolidated.txt', names=True, dtype=numpy.uint64)
ids = catalog_int['objid']
unique_ids = numpy.unique( ids )
print(len(unique_ids))

used_ids = []

ri_n = []
ri_n_sim = []

ri_nb = []

ri_BTn4 = []

ri_BTnf = []

ri_Reff = []
ri_Reff_ang = []
ri_Reff_sim = []

ri_disk_Reff = []
ri_disk_ellip = []
ri_disk_flux = []

expts = []
mags = []

ri_ssfr = []
ri_mstar = []

gswlc = numpy.genfromtxt('data/crossmatch_GSW_NYU_inclusive.dat', names=True)
gswlc_int = numpy.genfromtxt('data/crossmatch_GSW_NYU_inclusive.dat', names=True, dtype=numpy.uint64)
gswid = gswlc_int['objid']
gsw_z = gswlc['z']
gsw_mstar = gswlc['Mstar']
gsw_ssfr = gswlc['sSFR']
sim_n = gswlc['n']
sim_Reff = gswlc['totalRe']
 
for i in range(0,len(unique_ids)):
    
    has_ri_filters = False
    filters_test = catalog_str['filter'][catalog_int['objid'] == unique_ids[i]]
    numfilters = len(filters_test[(filters_test != 'F390W') & (filters_test != 'F435W') & (filters_test != 'F438W') & (filters_test != 'F450W') & (filters_test != 'F475W')])
    if numfilters > 0:
        has_ri_filters = True
    
    if unique_ids[i] not in used_ids and has_ri_filters:
        
        gsw_matchid = numpy.where( gswid == unique_ids[i]  )[0][0]
        redshift = gsw_z[gsw_matchid]
        n = sim_n[gsw_matchid]
        arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(redshift) * (apu.kpc / apu.arcsec)
        luminosity_distance = apu.Mpc.to(apu.pc, cosmo.luminosity_distance(redshift)) / apu.pc
        Reff = sim_Reff[gsw_matchid] / arcsec_per_kpc
        ssfr = gsw_ssfr[gsw_matchid]
        mstar = gsw_mstar[gsw_matchid]
        
        matchids = numpy.where( catalog_int['objid'] == unique_ids[i] )
        matchids = matchids[0]
        
        temp_filters = catalog_str['filter'][matchids]
        
        temp_mag = catalog['s_mag'][matchids]
        temp_exptimes = catalog['exptime'][matchids]
        temp_n = catalog['s_n'][matchids]
        temp_nb = catalog['fb_n'][matchids]
        temp_BTn4 = catalog['4_BT'][matchids]
        temp_diskEllip = catalog['4d_e'][matchids]
        temp_BTnf = catalog['f_BT'][matchids]
        temp_pixScale = catalog['pixScale'][matchids]
        temp_Reff = catalog['s_Reff'][matchids] * temp_pixScale / arcsec_per_kpc
        temp_diskReff = catalog['4d_Reff'][matchids] * temp_pixScale / arcsec_per_kpc
        temp_Reff_ang = catalog['s_Reff'][matchids] * temp_pixScale
        
        temp_4mag = catalog['4_mag'][matchids] - (5 * numpy.log10(luminosity_distance / 10))
        temp_4mag = 10**(-0.4 * temp_4mag)
        temp_4mag = numpy.log10( temp_4mag * (1. - temp_BTn4) )
        
        temp_dict = {'filter': temp_filters,
                     'exptime': temp_exptimes,
                     's_mag': temp_mag,
                     's_n': temp_n,
                     's_Reff': temp_Reff,
                     's_Reff_ang': temp_Reff_ang,
                     'fb_n': temp_nb,
                     '4_BT': temp_BTn4,
                     '4d_Reff': temp_diskReff,
                     '4d_e': temp_diskEllip,
                     '4d_flux': temp_4mag,
                     'f_BT': temp_BTnf,
                     'pixScale': temp_pixScale
            }
        
        all_mag = temp_dict['s_mag'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_n = temp_dict['s_n'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_nb = temp_dict['fb_n'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_BTn4 = temp_dict['4_BT'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_BTnf = temp_dict['f_BT'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_Reff = temp_dict['s_Reff'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_expt = temp_dict['exptime'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_Reff_ang = temp_dict['s_Reff_ang'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_diskReff = temp_dict['4d_Reff'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_diskEllip = temp_dict['4d_e'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        all_diskFlux = temp_dict['4d_flux'][ numpy.where( (temp_dict['filter'] != 'F390W') & (temp_dict['filter'] != 'F435W') & (temp_dict['filter'] != 'F438W') & (temp_dict['filter'] != 'F450W') & (temp_dict['filter'] != 'F475W') )[0] ]
        
        all_weight = all_expt * 10**(-0.4*all_mag)
        
        ri_n.append( weighted_mean(all_n, all_weight) )
        ri_n_sim.append( n )
        
        ri_nb.append( weighted_mean(all_nb, all_weight) )
        
        ri_BTn4.append( weighted_mean(all_BTn4, all_weight) )
        
        ri_BTnf.append( weighted_mean(all_BTnf, all_weight) )
        
        ri_Reff.append( weighted_mean(all_Reff, all_weight) )
        ri_Reff_ang.append( weighted_mean(all_Reff_ang, all_weight) )
        ri_Reff_sim.append( Reff )
        
        ri_disk_Reff.append( weighted_mean(all_diskReff, all_weight) )
        ri_disk_ellip.append( weighted_mean(all_diskEllip, all_weight) )
        ri_disk_flux.append( weighted_mean(all_diskFlux, all_weight) )
        
        ri_ssfr.append(ssfr)
        ri_mstar.append(mstar)
        
        used_ids.append(unique_ids[i])
        
    else:
        continue

ri_n = npaa(ri_n)
ri_n_sim = npaa(ri_n_sim)

ri_nb = npaa(ri_nb)

ri_BTn4 = npaa(ri_BTn4)

ri_BTnf = npaa(ri_BTnf)

ri_Reff = npaa(ri_Reff)
ri_Reff_ang = npaa(ri_Reff_ang)
ri_Reff_sim = npaa(ri_Reff_sim)

ri_disk_Reff = npaa(ri_disk_Reff)
ri_disk_ellip = npaa(ri_disk_ellip)
ri_disk_flux = npaa(ri_disk_flux)

ri_ssfr = npaa(ri_ssfr)
ri_mstar = npaa(ri_mstar)

ssfr_MS = (-0.46*ri_mstar) - 5.81
delta_ssfr = ri_ssfr - ssfr_MS
    
x1to1 = numpy.linspace(-50, 50, 100)
    
plt.figure(figsize=(4,4))
plt.scatter(numpy.log10(ri_n), numpy.log10(ri_n_sim), s=8, c='b')
plt.plot(x1to1, x1to1, 'k-', linewidth=1)
plt.xlabel('log $n$ (HST)', size=18)
plt.ylabel('log $n$ (NYU)', size=18)
medbins = numpy.arange(start=-0.3, stop=1, step=0.1)
medrange = (-0.3, 0.9)
counts, bin_edges, binnumber = binned_statistic(x=numpy.log10(ri_n_sim)[ri_n_sim > -99.], values=numpy.log10(ri_n)[ri_n_sim > -99.], statistic='count', bins=medbins, range=medrange)
medians, bin_edges, binnumber = binned_statistic(x=numpy.log10(ri_n_sim)[ri_n_sim > -99.], values=numpy.log10(ri_n)[ri_n_sim > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.plot(medians[counts>5], bin_centers[counts>5], c='#ff9900', linestyle='-', linewidth=4)
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
td = numpy.log10(ri_n_sim) - numpy.log10(ri_n)
td = td[td > -99.]
sigma = 0.5*(np.percentile(td,84)-np.percentile(td,16))
plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(numpy.nanmedian(td),2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/totaln_1to1_NYU_vs_hst.pdf', bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()

plt.figure(figsize=(4,4))
plt.scatter(numpy.log10(ri_Reff), numpy.log10(ri_Reff_sim), s=8, c='b')
plt.plot(x1to1, x1to1, 'k-', linewidth=1)
plt.xlabel('log $R_{eff}$ (HST) [kpc]', size=18)
plt.ylabel('log $R_{eff}$ (NYU) [kpc]', size=18)
medbins = numpy.arange(start=-0.6, stop=2, step=0.1)
medrange = (-0.6, 2)
counts, bin_edges, binnumber = binned_statistic(x=numpy.log10(ri_Reff_sim)[ri_Reff_sim > -99.], values=numpy.log10(ri_Reff)[ri_Reff_sim > -99.], statistic='count', bins=medbins, range=medrange)
medians, bin_edges, binnumber = binned_statistic(x=numpy.log10(ri_Reff_sim)[ri_Reff_sim > -99.], values=numpy.log10(ri_Reff)[ri_Reff_sim > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.plot(medians[counts>5], bin_centers[counts>5], c='#ff9900', linestyle='-', linewidth=4)
ax = plt.gca()
major_xticks = [-1, 0.0, 1.0, 2.0]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
ax.set_yticks(major_xticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(-1, 2.5)
plt.ylim(-1, 2.5)
ax.set_aspect('equal', 'box')
td = numpy.log10(ri_Reff_sim) - numpy.log10(ri_Reff)
td = td[td > -99.]
sigma = 0.5*(np.percentile(td,84)-np.percentile(td,16))
plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(numpy.nanmedian(td),2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/R50_1to1_NYU_vs_hst.pdf', bbox_inches='tight', format='pdf', dpi=500)
plt.show()
plt.close()

# delta vs various 

delta_n = numpy.log10(ri_n_sim) - numpy.log10(ri_n)

medbins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
medrange = (0, 1)
medians, bin_edges, binnumber = binned_statistic(x=ri_BTnf[delta_n > -99.], values=delta_n[delta_n > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.scatter(ri_BTnf, delta_n, s=8, c='b')
plt.ylabel('$\Delta$ log $n$ (NYU - HST)', size=18)
plt.xlabel('B/T n+1 (HST)', size=18)
plt.axhline(y=0, c='k', linestyle='-', linewidth=1)
plt.plot(bin_centers, medians, c='r', linestyle='-', linewidth=2)
ax = plt.gca()
major_xticks = [0.0, 0.25, 0.5, 0.75, 1]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
major_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(0, 1)
plt.ylim(-0.6, 0.6)
#plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
#plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(median,2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/delta_totaln_vs_BTnf.png', bbox_inches='tight')
plt.show()
plt.close()

medbins = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
medrange = (-1, 2.1)
medians, bin_edges, binnumber = binned_statistic(x=numpy.log10(ri_Reff_ang)[delta_n > -99.], values=delta_n[delta_n > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.scatter(numpy.log10(ri_Reff_ang), delta_n, s=8, c='b')
plt.ylabel('$\Delta$ log $n$ (NYU - HST)', size=18)
plt.xlabel('log $R_{eff}$ (HST) ["]', size=18)
plt.axhline(y=0, c='k', linestyle='-', linewidth=1)
plt.plot(bin_centers, medians, c='r', linestyle='-', linewidth=2)
ax = plt.gca()
major_xticks = [-1, 0, 1, 2]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
major_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(-1, 2.1)
plt.ylim(-0.6, 0.6)
#plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
#plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(median,2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/delta_totaln_vs_R50.png', bbox_inches='tight')
plt.show()
plt.close()


medbins = [8, 9, 10, 11, 12, 13]
medrange = (8, 13)
medians, bin_edges, binnumber = binned_statistic(x=ri_mstar[delta_n > -99.], values=delta_n[delta_n > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.scatter(ri_mstar, delta_n, s=8, c='b')
plt.ylabel('$\Delta$ log $n$ (NYU - HST)', size=18)
plt.xlabel('log $M_{*}$', size=18)
plt.axhline(y=0, c='k', linestyle='-', linewidth=1)
plt.plot(bin_centers, medians, c='r', linestyle='-', linewidth=2)
ax = plt.gca()
major_xticks = [8, 10, 12]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
major_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(8, 13)
plt.ylim(-0.6, 0.6)
#plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
#plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(median,2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/delta_totaln_vs_mstar.png', bbox_inches='tight')
plt.show()
plt.close()

medbins = [-14, -13, -12, -11.5, -11, -10.5, -10, -9.5, -9, -8]
medrange = (-14, -7.8)
medians, bin_edges, binnumber = binned_statistic(x=ri_ssfr[delta_n > -99.], values=delta_n[delta_n > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.scatter(ri_ssfr, delta_n, s=8, c='b')
plt.ylabel('$\Delta$ log $n$ (NYU - HST)', size=18)
plt.xlabel('log sSFR', size=18)
plt.axhline(y=0, c='k', linestyle='-', linewidth=1)
plt.plot(bin_centers, medians, c='r', linestyle='-', linewidth=2)
ax = plt.gca()
major_xticks = [-14, -12, -10, -8]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
major_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(-14, -7.8)
plt.ylim(-0.6, 0.6)
#plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
#plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(median,2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/delta_totaln_vs_ssfr.png', bbox_inches='tight')
plt.show()
plt.close()

# morphologies versus delta log ssfr
# not size, not sure why samir suggested against
    
medbins = [-4, -3, -2, -1, 0, 1, 2]
medrange = (-4, 2)
medians, bin_edges, binnumber = binned_statistic(x=delta_ssfr[delta_n > -99.], values=delta_n[delta_n > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.scatter(delta_ssfr, delta_n, s=8, c='b')
plt.ylabel('$\Delta$ log $n$ (NYU - HST)', size=18)
plt.xlabel('$\Delta$ log sSFR', size=18)
plt.axhline(y=0, c='k', linestyle='-', linewidth=1)
plt.plot(bin_centers, medians, c='r', linestyle='-', linewidth=2)
ax = plt.gca()
major_xticks = [-4, -3, -2, -1, 0, 1, 2]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
major_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(-4, 2)
plt.ylim(-0.6, 0.6)
#plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
#plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(median,2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/delta_ssfr_totaln.png', bbox_inches='tight')
plt.show()
plt.close()

# vs delta log ssfr

medbins = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2]
medrange = (-4, 2)
medians, bin_edges, binnumber = binned_statistic(x=delta_ssfr[numpy.log10(ri_n_sim) > -99.], values=numpy.log10(ri_n_sim)[numpy.log10(ri_n_sim) > -99.], statistic='median', bins=medbins, range=medrange)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.scatter(delta_ssfr, numpy.log10(ri_n_sim), s=8, c='b')
plt.ylabel('log $n$ (NYU)', size=18)
plt.xlabel('$\Delta$ log sSFR', size=18)
#plt.axhline(y=0, c='k', linestyle='-', linewidth=1)
plt.plot(bin_centers, medians, c='r', linestyle='-', linewidth=2)
ax = plt.gca()
major_xticks = [-4, -3, -2, -1, 0, 1, 2]
ax.set_xticks(major_xticks)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_tick_params(direction='in', size=10, top=True, bottom=True, which='major', labelsize=16)
ax.xaxis.set_tick_params(direction='in', size=4, top=True, bottom=True, which='minor', labelsize=16)
major_yticks = [-0.5, 0.0, 0.5, 1]
ax.set_yticks(major_yticks)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_tick_params(direction='in', size=10, left=True, right=True, which='major', labelsize=16)
ax.yaxis.set_tick_params(direction='in', size=4, left=True, right=True, which='minor', labelsize=16)
plt.xlim(-4, 2)
plt.ylim(-0.5, 1.0)
#plt.text(x=0.075, y=0.875, s='$\sigma$ = {}'.format(round(sigma,2)), fontsize=16, transform=ax.transAxes)
#plt.text(x=0.075, y=0.775, s='$\Delta$ = {}'.format(round(median,2)), fontsize=16, transform=ax.transAxes)
plt.savefig('Plots/analysis/RI_sample_NYU/totaln_vs_ssfr_NYU.png', bbox_inches='tight')
plt.show()
plt.close()