# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:13:22 2022

@author: sunlu
"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

plt.rcParams.update({'font.size': 26})


def main():
    
    print('Sample breakdown')
    
    main_subsample = numpy.genfromtxt('data/subsample.dat', names=True)
    main_subsample_str = numpy.genfromtxt('data/subsample.dat', names=True, dtype=None, encoding=None)
    main_subsample_int = numpy.genfromtxt('data/subsample.dat', names=True, dtype=numpy.uint64)
    
    objid = main_subsample_int['objID']
    matchid = main_subsample_int['matchid']
    ptype = main_subsample_int['type']
    
    imname = main_subsample_str['imname']
    instrument = main_subsample_str['instrument']
    detector = main_subsample_str['detector']
    sclass = main_subsample_str['class']
    subclass = main_subsample_str['subclass']
    survey = main_subsample_str['survey']
    program = main_subsample_str['program']
    pfilter = main_subsample_str['filter']
    
    z = main_subsample['z']
    ra = main_subsample['ra']
    dec = main_subsample['dec']
    rmag = main_subsample['rpetro']
    exptime = main_subsample['exptime']
    
    survey_list = []
    class_list = []
    subclass_list = []
    program_list = []
    
    QSO_redshifts = []
    
    galaxy_mags = []
    galaxy_z = []
    gal_xt = []
    gsw_z = []
    gsw_mags = []
    gsw_xt = []
    
    gal_filt = []
    gsw_filt = []
    
    objid_used = []
    
    greenpea_mags = []
    greenpea_z = []
    gp_xt = []
    
    # First remove dupes and extract useful info
    
    for i in range(0, len(objid)):
        
        if objid[i] not in objid_used:
            
            if sclass[i] == 'QSO':
                QSO_redshifts.append(z[i])
            else:
                galaxy_mags.append(rmag[i])
                galaxy_z.append(z[i])
                
                # Select best exp time
                exptime_temp = exptime[objid == objid[i]]
                pfilter_temp = pfilter[objid == objid[i]]
                best_xt_index = 0
                best_xt = 0
                best_filt = 'None'
                for k in range(0,len(exptime_temp)):
                    if exptime_temp[k] > best_xt:
                        best_xt_index = k
                        best_xt = exptime_temp[k]
                        best_filt = pfilter_temp[k]
                gal_xt.append(best_xt)
                gal_filt.append(best_filt)
                
                if rmag[i] < 18 and z[i] > 0.01 and z[i] < 0.3:
                    gsw_mags.append(rmag[i])
                    gsw_z.append(z[i])
                    gsw_xt.append(best_xt)
                    gsw_filt.append(best_filt)
                
                if ptype[i] == 6:
                    greenpea_mags.append(rmag[i])
                    greenpea_z.append(z[i])
                    gp_xt.append(best_xt)
        
            if sclass[i] not in class_list:
                class_list.append(sclass[i])
            if subclass[i] not in subclass_list:
                subclass_list.append(subclass[i])
            if survey[i] not in survey_list:
                survey_list.append(survey[i])
            if program[i] not in program_list:
                program_list.append(program[i])
                
            objid_used.append(objid[i])
    
    galaxy_mags = numpy.asarray(galaxy_mags)
    galaxy_z = numpy.asarray(galaxy_z)
    gal_xt = numpy.asarray(gal_xt)
    gal_filt = numpy.asarray(gal_filt)
    gsw_filt = numpy.asarray(gsw_filt)
    
    print('Spec classes:')
    print(class_list)
    print('Spec sub classes:')
    print(subclass_list)
    print('Surveys:')
    print(survey_list)
    print('Programs:')
    print(program_list)
    
    plt.figure(figsize=(12,9))
    plt.hist(QSO_redshifts)
    plt.title('QSO redshifts, N = {}'.format(len(QSO_redshifts)))
    plt.xlabel('z')
    plt.show()
    plt.close()
    
    minx = numpy.min(galaxy_z)
    maxx = numpy.max(galaxy_z)
    miny = numpy.min(galaxy_mags)
    maxy = numpy.max(galaxy_mags)
    
    xt_mean = round( 10**numpy.mean( numpy.log10(gal_xt) ) )
    plt.figure(figsize=(12,9))
    plt.hist(numpy.log10(gal_xt))
    plt.title('all galaxies, mean = {}'.format(xt_mean))
    plt.xlabel('log exp. time')
    plt.show()
    plt.close()
    
    xt_mean = round( 10**numpy.mean( numpy.log10(gsw_xt) ) )
    plt.figure(figsize=(12,9))
    plt.hist(numpy.log10(gsw_xt), color='#00bfff', bins=15, edgecolor='black', linewidth=1.2)
    #plt.title('GSW-range galaxies, mean = {}'.format(xt_mean))
    plt.xlabel('Log Exposure Time [s]')
    plt.ylabel('Number')
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(direction='in', size=10, top=True)
    ax.yaxis.set_tick_params(direction='in', size=10, right=True)
    ax.xaxis.set_tick_params(direction='in', size=5, top=True, bottom=True, which='minor')
    ax.yaxis.set_tick_params(direction='in', size=5, right=True, left=True, which='minor')
    plt.axvline(x=numpy.log10(xt_mean), c='r', linewidth=4, label='Mean = {}s'.format(round(xt_mean)))
    plt.legend()
    plt.savefig('Plots/GSW_exp_times.png')
    plt.show()
    plt.close()
    
    xt_mean = round( 10**numpy.mean( numpy.log10(gp_xt) ) )
    plt.figure(figsize=(12,9))
    plt.hist(numpy.log10(gp_xt))
    plt.title('GPs, mean = {}'.format(xt_mean))
    plt.xlabel('log exp. time')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(galaxy_z, galaxy_mags, c='b', s=8, label='All')
    plt.scatter(gsw_z, gsw_mags, c='r', s=8, label='GSW')
    plt.title('All = {}, GSW = {}'.format(len(galaxy_z), len(gsw_z)))
    plt.xlabel('z')
    plt.ylabel('petro r')
    plt.xlim( minx, maxx )
    plt.ylim( miny, maxy )
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.scatter(greenpea_z, greenpea_mags, c='g', s=8, label='All')
    plt.title('GP = {}'.format(len(greenpea_z)))
    plt.xlabel('z')
    plt.ylabel('petro r')
    plt.xlim( minx, maxx )
    plt.ylim( miny, maxy )
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,9))
    plt.hist(gal_filt)
    plt.title('Best filters, all')
    plt.show()
    plt.close()
    
    bins = [100, 100]
    maxx = 1.0
    maxy = 25.0
    galaxy_mags = galaxy_mags[galaxy_z > minx]
    galaxy_z = galaxy_z[galaxy_z > minx]
    galaxy_mags = galaxy_mags[galaxy_z < maxx]
    galaxy_z = galaxy_z[galaxy_z < maxx]
    
    galaxy_z = galaxy_z[galaxy_mags > miny]
    galaxy_mags = galaxy_mags[galaxy_mags > miny]
    galaxy_z = galaxy_z[galaxy_mags < maxy]
    galaxy_mags = galaxy_mags[galaxy_mags < maxy]
    
    gsw_mags = galaxy_mags[galaxy_z >= 0.01]
    gsw_z = galaxy_z[galaxy_z >= 0.01]
    gsw_mags = gsw_mags[gsw_z < 0.3]
    gsw_z = gsw_z[gsw_z < 0.3]
    gsw_z = gsw_z[gsw_mags < 18]
    gsw_mags = gsw_mags[gsw_mags < 18]
    
    brange = [[minx, maxx], [miny, maxy]]
    plt.figure(figsize=(12,9))
    plt.hist2d(galaxy_z, galaxy_mags, bins=bins, range=brange, cmap='gray_r')
    plt.xlabel('z')
    plt.ylabel('SDSS r-band Magnitude')
    plt.xlim( minx, maxx )
    plt.ylim( miny, maxy )
    #plt.title('Total = {}, GSW = {}'.format(len(galaxy_z), len(gsw_z)))
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(direction='in', size=16, top=True)
    ax.yaxis.set_tick_params(direction='in', size=10, right=True)
    ax.xaxis.set_tick_params(direction='in', size=8, top=True, bottom=True, which='minor')
    ax.yaxis.set_tick_params(direction='in', size=5, right=True, left=True, which='minor')
    plt.axvline(x=0.3, ymin=0, ymax=(18-miny)/(maxy-miny), color='r', linewidth=1)
    plt.axvline(x=0.01, ymin=0, ymax=(18-miny)/(maxy-miny), color='r', linewidth=1)
    plt.axhline(y=18, xmin=(0.01-minx)/(maxx-minx), xmax=(0.3-minx)/(maxx-minx), color='r', linewidth=1)
    plt.savefig('Plots/GSW_rmag_v_z.png')
    plt.show()
    plt.close()
    
    griz_filters = ['F390W', 'F435W', 'F438W', 'F450W', 'F475W', 'F555W', 'F569W', 'F606W', 'F622W', 'F625W', 'F675W', 
                    'F702W', 'F775W', 'F791W', 'F814W', 'F850LP']
    griz_colors = ['#00ff00', '#00ff00', '#00ff00', '#00ff00', '#00ff00', 'r', 'r', 'r', 'r', 'r', 'r', 'm', 'm', 'm', 'm', 'k']
    filter_counts = []
    
    for f in griz_filters:
        filter_counts.append( len(gsw_filt[gsw_filt == f]) )
    
    plt.figure(figsize=(11,9))
    plt.grid(axis='x')
    plt.barh(griz_filters, filter_counts, color=griz_colors)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_tick_params(direction='in', size=10, top=True)
    ax.xaxis.set_tick_params(direction='in', size=8, top=True, bottom=True, which='minor')
    ax.yaxis.set_ticks_position('none')
    plt.savefig('Plots/GSW_filter_dist.png', bbox_inches = "tight")
    plt.show()
    plt.close()
    
main()