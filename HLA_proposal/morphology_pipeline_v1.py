# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:07:28 2022

@author: sunlu
"""

import astropy, pylab, time, sys, os, requests, json
import numpy
from photutils.segmentation import detect_sources
import statmorph
import matplotlib.pyplot as plt
from astropy.io import fits
import statmorph
from astropy.stats import sigma_clipped_stats
import astropy.visualization as apv
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Sersic2D
from petrofit.modeling import print_model_params
from petrofit.modeling import fit_model
from petrofit.modeling import plot_fit
from petrofit.modeling import PSFConvolvedModel2D
from scipy import ndimage

def rescale_image(image):
    stretch = apv.AsinhStretch(0.01)
    image = image / numpy.nanmax(image)
    image[image > 0.95] = 0.95
    image = image / numpy.nanmax(image)
    image = stretch(image)
    return image

def get_weights(image):
    wmap = numpy.sqrt(image)
    return wmap

def process(imname, outfile):
    
    image = fits.open(imname)
    image = image[0].data
    
    psf_filepath = None
    imname_split = imname.split('_')
    # this encompasses all GSW crossmatch combos, may need more PSFs for entire HST-SDSS subsample
    if 'acs' in imname:
        # all are wfc
        pfilter = imname_split[-1]
        pfilter = pfilter.split('.')[0].upper()
        det = imname_split[-2]
        psf_filepath = 'HST_PSF/ACS/' + pfilter + '00_psf.fits'
    if 'wfc3' in imname:
        # all are uvis
        pfilter = imname_split[-1]
        pfilter = pfilter.split('.')[0].upper()
        det = imname_split[-2]
        psf_filepath = 'HST_PSF/WFC3/WFC3_UVIS_' + pfilter + '_100_psf.fits'
    if 'wfpc2' in imname:
        # got some wf and some pc
        pfilter = imname_split[-2]
        pfilter = pfilter.split('.')[0].upper()
        det = imname_split[-1]
        if 'wf' in det:
            psf_filepath = 'HST_PSF/WFPC2/WF/' + pfilter + '00.fits'
        if 'pc' in det:
            psf_filepath = 'HST_PSF/WFPC2/PC/' + pfilter + '00.fits'
    
    if psf_filepath is None:
        print('Error 1: No appropriate PSF')
        return 1
    
    psf = fits.open(psf_filepath)
    psf = psf[0].data
    psf /= numpy.sum(psf)
    psf /= numpy.sum(psf)

    plt.figure()
    plt.imshow(rescale_image(image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    #numpy.full(shape=(tdim,tdim), fill_value=1.0)
    wmap = numpy.full(shape=(image.shape[0],image.shape[0]), fill_value=1.0) #get_weights(image) 
    
    #print(numpy.max(image))
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=10)
    image_bgsub = image - mean

    threshold = 1.0 * mean # 1 sigma
    segmap = detect_sources(image_bgsub, threshold, npixels=100)
    segmap = numpy.asarray(segmap)
    
    # use scipy to make distance map
    # distance map is needed to exclude sky pixels too close to objects
    tdim = image.shape[0]
    inverted_segmap = numpy.full(shape=(tdim,tdim), fill_value=1.)
    inverted_segmap[segmap > 0] = 0.
    sky_distance_map = ndimage.distance_transform_edt(inverted_segmap)
    
    infotable = numpy.loadtxt(imname[:-5] + '_info.dat', usecols=(0,1))
    pixScale = infotable[0]
    print('Image pixel scale = ', pixScale)
    sky_cutoff_distance = 4.0 / pixScale # 4 arcsec, Simard 2011 used 4 arcseconds, pixel scale in "/pix
    print('Sky cutoff distance = ', sky_cutoff_distance)
    
    sky_distance_map[sky_distance_map <= sky_cutoff_distance] = 0
    sky_distance_map[sky_distance_map > sky_cutoff_distance] = 1
    #sky_distance_map[segmap > 0] = 2
    
    plt.figure()
    plt.imshow(sky_distance_map,cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    mean, median, std = sigma_clipped_stats(image[sky_distance_map == 1], sigma=3.0, maxiters=10)
    print('Updated background = ', mean)
    print('Number of sky pixels = ', len(image[sky_distance_map == 1]))
    image = image - mean
    
    hdu = fits.PrimaryHDU(segmap)
    hdu.writeto(imname[:-5] + '_segmap.fits', overwrite=True)
    
    source_morphs = statmorph.source_morphology(image, segmap, weightmap=wmap, psf=psf)
    
    target_index = segmap[int(tdim/2.)][int(tdim/2.)]
    morph = source_morphs[target_index-1]
    print('Statmorph sersic_n =', morph.sersic_n)
    print('SM amplitude = ', morph.sersic_amplitude)
    print('SM Reff = ', morph.sersic_rhalf)
    print('SM ellip = ', morph.sersic_ellip)
    print('SM theta = ', morph.sersic_theta)
    
    model_image = numpy.zeros(shape=(tdim, tdim))
    model = Sersic2D(amplitude=morph.sersic_amplitude, r_eff=morph.sersic_rhalf, n=morph.sersic_n, x_0=morph.sersic_xc, y_0=morph.sersic_yc, ellip=morph.sersic_ellip, theta=morph.sersic_theta)
    synth_xvals = numpy.zeros(shape=(tdim, tdim))
    synth_yvals = numpy.zeros(shape=(tdim, tdim))
    for i in range(0, tdim):
        for j in range(0, tdim):
            synth_xvals[i][j] = j
            synth_yvals[i][j] = i
    model_image += model(synth_xvals, synth_yvals)
    
    residual = image - model_image
    
    plt.figure()
    plt.imshow(rescale_image(image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(segmap,cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(rescale_image(model_image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # now we test petrofit
    
    print('Segmap indices =', numpy.nanmin(segmap), numpy.nanmax(segmap))
    
    maxiters = 10000
    accuracy = 1e-09
    
    masked_image = numpy.copy(image)
    masked_wmap = 1. / wmap
    
    masked_image[(segmap != target_index) & (segmap != 0)] = numpy.nan # mask other objects, dont mask sky
    masked_image[(sky_distance_map == 1)] = numpy.nan # mask sky used for background
    
    masked_wmap[(segmap != target_index) & (segmap != 0)] = numpy.nan
    masked_wmap[(sky_distance_map == 1)] = numpy.nan
    
    plt.figure()
    plt.imshow(rescale_image(masked_image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    hdu = fits.PrimaryHDU(masked_image)
    hdu.writeto(imname[:-5] + '_masked.fits', overwrite=True)

    center_slack = 10
    petrofit_model = Sersic2D(
        amplitude=morph.sersic_amplitude,
        r_eff=morph.sersic_rhalf,
        n=morph.sersic_n,
        x_0=morph.sersic_xc,
        y_0=morph.sersic_yc,
        ellip=morph.sersic_ellip,
        theta=morph.sersic_theta,
        bounds = {
            'amplitude': (0, None),
            'r_eff': (0, None),
            'n': (0.5, 8.0),
            'ellip': (0, 1),
            'theta': (-2*numpy.pi, 2*numpy.pi),
            'x_0': (morph.sersic_xc - center_slack/2, morph.sersic_xc + center_slack/2),
            'y_0': (morph.sersic_yc - center_slack/2, morph.sersic_yc + center_slack/2),
        }
    )
    petrofit_model = PSFConvolvedModel2D(petrofit_model, psf=psf, oversample=None)
    
    fitted_model, fit_info = fit_model(
    image=masked_image, model=petrofit_model,
    weights=masked_wmap,
    maxiter=maxiters,
    acc=accuracy
    )
    
    print('petrofit model results')
    print_model_params(fitted_model)
    
    plt.figure()
    plt.imshow(rescale_image(image - fitted_model(synth_xvals, synth_yvals)),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # Now try a compound model
    
    disk_model = Sersic2D(
        amplitude=morph.sersic_amplitude/2.,
        r_eff=morph.sersic_rhalf,
        n=1.0,
        x_0=morph.sersic_xc,
        y_0=morph.sersic_yc,
        ellip=morph.sersic_ellip,
        theta=morph.sersic_theta,
        bounds = {
            'amplitude': (0, None),
            'r_eff': (0, None),
            'n': (1.0, 1.0),
            'ellip': (0, 1),
            'theta': (-2*numpy.pi, 2*numpy.pi),
            'x_0': (morph.sersic_xc - center_slack/2, morph.sersic_xc + center_slack/2),
            'y_0': (morph.sersic_yc - center_slack/2, morph.sersic_yc + center_slack/2),
        }
    )
    
    bulge_model = Sersic2D(
        amplitude=morph.sersic_amplitude/2.,
        r_eff=morph.sersic_rhalf/2.,
        n=4.0,
        x_0=morph.sersic_xc,
        y_0=morph.sersic_yc,
        ellip=morph.sersic_ellip,
        theta=morph.sersic_theta,
        bounds = {
            'amplitude': (0, None),
            'r_eff': (0, None),
            'n': (0.5, 8.0),
            'ellip': (0, 1),
            'theta': (-2*numpy.pi, 2*numpy.pi),
            'x_0': (morph.sersic_xc - center_slack/2, morph.sersic_xc + center_slack/2),
            'y_0': (morph.sersic_yc - center_slack/2, morph.sersic_yc + center_slack/2),
        }
    )
    compound_model = numpy.array([disk_model, bulge_model]).sum()
    compound_model = PSFConvolvedModel2D(compound_model, psf=psf, oversample=None)
    
    fitted_model, fit_info = fit_model(
    image=masked_image, model=compound_model,
    weights=masked_wmap,
    maxiter=maxiters, #10000
    acc=accuracy
    )
    
    print('Compound bulge+disk model results')
    print_model_params(fitted_model)
    print('Compound bulge n = ', fitted_model.n_1)
    
    plt.figure()
    plt.imshow(rescale_image(image - fitted_model(synth_xvals, synth_yvals)),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # Now try a compound model with bulge component fixed at n = 4
    
    disk_model = Sersic2D(
        amplitude=morph.sersic_amplitude/2.,
        r_eff=morph.sersic_rhalf,
        n=1.0,
        x_0=morph.sersic_xc,
        y_0=morph.sersic_yc,
        ellip=morph.sersic_ellip,
        theta=morph.sersic_theta,
        bounds = {
            'amplitude': (0, None),
            'r_eff': (0, None),
            'n': (1.0, 1.0),
            'ellip': (0, 1),
            'theta': (-2*numpy.pi, 2*numpy.pi),
            'x_0': (morph.sersic_xc - center_slack/2, morph.sersic_xc + center_slack/2),
            'y_0': (morph.sersic_yc - center_slack/2, morph.sersic_yc + center_slack/2),
        }
    )
    
    bulge_model = Sersic2D(
        amplitude=morph.sersic_amplitude/2.,
        r_eff=morph.sersic_rhalf/2.,
        n=4.0,
        x_0=morph.sersic_xc,
        y_0=morph.sersic_yc,
        ellip=morph.sersic_ellip,
        theta=morph.sersic_theta,
        bounds = {
            'amplitude': (0, None),
            'r_eff': (0, None),
            'n': (4.0, 4.0),
            'ellip': (0, 1),
            'theta': (-2*numpy.pi, 2*numpy.pi),
            'x_0': (morph.sersic_xc - center_slack/2, morph.sersic_xc + center_slack/2),
            'y_0': (morph.sersic_yc - center_slack/2, morph.sersic_yc + center_slack/2),
        }
    )
    compound_model = numpy.array([disk_model, bulge_model]).sum()
    compound_model = PSFConvolvedModel2D(compound_model, psf=psf, oversample=None)
    
    fitted_model, fit_info = fit_model(
    image=masked_image, model=compound_model,
    weights=masked_wmap,
    maxiter=maxiters,
    acc=accuracy
    )
    
    print('Compound n1n4 bulge+disk model results')
    print_model_params(fitted_model)
    
    plt.figure()
    plt.imshow(rescale_image(image - fitted_model(synth_xvals, synth_yvals)),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    
def main():
    
    data = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True)
    data_int = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=numpy.uint64)
    data_str = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=None, encoding=None)
    
    objid = data_int['objid']
    exptimes = data['exptime']
    imageNames = data_str['imname']
    instrumentNames = data_str['instrument']
    detectorNames = data_str['detector']
    filterNames = data_str['filter']
    aperNames = data_str['aperture']
    
    objid_unique = numpy.unique(objid)
    
    outfile = open('CO_HST_morphology_catalog.dat', 'w')
    
    for i in range(2,3):#len(objid_unique)):
        
        #if str(objid_unique[i]) == '1237648721223352487':
       
        all_indices = numpy.where(objid == objid_unique[i])
        all_imageNames = imageNames[all_indices[0]]
        all_exptimes = exptimes[all_indices[0]]
        all_filterNames = filterNames[all_indices[0]]
        
        images_unique = numpy.unique(all_imageNames)
    
        # process('HST_image_repo/' + str(objid_unique[i]) + '/SDSSr.fits')
        for name in images_unique:
            
            temp_index = numpy.where( all_imageNames == name )[0][0]
            exptime = all_exptimes[temp_index]
            pfilter = all_filterNames[temp_index]
            
            outfile.write('{} {} {} '.format(objid_unique[i], pfilter, exptime))
            
            impath = 'HST_image_repo/' + str(objid_unique[i]) + '/' + name + '.fits'
            if os.path.exists(impath):
                process(impath)
                
            outfile.write('\n')
                
    outfile.close()
         
    return 0

main()