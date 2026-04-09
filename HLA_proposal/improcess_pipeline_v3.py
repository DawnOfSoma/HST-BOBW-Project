# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:25:28 2023

@author: sunlu
"""

import os
import numpy
from photutils.segmentation import detect_sources
import statmorph
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astropy.visualization as apv
#from astropy.convolution import convolve
#from photutils.segmentation import make_2dgaussian_kernel
#from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Sersic2D
#from petrofit.modeling import print_model_params
from petrofit.modeling import fit_model
#from petrofit.modeling import plot_fit
from petrofit.modeling import PSFConvolvedModel2D
from scipy import ndimage
from photutils.segmentation import deblend_sources
from io import BytesIO
import requests
import astropy.units as apu
from astropy.cosmology import WMAP9 as cosmo

def get_hla_cutout(imagename,ra,dec,size=50,autoscale=100,asinh=False,zoom=1):
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="fits", red=imagename, autoscale=autoscale, asinh=asinh, zoom=zoom))
    im = fits.open(BytesIO(r.content))
    return im

def rescale_image(image):
    stretch = apv.LogStretch(10000) #apv.AsinhStretch(0.01)
    image = image / numpy.nanmax(image)
    image[image > 0.95] = 0.95
    image = image / numpy.nanmax(image)
    image = stretch(image)
    return image

def get_weights(image, sky):
    # image should be background subtracted and in electrons
    # see #sigmamap on galfit facebook page, or galfit FAQs
    image_ph = numpy.copy(image) # avoid modifying original image
    
    # avoid negative sqrt; can't define object var for negative values anyway
    # negative pixels should be attributable to sky var
    image_ph[image_ph <= 0] = 0 
    objvar = numpy.sqrt(image_ph)
    skyvar = numpy.sqrt(sky)
    
    wmap = numpy.sqrt( (objvar**2.) + (skyvar**2.) )
    return wmap

def getskymap(image, segmap, pixScale, arcsec):
    
    tdim = image.shape[0]
    
    inverted_segmap = numpy.full(shape=(tdim,tdim), fill_value=1.)
    inverted_segmap[segmap > 0] = 0.
    sky_distance_map = ndimage.distance_transform_edt(inverted_segmap)
    
    sky_cutoff_distance = arcsec / pixScale # Simard 2011 used 4 arcseconds, pixel scale in "/pix
    
    sky_distance_map[sky_distance_map <= sky_cutoff_distance] = 0
    sky_distance_map[sky_distance_map > sky_cutoff_distance] = 1
    sky_distance_map[numpy.isnan(image) == True] = 0
    number_of_sky_pixels = len(image[sky_distance_map == 1])
    
    return sky_distance_map, number_of_sky_pixels
    

def process(imname, outfile, exptime, ra, dec, imfilename, z):
    
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z) * (apu.kpc / apu.arcsec)
    arcsec_limdist = 6.0 * arcsec_per_kpc
    
    image = fits.open(imname)
    header = image[0].header
    image = image[0].data
    tdim = image.shape[0]
    
    ncombine = 1
    ncombine_flag = 1
    try:
        ncombine = header['NCOMBINE']
    except KeyError:
        ncombine = 1 # Should be true, since ncombine keyword is absent for WFPC2 images
    # Flag if we don't have multiple images; may mean no CR rejection (but uncertain; can assess later)
    if ncombine > 1:
        ncombine_flag = 0
        
    # Image is in either counts or electrons
    # either way, scale by ncombine to get accurate weight map later
    if ncombine <= 0:
        ncombine = 1
        ncombine_flag = 2 # probably won't happen, but just in case
    image = ncombine * image
        
    units = 'N/A'
    unit_flag = 0
    try:
        units = header['BUNIT']
    except KeyError:
        units = 'COUNTS/S'
        unit_flag = 1
    # So now we have correct units
    # If not defined in header, we assume counts/s as this seems appropriate for wfpc2
    # Only wfpc2 should have undefined units
    # Anyway, we flag if units is undefined
    
    gain = 1 # default to 1 since most images are in e-/s anyway
    gain_flag = 0
    if units == 'COUNTS/S':
        try:
            photmode = header['PHOTMODE']
            if 'A2D7' in photmode:
                gain = 7
            if 'A2D15' in photmode:
                gain = 14
        except KeyError:
            gain_flag = 1
            if 'wfpc2' in imname:
                gain = 7 # good guess, true for most objects
    # gain flag = 1 means the gain is not properly defined
    # May only be true if unit flag = 1 anyway, but we will see
    # Ncombine, unit and gain flags should tell us when the weight map may be poorly defined
    
    # Convert image into electrons
    # I already multiply by exposure time when downloading the images
    if units == 'COUNTS/S':
        image = gain * image
    
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=100)
    image_bgsub = image - mean

    threshold = 1.0 * std # sigma
    segmap = detect_sources(image_bgsub, threshold, npixels=100)
    segmap = deblend_sources(image_bgsub, segmap,
                                   npixels=100, nlevels=32, contrast=0.1,
                                   progress_bar=False)
    segmap = numpy.asarray(segmap)
    
    """plt.figure()
    plt.imshow(segmap,cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()"""
    
    # Update background
    # This should be written to output maybe
    sky_flag = 0
    mean_init = mean
    mean, median, std = sigma_clipped_stats(image[segmap == 0], sigma=1.0, maxiters=100)
    #mean_dcut, median_dcut, std_dcut = sigma_clipped_stats(image[sky_distance_map == 1], sigma=1.0, maxiters=100)
    
    """image_rounded = image.round(decimals=0)
    image_rounded.flatten()
    image_rounded = image_rounded[numpy.isnan(image_rounded) == False]
    unique_pixels, unique_pixel_counts = numpy.unique(image_rounded, return_counts=True)
    mode_indices = numpy.where( unique_pixel_counts == numpy.max(unique_pixel_counts) )
    mode_indices = mode_indices[0]
    mode_index = mode_indices[0]
    print(mode_indices, mode_index)
    mode_single = unique_pixels[mode_index]
    modes = unique_pixels[mode_indices]
    mode = numpy.nanmean( modes )
    if numpy.isnan(mode):
        raise Exception('Mode is nan')"""
    
    number_of_sky_pixels = len(segmap[segmap == 0])
    if number_of_sky_pixels < 20000 or numpy.isnan(mean):
        sky_flag = 1
    
    #print('Sky pixels: ', number_of_sky_pixels_dcut, number_of_sky_pixels)
    #print(mean_init, mean_dcut, mean, mode)
    
    """plt.figure()
    plt.hist(image.flatten(), log=False, bins=100, range=(10**(numpy.log10(mean_init)-0.5), 10**(numpy.log10(mean_init)+0.5)))
    plt.axvline(x=mean_dcut, color='k')
    for m in modes:
        plt.axvline(x=m, color='r')
    plt.axvline(x=mean, color='b')
    plt.show()
    plt.close()
    
    plt.figure()
    plt.hist(image.flatten(), log=False, bins=100, range=(10**(numpy.log10(mean_init)-0.25), 10**(numpy.log10(mean_init)+0.25)))
    plt.axvline(x=mean_dcut, color='k')
    for m in modes:
        plt.axvline(x=m, color='r')
    plt.axvline(x=mean, color='b')
    plt.show()
    plt.close()
    
    plt.figure()
    plt.hist(image.flatten(), log=False, bins=100, range=(10**(numpy.log10(mean_init)-0.1), 10**(numpy.log10(mean_init)+0.1)))
    plt.axvline(x=mean, color='b')
    plt.axvline(x=mean_dcut, color='k')
    for m in modes:
        plt.axvline(x=m, color='r')
    plt.show()
    plt.close()"""
    
    # need to extend cutout if background is not good enough
    #if numpy.isnan(mean) or number_of_sky_pixels < 20000:
    
    # save segmap
    hdu = fits.PrimaryHDU(segmap)
    hdu.writeto(imname[:-5] + '_segmap.fits', overwrite=True)
    
    # finally, subtract refined background
    image = image - mean
    
    # Save rescaled and bgsubbed image
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(imname[:-5] + '_bgsub.fits', overwrite=True)
    
    """plt.figure()
    plt.imshow(rescale_image(image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()"""
    
    # Now get weightmap since we have sky
    wmap = get_weights(image, mean) 
    hdu = fits.PrimaryHDU(wmap)
    hdu.writeto(imname[:-5] + '_wmap.fits', overwrite=True)
    
    # use scipy to make distance map
    # distance map is needed to exclude sky pixels too close to objects
    #infotable = numpy.loadtxt(imname[:-5] + '_info.dat', usecols=(0,1))
    #pixScale = infotable[0]
    #sky_distance_map, number_of_sky_pixels_dcut = getskymap(image, segmap, pixScale, arcsec_limdist)
    
    # I assume here that the center pixel belongs to the object
    # There may be a safer way to do this
    target_index = segmap[int(tdim/2.)][int(tdim/2.)]
    if target_index == 0:
        raise Exception('Center pixel is sky')
    
    # Create masked images
    masked_image = numpy.copy(image)
    
    masked_image[(segmap != target_index) & (segmap != 0)] = numpy.nan # mask other objects, dont mask sky
    #masked_image[(sky_distance_map == 1)] = numpy.nan # mask sky used for background
    
    hdu = fits.PrimaryHDU(masked_image)
    hdu.writeto(imname[:-5] + '_masked.fits', overwrite=True)
    
    plt.figure()
    plt.imshow(rescale_image(image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(rescale_image(masked_image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    outfile.write('{} {} {} {} {} {} {}\n'.format(ncombine_flag, unit_flag, gain_flag, number_of_sky_pixels, mean, mean_init, sky_flag))
    
    return 0

def main():
    
    data = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True)
    data_int = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=numpy.uint64)
    data_str = numpy.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=None, encoding=None)
    
    ras = data['ra']
    decs = data['dec']
    redshifts = data['z']
    objid = data_int['objid']
    exptimes = data['exptime']
    imageNames = data_str['imname']
    instrumentNames = data_str['instrument']
    detectorNames = data_str['detector']
    filterNames = data_str['filter']
    aperNames = data_str['aperture']
    
    objid_unique = numpy.unique(objid)
    
    outfile = open('CO_HST_improcess_infocat.txt', 'w')
    
    outfile.write('#objid instrument detector aperture filter exptime ')
    outfile.write('ncombine_flag unit_flag gain_flag skybg_count sky sky_init sky_flag\n')
    
    for i in range(0,25):
        
        """if i % 10 == 0:
            print('')
            print('')
            print(i, objid_unique[i])
            print('')
            print('')"""
        
        if i > -99.:
        #if str(objid_unique[i]) == '1237648721223352487' or str(objid_unique[i]) == '1237648705663205475' or str(objid_unique[i]) == '1237648703518998783':
        #if str(objid_unique[i]) == '1237648705663205475':
        #if str(objid_unique[i]) == '1237648703518998783':
            #
       
            all_indices = numpy.where(objid == objid_unique[i])
            
            all_ra = ras[all_indices[0]]
            all_dec = decs[all_indices[0]]
            all_redshifts = redshifts[all_indices[0]]
            all_imageNames = imageNames[all_indices[0]]
            all_exptimes = exptimes[all_indices[0]]
            all_filterNames = filterNames[all_indices[0]]
            all_instruments = instrumentNames[all_indices[0]]
            all_detectors = detectorNames[all_indices[0]]
            all_apertures = aperNames[all_indices[0]]
            
            images_unique = numpy.unique(all_imageNames)
        
            # process('HST_image_repo/' + str(objid_unique[i]) + '/SDSSr.fits')
            for name in images_unique:
                
                temp_index = numpy.where( all_imageNames == name )[0][0]
                
                exptime = all_exptimes[temp_index]
                pfilter = all_filterNames[temp_index]
                instr = all_instruments[temp_index]
                det = all_detectors[temp_index]
                aper = all_apertures[temp_index]
                ra = all_ra[temp_index]
                dec = all_dec[temp_index]
                z = all_redshifts[temp_index]
                
                impath = 'HST_image_repo/' + str(objid_unique[i]) + '/' + name + '.fits'
                if os.path.exists(impath):
                    outfile.write('{} {} {} {} {} {} '.format(objid_unique[i], instr, det, aper, pfilter, exptime))
                    process(impath, outfile, exptime, ra, dec, name, z)
    
main()