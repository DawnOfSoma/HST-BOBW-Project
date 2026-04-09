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
import warnings

warnings.filterwarnings('ignore', category=Warning)

def get_segmap(image_bgsub, threshold):
    segmap = detect_sources(image_bgsub, threshold, npixels=1000)
    segmap = deblend_sources(image_bgsub, segmap, npixels=1000, nlevels=32, contrast=0.1, progress_bar=False)
    segmap = numpy.asarray(segmap)
    return segmap

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
    sky_cutoff_distance = arcsec / pixScale # Simard 2011 used 4 arcseconds, pixel scale in "/pix
    
    inverted_segmap = numpy.full(shape=(tdim,tdim), fill_value=1.)
    inverted_segmap[segmap > 0] = 0.
    
    sdm = ndimage.distance_transform_edt(inverted_segmap)
    
    sdm[sdm <= sky_cutoff_distance] = 0
    sdm[sdm > sky_cutoff_distance] = 1
    sdm[numpy.isnan(image) == True] = 0
    
    return sdm
    

def process(imname, outfile, exptime, ra, dec, imfilename, z):
    
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z) * (apu.kpc / apu.arcsec)
    arcsec_limdist = 6.0 * arcsec_per_kpc
    
    image = fits.open(imname)
    header = image[0].header
    image = image[0].data
    
    tdim = image.shape[0]
    infotable = numpy.loadtxt(imname[:-5] + '_info.dat', usecols=(0,1))
    pixScale = infotable[0]
    
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
    # get the segmentation map
    threshold = 1.0 * std # sigma
    try:
        segmap = get_segmap(image_bgsub, threshold)
    except ValueError:
        outfile.write('{} {} {} {} {} {} {} '.format(ncombine_flag, unit_flag, gain_flag, -99., mean, mean, -99.))
        return 1
    
    # Update background
    # This should be written to output maybe
    sky_flag = 0
    mean_init = mean
    mean, median, std = sigma_clipped_stats(image[segmap == 0], sigma=1.0, maxiters=100)
    image_bgsub = image - mean
    
    number_of_sky_pixels = len(segmap[segmap == 0])
    if number_of_sky_pixels < 20000 or numpy.isnan(mean):
        sky_flag = 1
        
    # I assume here that the center pixel belongs to the object
    # There may be a safer way to do this
    target_index = segmap[int(tdim/2.)][int(tdim/2.)]
    if target_index == 0:
        #raise Exception('Center pixel is sky')
        outfile.write('{} {} {} {} {} {} {} '.format(ncombine_flag, unit_flag, gain_flag, -99., mean, mean_init, -99.))
        return 2
    segmap_target = numpy.copy(segmap)
    segmap_target[(segmap != 0) & (segmap != target_index)] = 0
    
    # save segmap
    hdu = fits.PrimaryHDU(segmap)
    hdu.writeto(imname[:-5] + '_segmap.fits', overwrite=True)
    
    # finally, subtract refined background
    image = image - mean
    
    # Save rescaled and bgsubbed image
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(imname[:-5] + '_bgsub.fits', overwrite=True)
    
    # Now get weightmap since we have sky
    wmap = get_weights(image, mean) 
    hdu = fits.PrimaryHDU(wmap)
    hdu.writeto(imname[:-5] + '_wmap.fits', overwrite=True)
        
    #skydm_all = getskymap(image, segmap, pixScale, arcsec_limdist)
    skydm_target = getskymap(image, segmap_target, pixScale, arcsec_limdist)
    
    # Create masked images
    masked_image = numpy.copy(image)
    masked_image[(segmap != target_index) & (segmap != 0)] = numpy.nan # mask other objects, dont mask sky
    masked_image[skydm_target == 1] = numpy.nan # mask pixels far from target
    #masked_image[(skydm_all == 0) & (skydm_target != 0)] = numpy.nan # mask sky pixels around other objects
    
    hdu = fits.PrimaryHDU(masked_image)
    hdu.writeto(imname[:-5] + '_masked.fits', overwrite=True)
    
    outfile.write('{} {} {} {} {} {} {} '.format(ncombine_flag, unit_flag, gain_flag, number_of_sky_pixels, mean, mean_init, sky_flag))
    
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
    outfile.write('ncombine_flag unit_flag gain_flag skybg_count sky sky_init sky_flag ')
    outfile.write('errcode\n')
    
    for i in range(0,len(objid_unique)):
        
        if i % 10 == 0:
            print(i, '/', len(objid_unique))
       
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
                retval = process(impath, outfile, exptime, ra, dec, name, z)
                outfile.write('{}\n'.format(retval))
    
main()