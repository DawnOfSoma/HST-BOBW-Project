# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:07:28 2022

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

def hst_flux_to_abmag(flux, header):
    # copied from petrofit API, modified slightly
    if flux <= 0:
        return numpy.nan
    PHOTFLAM = header['PHOTFLAM']
    PHOTZPT = header['PHOTZPT']
    PHOTPLAM = header['PHOTPLAM']

    STMAG_ZPT = (-2.5 * numpy.log10(PHOTFLAM)) + PHOTZPT
    ABMAG_ZPT = STMAG_ZPT - (5. * numpy.log10(PHOTPLAM)) + 18.692
    return -2.5 * numpy.log10(flux) + ABMAG_ZPT

def rescale_image(image):
    stretch = apv.AsinhStretch(0.01)
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

def get_chi2(residual, sigma):
    df = len(sigma[numpy.isnan(sigma) == False])
    chi2 = numpy.nansum( (residual[numpy.isnan(sigma) == False]**2.) / (sigma[numpy.isnan(sigma) == False]**2.) ) / df
    return chi2

def process(imname, outfile, exptime):
    
    image = fits.open(imname)
    header = image[0].header
    image = image[0].data
    
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
    
    # Now find the appropriate PSF file
    # PSFs are for detector center, sun-like star
    # For WFPC2, we may want to make this position-dependent (on detector)
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
    
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=10)
    image_bgsub = image - mean

    threshold = 1.0 * std # 1 sigma
    segmap = detect_sources(image_bgsub, threshold, npixels=100)
    segmap = numpy.asarray(segmap)
    hdu = fits.PrimaryHDU(segmap)
    hdu.writeto(imname[:-5] + '_segmap.fits', overwrite=True)
    
    plt.figure()
    plt.imshow(segmap,cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # use scipy to make distance map
    # distance map is needed to exclude sky pixels too close to objects
    tdim = image.shape[0]
    inverted_segmap = numpy.full(shape=(tdim,tdim), fill_value=1.)
    inverted_segmap[segmap > 0] = 0.
    sky_distance_map = ndimage.distance_transform_edt(inverted_segmap)
    
    infotable = numpy.loadtxt(imname[:-5] + '_info.dat', usecols=(0,1))
    pixScale = infotable[0]
    sky_cutoff_distance = 4.0 / pixScale # 4 arcsec, Simard 2011 used 4 arcseconds, pixel scale in "/pix
    
    sky_distance_map[sky_distance_map <= sky_cutoff_distance] = 0
    sky_distance_map[sky_distance_map > sky_cutoff_distance] = 1
    sky_distance_map[numpy.isnan(image) == True] = 0
    number_of_sky_pixels = len(image[sky_distance_map == 1])
    print('# of sky pixels = ', number_of_sky_pixels)
    
    plt.figure()
    plt.imshow(sky_distance_map,cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # Update background
    # This should be written to output maybe
    sky_flag = 0
    mean_init = mean
    mean, median, std = sigma_clipped_stats(image[sky_distance_map == 1], sigma=3.0, maxiters=10)
    
    if numpy.isnan(mean) or number_of_sky_pixels < 20000:
        mean = mean_init
        sky_flag = 1
        
    image = image - mean
    
    plt.figure()
    plt.imshow(rescale_image(image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # Now get weightmap since we have sky
    wmap = get_weights(image, mean) 
    hdu = fits.PrimaryHDU(wmap)
    hdu.writeto(imname[:-5] + '_wmap.fits', overwrite=True)
    
    # I assume here that the center pixel belongs to the object
    # There may be a safer way to do this
    target_index = segmap[int(tdim/2.)][int(tdim/2.)]
    if target_index == 0:
        raise Exception('Center pixel is sky')
    
    # configure segmentation map for statmorph
    SM_segmap = numpy.copy(segmap)
    SM_segmap[(segmap != target_index)] = 0 # only target is segmented; all else is 'sky'
    
    SM_invseg = numpy.full(shape=(tdim,tdim), fill_value=1.)
    SM_invseg[SM_segmap > 0] = 0.
    SM_skydm = ndimage.distance_transform_edt(SM_invseg)
    
    SM_skydm[SM_skydm <= sky_cutoff_distance] = 0
    SM_skydm[SM_skydm > sky_cutoff_distance] = 1
    SM_skydm[(segmap != target_index) & (segmap != 0)] = 1
    
    SM_segmap = numpy.full(shape=(tdim,tdim), fill_value=0.)
    SM_segmap[(SM_skydm == 0)] = 1
    SM_segmap = SM_segmap.astype(int)
    
    # mask other objects out
    SM_mask = numpy.full(shape=(tdim,tdim), fill_value=False)
    SM_mask[(segmap != 0) & (segmap != target_index)] = True
    SM_mask[numpy.isnan(image) == True] = True # mask nans
    
    morph = statmorph.SourceMorphology(image=image, segmap=segmap, label=target_index, weightmap=wmap, psf=psf, mask=SM_mask)
    #statmorph.source_morphology(image, SM_segmap, weightmap=wmap, psf=psf, mask=SM_mask)
    #morph = source_morphs[0] # Will throw error if sky is selected (segmap = 0)
    
    model = Sersic2D(amplitude=morph.sersic_amplitude, r_eff=morph.sersic_rhalf, n=morph.sersic_n, x_0=morph.sersic_xc, y_0=morph.sersic_yc, ellip=morph.sersic_ellip, theta=morph.sersic_theta)
    synth_yvals, synth_xvals = numpy.mgrid[0:tdim, 0:tdim]
    model_image = model(synth_xvals, synth_yvals)
    residual = image - model_image
    
    plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    #outfile.write('SM_mag SM_Reff SM_n SM_ellip SM_theta SM_dx SM_dy ')
    model_image_rescaled = model_image / (ncombine * exptime)
    fit_magnitude = hst_flux_to_abmag( numpy.nansum(model_image_rescaled), header )
    xdiff = morph.sersic_xc - (tdim/2)
    ydiff = morph.sersic_yc - (tdim/2)
    outfile.write('{} {} {} {} {} {} {} '.format(fit_magnitude, morph.sersic_rhalf, morph.sersic_n, morph.sersic_ellip, morph.sersic_theta, xdiff, ydiff))
    
    #outfile.write('C A S r20 r50 r80 Gini M20 rpc rpe rhc rhe M I D mag_c mag_e ')
    mag_c = hst_flux_to_abmag( morph.flux_circ / (ncombine * exptime), header )
    mag_e = hst_flux_to_abmag( morph.flux_ellip / (ncombine * exptime), header )
    outfile.write('{} {} {} '.format(morph.concentration, morph.asymmetry, morph.smoothness) )
    outfile.write('{} {} {} '.format(morph.r20, morph.r50, morph.r80) )
    outfile.write('{} {} '.format(morph.gini, morph.m20) )
    outfile.write('{} {} {} {} '.format(morph.rpetro_circ, morph.rpetro_ellip, morph.rhalf_circ, morph.rhalf_ellip) )
    outfile.write('{} {} {} '.format(morph.multimode, morph.intensity, morph.deviation) )
    outfile.write('{} {} '.format(mag_c, mag_e) )
    
    SM_flag = morph.flag
    SM_flag_sersic = morph.flag_sersic
    SM_flag_catastrophic = morph.flag_catastrophic
    outfile.write('{} {} {} {} {} {} {} {} {} {} '.format(SM_flag, SM_flag_sersic, SM_flag_catastrophic, ncombine_flag, unit_flag, gain_flag, number_of_sky_pixels, mean, mean_init, sky_flag))
    
    #print('SM done, moving to petrofit')
    
    # now we test petrofit
    
    maxiters = 10000
    
    # Create masked images
    masked_image = numpy.copy(image)
    # Petrofit wants 1 / sigma weights
    masked_wmap = 1. / wmap
    
    masked_image[(segmap != target_index) & (segmap != 0)] = numpy.nan # mask other objects, dont mask sky
    masked_image[(sky_distance_map == 1)] = numpy.nan # mask sky used for background
    
    masked_wmap[(segmap != target_index) & (segmap != 0)] = numpy.nan
    masked_wmap[(sky_distance_map == 1)] = numpy.nan
    
    hdu = fits.PrimaryHDU(masked_image)
    hdu.writeto(imname[:-5] + '_masked.fits', overwrite=True)

    # First run petrofit and get a single-component profile
    center_slack = 10
    
    amp_init = morph.sersic_amplitude
    reff_init = morph.sersic_rhalf
    n_init = morph.sersic_n
    x0_init = morph.sersic_xc
    y0_init = morph.sersic_yc
    e_init = morph.sersic_ellip
    theta_init = morph.sersic_theta
    
    petrofit_model = Sersic2D(
        amplitude=amp_init,
        r_eff=reff_init,
        n=n_init,
        x_0=y0_init,
        y_0=y0_init,
        ellip=e_init,
        theta=theta_init,
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
    maxiter=maxiters
    )
    
    # Calculate chi2
    residual = image - fitted_model(synth_xvals, synth_yvals)
    chi2 = get_chi2(residual, 1. / masked_wmap)
    
    plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # Write model and residual images to output
    hdu = fits.PrimaryHDU(residual)
    hdu.writeto(imname[:-5] + '_s_residual.fits', overwrite=True)
    hdu = fits.PrimaryHDU(fitted_model(synth_xvals, synth_yvals))
    hdu.writeto(imname[:-5] + '_s_model.fits', overwrite=True)
    
    #outfile.write('nf_mag nf_Reff nf_n nf_ellip nf_theta nf_dx nf_dy nf_chir ')
    model_image_rescaled = fitted_model(synth_xvals, synth_yvals) / (ncombine * exptime)
    fit_magnitude = hst_flux_to_abmag( numpy.nansum(model_image_rescaled), header )
    xdiff = fitted_model.x_0.value - (tdim/2)
    ydiff = fitted_model.y_0.value - (tdim/2)
    outfile.write('{} {} {} {} {} {} {} {} '.format(fit_magnitude, fitted_model.r_eff.value, fitted_model.n.value, fitted_model.ellip.value, fitted_model.theta.value, xdiff, ydiff, chi2))
    
    # Now try a compound model
    amp_init = fitted_model.amplitude.value
    reff_init = fitted_model.r_eff.value
    x0_init = fitted_model.x_0.value
    y0_init = fitted_model.y_0.value
    e_init = fitted_model.ellip.value
    theta_init = fitted_model.theta.value
    
    disk_model = Sersic2D(
        amplitude=amp_init/2.,
        r_eff=reff_init,
        n=1.0,
        x_0=x0_init,
        y_0=y0_init,
        ellip=e_init,
        theta=theta_init,
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
        amplitude=amp_init/2.,
        r_eff=reff_init/2.,
        n=4.0,
        x_0=x0_init,
        y_0=y0_init,
        ellip=e_init,
        theta=theta_init,
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
    maxiter=maxiters
    )
    
    residual = image - fitted_model(synth_xvals, synth_yvals)
    chi2 = get_chi2(residual, 1. / masked_wmap)
    
    plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # Write model and residual images to output
    hdu = fits.PrimaryHDU(residual)
    hdu.writeto(imname[:-5] + '_f_residual.fits', overwrite=True)
    hdu = fits.PrimaryHDU(fitted_model(synth_xvals, synth_yvals))
    hdu.writeto(imname[:-5] + '_f_model.fits', overwrite=True)
    
    #outfile.write('f_mag f_BT fd_Reff fd_e fd_th fd_dx fd_dy ')
    #outfile.write('fb_Reff fb_n fb_e fb_th fb_dx fb_dy f_chir ')
    model_image_rescaled = fitted_model(synth_xvals, synth_yvals) / (ncombine * exptime)
    fit_magnitude = hst_flux_to_abmag( numpy.nansum(model_image_rescaled), header )
    xdiff0 = fitted_model.x_0_0.value - (tdim/2)
    ydiff0 = fitted_model.y_0_0.value - (tdim/2)
    xdiff1 = fitted_model.x_0_1.value - (tdim/2)
    ydiff1 = fitted_model.y_0_1.value - (tdim/2)
    
    temp_disk = Sersic2D(amplitude=fitted_model.amplitude_0.value, 
                          r_eff=fitted_model.r_eff_0.value, 
                          n=fitted_model.n_0.value, 
                          x_0=fitted_model.x_0_0.value, y_0=fitted_model.y_0_0.value, 
                          ellip=fitted_model.ellip_0.value, 
                          theta=fitted_model.theta_0.value)
    temp_bulge = Sersic2D(amplitude=fitted_model.amplitude_1.value, 
                          r_eff=fitted_model.r_eff_1.value, 
                          n=fitted_model.n_1.value, 
                          x_0=fitted_model.x_0_1.value, y_0=fitted_model.y_0_1.value, 
                          ellip=fitted_model.ellip_1.value, 
                          theta=fitted_model.theta_1.value)
    tBT = numpy.nansum(temp_bulge(synth_xvals, synth_yvals)) / numpy.nansum(temp_disk(synth_xvals, synth_yvals) + temp_bulge(synth_xvals, synth_yvals))
    
    outfile.write('{} {} {} {} {} {} {} '.format(fit_magnitude, tBT, fitted_model.r_eff_0.value, fitted_model.ellip_0.value, fitted_model.theta_0.value, xdiff0, ydiff0))
    outfile.write('{} {} {} {} {} {} {} '.format(fitted_model.r_eff_1.value, fitted_model.n_1.value, fitted_model.ellip_1.value, fitted_model.theta_1.value, xdiff1, ydiff1, chi2))
    
    # Now try a compound model with bulge component fixed at n = 4
    
    disk_model = Sersic2D(
        amplitude=amp_init/2.,
        r_eff=reff_init,
        n=1.0,
        x_0=x0_init,
        y_0=y0_init,
        ellip=e_init,
        theta=theta_init,
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
        amplitude=amp_init/2.,
        r_eff=reff_init/2.,
        n=4.0,
        x_0=x0_init,
        y_0=y0_init,
        ellip=e_init,
        theta=theta_init,
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
    maxiter=maxiters
    )
    
    residual = image - fitted_model(synth_xvals, synth_yvals)
    chi2 = get_chi2(residual, 1. / masked_wmap)
    
    plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    # Write model and residual images to output
    hdu = fits.PrimaryHDU(residual)
    hdu.writeto(imname[:-5] + '_4_residual.fits', overwrite=True)
    hdu = fits.PrimaryHDU(fitted_model(synth_xvals, synth_yvals))
    hdu.writeto(imname[:-5] + '_4_model.fits', overwrite=True)
    
    #outfile.write('f_mag f_BT fd_Reff fd_e fd_th fd_dx fd_dy ')
    #outfile.write('fb_Reff fb_e fb_th fb_dx fb_dy ')
    model_image_rescaled = fitted_model(synth_xvals, synth_yvals) / (ncombine * exptime)
    fit_magnitude = hst_flux_to_abmag( numpy.nansum(model_image_rescaled), header )
    xdiff0 = fitted_model.x_0_0.value - (tdim/2)
    ydiff0 = fitted_model.y_0_0.value - (tdim/2)
    xdiff1 = fitted_model.x_0_1.value - (tdim/2)
    ydiff1 = fitted_model.y_0_1.value - (tdim/2)
    
    temp_disk = Sersic2D(amplitude=fitted_model.amplitude_0.value, 
                          r_eff=fitted_model.r_eff_0.value, 
                          n=fitted_model.n_0.value, 
                          x_0=fitted_model.x_0_0.value, y_0=fitted_model.y_0_0.value, 
                          ellip=fitted_model.ellip_0.value, 
                          theta=fitted_model.theta_0.value)
    temp_bulge = Sersic2D(amplitude=fitted_model.amplitude_1.value, 
                          r_eff=fitted_model.r_eff_1.value, 
                          n=fitted_model.n_1.value, 
                          x_0=fitted_model.x_0_1.value, y_0=fitted_model.y_0_1.value, 
                          ellip=fitted_model.ellip_1.value, 
                          theta=fitted_model.theta_1.value)
    tBT = numpy.nansum(temp_bulge(synth_xvals, synth_yvals)) / numpy.nansum(temp_disk(synth_xvals, synth_yvals) + temp_bulge(synth_xvals, synth_yvals))
    
    outfile.write('{} {} {} {} {} {} {} '.format(fit_magnitude, tBT, fitted_model.r_eff_0.value, fitted_model.ellip_0.value, fitted_model.theta_0.value, xdiff0, ydiff0))
    outfile.write('{} {} {} {} {} {}\n'.format(fitted_model.r_eff_1.value, fitted_model.ellip_1.value, fitted_model.theta_1.value, xdiff1, ydiff1, chi2))
    
    
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
    outfile.write('#objid instrument detector aperture filter exptime ')
    
    outfile.write('SM_mag SM_Reff SM_n SM_ellip SM_theta SM_dx SM_dy ')
    outfile.write('C A S r20 r50 r80 Gini M20 rpc rpe rhc rhe M I D mag_c mag_e ')
    outfile.write('SM_flag SM_flag_sersic SM_flag_catastrophic ncombine_flag unit_flag gain_flag skybg_count sky sky_init sky_flag ')
    outfile.write('s_mag s_Reff s_n s_ellip s_theta s_dx s_dy s_chir ')
    
    outfile.write('f_mag f_BT fd_Reff fd_e fd_th fd_dx fd_dy ')
    outfile.write('fb_Reff fb_n fb_e fb_th fb_dx fb_dy f_chir ')
    
    outfile.write('4_mag 4_BT 4d_Reff 4d_e 4d_th 4d_dx 4d_dy ')
    outfile.write('4b_Reff 4b_e 4b_th 4b_dx 4b_dy 4_chir\n')
    
    for i in range(0,len(objid_unique)):
        
        #1237648704586514654
        """if i % 100 == 0:
            print('')
            print('')
            print('')
            print(i, objid_unique[i])
            print('')
            print('')
            print('')"""
        
        if str(objid_unique[i]) == '1237648721223352487':
        #if str(objid_unique[i]) == '1237648705663205475':
       
            all_indices = numpy.where(objid == objid_unique[i])
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
                
                impath = 'HST_image_repo/' + str(objid_unique[i]) + '/' + name + '.fits'
                if os.path.exists(impath):
                    outfile.write('{} {} {} {} {} {} '.format(objid_unique[i], instr, det, aper, pfilter, exptime))
                    process(impath, outfile, exptime)
                
    outfile.close()
         
    return 0

main()