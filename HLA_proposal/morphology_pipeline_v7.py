# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:07:28 2022

@author: sunlu
"""

import os
import numpy
#from photutils.segmentation import detect_sources
import statmorph
import matplotlib.pyplot as plt
from astropy.io import fits
#from astropy.stats import sigma_clipped_stats
import astropy.visualization as apv
#from astropy.convolution import convolve
#from photutils.segmentation import make_2dgaussian_kernel
#from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Sersic2D
#from petrofit.modeling import print_model_params
from petrofit.modeling import fit_model
#from petrofit.modeling import plot_fit
from petrofit.modeling import PSFConvolvedModel2D
#from scipy import ndimage
import warnings

warnings.filterwarnings('ignore', category=Warning)

def validate_files(imname):
    try:
        image = fits.open(imname[:-5] + '_bgsub.fits')
        image = image[0].data
        
        segmap = fits.open(imname[:-5] + '_segmap.fits')
        segmap = segmap[0].data
        
        wmap = fits.open(imname[:-5] + '_wmap.fits')
        wmap = wmap[0].data
        
        masked_image = fits.open(imname[:-5] + '_masked.fits')
        masked_image = masked_image[0].data
    except FileNotFoundError:
        return 1
    return 0

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

    image_original = fits.open(imname)  
    header = image_original[0].header
    image_original = image_original[0].data
    
    ncombine = 1
    try:
        ncombine = header['NCOMBINE']
    except KeyError:
        ncombine = 1 # Should be true, since ncombine keyword is absent for WFPC2 images
    if ncombine < 1:
        ncombine = 1
    
    image = fits.open(imname[:-5] + '_bgsub.fits')
    image = image[0].data
    tdim = image.shape[0]
    
    segmap = fits.open(imname[:-5] + '_segmap.fits')
    segmap = segmap[0].data
    
    wmap = fits.open(imname[:-5] + '_wmap.fits')
    wmap = wmap[0].data
    
    masked_image = fits.open(imname[:-5] + '_masked.fits')
    masked_image = masked_image[0].data
    
    masked_wmap = 1. / wmap
    masked_wmap[numpy.isnan(masked_image)] = numpy.nan
    
    """plt.figure()
    plt.imshow(rescale_image(image_original),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
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
    plt.imshow(rescale_image(masked_image),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(rescale_image(wmap),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()
    
    plt.figure()
    plt.imshow(rescale_image(masked_wmap),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()"""
    
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
    
    # I assume here that the center pixel belongs to the object
    # There may be a safer way to do this
    target_index = segmap[int(tdim/2.)][int(tdim/2.)]
    if target_index == 0:
        raise Exception('Center pixel is sky')
    
    # statmorph gets wmap, which is units of std, not 1 / std like petrofit wants
    morph = statmorph.SourceMorphology(image=image, segmap=segmap, label=target_index, weightmap=wmap, psf=psf)
    #statmorph.source_morphology(image, SM_segmap, weightmap=wmap, psf=psf, mask=SM_mask)
    #morph = source_morphs[0] # Will throw error if sky is selected (segmap = 0)
    
    model = Sersic2D(amplitude=morph.sersic_amplitude, r_eff=morph.sersic_rhalf, n=morph.sersic_n, x_0=morph.sersic_xc, y_0=morph.sersic_yc, ellip=morph.sersic_ellip, theta=morph.sersic_theta)
    synth_yvals, synth_xvals = numpy.mgrid[0:tdim, 0:tdim]
    model_image = model(synth_xvals, synth_yvals)
    residual = image - model_image
    
    """plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()"""
    
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
    #SM_flag_catastrophic = morph.flag_catastrophic
    outfile.write('{} {} '.format(SM_flag, SM_flag_sersic))
    
    #print('SM done, moving to petrofit')
    
    # now we run petrofit
    
    maxiters = 10000

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
    
    """plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()"""
    
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
    
    """plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()"""
    
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
    
    """plt.figure()
    plt.imshow(rescale_image(residual),cmap="gray")
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    plt.show()
    plt.close()"""
    
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
    
    return 0
    
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
    outfile.write('SM_flag SM_flag_sersic ')
    outfile.write('s_mag s_Reff s_n s_ellip s_theta s_dx s_dy s_chir ')
    
    outfile.write('f_mag f_BT fd_Reff fd_e fd_th fd_dx fd_dy ')
    outfile.write('fb_Reff fb_n fb_e fb_th fb_dx fb_dy f_chir ')
    
    outfile.write('4_mag 4_BT 4d_Reff 4d_e 4d_th 4d_dx 4d_dy ')
    outfile.write('4b_Reff 4b_e 4b_th 4b_dx 4b_dy 4_chir\n')
    
    for i in range(0,len(objid_unique)):
        
        #1237648704586514654
        if i % 10 == 0:
            print(i, '/', len(objid_unique))
        
        if i > -99.:
        #if str(objid_unique[i]) == '1237648721223352487':
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
                    retval = validate_files(impath)
                    if retval == 0:
                        outfile.write('{} {} {} {} {} {} '.format(objid_unique[i], instr, det, aper, pfilter, exptime))
                        process(impath, outfile, exptime)
                
    outfile.close()
         
    return 0

main()