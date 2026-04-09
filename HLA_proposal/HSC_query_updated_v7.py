import astropy, pylab, time, sys, os, requests, json
from astroquery.skyview import SkyView
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

from astropy.table import Table, join
from astropy.io import ascii
import astropy.units as u
from astropy.io import fits

from astropy.cosmology import WMAP9 as cosmo

from matplotlib.patches import Ellipse

from astropy.stats import sigma_clipped_stats

from astropy.nddata import block_reduce

import astropy.visualization as apv

from PIL import FitsStubImagePlugin

from photutils.segmentation import detect_sources

import statmorph

FitsStubImagePlugin.register_handler(astropy.io.fits)

plt.rcParams.update({'font.size': 26})

hscapiurl = "https://catalogs.mast.stsci.edu/api/v0.1/hsc"

def rescale_image(image):
    stretch = apv.AsinhStretch(0.01)
    image = image / np.max(image)
    image[image > 0.95] = 0.95
    image = image / np.max(image)
    image = stretch(image)
    return image

def main():
    
    print('HSC Query Script')
    
    # Source:
    # https://archive.stsci.edu/hst/hsc/help/HCV/
    
    data = np.genfromtxt('data/crossmatch_GSW.dat', names=True)
    data_int = np.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=np.uint64)
    data_str = np.genfromtxt('data/crossmatch_GSW.dat', names=True, dtype=None, encoding=None)
    
    objid = data_int['objid']
    ra = data['ra']
    dec = data['dec']
    z = data['z']
    prad = data['prad']
    pr50 = data['pr50']
    pr90 = data['pr90']
    #mstar = data['Mstar']
    #sSFR = data['sSFR']
    exptimes = data['exptime']
    imageNames = data_str['imname']
    instrumentNames = data_str['instrument']
    detectorNames = data_str['detector']
    filterNames = data_str['filter']
    aperNames = data_str['aperture']
    
    objid_unique = np.unique(objid)
    
    print(len(objid), len(objid_unique), ' objects')
    
    # https://hst-docs.stsci.edu/acsihb/chapter-5-imaging/5-3-wide-field-optical-ccd-imaging
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-3-choosing-the-optimum-hst-instrument/3-3-comparison-of-wfc3-with-other-hst-imaging-instruments
    ACS_WFC_pixscale = 0.05 # "/pix
    WFC3_UVIS_pixscale = 0.04
    WFC3_IR_pixscale = 0.13 
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/wfpc2/_documents/wfpc2_dhb.pdf
    WFPC2_WF_pixscale = 0.1
    WFPC2_PC_pixscale = 0.0456 #TinyTim doc
    SDSS_pixScale = 0.396127
    
    imname_used = []
    
    for i in range(0,len(objid_unique)):
        
        if i % 10 == 0:
            print(i, '/', len(objid_unique))
            
        #if str(objid[i]) == '1237657192516354264' or str(objid[i]) == '1237648721223352487': 
        
        path = 'HST_image_repo/{}/'.format(objid_unique[i])
        if not os.path.exists(path):
            os.makedirs(path)
        
        sdss_id = objid_unique[i]
        
        all_indices = np.where(objid == objid_unique[i])
        
        mra = ra[all_indices[0][0]]
        mdec = dec[all_indices[0][0]]
        mz = z[all_indices[0][0]]
        mprad = prad[all_indices[0][0]]
        
        all_exptimes = exptimes[all_indices[0]]
        all_imageNames = imageNames[all_indices[0]]
        all_instrumentNames = instrumentNames[all_indices[0]]
        all_detNames = detectorNames[all_indices[0]]
        all_filterNames = filterNames[all_indices[0]]
        all_aperNames = aperNames[all_indices[0]]
        
        images_unique = np.unique(all_imageNames)
        #print('# of unique images: ', len(images_unique))

        arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(mz) * (u.kpc / u.arcsec)
        
        for j in range(0,len(images_unique)):
            
            temp_index = np.where( all_imageNames == images_unique[j] )[0][0]
    
            imname = images_unique[j]  # all_imageNames[temp_index]
    
            exptime = all_exptimes[temp_index]
            instr = all_instrumentNames[temp_index]
            det = all_detNames[temp_index]
            pfilter = all_filterNames[temp_index]
            aper = all_aperNames[temp_index]
            
            """if not os.path.exists(path + 'SDSSr' + '.fits'):
                imsize_SDSS = int( round(4 * mprad / SDSS_pixScale) )
                SDSSpaths = SkyView.get_images('{} {}'.format(mra,mdec), survey=['SDSSr'], pixels=imsize_SDSS, deedger="skyview.process.Deedger", sampler="LI", projection="Tan", coordinates="J2000")
                for SDSSpath in SDSSpaths:
                    SDSS_imdata = SDSSpath[0].data
                    SDSS_header = SDSSpath[0].header
                hdu = fits.PrimaryHDU(data=SDSS_imdata, header=SDSS_header)
                hdu.writeto(path + 'SDSSr.fits', overwrite=True)"""
    
            pixScale = 0.01 # Overestimate resolution by default
            if 'ACS' in instr:
                if 'WFC' in det:
                    pixScale = ACS_WFC_pixscale
                elif 'multi' in det:
                    pixScale = ACS_WFC_pixscale
                else:
                    print('Error:  No appropriate pixel scale found for object ', sdss_id)
                    return 1
            elif 'WFC3' in instr:
                if 'UVIS' in det:
                    pixScale = WFC3_UVIS_pixscale
                elif 'IR' in det:
                    pixScale = WFC3_IR_pixscale
                else:
                    print('Error:  No appropriate pixel scale found for object ', sdss_id)
                    return 1
            elif 'WFPC2' in instr:
                if 'PC' in aper:
                    pixScale = WFPC2_PC_pixscale
                elif 'WF' in aper:
                    pixScale = WFPC2_WF_pixscale
                elif 'multi' in aper:
                    pixScale = WFPC2_WF_pixscale
                else:
                    #print(aper)
                    print('Error:  No appropriate WFPC2 scale for ', sdss_id, i)
                    return 1
            else:
                print('Error:  No appropriate pixel scale found for object ', sdss_id)
                return 1
                
            rad_tot = mprad / pixScale
            imsize = int( round(4 * rad_tot) ) 
     
            # http://hla.stsci.edu/fitscutcgi_interface.html
            # ^ info on cutout service
            
            found_image = False
            bad_pix_threshold = 0.05
            
            fitsfile = get_hla_cutout(imname,mra,mdec,size=imsize,autoscale=100,asinh=False)
            image = fitsfile[0].data * exptime # electrons / s * exptime
            hdr_out = fitsfile[0].header
                
            xmin = int((imsize/2) - rad_tot)
            xmax = int((imsize/2) + rad_tot)
            
            all_pixels = image[xmin:xmax][xmin:xmax]
            bad_pixels = all_pixels[np.isnan(all_pixels) == True]
            
            """plt.figure()
            plt.imshow(image,cmap="gray")
            plt.axhline(y=xmax, xmin=xmin, xmax=xmax, color='r', linewidth=2)
            plt.axhline(y=xmin, xmin=xmin, xmax=xmax, color='r', linewidth=2)
            plt.axvline(x=xmax, ymin=xmin, ymax=xmax, color='r', linewidth=2)
            plt.axvline(x=xmin, ymin=xmin, ymax=xmax, color='r', linewidth=2)
            ax = plt.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_tick_params(left=False, labelleft=False)
            plt.show()
            plt.close()"""
            
            if (bad_pixels.size/all_pixels.size) < bad_pix_threshold:
                found_image = True
        
            if found_image:
                hdu = fits.PrimaryHDU(data=image, header=hdr_out)
                hdu.writeto(path + imname + '.fits', overwrite=True)
                
                with open(path + imname + '_info.dat', 'w') as outfile:
                    outfile.write('{} {} {} {} {}'.format(pixScale, exptime, instr, det, pfilter))
                    
                # download extended cutout just in case it is needed
                """imsize = int(10 * imsize) # 100x area
                fitsfile = get_hla_cutout(imname,mra,mdec,size=imsize,autoscale=100,asinh=False)
                image = fitsfile[0].data * exptime # electrons / s * exptime
                hdr_out = fitsfile[0].header
                hdu = fits.PrimaryHDU(data=image, header=hdr_out)
                hdu.writeto(path + imname + '_extended.fits', overwrite=True)"""
                
    return 0
    
def get_hla_cutout(imagename,ra,dec,size=33,autoscale=99.5,asinh=True,zoom=1):
    
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="fits", red=imagename, autoscale=autoscale, asinh=asinh, zoom=zoom))
    im = fits.open(BytesIO(r.content))
    
    return im

main()