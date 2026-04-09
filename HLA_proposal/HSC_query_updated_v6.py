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
    mstar = data['Mstar']
    sSFR = data['sSFR']
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
    
    for i in range(0,len(objid)):
        
        if i % 10 == 0:
            print(i)
            
        if str(objid[i]) == '1237657192516354264' or str(objid[i]) == '1237648721223352487': 
        
            path = 'HST_image_repo/{}/'.format(objid[i])
            if not os.path.exists(path):
                os.makedirs(path)
            
            sdss_id = objid[i]
            mra = ra[i]
            mdec = dec[i]
            mz = z[i]
            mSSFR = sSFR[i]
            mMstar = mstar[i]
            mprad = prad[i]
    
            arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(mz) * (u.kpc / u.arcsec)
        
            exptime = exptimes[i]
            imname = imageNames[i]
            instr = instrumentNames[i]
            det = detectorNames[i]
            pfilter = filterNames[i]
            aper = aperNames[i]
    
            pixScale = 0.01 # Overestimate resolution by default
            if 'ACS' in instr:
                if 'WFC' in det:
                    pixScale = ACS_WFC_pixscale
                else:
                    print('Error:  No appropriate pixel scale found for object ', objid[i])
                    return 1
            elif 'WFC3' in instr:
                if 'UVIS' in det:
                    pixScale = WFC3_UVIS_pixscale
                elif 'IR' in det:
                    pixScale = WFC3_IR_pixscale
                else:
                    print('Error:  No appropriate pixel scale found for object ', objid[i])
                    return 1
            elif 'WFPC2' in instr:
                if 'PC' in aper:
                    pixScale = WFPC2_PC_pixscale
                elif 'WF' in aper:
                    pixScale = WFPC2_WF_pixscale
                else:
                    print('Error:  No appropriate WFPC2 scale for ', objid[i], i)
            else:
                print('Error:  No appropriate pixel scale found for object ', objid[i])
                return 1
            
            with open(path + imname + '_info.dat', 'w') as outfile:
                outfile.write('{} {} {} {} {}'.format(pixScale, exptime, instr, det, pfilter))
                
            rad_tot = mprad / pixScale
            #imsize = int( 50 * arcsec_per_kpc / pixScale ) 
            imsize = int( round(4 * rad_tot) ) 
            
            #print(rad_tot, imsize, pixScale)
            
            # http://hla.stsci.edu/fitscutcgi_interface.html
            # ^ info on cutout service
            
            found_image = False
            bad_pix_threshold = 0.05
            
            fitsfile = get_hla_cutout(imname,mra,mdec,size=imsize,autoscale=100,asinh=False)
            image = fitsfile[0].data * exptime # electrons / s * exptime
            
            hdr_out = fitsfile[0].header
            
            """ncombined = header['NCOMBINE']
            hdr_out = fits.Header()
            hdr_out['NCOMBINE'] = ncombined
            hdr_out['EXPTIME'] = exptime
            hdr_out['SDSS_ID'] = sdss_id
            hdr_out['PIXSCALE'] = pixScale"""
            #hdr_out['GAIN'] = 1
            #print(hdr_out)
                
            angle = np.linspace(0, 2*np.pi, 150)
            xtot = (imsize/2) + rad_tot * np.cos(angle)
            ytot = (imsize/2) + rad_tot * np.sin(angle)
            
            xmin = max( [int(np.min(xtot)), 0] )
            xmax = min( [int(np.max(xtot)), imsize] )
            ymin = max( [int(np.min(ytot)), 0] )
            ymax = min( [int(np.max(ytot)), imsize] )
            
            ellipse_tot = Ellipse( xy=(imsize/2,imsize/2), width=2*rad_tot, height=2*rad_tot, angle=0 )
            is_bounded_tot = np.zeros(shape=(imsize,imsize))
            bad_pixel_tuples = []
            for tx in range(xmin,xmax):#imsize):
                for ty in range(ymin,ymax):#imsize):
                    is_bounded_tot[ty][tx] = ellipse_tot.contains_point( point=(tx, ty) )
                    if image[ty][tx] == 'nan':
                        bad_pixel_tuples.append( (tx, ty) )
            
            all_pixels = image[is_bounded_tot == True]
            bad_pixels = all_pixels[all_pixels == 'nan']
            #print(100*bad_pixels.size/all_pixels.size, '% bad pixels' )
            
            if (bad_pixels.size/all_pixels.size) < bad_pix_threshold:
                found_image = True
                #print(imname, ' is good!')
            else:
                print(imname, ' is bad')
        
            if found_image:
                hdu = fits.PrimaryHDU(data=image, header=hdr_out)
                hdu.writeto(path + imname + '.fits', overwrite=True)
                
    return 0
    
def get_hla_cutout(imagename,ra,dec,size=33,autoscale=99.5,asinh=True,zoom=1):
    
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="fits", red=imagename, autoscale=autoscale, asinh=asinh, zoom=zoom))
    #print(BytesIO(r.content))
    #im = Image.open(BytesIO(r.content))
    im = fits.open(BytesIO(r.content))
    
    return im

main()