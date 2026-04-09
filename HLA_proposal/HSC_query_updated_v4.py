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

FitsStubImagePlugin.register_handler(astropy.io.fits)

plt.rcParams.update({'font.size': 26})

hscapiurl = "https://catalogs.mast.stsci.edu/api/v0.1/hsc"

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
    
    print(len(objid), ' objects')
    
    # https://hst-docs.stsci.edu/acsihb/chapter-5-imaging/5-3-wide-field-optical-ccd-imaging
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-3-choosing-the-optimum-hst-instrument/3-3-comparison-of-wfc3-with-other-hst-imaging-instruments
    ACS_WFC_pixscale = 0.05 # "/pix
    WFC3_UVIS_pixscale = 0.04
    WFC3_IR_pixscale = 0.13 
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/wfpc2/_documents/wfpc2_dhb.pdf
    WFPC2_WF_pixscale = 0.1
    
    #has_good_image_count = 0
    
    used_IDs = []
    
    for i in range(0,len(objid)):
        
        sdss_id = objid[i]
        mra = ra[i]
        mdec = dec[i]
        mz = z[i]
        mSSFR = sSFR[i]
        mMstar = mstar[i]
        mprad = prad[i]

        arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(mz) * (u.kpc / u.arcsec)
        
        #final_image = np.zeros(shape=(imsize,imsize))
        image_list = []
        exptime_list = []
        pixScale_list = []
        
        selval = 1237661874024677506 #1237661069263765721
        if str(sdss_id) == str(selval) and sdss_id not in used_IDs:
        
            for j in range(0,len(objid)):
                
                if str(objid[j]) == str(sdss_id):
            
                    exptime = exptimes[j]
                    imname = imageNames[j]
                    instr = instrumentNames[j]
                    det = detectorNames[j]
                    pfilter = filterNames[j]
            
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
                        pixScale = WFPC2_WF_pixscale
                    else:
                        print('Error:  No appropriate pixel scale found for object ', objid[i])
                        return 1
                    
                    
                    
                    rad_tot = mprad / pixScale
                    #imsize = int( 50 * arcsec_per_kpc / pixScale ) 
                    imsize = int( round(5 * rad_tot) ) 
                    
                    print(rad_tot, imsize, pixScale)
                    
                    # http://hla.stsci.edu/fitscutcgi_interface.html
                    # ^ info on cutout service
                    
                    
                    found_image = False
                    bad_pix_threshold = 0.05
                        
                    download_successful = False
                    while not download_successful:
                        try:
                            #image = get_hla_cutout(imname,mra,mdec,size=imsize, autoscale=97.5)
                            image = np.asarray(get_hla_cutout(imname,mra,mdec,size=imsize,autoscale=100,asinh=False))
                            download_successful = True
                        except:
                            download_successful = False
                            continue
                        
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
                            if image[ty][tx] == 255:
                                bad_pixel_tuples.append( (tx, ty) )
                    
                    all_pixels = image[is_bounded_tot == True]
                    bad_pixels = all_pixels[all_pixels == 255]
                    print(100*bad_pixels.size/all_pixels.size, '% bad pixels' )
                    
                    if (bad_pixels.size/all_pixels.size) < bad_pix_threshold:
                        found_image = True
                        print(imname, ' is good!')
                    else:
                        print(imname, ' is bad, checking next image')
                        break
                
                    if found_image:
                        
                        # get background
                        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
                        image[image != 255] = image[image != 255] - mean
                        image[image == 255] = 0
                        
                        image_list.append(image)
                        exptime_list.append(exptime)
                        pixScale_list.append(pixScale)
                        
                        """stretch = apv.AsinhStretch(0.01)
                        image = image / np.max(image)
                        image[image > 0.95] = 0.95
                        image = image / np.max(image)
                        image = stretch(image)
                        
                        plt.figure()
                        #plt.add_axes([objx + 0.2, objy*1.2, 1.0, 1.0])
                        plt.imshow(image,cmap="gray")
                        #plt.title('BG subbed')
                        ax = plt.gca()
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_tick_params(left=False, labelleft=False)
                        xmax = float(0.1 + (10*arcsec_per_kpc/pixScale/imsize))
                        plt.axhline(y=0.15*imsize, xmin=0.1, xmax=xmax, color='#ffff00', linewidth=6)
                        #fig.savefig('images/all_combined.png', bbox_inches = "tight", dpi=500)
                        plt.show()
                        plt.close()"""
                    
            """final_pixScale = np.max(np.asarray(pixScale_list))
            imsize = int( round(4 * mprad / final_pixScale) )  # rounding of image pixels may cause issues, should follow up
            
            final_image = np.zeros(shape=(imsize,imsize))
            
            print('All images downloaded, making final image')
            
            number_of_images = 0
            for k in range(0,len(image_list)):
                image_temp = Image.fromarray(image_list[k])
                if pixScale_list[k] < final_pixScale:
                    image_temp = image_temp.resize((imsize, imsize)) #block_reduce(image_temp, final_pixScale/pixScale_list[k])
                final_image += image_temp
                number_of_images += 1
            
            final_image /= number_of_images # saturated or bad pixels may have weird values
            
            plt.figure()
            #plt.add_axes([objx + 0.2, objy*1.2, 1.0, 1.0])
            plt.imshow(final_image,cmap="gray")
            ax = plt.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_tick_params(left=False, labelleft=False)
            xmax = float(0.1 + (10*arcsec_per_kpc/pixScale/imsize))
            plt.axhline(y=0.15*imsize, xmin=0.1, xmax=xmax, color='#ffff00', linewidth=6)
            #fig.savefig('images/all_combined.png', bbox_inches = "tight", dpi=500)
            plt.show()
            plt.close()"""
            
        used_IDs.append(sdss_id)
                    
    return 0
    
def get_hla_cutout(imagename,ra,dec,size=33,autoscale=99.5,asinh=True,zoom=1):
    
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="png", red=imagename, autoscale=autoscale, asinh=asinh, zoom=zoom))
    im = Image.open(BytesIO(r.content))
    
    return im

def get_hla_cutout_custom(imagename,ra,dec,size=33,asinh=False,zoom=1,autoscalemax=100.0, autoscalemin=0.0):
    
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="png", red=imagename, autoscalemax=autoscalemax, autoscalemin=autoscalemin, asinh=asinh, zoom=zoom))
    im = Image.open(BytesIO(r.content))
    
    return im

main()