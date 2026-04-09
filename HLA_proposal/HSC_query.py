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

from PIL import FitsStubImagePlugin

FitsStubImagePlugin.register_handler(astropy.io.fits)

hscapiurl = "https://catalogs.mast.stsci.edu/api/v0.1/hsc"

def main():
    
    print('HSC Query Script')
    
    # Source:
    # https://archive.stsci.edu/hst/hsc/help/HCV/
    
    data = np.genfromtxt('data/crossmatch_GSW_Simard_Group.dat', names=True)
    data_int = np.genfromtxt('data/crossmatch_GSW_Simard_Group.dat', names=True, dtype=np.uint64)
    data_str = np.genfromtxt('data/crossmatch_GSW_Simard_Group.dat', names=True, dtype=None, encoding=None)
    
    objid = data_int['objid']
    ra = data['ra']
    dec = data['dec']
    z = data['z']
    n = data['n']
    pra = data['pra']
    pdec = data['pdec']
    prad = data['prad']
    pr50 = data['pr50']
    pr90 = data['pr90']
    sSFR = data['sSFR']
    exptimes = data['exptime']
    imageNames = data_str['imname']
    instrumentNames = data_str['instrument']
    detectorNames = data_str['detector']
    filterNames = data_str['filter']
    
    # Apply cuts
    
    objid = objid[sSFR < -10.8]
    ra = ra[sSFR < -10.8]
    dec = dec[sSFR < -10.8]
    z = z[sSFR < -10.8]
    n = n[sSFR < -10.8]
    exptimes = exptimes[sSFR < -10.8]
    imageNames = imageNames[sSFR < -10.8]
    instrumentNames = instrumentNames[sSFR < -10.8]
    detectorNames = detectorNames[sSFR < -10.8]
    filterNames = filterNames[sSFR < -10.8]
    sSFR = sSFR[sSFR < -10.8]
    
    """objid = objid[n >= 2]
    ra = ra[n >= 2]
    dec = dec[n >= 2]
    z = z[n >= 2]
    exptimes = exptimes[n >= 2]
    imageNames = imageNames[n >= 2]
    instrumentNames = instrumentNames[n >= 2]
    detectorNames = detectorNames[n >= 2]
    filterNames = filterNames[n >= 2]
    sSFR = sSFR[n >= 2]
    n = n[n >= 2]"""
    
    objid_used = []
    for i in range(0,len(objid)):
        if objid[i] not in objid_used:
            objid_used.append(objid[i])
    
    print(len(objid_used), ' objects after cuts')
    
    # https://hst-docs.stsci.edu/acsihb/chapter-5-imaging/5-3-wide-field-optical-ccd-imaging
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-3-choosing-the-optimum-hst-instrument/3-3-comparison-of-wfc3-with-other-hst-imaging-instruments
    ACS_WFC_pixscale = 0.05 # "/pix
    WFC3_UVIS_pixscale = 0.04
    WFC3_IR_pixscale = 0.13 
    # https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/legacy/wfpc2/_documents/wfpc2_dhb.pdf
    WFPC2_WF_pixscale = 0.1
    
    used_IDs = []
    
    for i in range(0,len(n)):
        
        sdss_id = objid[i]
        mra = ra[i]
        mdec = dec[i]
        mz = z[i]
        mSSFR = sSFR[i]
        mn = n[i]
        
        if sdss_id not in used_IDs:
            
            # Select best exp time
            exptime_temp = exptimes[objid == sdss_id]
            imname_temp = imageNames[objid == sdss_id]
            instr_temp = instrumentNames[objid == sdss_id]
            det_temp = detectorNames[objid == sdss_id]
            filters_temp = filterNames[objid == sdss_id]
            
            best_xt_index = 0
            best_xt = 0
            for k in range(0,len(exptime_temp)):
                if exptime_temp[k] > best_xt:
                    best_xt_index = k
                    best_xt = exptime_temp[k]
            
            imname = imname_temp[best_xt_index]
            instr = instr_temp[best_xt_index]
            det = det_temp[best_xt_index]
            pfilter = filters_temp[best_xt_index]
            
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
        
            arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(mz) * (u.kpc / u.arcsec)
            
            imsize = int( 50 * arcsec_per_kpc / pixScale ) 
            
            # http://hla.stsci.edu/fitscutcgi_interface.html
            # ^ info on cutout service
            
            rad_tot = prad[i] / pixScale
            rad_50 = pr50[i] / pixScale
            rad_90 = pr90[i] / pixScale
            
            angle = np.linspace(0, 2*np.pi, 150)
            
            xtot = (imsize/2) + rad_tot * np.cos(angle)
            ytot = (imsize/2) + rad_tot * np.sin(angle)
            
            x50 = (imsize/2) + rad_50 * np.cos(angle)
            y50 = (imsize/2) + rad_50 * np.sin(angle)
            
            x90 = (imsize/2) + rad_90 * np.cos(angle)
            y90 = (imsize/2) + rad_90 * np.sin(angle)
            
            # Iterate until good image is found
            
            found_image = False
            
            while not found_image:
                image = get_hla_cutout(imname,mra,mdec,size=imsize)
                found_image = True
            
            print(sdss_id, imname, instr, det, pixScale)
            print(pfilter, best_xt)
            print(mra, mdec)
            print(mz)
            
            hstpath = 'images/{}/HST/'.format(sdss_id)
            sdsspath = 'images/{}/SDSS/'.format(sdss_id)
            
            if not os.path.exists(hstpath):
                os.makedirs(hstpath)
            if not os.path.exists(sdsspath):
                os.makedirs(sdsspath)
            
            plt.figure(figsize=(12,9))
            plt.title('z = {}, sSFR = {}, n = {}'.format(round(mz,2), round(mSSFR,2), round(mn,2)))
            plt.imshow(image,cmap="gray")
            plt.plot(imsize/2, imsize/2, 'b+', linewidth=5)
            plt.plot(xtot, ytot, 'r', linewidth=2)
            plt.plot(x50, y50, 'r', linewidth=2)
            plt.plot(x90, y90, 'r', linewidth=2)
            ax = plt.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.show()
            plt.close()
            
            print(imsize)
            
            # Download SDSS image
            SDSS_pixScale = 0.4
            imsize_SDSS = int( 50 * arcsec_per_kpc / SDSS_pixScale ) 
            paths = SkyView.get_images('{} {}'.format(mra,mdec), survey=['SDSSr'], pixels=imsize_SDSS)
            for path in paths:
                data = path[0].data
                print(imsize_SDSS/2)
            
                plt.figure(figsize=(12,9))
                plt.title('z = {}, sSFR = {}, n = {}'.format(round(mz,2), round(mSSFR,2), round(mn,2)))
                plt.imshow(np.arcsinh(data),origin='lower',cmap="gray")
                ax = plt.gca()
                #ax.invert_xaxis()
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                plt.show()
                plt.close()
                
                #hdu = fits.PrimaryHDU(data)
                #hdu.writeto(sdss_out_path, overwrite=True)
            
            used_IDs.append(sdss_id)
    
    return 0
    
def get_hla_cutout(imagename,ra,dec,size=33,autoscale=99.5,asinh=True,zoom=1):
    
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="pdf", red=imagename, autoscale=autoscale, asinh=asinh, zoom=zoom))
    
    im = Image.open(BytesIO(r.content))
    
    return im

main()