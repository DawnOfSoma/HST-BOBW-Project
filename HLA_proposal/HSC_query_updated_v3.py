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
    
    data = np.genfromtxt('data/crossmatch_GSW_Simard_Group_Gzoo.dat', names=True)
    data_int = np.genfromtxt('data/crossmatch_GSW_Simard_Group_Gzoo.dat', names=True, dtype=np.uint64)
    data_str = np.genfromtxt('data/crossmatch_GSW_Simard_Group_Gzoo.dat', names=True, dtype=None, encoding=None)
    
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
    mstar = data['Mstar']
    sSFR = data['sSFR']
    fbar = data['fbar']
    fsmooth = data['fsmooth']
    exptimes = data['exptime']
    imageNames = data_str['imname']
    instrumentNames = data_str['instrument']
    detectorNames = data_str['detector']
    filterNames = data_str['filter']
    
    # Apply cuts
    
    """objid = objid[sSFR < -10.8]
    ra = ra[sSFR < -10.8]
    dec = dec[sSFR < -10.8]
    z = z[sSFR < -10.8]
    n = n[sSFR < -10.8]
    exptimes = exptimes[sSFR < -10.8]
    imageNames = imageNames[sSFR < -10.8]
    instrumentNames = instrumentNames[sSFR < -10.8]
    detectorNames = detectorNames[sSFR < -10.8]
    filterNames = filterNames[sSFR < -10.8]
    fsmooth = fsmooth[sSFR < -10.8]
    sSFR = sSFR[sSFR < -10.8]"""
    
    objid = objid[fsmooth > 0.7]
    ra = ra[fsmooth > 0.7]
    dec = dec[fsmooth > 0.7]
    z = z[fsmooth > 0.7]
    n = n[fsmooth > 0.7]
    exptimes = exptimes[fsmooth > 0.7]
    imageNames = imageNames[fsmooth > 0.7]
    instrumentNames = instrumentNames[fsmooth > 0.7]
    detectorNames = detectorNames[fsmooth > 0.7]
    filterNames = filterNames[fsmooth > 0.7]
    sSFR = sSFR[fsmooth > 0.7]
    fsmooth = fsmooth[fsmooth > 0.7]
    
    """objid = objid[n < 3]
    ra = ra[n < 3]
    dec = dec[n < 3]
    z = z[n < 3]
    exptimes = exptimes[n < 3]
    imageNames = imageNames[n < 3]
    instrumentNames = instrumentNames[n < 3]
    detectorNames = detectorNames[n < 3]
    filterNames = filterNames[n < 3]
    sSFR = sSFR[n < 3]
    n = n[n < 3]"""
    
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
    
    total_count = 0
    has_good_image_count = 0
    
    subsample = [1237658629695799458, 1237654670814675063, 1237655349431959652, 1237674478123417869, 1237661416602534162, 1237653665795801308]
    
    fig = plt.figure()
    
    objno = 0
    objx = 0
    objy = 0
    
    for entry in subsample:
    
        for i in range(0,len(n)):
            
            sdss_id = objid[i]
            mra = ra[i]
            mdec = dec[i]
            mz = z[i]
            mSSFR = sSFR[i]
            mMstar = mstar[i]
            mn = n[i]
            
            if sdss_id not in used_IDs and sdss_id == entry:
                
                total_count += 1
                
                objno += 1
                if (objno % 2 == 0):
                    objx = 1.5
                if (objno % 2 != 0 and objno > 1):
                    objy += 1
                    objx = 0.0
                
                # Select best exp time
                exptime_temp = exptimes[objid == sdss_id]
                imname_temp = imageNames[objid == sdss_id]
                instr_temp = instrumentNames[objid == sdss_id]
                det_temp = detectorNames[objid == sdss_id]
                filters_temp = filterNames[objid == sdss_id]
                
                # Iterate until good image is found
                
                found_image = False
                bad_indices = []
                bad_pix_threshold = 0.05
                
                while not found_image:
                
                    best_xt_index = 0
                    best_xt = 0
                    for k in range(0,len(exptime_temp)):
                        if k not in bad_indices:
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
                    
                    # http://hla.stsci.edu/fitscutcgi_interface.html
                    # ^ info on cutout service
                    
                    rad_tot = prad[i] / pixScale
                    rad_50 = pr50[i] / pixScale
                    rad_90 = pr90[i] / pixScale
                    
                    angle = np.linspace(0, 2*np.pi, 150)
                    
                    #imsize = int( 50 * arcsec_per_kpc / pixScale ) 
                    imsize = int( round(2.5 * rad_tot) ) 
                    
                    xtot = (imsize/2) + rad_tot * np.cos(angle)
                    ytot = (imsize/2) + rad_tot * np.sin(angle)
                    
                    x50 = (imsize/2) + rad_50 * np.cos(angle)
                    y50 = (imsize/2) + rad_50 * np.sin(angle)
                    
                    x90 = (imsize/2) + rad_90 * np.cos(angle)
                    y90 = (imsize/2) + rad_90 * np.sin(angle)
                
                    download_successful = False
                    while not download_successful:
                        try:
                            image = np.asarray(get_hla_cutout(imname,mra,mdec,size=imsize,autoscale=100,asinh=False))
                            download_successful = True
                        except:
                            download_successful = False
                            continue
                    
                    # sample url:  https://hla.stsci.edu/cgi-bin/fitscut.cgi?red=hst_13695_58_wfc3_uvis_f606w&size=750,750&x=225.153590&y=16.172324&wcs=1&format=fits&config=ops
                    
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
                        bad_indices.append(best_xt_index)
                        print(imname, ' is bad, checking next best image')
                        if len(bad_indices) >= len(exptime_temp):
                            print('Images exhausted, adjusting threshold')
                            bad_indices = []
                            bad_pix_threshold += 0.05
                        if bad_pix_threshold > 0.05:#0.25:
                            print('No good image for this object')
                            break
                
                if found_image:
                    
                    has_good_image_count += 1
                
                    if mSSFR >= -10.8:
                        outpath = 'images/SFG/'
                    elif mSSFR <= -11.8:
                        outpath = 'images/Q/'
                    else:
                        outpath = 'images/GV/'
                    
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
                    
                    if entry == 1237658629695799458 or entry == 1237655349431959652:
                        
                        download_successful = False
                        while not download_successful:
                            try:
                                #image = get_hla_cutout(imname,mra,mdec,size=imsize, autoscale=97.5)
                                image = np.asarray(get_hla_cutout(imname,mra,mdec,size=imsize,autoscale=100,asinh=False))
                                download_successful = True
                            except:
                                download_successful = False
                                continue
                        
                        stretch = apv.AsinhStretch(0.01)
                        image = image / np.max(image)
                        image[image > 0.95] = 0.95
                        image = image / np.max(image)
                        image = stretch(image)
                        
                    else:
                        
                        download_successful = False
                        while not download_successful:
                            try:
                                #image = get_hla_cutout(imname,mra,mdec,size=imsize, autoscale=97.5)
                                image = np.asarray(get_hla_cutout(imname,mra,mdec,size=imsize,autoscale=100,asinh=False))
                                download_successful = True
                            except:
                                download_successful = False
                                continue
                        
                        stretch = apv.AsinhStretch(0.01)
                        image = image / np.max(image)
                        image[image > 0.95] = 0.95
                        image = image / np.max(image)
                        image = stretch(image)
                        
                    HST_plot = fig.add_axes([objx + 0.2, objy*1.2, 1.0, 1.0])
                    HST_plot.imshow(image,cmap="gray")
                    ax = plt.gca()
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_tick_params(left=False, labelleft=False)
                    xmax = float(0.1 + (10*arcsec_per_kpc/pixScale/imsize))
                    HST_plot.axhline(y=0.15*imsize, xmin=0.1, xmax=xmax, color='#ffff00', linewidth=6)
                    
                    # Download SDSS image
                    SDSS_pixScale = 0.396127
                    imsize_SDSS = int( round(2.5 * prad[i] / SDSS_pixScale) )
                    #imsize_SDSS = int( 50 * arcsec_per_kpc / SDSS_pixScale ) 
                    download_successful = False
                    
                    print(imsize_SDSS, mra, mdec)
                    
                    while not download_successful:
                        try:
                            # see https://skyview.gsfc.nasa.gov/current/cgi/query.pl
                            paths = SkyView.get_images('{} {}'.format(mra,mdec), survey=['SDSSr'], pixels=imsize_SDSS, deedger="skyview.process.Deedger", sampler="LI", projection="Tan", coordinates="J2000")
                            for path in paths:
                                data = path[0].data
                                
                                rad_tot = prad[i] / SDSS_pixScale
                                xtot = (imsize_SDSS/2) + rad_tot * np.cos(angle)
                                ytot = (imsize_SDSS/2) + rad_tot * np.sin(angle)
                                
                                # see http://ds9.si.edu/doc/ref/how.html#Scales
                                #data = np.arcsinh(data)
                                """data = data / np.max(data)
                                data[data > 0.9] = 0.9
                                data = np.arcsinh(10*data) / 3"""
                                
                                stretch = apv.AsinhStretch(0.05)
                                data = stretch(data/np.max(data))
                            
                                SDSS_plot = fig.add_axes([objx - 0.5, objy*1.2, 1.0, 1.0])
                                SDSS_plot.imshow(data,origin='lower',cmap="gray")
                                ax = plt.gca()
                                ax.xaxis.set_visible(False)
                                ax.yaxis.set_tick_params(left=False, labelleft=False)
                                xmax = float(0.1 + (10*arcsec_per_kpc/SDSS_pixScale/imsize_SDSS))
                                SDSS_plot.axhline(y=0.85*imsize_SDSS, xmin=0.1, xmax=xmax, color='#ffff00', linewidth=6)
                                fig.text(s='z = {}, log $M_*$ = {}, log sSFR = {}, n = {}'.format(round(mz,2), round(mMstar,1), round(mSSFR,1), round(mn,1)), size=24, y=1.05, x=0.0, horizontalalignment='left', transform=ax.transAxes)
                
                            download_successful = True
                        except:
                            download_successful = False
                            continue
                        
                used_IDs.append(sdss_id)
    
    fig.savefig('images/all_combined.png', bbox_inches = "tight", dpi=500)
    fig.clf()
    
    return 0
    
def get_hla_cutout(imagename,ra,dec,size=33,autoscale=99.5,asinh=True,zoom=1):
    
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="png", red=imagename, autoscale=autoscale, asinh=asinh, zoom=zoom))
    im = Image.open(BytesIO(r.content))
    
    return im

def get_hla_cutout_custom(imagename,ra,dec,size=33,asinh=True,zoom=1,autoscalemax=99.5, autoscalemin=0.0):
    
    url = "https://hla.stsci.edu/cgi-bin/fitscut.cgi"
    r = requests.get(url, params=dict(ra=ra, dec=dec, size=size, 
            format="png", red=imagename, autoscalemax=autoscalemax, autoscalemin=autoscalemin, asinh=asinh, zoom=zoom))
    im = Image.open(BytesIO(r.content))
    
    return im

main()