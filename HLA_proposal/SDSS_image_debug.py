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

from PIL import FitsStubImagePlugin

FitsStubImagePlugin.register_handler(astropy.io.fits)

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
    
    used_IDs = []
    
    angle = np.linspace(0, 2*np.pi, 150)
    
    for i in range(0,20):
        
        sdss_id = objid[i]
        mra = ra[i]
        mdec = dec[i]
        mz = z[i]
        mSSFR = sSFR[i]
        mn = n[i]
        
        if sdss_id not in used_IDs:
            
            # Download SDSS image
            SDSS_pixScale = 0.396127
            imsize_SDSS = 75 #int( round(4 * prad[i] / SDSS_pixScale) )
            #imsize_SDSS = int( 50 * arcsec_per_kpc / SDSS_pixScale ) 
            download_successful = False
            
            print(sdss_id, mra, mdec)

            # see https://skyview.gsfc.nasa.gov/current/cgi/query.pl

            print('Downloading SDSS image')
            paths = SkyView.get_images('{} {}'.format(mra,mdec), survey=['SDSSr'], pixels=imsize_SDSS, deedger="skyview.process.Deedger", sampler="LI", projection="Tan", coordinates="J2000")
            print(paths)
            for path in paths:
                data = path[0].data
                
                rad_tot = prad[i] / SDSS_pixScale
                xtot = (imsize_SDSS/2) + rad_tot * np.cos(angle)
                ytot = (imsize_SDSS/2) + rad_tot * np.sin(angle)
            
                plt.figure(figsize=(12,9))
                plt.title('z = {}, sSFR = {}, n = {}'.format(round(mz,2), round(mSSFR,2), round(mn,2)))
                plt.imshow(np.arcsinh(data),origin='lower',cmap="gray")
                #plt.plot(xtot, ytot, 'r', linewidth=1)
                ax = plt.gca()
                #ax.invert_xaxis()
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                plt.show()
                plt.close()
                        
            download_successful = True
                
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