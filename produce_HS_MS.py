#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:06:54 2019
@author: cguillot3
Ref 1 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
“Simulated JWST datasets for multispectral and hyperspectral image fusion”
The Astronomical Journal, vol. 160, no. 1, p. 28, Jun. 2020.
Ref 2 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
"Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs – Application to High Dimensional Infrared Astronomical Imaging"
IEEE Transactions on Computatonal Imaging, vol.6, Sept. 2020.
This code implements forward models of the NIRCam imager and the NIRSpec IFU embedded in the JWST as described in the references above.
"""

import warnings

import numpy as np
import os.path

from astropy.io        import fits
from tools             import compute_symmpad_3d, _centered, compute_symmpad
from skimage.transform import resize
from scipy.signal      import convolve2d
from main              import load_config
# from pandeia.engine.instrument_factory import InstrumentFactory

warnings.filterwarnings('ignore')


############ Produce Multispectral Image with NIRCam Forward Model ############





##### In Ref 2, section V : Robustness with respect to model mismatch.
# Add white gaussian noise to the spectral degradation operator Lm
def calc_sigma(Lm, snr=50):
    return np.linalg.norm(Lm) ** 2 * 10 ** (-0.1 * snr) * (1 / np.prod(Lm.shape))




def get_spa_bandpsf_ms(band, PSF, sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    h_ = fits.getdata(PSF)[:, band]
    k, m, n = h_.shape
    h = h_[0] + h_[1] * 1.j
    return np.reshape(h, (m, n))




def produce_MS_nir_bis(M, A, Data, fact_pad, PSF,  sigma=0):
    """
    Use this function if the product MA, i.e. the whole scene with full spectral and spatial resolutions, cannot be stored in memory.
    """

    Lm = fits.getdata(Data + 'Lm.fits')

    # Number of MS bands
    lm = Lm.shape[0]

    # Shapes
    lh, m = M.shape
    n, p, q = A.shape

    # Initialize MS image
    Y = np.zeros((lm, p, q))
    # Compute MS bands
    for i in range(lh):
        # Compute the scene at a wavelength
        X = compute_symmpad(np.reshape(np.dot(M[i], np.reshape(A, (n, p * q))), (p, q)), fact_pad)
        # Get PSF at this band
        H = get_spa_bandpsf_ms(i, PSF, sigma)
        # Get spectral degradation at this wavelength
        filt = np.reshape(Lm[:, i], (lm, 1))
        # Convolve and spectrally degrade
        temp = np.reshape(_centered(np.fft.ifft2(H * np.fft.fft2(X, norm='ortho'), norm='ortho')[:-2, :-2], (p, q)),
                          (p * q))
        # Update MS image
        Y += np.real(np.reshape(filt * temp, (lm, p, q)))
    return Y


########### Produce Hyperspectral Image with NIRSpec Forward Model ############




########


def get_spec_psf(Data):
    # Get spectral PSF (1-D gaussian blur)
    return fits.getdata(os.path.join(Data, 'PSF_spec.fits'))


def get_spa_bandpsf_hs(band, PSF,sigma=0):
    # Get a spatial Point Spread Function at wavelength number 'band'. It has to be calculated in the Fourier domain and saved in advance with webbpsf.
    g_ = fits.getdata(PSF)[:, band]
    k, m, n = g_.shape
    g = g_[0] + g_[1] * 1.j
    return np.reshape(g, (m, n))

def subsample2d(X, d=0.31):
    # 2D Subsampling function with Nircam/Nirspec pixel size ratio.
    m, n = X.shape
    return resize(X, np.round((m * d, n * d)))



def produce_HS_nir_bis(Data: str, M, A, fact_pad, PSF, sigma=0, d=0.31):
    """
    Use this function if the product MA, i.e. the whole scene with full spectral and spatial resolutions, cannot be stored in memory.
    M in Mjy/sr
    !!!!! TABWAVE IN MICRONS !!!!!
    """
    Lh = fits.getdata(os.path.join(Data, 'Lh.fits'))

    # Get spectral PSF
    L = get_spec_psf(Data)

    # Shapes
    lh, m = M.shape
    n, p, q = A.shape
    p_ = int(p // (1 / d) + 1)
    q_ = int(q // (1 / d) + 1)
    print(p_,q_)

    # Initialize HS image
    Y = np.zeros((lh, p_, q_))

    for i in range(lh):
        # Compute the 3D scene at a wavelength
        X = compute_symmpad(np.reshape(np.dot(M[i], np.reshape(A, (n, p * q))), (p, q)), fact_pad)
        # Get spatial PSF at that band
        H = get_spa_bandpsf_hs(i, PSF, sigma)
        # Convolve with PSF and subsample
        Y[i] = subsample2d(
            np.real(_centered(np.fft.ifft2(H * np.fft.fft2(X, norm='ortho'), norm='ortho')[:-2, :-2], (p, q))))
    # Spectral covolution and spectral throughput
    Y = np.reshape(Y, (lh, p_ * q_))
    Y = np.reshape((Lh * np.apply_along_axis(lambda x: np.convolve(x, L, mode='same'), axis=0, arr=Y).T).T,
                   (lh, p_, q_))
    return Y


############################## Noise Model ####################################

def mult_noise(Y):
    # Compute multiplicative noise ~ N(0,sqrt(|Y|))
    L, M, N = Y.shape
    noise = np.sqrt(abs(Y)) * np.random.randn(L, M, N)
    return noise


def add_noise_nocorr(Y, sigma2):
    # Compute additive noise with no spatial correlation
    L, M, N = Y.shape
    noise = np.sqrt(sigma2) * np.random.randn(L, M, N)
    return noise


def add_noise_corr_ns(Y, Data, sigma2 = 2*(16.2)**2/2, nframes=5):
    # Compute additive noise with spatial correlation for HS image from NIRSpec
    L, M, N = Y.shape
    # Additive Gaussian Noise
    noise = add_noise_nocorr(Y, sigma2)
    # Get correlation matrix
    if nframes > 18:
        nframes = 18
    cormat = fits.getdata(os.path.join(Data, 'h2rg_corr.fits'))[nframes]
    cormat = cormat / np.sum(cormat)
    noisecorr = np.zeros(Y.shape)
    # Apply additive correlated noise as described in [1]
    for i in range(N // 30):
        temp = compute_symmpad(np.reshape(noise[:, :, 30 * i:30 * (i + 1)], (L, M * 30)), 10)[:-2, :-2]
        noisecorr[:, :, 30 * i:30 * (i + 1)] = np.reshape(_centered(convolve2d(temp, cormat, mode='same'), (L, M * 30)),
                                                          (L, M, 30))
    n = N - 30 * (i + 1)
    temp = compute_symmpad(np.reshape(noise[:, :, 30 * (i + 1):], (L, M * n)), min(10, n))[:-2, :-2]
    noisecorr[:, :, 30 * (i + 1):] = np.reshape(_centered(convolve2d(temp, cormat, mode='same'), (L, M * n)), (L, M, n))
    return noisecorr


def add_noise_corr_nc(Y, Data, sigma2=2*(16.2)**2/2 , nframes=2):
    # Compute additive noise with spatial correlation for MS image from NIRCam
    L, M, N = Y.shape
    # Additive Gaussian Noise
    noise = add_noise_nocorr(Y, sigma2)
    # Get correlation matrix
    if nframes > 18:
        nframes = 18
    cormat = fits.getdata(os.path.join(Data, 'h2rg_corr.fits'))[nframes]
    cormat = cormat / np.sum(cormat)
    noisecorr = np.zeros(Y.shape)
    # Apply additive correlated noise as described in [1]
    for i in range(L):
        temp = compute_symmpad(noise[i, :, :], 10)[:-2, :-2]
        noisecorr[i, :, :] = _centered(convolve2d(temp, cormat, mode='same'), (M, N))
    return noisecorr


def apply_noise(Y, instrument, Data):
    # Apply noise to either HS or MS image.
    Y_noise = Y + mult_noise(Y)
    if 'nirspec' in instrument:
        Y_noise += add_noise_corr_ns(Y, Data)
    elif 'nircam' in instrument:
        Y_noise += add_noise_corr_nc(Y, Data)
    return Y_noise


############################## Simulation and saving ####################################

def main(config, file_data, snr=50, sigma=0):
    ##### Get the 3D scene with full resolution
    M = fits.getdata(os.path.join(file_data, 'M_1.fits')).T
    A = fits.getdata(os.path.join(file_data, 'A.fits'))
    #####

    ##### Get specification : wavelength table, instrument spec.
    tabwave = fits.getdata(os.path.join(file_data, 'tabwave.fits'))[:, 0]
    channel = 'short'
    fname = 'na'
    dname = 'na'
    SIG2READ_NS = 1.6 * (6 / 88) ** 2
    #####

    ##### Compute images
    print('Simulating HS and MS images ...')
    Yh_highsnr = apply_noise(
        produce_HS_nir_bis(config['DataDir'],
            M, A, config['fact_pad'], config['PSF_HS']
        )
        , 'nirspec',
        config['DataDir']
    )
    Ym_highsnr = apply_noise(
        produce_MS_nir_bis(
            M, A, config['DataDir'], config['fact_pad'], config['PSF_MS']
        ),
        'nircam',
        config['DataDir']
    )
    #####

    ##### Save images
    print('Saving HS and MS images ...')
    # Save high snr images
    hdu = fits.PrimaryHDU(Yh_highsnr)
    hdu.writeto(os.path.join(file_data, 'Yh_highsnr.fits'), overwrite=True)
    hdu = fits.PrimaryHDU(Ym_highsnr)
    hdu.writeto(os.path.join(file_data, 'Ym_highsnr.fits'), overwrite=True)
    ######

file_data = '/Users/lina/Documents/MyFusion/input_data/simulated_images'
config = load_config('config.yaml')
main(config, file_data)