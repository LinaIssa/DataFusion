#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:03:21 2018

@author: cguillot3
"""

import numpy as np
from astropy.io import fits
from skimage.transform import resize
import cmath
########################### Convolution functions #############################


def compute_symmpad(X, npix):
    """
    Compute a symmetric padding of npix and add 2 rows and 2 columns of zero padding (for aliasing)
    """
    M = X.shape[0]
    N = X.shape[1]
    symm_pad = np.zeros((M+2*npix, N+2*npix), dtype='float64')
    symm_pad[npix:M+npix, npix:N+npix] = X

    symm_pad[npix:M+npix, 0:npix] = X[:, npix-1::-1]
    symm_pad[0:npix, npix:N+npix] = X[npix-1::-1]
    symm_pad[M+npix:, npix:N+npix] = X[M-1:M-1-npix:-1]
    symm_pad[npix:M+npix, N+npix:] = X[:, N-1:N-1-npix:-1]

    symm_pad[0:npix, 0:npix] = symm_pad[2*npix-1:npix-1:-1, 0:npix]
    symm_pad[M+npix:M+2*npix, 0:npix] = symm_pad[M+npix-1:M-1:-1, 0:npix]
    symm_pad[0:npix, N+npix:N+2*npix] = symm_pad[2*npix-1:npix-1:-1, N+npix:N+2*npix]
    symm_pad[M+npix:M+2*npix, N+npix:N+2*npix] = symm_pad[M+npix-1:M-1:-1, N+npix:N+2*npix]

    symm_pad_ = np.zeros((M+2*npix+2, N+2*npix+2), dtype='float64')
    symm_pad_[:M+2*npix, :N+2*npix] = symm_pad
    return symm_pad_


def compute_symmpad_3d(X, npix):
    L, M, N = X.shape
    Y = np.zeros((L, M+2*npix+2, N+2*npix+2))
    for i in range(L):
        Y[i] = compute_symmpad(X[i], npix)
    return Y


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def convolve_nc(X, H, fact_padding : int, mode='direct'): # not used
    npix = fact_padding
    Y = compute_symmpad(X, npix)
    Yfft = np.fft.rfft2(Y)
    if 'adj' in mode:
        H *= get_trans('nircam')
    ret = np.fft.irfft2(Yfft*H)[:-2, :-2]
    return ret[npix:X.shape[0]+npix, npix:X.shape[1]+npix]


def convolve_ns(X, G, mode='direct'): # not used
    npix = 10 # dépend de la taille de la fonction dede transfert +1
    Y = compute_symmpad(X, npix)
    Yfft = np.fft.rfft2(Y)
    if 'adj' in mode:
        G *= get_trans('nirspec')
    ret = np.fft.irfft2(Yfft*G)[:-2, :-2]
    return ret[npix:X.shape[0]+npix, npix:X.shape[1]+npix]


def get_fft(Kernel, shape_im): # not used
    s1 = np.array(Kernel[0].shape)+1
    s2 = np.array(shape_im[1:])+10*2
    sh = s1+s2-1
    Kernel_ = np.zeros((Kernel.shape[0], Kernel.shape[1]+1, Kernel.shape[2]+1))
    Kernel_[:, 1:, 1:] = Kernel
    return np.fft.rfft2(Kernel_, s=sh, axes=(1, 2))


def fft_ns_ac(tabwave): # not used
    tb = np.arange(1842,3442)
    psf = np.zeros((2,3574-1842,181,181))
    for i in np.append(tb):
        cube = np.fft.fft2(np.fft.fftshift(fits.getdata('psf_ns_'+str(i)+'_'+str(i+100)+'.fits'), axis=(1,2)), axis=(1,2))
        psf[0] = np.real(cube)
        psf[1] = np.imag(cube)
    hdu = fits.PrimaryHDU(psf)
    hdu.writeto('psf_fft_g140h_f070lp.fits')


def compute_translation(sh): # not used anywhere
    K, L = sh
    trans = np.zeros(sh, dtype=complex)
    for k in range(K):
        for l in range(L):
            trans[k, l] = cmath.exp(-2*np.pi*k*(1/2)*1.j)*cmath.exp(-2*np.pi*l*(1/2)*1.j)
    return trans


def save_trans(trans, name): # not used
    M, N = trans.shape()
    t = np.zeros((2, M, N))
    t[0] = np.real(trans)
    t[1] = np.imag(trans)
    hdu = fits.PrimaryHDU(t)
    hdu.writeto('/usr/local/home/cguillot3/Ech_fraction/trans_'+name+'.fits')


def get_trans(name): # used in convolve_nc and convolve_ns
    t = fits.getdata('/usr/local/home/cguillot3/Ech_fraction/trans_'+name+'.fits')
    L, M, N = t.shape()
    trans = np.zeros((M, N), dtype=complex)
    trans = t[0]+t[1]*1.j
    return trans

####################### Getters for NIRCam direct model #######################


def getline(M, A, l): # not used
    """
    Calcule une ligne de la scène à observer (à une longueur d'onde donnée)
    ---------------------------------------------------------------------------
    Entrées:
        M : np.array 2D - matrice des spectres sources
        A : np.array 3D - matrice des abondances
        l : int - numero de la ligne
    Sortie :
        X : np.array 2D - scène à la longueur d'onde l
    """
    i, m, n = A.shape
    A = np.reshape(A, (i, m*n))
    return np.reshape(np.dot(M[l], A), (m, n))

####################### Getters for NIRSpec direct model ######################


def subsample(X): #  used only in produce_HS_MS.py
    """
    """
    Shy = np.around(X.shape[0]*31/100)
    Shx = np.around(X.shape[1]*31/100)
    return resize(X, (Shy, Shx), order=1)

####################### Getters for NC/NS PSFs ################################
# not used anywhere

def get_psf_ns(i):  #not used anywhere
    i_ = fits.getdata('/usr/local/home/cguillot3/Ech_fraction/G_fft_1.fits').shape[0]//2+i
    g = np.array(fits.getdata('/usr/local/home/cguillot3/Ech_fraction/G_fft_1.fits')[i]+fits.getdata('/usr/local/home/cguillot3/Ech_fraction/G_fft_1.fits')[i_]*1.j, dtype=complex)
#    i_=fits.getdata('/Users/cguillot/Documents/Ech_fraction/G_fft.fits').shape[0]//2+i
#    g=np.array(fits.getdata('/Users/cguillot/Documents/Ech_fraction/G_fft.fits')[i]+fits.getdata('/Users/cguillot/Documents/Ech_fraction/G_fft.fits')[i_]*1.j,dtype=np.complex_)
    return g


def get_psf_nc(i):  #not used anywhere
    i_ = fits.getdata('/usr/local/home/cguillot3/Ech_fraction/H_fft_1.fits').shape[0]//2+i
    h = np.array(fits.getdata('/usr/local/home/cguillot3/Ech_fraction/H_fft_1.fits')[i]+fits.getdata('/usr/local/home/cguillot3/Ech_fraction/H_fft_1.fits')[i_]*1.j, dtype=complex)
#    i_=fits.getdata('/Users/cguillot/Documents/Ech_fraction/H_fft.fits').shape[0]//2+i
#    h=np.array(fits.getdata('/Users/cguillot/Documents/Ech_fraction/H_fft.fits')[i]+fits.getdata('/Users/cguillot/Documents/Ech_fraction/H_fft.fits')[i_]*1.j,dtype=np.complex_)
    return h


def get_h(PSF_MS : str, mode='direct'):  #not used anywhere
    # h_ = fits.getdata('H_fft_1.fits')
    # h_ = fits.getdata('/usr/local/home/cguillot3/SAM20/EXPE/H_fft.fits')
    h_ = fits.getdata(PSF_MS)
    k, l, m, n = h_.shape
    h = np.zeros((l, m, n), dtype=complex)
    h = h_[0]+h_[1]*1.j
    if 'adj' in mode:
        h = np.conj(h)
    return np.reshape(h, (l, m*n))


def get_h_mean(PSF_MS: str, mode='direct'):
    # h_ = fits.getdata(PSF+'H_fft_1.fits')[:, :, 0, 0]
    # h_ = fits.getdata('/usr/local/home/cguillot3/SAM20/EXPE/H_fft.fits')[:, :, 0, 0]
    # h_ = fits.getdata(DATA+'H_fft.fits')[:, :, 0, 0]
    h_ = fits.getdata(PSF_MS)[:, :, 0, 0]
    # h_=fits.getdata('/usr/local/home/cguillot3/Fourier_Res/H_fft.fits')[:,:,0,0]
    k, l = h_.shape
    h = np.zeros((l), dtype=complex)
    h = h_[0]+h_[1]*1.j
    if 'adj' in mode:
        h = np.conj(h)
    return h


def get_h_band(PSF_MS : str, band, mode='direct'):
    h_ = fits.getdata(PSF_MS)[:, band]  #PSF_MS is defined in CONSTANTS
    k, m, n = h_.shape
    #h = np.zeros((m, n), dtype=complex)
    h = h_[0]+h_[1]*1.j
    if 'adj' in mode:
        h = np.conj(h)
    return np.reshape(h, (m*n))


def get_h_bands(lh, l_1000, mode='direct'): #not used anywhere
    if l_1000 == (lh//1000)+1:
        h_ = fits.getdata('H_fft_1.fits')[:, l_1000*1000:]
    else:
        h_ = fits.getdata('H_fft_1.fits')[:, l_1000*1000:(l_1000+1)*1000]
    k, l, m, n = h_.shape
    h = np.zeros((l, m, n), dtype=complex)
    h = h_[0]+h_[1]*1.j
    if 'adj' in mode:
        h = np.conj(h)
    return np.reshape(h, (l, m*n))


def get_g(PSF_HS : str, mode='direct'): # not used anywhere
    # g_=fits.getdata('/usr/local/home/cguillot3/Fourier_Res/G_fft.fits')
    # g_ = fits.getdata('G_fft_1.fits')
    # g_ = fits.getdata('/usr/local/home/cguillot3/SAM20/EXPE/G_fft.fits')
    g_ = fits.getdata(PSF_HS)
    k, l, m, n = g_.shape
    g = np.zeros((l, m, n), dtype=complex)
    g = g_[0]+g_[1]*1.j
    if 'adj' in mode:
        g = np.conj(g)
    return np.reshape(g, (l, m*n))


def get_g_mean(PSF_HS: str, mode='direct'):
    # g_ = fits.getdata(PSF+'G_fft_1.fits')[:, :, 0, 0]
    # g_=fits.getdata('/usr/local/home/cguillot3/Fourier_Res/G_fft.fits')[:,:,0,0]
    # g_ = fits.getdata('/usr/local/home/cguillot3/SAM20/EXPE/G_fft.fits')[:,:,0,0]
    # g_ = fits.getdata(DATA+'G_fft.fits')[:,:,0,0]
    g_ = fits.getdata(PSF_HS)[:,:,0,0]
    k, l = g_.shape
    g = np.zeros((l), dtype=complex)
    g = g_[0]+g_[1]*1.j
    if 'adj' in mode:
        g = np.conj(g)
    return g


def get_g_band(PSF_HS: str, band, mode='direct'):
    # g_ = fits.getdata(PSF+'G_fft_1.fits')[:, band]
    # g_=fits.getdata('/usr/local/home/cguillot3/Fourier_Res/G_fft.fits')[:,band]
    # g_ = fits.getdata('/usr/local/home/cguillot3/SAM20/EXPE/G_fft.fits')[:,band]
    # g_ = fits.getdata(DATA+'G_fft.fits')[:,band]
    g_ = fits.getdata(PSF_HS)[:, band]
    k, m, n = g_.shape
    g = np.zeros((m, n), dtype=complex)
    g = g_[0]+g_[1]*1.j
    if 'adj' in mode:
        g = np.conj(g)
    return np.reshape(g, (m*n))

########################### Aliasing functions ################################


def aliasing(Z : np.array, shape3d : tuple, downsampling : int):
    d = downsampling
    l, m, n = shape3d
    Z = np.reshape(Z, shape3d)
    Z_ = np.zeros((l, m//d, n//d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Z_ += Z[:, i*m//d:(i+1)*m//d, j*n//d:(j+1)*n//d]
    return np.reshape(Z_, (l, (m//d)*(n//d)))/d


def aliasing_adj(Z, shape3d : tuple, downsampling : int): #  used in tools.build_b
    d = downsampling
    Z_ = np.zeros(shape3d, dtype=complex)
    M = shape3d[1]//d
    N = shape3d[2]//d
    Z = np.reshape(Z, (Z.shape[0], M, N))
    for i in range(d):
        for j in range(d):
            Z_[:, i*M:(i+1)*M, j*N:(j+1)*N] = Z
    return np.reshape(Z_/d, (Z_.shape[0], Z_.shape[1]*Z_.shape[2]))
