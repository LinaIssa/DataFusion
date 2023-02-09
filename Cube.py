#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from skimage.transform import resize
from abc import ABC, abstractmethod
from typing import Union, Optional
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from tools import compute_symmpad_3d,  get_g_mean, aliasing, get_h_mean

class Cube(ABC): #si heritage mettre le nom de l heritage ceci est une classe abstraite
    """
    @author: Lina Issa, adapted from FRHOMAGE code developed by Claire Guilloteau
    We define here an abstract cube class that gathers all the pre-processing and post-processing methods used
    in the fusion framework. These methods apply to the hyperspectral and multispectral data cubes
    and are designed to be applied to JWST datacubes.Date must be loaded beforehand and be a numpy array type.
    The expected shape is (l, x, y) where l corresponds to the spectral dimension and x,y the spatial ones.
    """

    def __init__(self, data: np.ndarray, fact_pad : int, **kwargs) -> None:

        if not isinstance(data, np.ndarray):
            raise TypeError(f'data has type {type(data)} but it must be a numpy array ')

        if not isinstance(fact_pad, int) :
            raise TypeError(f'factor padding has type {type(fact_pad)} but it must be an integer ')

        self.data = data

        self.dim  = data.shape
        self.fact_pad = fact_pad

    @abstractmethod
    def __call__(self,*args, **kwargs):
        return

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        return


class CubeHyperSpectral(Cube):
    """
    @author Lina Issa
    :param data: the hyperspectral image in a numpy array of shape (spectral dimension, spatial dimension 1, spatial dimenson 2)
    :param fact_pad: a padding factor parameter from the configuration file
    :param downsampling: a downsampling factor from the configuration file
    :param fluxConv: a flux conversion parameter from the configuration file
    :param lacp: the dimension of the reduced spectral space
    :param PSF_HS : the path to the PSF file of NirSpec
    :param kwargs: optional parameters
    """

    def __init__(self, data : np.ndarray, YnirCam : np.ndarray, fact_pad : int,  downsampling : int, fluxConv : float,
                 PSF_HS : str, lacp : int = 10, **kwargs) -> None:

        super().__init__(data, fact_pad)
        self.lacp = self._checklacp(lacp)
        self.d    = self._checkdownsampling(downsampling)
        self.fluxConv = fluxConv
        #self.wave, self.x_hyper, self.y_hyper = self.dim[0], self.dim[1], self.dim[2]
        self.x_multi, self.y_multi = YnirCam.shape[1], YnirCam.shape[2]
        self.PSF_file = PSF_HS
        self.V, self.Z, self.mean = None, None, None
        self.Yns = None
        self.sig2 = None
        self.data = self._downsizing(self.x_multi, self.y_multi)

        #self._Yns = self.__call__(Lh)

    def __call__(self, Lh: np.ndarray, **kwargs):


        ### step 1: initialisation with PCA projection
        self.V, self.Z, self.mean = self.initialisation_HS(Lh)

        ### step 2: preprocess !
        self.Yns = self.preprocess(Lh, self.mean)
        return self.Yns

    def _checklacp(self, lacp : int) -> int:

        if not isinstance(lacp, int):
            raise TypeError(f'You provided a {type(lacp)} type instead of an integer for lacp.')

        if lacp < 0 or lacp > self.dim[2]:
            raise ValueError('lacp should be a positive integer and below the spectral dimension of the datacube.')

        return lacp
    def _checkdownsampling(self, downsampling : int) -> int:

        if not isinstance(downsampling, int):
            raise TypeError(f'You provided a {type(downsampling)} type instead of an integer for downsampling.')

        if (2*self.fact_pad+2) % downsampling != 0  or (self.fact_pad//downsampling + 1)%2 != 0:
            raise ValueError('The downsampling and the padding factors does not obey the following relations:  ('
                             '2*fact_pad+2) % d = 0  and (fact_pad//d + 1)%2 =0 . See the docstring in '
                             'create_ConfigFile.py for more information')

        return downsampling




    def _downsizing(self, x_multi: int, y_multi : int, *args, **kwargs) -> np.ndarray :
        """
        @author Lina Issa
        Reshape HS image in order to set the ratio between MS and HS spatial pixel sizes to an integer
       : param x_multi : first  spatial dimension of the Multispectral image
       : param y_multi : second spatial dimension of the Multispectral image
       : returns: downsized np.array
       
       """
        wave = self.data.shape[0]
        return resize(self.data,
                      (wave, x_multi//self.d, y_multi//self.d),
                      order=3,
                      mode='symmetric') * (self.d**2*self.fluxConv)


    def initialisation_HS(self,  Lh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
       : param YnirSpec: the hyperspectral image
       : param Lh      : the spectral operator retrieved from LH.fits
       : returns       : V - the PCA projection matrix, Z - the datacube prepared for the initialisation, mean - the centered data

       """

        print(' PCA on the HS image : ')
        #############################################
        #              Flattening  YnirSpec
        #############################################
        YnirSpec  = self.data
        self.sig2 = np.mean(YnirSpec)
        wave, x_hyper, y_hyper = YnirSpec.shape

        x_multi, y_multi = self.x_multi, self.y_multi
        X = np.reshape(
            np.dot(
                np.diag(Lh**-1), np.reshape(
                    YnirSpec, (wave, x_hyper*y_hyper)
                )
            ),
            (wave, x_hyper, y_hyper)
        )
        X = np.reshape(X.copy(), (wave, x_hyper * y_hyper)) # depliqage du cube X
        #############################################
        #              PCA projection of  Z
        #############################################

        V, Z, mean = self._pca_projection(X.T, self.lacp)

        #############################################
        #              Retroprojection of  Z
        #############################################

        Z = np.reshape(Z.T, (self.lacp, x_hyper, y_hyper))

        #############################################
        #              Upsampling and FFT of  Z
        #############################################

        Z = self._upsampling(Z, x_multi, y_multi, self.lacp)
        Z = compute_symmpad_3d(Z, self.fact_pad)  # applies symmetric padding to the datacube with the help of tools.compute_symmpad_3d
        Z = np.fft.fft2(Z, norm='ortho')    # applies symmetric boundaries condtions to the upsampled hyperspectral datacubes

        mean = self._meanSpectrumFourier(mean, Z)

        return V, Z, mean

    @staticmethod
    def _pca_projection(X: np.ndarray, lacp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs a PCA decomposition on the hyperspectral image in order to retrieve the matrixes V, Z and mean.
        The PCA is applied for preprocessing the hyperspectral datacubes.
        :param X   : the flattened hyperspectral datacubes
        :param lacp:  number of dimension for the reduced spectral space.
        :return    :  V - the PCA projection matrix, Z- the projected cube,  mean - the centered data
        """



        #############################################
        #               PCA Projection
        #############################################
        print(X.T.shape)

        X_mean = np.mean(X.T, axis=0)
        print(X_mean.shape)
        X -= X_mean
        U, S, V = linalg.svd(X.T, full_matrices=False)
        U, V = svd_flip(U, V)
        S = S[:lacp]


        #############################################
        #               PCA Decomposition
        #############################################

        Z = U[:, :lacp] * (S ** (1 / 2))
        V = np.dot(np.diag(S ** (1 / 2)), V[:lacp])

        return V.T, Z, X_mean

    @staticmethod
    def _upsampling(Z: np.ndarray, x_multi: int, y_multi: int, lacp:int) -> np.ndarray:
        """
        Performs a bi-cubic interpolation under symmetric boundaries conditions
        : param Z   : the PCA retroprojected datacube
        : param x_M : first  spatial dimension of the Multispectral image
        : param y_M : second spatial dimension of the Multispectral image
        : returns   : the resized and interpolated datacube Z_interpol
        """
        Z_upsampled = resize(Z, (lacp, x_multi, y_multi), order=3, mode='symmetric')
        return Z_upsampled


# not sure about  keeping these two ...
#    @staticmethod
#    def _padding(Z: np.array, fact_pad: int)-> np.array:
#        """
#        Applies symmetric padding to the datacube with the help of tools.compute_symmpad_3d
#        :param Z_upsampled : upsampled hyperspectral datacubes
#        :param fact_pad    : padding factor for image processing defined in the config file
#        :return            : datacube with a symmetric padding
#        """
#        return compute_symmpad_3d(Z, fact_pad)
#
#    @staticmethod
#    def _fft(Z_upsampled: np.array)-> np.array:
#        """
#        Applies symmetric boundaries condtions to the upsampled hyperspectral datacubes
#        :param Z_upsampled : the upsampled hyperspectral datacubes with symmetric padding
#        :return            : The FFT projection of the datacube Z
#        """
#        Z = np.fft.fft2(Z_upsampled, norm='ortho')
#        return Z

    @staticmethod
    def _meanSpectrumFourier(X_mean, Z):

        """
        Computes the mean spectrum in the Fourier domain
        : param X_mean: the centered data computed by the PCA
        : param Z     : the resized and interpolated datacube Z_interpol
        : returns     : mean, the mean spectrum in the Fourier domain

        """
        N = Z.shape[1] * Z.shape[2]
        mean = np.zeros((X_mean.shape[0], N))
        mean[:, 0] = X_mean * np.sqrt(N)
        return mean

    def preprocess(self, Lh: np.ndarray, mean: np.ndarray) -> np.ndarray :
        """
        Preprocessing for the hyperspectral image only. Needs the mean form the PCA decomposition to substract the mean spectrum to each pixel.

        :param YnirCam : the multispectral image
        :param Lh: the spectral operator retrieved from LH.fits
        :param mean: from the PCA projection on the hyperspectral datacubes
        :return: the fusion-ready hyperspectral image
        """
        print(' Operators and data preprocessing : ')
        YnirSpec = self.data
        x_multi, y_multi = self.x_multi, self.y_multi

        #############################################
        #               FFT on YnirSpec
        #############################################

        Yns = compute_symmpad_3d(YnirSpec,  self.fact_pad//self.d+1)
        Yns = np.fft.fft2(Yns[:, :-2, :-2], axes=(1, 2), norm='ortho')

        #############################################
        #               Substracting the mean image
        #############################################
        wave, x_hyper, y_hyper = Yns.shape
        mean[:, 0] = mean[:, 0] * get_g_mean(self.PSF_file) # Applying the NirSpec PSF to the mean
        self.Yns   = np.reshape(Yns, (wave, x_hyper * y_hyper)) - np.dot(np.diag(Lh),
                                                                               aliasing(
                                                                                   mean, (wave, x_multi,y_multi),
                                                                                   self.d
                                                                               )
                                                                         )
        return self.Yns

class CubeMultiSpectral(Cube):
    """
    : param data    : the multispectral image stored in a numpy array
    : param fact_pad: padding factor, a parameter in main
    : param PSF_MS : the path to the PSF file of NirCam

    : return: the fusion-ready multispectral image
    """

    def __init__(self, data : np.ndarray, fact_pad: int, PSF_MS : str, **kwargs) -> None:
        super().__init__(data, fact_pad)
        #self.wave, self.x_multi, self.y_multi = self.data.dim[0],self.dim[1], self.dim[2]
        self.PSF_file = PSF_MS
        self.Ync = None
        self.sig2 = np.mean(data)

    def __call__(self, cubeHyperSpectal: CubeHyperSpectral, Lm: np.ndarray):
        mean = cubeHyperSpectal.mean
        if mean is None :
            raise ValueError('The hyperspectral image should have been preprocessed first with the call method in order to perform the PCA projection and to compute the mean value.')
        self.Ync  = self.preprocess(Lm, mean)
        return self.Ync

    def preprocess(self, Lm: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """
        Performs preprocessing for the multispectral image. Needs the mean from the PCA decomposition performed on the hyperspectral image.
        :param Lm: the spectral operator retrieved from LM.fits
        :param mean: from the PCA projection on the hyperspectral datacubes
        :return: the fusion-ready multispectral image
        """
        print(' Operators and data preprocessing : ')

        #############################################
        #               FFT on YnirSpec
        #############################################
        YnirCam = self.data
        Ync = compute_symmpad_3d(YnirCam,  self.fact_pad)
        Ync = np.fft.fft2(Ync, axes=(1, 2), norm='ortho')

        #############################################
        #               Substracting the mean image
        #############################################
        wave, x_multi, y_multi = Ync.shape
        mean[:, 0] = mean[:, 0] * get_h_mean(self.PSF_file) # Applying the NirCam PSF to the mean
        Ync = np.reshape(Ync, (wave, x_multi * y_multi)) - np.dot(np.diag(Lm), mean)

        return Ync
#self = objet lui meme
#cls renvoie a la classe,  pas de self en static
#    def loading_datacubes(self, dataHS: str) -> np.array: # a mettre dans tools.py
    """
    Fetching the datacubes files from the given path

    : param dataHS: the path to the .fits hyperspectral image 
    : type data: str
    : return: np.array    
   
    """

    @property
    def Ync(self):
        return self.Ync

    @Ync.setter
    def Ync(self, value):
        self._Ync = value
#       YnirSpec = fits.getdata(HyperSpectral_Image)
#        return YnirSpec
#            @staticmethod
#     def make3d( data: np.array, mask, *args, **kwargs) -> np.array:
#         r"""
#         .. codeauthor:: Lina Issa - IRAP <lina.issa@irap.omp.eu>
#         Transform a 2d array into a 3d flatten array
#
#         : param data: input data
#         :type data: numpy array
#         :return: flatten array
#         :rtype: np.array
#         """
#         bands, pix1, pix2, = data.shape
#         x, y = np.arange(pix1), np.arange(pix2)
#         MX, MY = np.meshgrid(y, x)
#         X_cube = np.full((bands, pix1, pix2), np.nan)
#         if bands in data.shape:
#
#             if data.shape[0] == bands:
#                 for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[1])):
#                     X_cube[:, y, x] = data[:, z]
#             if data.shape[1] == bands:
#                 for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[0])):
#                     X_cube[:, y, x] = data[z, :]
#             return X_cube
#         else:
#             raise TypeError(
#                 f'The given number of bands {bands} is in conflict with the shape of the given data {X.shape[0], X.shape[1]}.')
#
#  #if mask is not None and not isinstance(mask, np.array): a mettre dans preprocess
#    raise TypeError(f'mask has type {type(mask)} but it must be a numpy array or a NoneType ')#