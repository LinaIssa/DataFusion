import numpy as np
from skimage.transform import resize
from abc import ABC, abstractmethod
from typing import Union
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from tools import compute_symmpad_3d,  get_g_mean, aliasing
"""
@author: Lina Issa 

We define here an abstract cube class that gathers all the pre-processing and post-processing methods used in the fusion framework. These methods apply to the hyperspectral and multispectral data cubes and are designed to be applied to JWST datacubes 
"""
class Cube(ABC): #si heritage mettre le nom de l heritage ceci est une classe abstraite
    '''
    Abstract class that contains the datacube and some fundamental operations
    '''

        def __init__(self, data:np.array, mask: np.array =None, **kwargs) -> None:
            if not isinstance(data, np.array):
                raise TypeError(f'data has type {type(data)} but it must be a numpy array ')
            self.data = data
            self.mask = mask
            self.dim  = data.shape

            # Flatten version of the data
            self.dataflat = self.flatten(data)

        @staticmethod
        def flatten( data: np.array, *args, **kwargs) -> np.array:
            r"""
            .. codeauthor:: Lina Issa - IRAP <lina.issa@irap.omp.eu>
            Transform a 3d array into a 2d flatten array

            : param data: input data
            :type data: numpy array
            :return: flatten array
            :rtype: np.array
            """
            x, y = np.arange(pix1), np.arange(pix2)
            MX, MY = np.meshgrid(y, x)
            X_cube = np.full((bands, pix1, pix2), np.nan)
            if bands in X.shape:

                if X.shape[0] == bands:
                    for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[1])):
                        X_cube[:, y, x] = X[:, z]
                if X.shape[1] == bands:
                    for x, y, z in zip(MX[~mask], MY[~mask], range(X.shape[0])):
                        X_cube[:, y, x] = X[z, :]
                return X_cube
            else:
                raise TypeError(
                    f'The given number of bands {bands} is in conflict with the shape of the given data {X.shape[0], X.shape[1]}.')

        @abstractmethod
        def postprocess(self, *args, **kwargs):
            return
        @abstractmethod
        def preprocess(self, *args, **kwargs):
            return
        @abstractmethod
        def loading_datacubes(self, data: str):
            return


class CubeHyperSpectral(Cube):
    def __init__(self,data:np.array, mask: np.array =None, **kwargs) -> None:
        super().__init__(data, mask, **kwargs)

    def loading_datacubes(self, dataHS: str) -> np.array:
    """
    Fetching the datacubes files from the given path

    : param dataHS: the path to the .fits hyperspectral image 
    : type data: str
    : return: np.array    
   
    """
        YnirSpec = fits.getdata(HyperSpectral_Image)
        return YnirSpec

    @staticmethod
    def downsizing(YnirSpec: np.array, d: int, x_ms: int, y_ms : int, *args, **kwargs) -> np.array :
        """
       : param YnirSpec: the hyperspectral image 
       : param d : the downsizing factor given in the config file
       : param x_ms : first  spatial dimension of the Multispectral image
       : param y_ms : second spatial dimension of the Multispectral image
       : returns: downsized np.array
       
       """
        YnirSpec = resize(YnirSpec, (YnirSpec.shape[0],x_ms//d, y_ms//d),order=3, mode='symmetric')*(d**2*FLUXCONV_NC)
        return YnirSpec

    def initialisation_HS(self, YnirSpec: np.array, YnirCam: np.array, Lh: np.array, lacp: int, fact_pad: int) -> Union[np.array, np.array, np.array]:
        """
       : param YnirSpec: the hyperspectral image
       : param YnirCam : the multispectral image
       : param Lh      : the spectral operator retrieved from LH.fits
       : param lacp    : number of dimension for the reduced spectral space.
       : param fact_pad: padding factor from the config file
       : returns       : V - the PCA projection matrix, Z - the datacube prepared for the initialisation, mean - the centered data

       """
        print(' PCA on the HS image : ')
        #############################################
        #              Flattening  YnirSpec
        #############################################

        z, x_H, y_H = YnirSpec.shape
        x_M, y_M    = YnirCam.shape[1],  YnirCam.shape[2]
        X           = np.reshape(np.dot(np.diag(Lh**-1), np.reshape(Yns, (z, x_H*y_H))), (z, x_H, y_H))

        #############################################
        #              PCA projection of  Z
        #############################################

        V, Z, mean = self._pca_projection(X, lacp)

        #############################################
        #              Retroprojection of  Z
        #############################################

        Z = np.reshape(Z.T, (lacp, x_H, y_H))

        #############################################
        #              Upsampling and FFT of  Z
        #############################################

        Z    = self._upsampling(Z, x_M, y_M)
        Z    = compute_symmpad_3d(Z,fact_pad)  # applies symmetric padding to the datacube with the help of tools.compute_symmpad_3d
        Z    = np.fft.fft2(Z, norm='ortho')    # applies symmetric boundaries condtions to the upsampled hyperspectral datacubes

        mean = self._meanSpectrumFourier(X_mean, Z)

        return V, Z, mean

    @staticmethod
    def _pca_projection(X: np.array, lacp: int ) ->  Union[np.array, np.array, np.array]:
        """
        Performs a PCA decomposition on the hyperspectral image in order to retrieve the matrixes V, Z and mean.
        The PCA is applied for preprocessing the hyperspectral datacubes.
        :param X   : the flattenned hyperspectral datacubes
        :param lacp:  number of dimension for the reduced spectral space.
        :return    :  V - the PCA projection matrix, Z- the projected cube,  mean - the centered data
        """

    #############################################
    #               PCA Projection
    #############################################

        X_mean = np.mean(X.T, axis=0)
        X -= X_mean
        U, S, V = linalg.svd(X.T, full_matrices=False)
        U, V = svd_flip(U, V)
        S = S[:lacp]


    #############################################
    #               PCA Decomposition
    #############################################

        Z = U[:, :lacp] * (S ** (1 / 2))
        V = np.dot(np.diag(S ** (1 / 2)), V[:lacp])

        return V.T, Z, mean

    @staticmethod
    def  _upsampling(Z: np.array, x_M: int, y_M: int)-> np.array:
        """
        Performs a bi-cubic interpolation under symmetric boundaries conditions
        : param Z   : the PCA retroprojected datacube
        : param x_M : first  spatial dimension of the Multispectral image
        : param y_M : second spatial dimension of the Multispectral image
        : returns   : the resized and interpolated datacube Z_interpol
        """
        Z_upsampled = resize(Z, (n_comp, x_M, y_M), order=3, mode='symmetric')
        return Z_upsampled


# not sure about about keeping these two ...
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

    def preprocess(YnirSpec: np.array, YnirCam: np.array, Lh: np.array, mean: np.array, fact_pad: int, d: int) -> np.array :
        """
        Preprocessing for the hyperspectral image only. Needs the mean form the PCA decomposition.

        :param YnirSpec: the hyperspectral image
        :param YnirCam : the multispectral image
        :param Lh: the spectral operator retrieved from LH.fits
        :param mean: from the PCA projection on the hyperspectral datacubes
        :return: the fusion-ready hyperspectral image
        """
        print(' Operators and data preprocessing : ')

        z, x_H, y_H = YnirSpec.shape
        x_M, y_M    = YnirCam.shape[1], YnirCam.shape[2]  # only used for the aliasing

        #############################################
        #               FFT on YnirSpec
        #############################################

        Yns = compute_symmpad_3d(YnirSpec,  fact_pad//d+1)
        Yns = np.fft.fft2(Yns[:, :-2, :-2], axes=(1, 2), norm='ortho')

        #############################################
        #               Substracting the mean image
        #############################################

        mean[:, 0] = mean[:, 0] * get_g_mean() # Applying the NirSpec PSF to the mean
        Yns        = np.reshape(Yns, (z, x_H * y_H)) - np.dot(np.diag(Lh), aliasing(mean_, (z, x_M, y_M)))
        return Yns

class CubeMultisSectral(Cube):

    def __init__(self,data:np.array, mask: np.array =None, **kwargs) -> None:
        super().__init__(data, mask, **kwargs) # appelle classe mere
        # truc supplementaires

    def loading_datacubes(self, dataMS: str) -> np.array :
    """
    :param dataHS: the path to the .fits multispectral image
    :type data: str
    :return: np.array

    """
        YnirCam  = fits.getdata(MultiSpectral_Image)
        return YnirCam

    def preprocess(YnirCam: np.array, Lm: np.array, mean: np.array, fact_pad: int) -> np.array:
        """
        Performs preprocessing for the multispectral image. Needs the mean from the PCA decomposition performed on the hyperspectral image.
        :param YnirCam : the multispectral image
        :param Lm: the spectral operator retrieved from LM.fits
        :param mean: from the PCA projection on the hyperspectral datacubes
        :param fact_pad : padding factor
        :return: the fusion-ready multispectral image
        """
        print(' Operators and data preprocessing : ')

        z, x, y = YnirCam.shape

        #############################################
        #               FFT on YnirSpec
        #############################################
        Ync = compute_symmpad_3d(YnirCam,  fact_pad)
        Ync = np.fft.fft2(Ync, axes=(1, 2), norm='ortho')

        #############################################
        #               Substracting the mean image
        #############################################
        mean[:, 0] = mean[:, 0] * get_h_mean() # Applying the NirSpec PSF to the mean
        Ync = np.reshape(Ync, (z, x * y)) - np.dot(np.diag(Lm), mean)

        return Ync
#self = objet lui meme
#cls renvoie a la classe,  pas de self en static
