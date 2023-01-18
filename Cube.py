import numpy as np
from skimage.transform import resize
from abc import ABC, abstractmethod
from typing import Union
from scipy import linalg
from sklearn.utils.extmath import svd_flip
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
        def loading_data(self, data: str):
            return


class CubeHyperSpectral(Cube):
    def __init__(self,data:np.array, mask: np.array =None, **kwargs) -> None:
        super().__init__(data, mask, **kwargs)

    def loading_data(self, dataHS: str) -> np.array:
    """
    Fetching the datacubes files from the given path

    : param dataHS: the path to the .fits hyperspectral image 
    : type data: str
    : return: np.array    
   
    """
        YnirSpec = fits.getdata(HyperSpectral_Image)
        return

    @staticmethod
    def downsizing(YnirSpec: np.array, d: int, m: int, n : int, *args, **kwargs) -> np.array :
        """
       : param YnirSpec: the hyperspectral image 
       : param d : the downsizing factor given in the config file
       : param m : first  spatial dimension of the Multispectral image
       : param n : second spatial dimension of the Multispectral image
       : returns: downsized np.array
       
       """
        YnirSpec = resize(YnirSpec, (YnirSpec.shape[0],m//d, n//d),order=3, mode='symmetric')*(d**2*FLUXCONV_NC)
        return YnirSpec

    def PCA_projection(self, YnirSpec: np.array,YnirCam: np.array, Lh: np.array, lacp: int ) -> Union[np.array, np.array, np.array]:
        """
       : param YnirSpec: the hyperspectral image
       : param YnirCam : the multispectral image
       : param Lh      : the spectral operator retrieved from LH.fits
       : param lacp    : number of dimension for the reduced spectral space.
       : returns       : V - the PCA projection matrix, Z - the retroprojected datacube, mean - the centered data

       """
        print(' PCA on the HS image : ')
        #############################################
        #               Depliage de YnirSpec
        #############################################

        l, m, n = YnirSpec.shape
        X = np.reshape(np.dot(np.diag(Lh**-1), np.reshape(Yns, (l, m*n))), (l, m, n))

        #############################################
        #               PCA Projection
        #############################################

        X_mean = np.mean(X.T, axis=0)
        X-=X_mean
        U, S, V = linalg.svd(X.T, full_matrices=False)
        U, V = svd_flip(U, V)
        S = S[:lacp]

        #############################################
        #               PCA Decomposition
        #############################################

        Z = U[:, :nb_comp] * (S ** (1 / 2))
        V = np.dot(np.diag(S ** (1 / 2)), V[:nb_comp])

        #############################################
        #              Retroprojection du cube  Z
        #############################################

        Z = np.reshape(Z.T, (lacp, m, n))

        #############################################
        #              Upsampling du cube  Z
        #############################################

        Z    = self._upsampling(Z, m, n)
        mean = self._meanSpectrumFourier(X_mean, Z)

        return V.T, Z, mean


    @staticmethod
    def  _upsampling(Z: np.array, m: int, n: int)-> np.array:
        """
        Performs a bi-cubic interpolation under symmetric boundaries conditions
        : param Z: the PCA retroprojected datacube
        : param m : first  spatial dimension of the Multispectral image
        : param n : second spatial dimension of the Multispectral image
        : returns : the resized and interpolated datacube Z_interpol
        """
        Z_resized = resize(Z, (n_comp, m, n), order=3, mode='symmetric')
        Z_interpol = np.fft.fft2(tools.compute_symmpad_3d(Z_resized, fact_pad), norm='ortho')  # Symmetric boundaries conditions
        return Z_interpol


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

    def preprocess(self):
        return

class CubeMultisSectral(Cube):

    def __init__(self,data:np.array, mask: np.array =None, **kwargs) -> None:
        super().__init__(data, mask, **kwargs) # appelle classe mere
        # truc supplementaires

    def loading_data(self, dataMS: str) -> np.array :
    """
    : param dataHS: the path to the .fits multispectral image 
    : type data: str
    : return: np.array    

    """
    YnirCam  = fits.getdata(MultiSpectral_Image)
    return

    def preprocess(self):
        return
#self = objet lui meme
#cls renvoie a la classe,  pas de self en static
