import warnings
import numpy as np

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from tools import cropping_Lm
from Cube   import CubeHyperSpectral, CubeMultiSpectral
from Fusion import Weighted_Sobolev_Regularisation
from astropy.io import fits


warnings.filterwarnings('ignore')
"""
@author: Lina Issa, adapted from Claire Guilloteau's FRHOMAGE algorithm.

Before running main.py, you should create or update a configuration file config.yaml in which all the important 
parameters used in this code are stored. This configuration file can be created with the module create_ConfigFile.py 
where we give the description of the parameters.
Then, the datacubes should be instantiated as Cube objects from Cube.py in which all the needed preprocessing takes 
place. 
 
By construction, regularisation methods are class objects with a set of attributes and methods. They can be found in 
Fusion.py. The chosen regularisation method needs to be instantiated so that the fusion code is performed with the 
embedded call method.

By default, this main will launch the weighted sobolev regularisation.  

Ref 1 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
“Simulated JWST datasets for multispectral and hyperspectral image fusion”
The Astronomical Journal, vol. 160, no. 1, p. 28, Jun. 2020.
Ref 2 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
"Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs – Application to High Dimensional 
Infrared Astronomical Imaging"
IEEE Transactions on Computatonal Imaging, vol.6, Sept. 2020.


"""

def load_config(filename, type : str = 'fusion' ):
    """
    Importing the yaml configuration file and returns it as a python dictionary.
    :param filename : The filename of the configuration file

    In observation_config.yaml we gather all the parameters related to the context of acquisition of the images.
    In fusion_config.yaml, we store paths to data and all parameters that dictate the fusion algorithm.
    :return: a python dictionary containing all the important parameter for the fusion code.

    """

    with open(filename) as f:
        stream = f.read()
    config = load(stream, Loader=Loader)

    if type == 'fusion':

        params = {"OutputDir": str, "PSF_HS": str, "PSF_MS": str, "multi_image": str, "hyper_image": str,
              "LM": str, "LH": str,
              "lacp": int, 'NIRCam_Filters' : list, 'Spectral_Scope' : list
              }

    elif type == 'observation':

        params = {'FLUXCONV_NC' : float, 'exp_time' : float, 'ConvConst' : int
               }

    else :

        raise ValueError(f'You provided {type} as type while it should have been either fusion or observation ')


    for va in params.keys():

        if va not in config.keys():
            raise IndexError(f"{va} is missing in the configuration file")

        typ = params[va]

        if not isinstance(config[va], typ):  # noqa
            raise TypeError(f"{va} given in the configuration file as {type(config[va])} but it should be {params[va]}")

    return config


def main(config: dict):
    """
    :param config : the parameter dictionary loaded automatically by running main.py

    :return Zfusion : the fusion product
    :return obj :  the objective function
    """
    print('Launching Fusion with Regularization Of Hyper and Multi-spectral imAGEs')

    ####################################################################################################################
    #                                               Data Loading & Preprocessing                                       #
    ####################################################################################################################

    # ----------- Retrieving Datafiles

    datafiles                = {"hyper_image": config["hyper_image"],
                                "multi_image": config["multi_image"]}

    SpectralDegradationFiles = {"Lh" : config["LH"],
                                "Lm" : config["LM"]}

    PSF_files                = {"PSF_HS" : config["PSF_HS"],
                                "PSF_MS" : config["PSF_MS"]}

    PSF_HS = PSF_files["PSF_HS"] # path to PSF_HS file
    PSF_MS = PSF_files["PSF_MS"] # path to PSF_MS file

    YnirSpec    = fits.getdata(datafiles["hyper_image"])
    YnirCam     = fits.getdata(datafiles["multi_image"])
    Lh          = fits.getdata(SpectralDegradationFiles["Lh"])
    Lm          = fits.getdata(SpectralDegradationFiles["Lm"])
    PSF_HS_data = fits.getdata(PSF_files["PSF_HS"])
    PSF_MS_data = fits.getdata(PSF_files["PSF_MS"])

    print(f'YnirCam shape: {YnirCam.shape}')
    print(f'YnirSpec shape: {YnirSpec.shape}')

    print(f'Lm shape: {Lm.shape}')
    print(f'Lh shape: {Lh.shape}')


    if not isinstance(YnirSpec, np.ndarray):
        raise TypeError(f'The hyperspectral image stored in {config["hyper_image"]} could not be stored in a numpy '
                        f'array')

    if not isinstance(YnirCam, np.ndarray):
        raise TypeError(f'The multi-spectral image stored in {config["multi_image"]} could not be stored in a numpy '
                        f'array')

    if Lm.shape[0] >= Lh.shape[0] :
        raise TypeError(f' Lm operator has more bands (lm = {Lm.shape[0]}) than Lh (lh = {Lh.shape[0]})')

    if not Lm.shape[1] == Lh.shape[0] :

        if Lm.shape[1] > Lh.shape[0] :
            Lm = cropping_Lm(Lm, (Lm.shape[0], Lh.shape[0]))
        else :
            raise ValueError(f'You should build Lh with a number of bands at most equal to the total number of filters '
                             f'in Lm.fits {Lm.shape[1]} ')

    if Lm.shape[0] != YnirCam.shape[0] or Lh.shape[0] != YnirSpec.shape[0]:
        raise TypeError(f'Either the number of bands in the multi-spectral image ({YnirCam.shape[0]}) does not '
                        f'correspond to that in the associated operator Lm ({Lm.shape[0]}) or the number of bands in'
                        f' the hyperspectral image ({YnirSpec.shape[0]}) is not in adequacy with that in the '
                        f'corresponding operator Lh ({Lh.shape[0]}).')

    # ----------- Retrieving Parameters from the config file

    mu       = config['mu']
    lacp     = config["lacp"]
    fluxConv = config["FLUXCONV_NC"]

    # ----------- Defining some factors from the retrieved data

    nr, nc       = PSF_HS_data.shape[2], PSF_HS_data.shape[3]
    fact_pad     = (nr - (YnirCam.shape[-2]+2))/2
    downsampling = YnirCam.shape[-1]/YnirSpec.shape[-1]

    if not isinstance(downsampling, int):
        downsampling = int(downsampling)

    if not isinstance(fact_pad, int):
        fact_pad = int(fact_pad)

    if not nr * nc == PSF_MS_data.shape[2] * PSF_MS_data.shape[3]:
        raise ValueError(f'There is a discrepancy between the value of (nr, nc) and that of the spatial dimensions of '
                         f'the PSF that is {PSF_MS_data.shape[2]} x {PSF_MS_data.shape[3]}')


    # -------- Datacubes preprocessing
    cubeHyperspectral = CubeHyperSpectral(YnirSpec, YnirCam,
                                          fact_pad,  downsampling,
                                          fluxConv, PSF_HS, lacp)

    cubeHyperspectral(Lh)

    cubeMultiSpectral = CubeMultiSpectral(YnirCam, fact_pad, PSF_MS)

    cubeMultiSpectral(cubeHyperspectral, Lm)

    ####################################################################################################################
    #                                                       Data Fusion                                                #
    ####################################################################################################################

    outputDir = config["OutputDir"]

    myFusion = Weighted_Sobolev_Regularisation(cubeMultiSpectral, cubeHyperspectral,
                                    Lm, Lh,
                                    PSF_MS, PSF_HS,
                                    nc, nr,
                                    outputDir,
                                    mu, first_run=False)

    myFusion(save_it=True)


if __name__ == "__main__":
    config = load_config('config.yaml')
    image = main(config)
