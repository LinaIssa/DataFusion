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

        params = {"OutputDir" : str, "lacp": int, "mu" : int,'Spectral_Scope': list
                  }

    elif type == 'observation':

        params = {"InputDir": str,
                  "multi_image": str, "hyper_image": str,
                  'TableWave' : str,
                  "PSF_HS": str, "PSF_MS": str,
                  'NIRCam_Filters' : list, 'NIRSpec_Filters' : str,
                  'FLUXCONV_NC' : float, 'exp_time' : float, 'ConvConst' : int,
                  'NIRCam_SpectralResponses'  : str,
                  'NIRSpec_SpectralResponses' : str
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


def main(observation_config: dict, fusion_conf: dict):
    """
    :param obsercation_conf : the parameter dictionary related to the context of the observation and loaded automatically
     by running main.py
    :param fusion_conf      : the parameter dictionary that control the fusion algorithm

    :return Zfusion : the fusion product
    :return obj :  the objective function

    """
    print('Launching Fusion with Regularization Of Hyper and Multi-spectral imAGEs')

    ####################################################################################################################
    #                                               Data Loading & Preprocessing                                       #
    ####################################################################################################################

    # ----------- Retrieving Datafiles

    datafiles                = {"hyper_image": observation_config["hyper_image"],
                                "multi_image": observation_config["multi_image"]}

    SpectralDegradationFiles = {"NIRSpec" : observation_config["NIRSpec_SpectralResponses"],
                                "NIRCam"  : observation_config["NIRCam_SpectralResponses"]}

    PSF_files                = {"PSF_HS" : observation_config["PSF_HS"],
                                "PSF_MS" : observation_config["PSF_MS"]}

    PSF_HS = PSF_files["PSF_HS"] # path to PSF_HS file
    PSF_MS = PSF_files["PSF_MS"] # path to PSF_MS file

    spectral_scope = fusion_conf["Spectral_Scope"]

    tablewave   = fits.getdata(observation_config["TableWave"])
    YnirSpec    = fits.getdata(datafiles["hyper_image"])
    YnirCam     = fits.getdata(datafiles["multi_image"])
    Lh          = fits.getdata(SpectralDegradationFiles["NIRSpec"])
    Lm          = fits.getdata(SpectralDegradationFiles["NIRCam"])
    PSF_HS_data = fits.getdata(PSF_files["PSF_HS"])
    PSF_MS_data = fits.getdata(PSF_files["PSF_MS"])

    print(f'YnirCam shape: {YnirCam.shape}')
    print(f'YnirSpec shape: {YnirSpec.shape}')

    print(f'Lm shape: {Lm.shape}')
    print(f'Lh shape: {Lh.shape}')


    if not isinstance(YnirSpec, np.ndarray):
        raise TypeError(f'The hyperspectral image stored in {observation_config["hyper_image"]} could not be stored in a numpy '
                        f'array')

    if not isinstance(YnirCam, np.ndarray):
        raise TypeError(f'The multi-spectral image stored in {observation_config["multi_image"]} could not be stored in a numpy '
                        f'array')

    if Lm.shape[0] >= Lh.shape[0] :
        raise TypeError(f' Lm operator has more bands (lm = {Lm.shape[0]}) than Lh (lh = {Lh.shape[0]})')

    if not Lm.shape[1] == Lh.shape[0] :

        raise ValueError(f'You should interpolate Lm into the spectrzl resolution of the hyperspectral image '
                         f' that is {Lh.shape[1]} ')

    if Lm.shape[0] != YnirCam.shape[0] or Lh.shape[0] != YnirSpec.shape[0]:
        raise TypeError(f'Either the number of bands in the multi-spectral image ({YnirCam.shape[0]}) does not '
                        f'correspond to that in the associated operator Lm ({Lm.shape[0]}) or the number of bands in'
                        f' the hyperspectral image ({YnirSpec.shape[0]}) is not in adequacy with that in the '
                        f'corresponding operator Lh ({Lh.shape[0]}).')

    # ----------- Constructing the spectral operators  --------------------------------
     #if spectral_scope != [] : #TODO: How to transform data to take into account the spectral range
     #   wave1, wave2 = spectral_scope[0], spectral_scope[1]
     #   if wave1 < tablewave[1] and wave2 > tablewave[1]:
     #       a, b = np.where(tablewave>wave1)[0][0], np.where(tablewave<wave2[0][-1])
     #       spectral_window = tablewave[a:b]

    # ----------- Retrieving Parameters from the config file --------------------------------

    mu       = fusion_conf['mu']
    lacp     = fusion_conf["lacp"]
    fluxConv = observation_config["FLUXCONV_NC"]

    # ----------- Defining some factors from the retrieved data --------------------------------

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


    # ---------------- Datacubes preprocessing  ----------------
    cubeHyperspectral = CubeHyperSpectral(YnirSpec, YnirCam,
                                          fact_pad,  downsampling,
                                          fluxConv, PSF_HS, lacp)

    cubeHyperspectral(Lh)

    cubeMultiSpectral = CubeMultiSpectral(YnirCam, fact_pad, PSF_MS)

    cubeMultiSpectral(cubeHyperspectral, Lm)

    ####################################################################################################################
    #                                                       Data Fusion                                                #
    ####################################################################################################################

    outputDir = fusion_conf["OutputDir"]
    outputFile = fusion_conf['OutputFileName']

    myFusion = Weighted_Sobolev_Regularisation(cubeMultiSpectral, cubeHyperspectral,
                                    Lm, Lh,
                                    PSF_MS, PSF_HS,
                                    nc, nr,
                                    outputDir, outputFile,
                                    mu, first_run=False)

    myFusion(save_it=True)


if __name__ == "__main__":
    fusion_config = load_config('fusion_config.yaml')
    obs_config = load_config('observation_config.yaml', type='observation')

    image = main(obs_config, fusion_config)
