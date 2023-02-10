import warnings
import numpy as np

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from Cube   import CubeHyperSpectral, CubeMultiSpectral
from Fusion import Weighted_Sobolev_Reg
from astropy.io import fits


warnings.filterwarnings('ignore')
"""
@author: Lina Issa, adapted from Claire Guilloteau's FRHOMAGE algorithm.

Before running main.py, you should create or update a configuration file config.yaml in which all the important 
parameters used in this code are stored. This configuration file can be created with the module create_ConfigFile.py 
where we give the description of the parameters.

Running the main function will launch the fusion procedure accordingly to the chosen regularisation.  
"""

def load_config(filename):
    """
    Importing the yaml configuration file and returns it as a python dictionary.
    :param filename : The filename of the configuration file
    :return: a python dictionary containing all the important parameter for the fusion code.
    """
    with open(filename) as f:
        stream = f.read()
    config = load(stream, Loader=Loader)
    params = {"DataDir": str, "OutputDir": str, "PSF_HS": str, "PSF_MS": str, "multi_image": str, "hyper_image": str,
              "LM": str, "LH": str, "nr": int, "nc": int, "fact_pad": int, "FLUXCONV_NC": float, "downsampling": int,
              "lacp": int  # , "WaveData": str, "CorrelationData": str,
              }
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

    ##############################################
    #              Data Loading                  #
    ##############################################

    datafiles = {"hyper_image": config["hyper_image"],
                 "multi_image": config["multi_image"]}
    SpectralDegradationFiles = {"Lh" : config["LH"],
                              "Lm" : config["LM"]}

    YnirSpec = fits.getdata(datafiles["hyper_image"])
    YnirCam = fits.getdata(datafiles["multi_image"])
    Lh = fits.getdata(SpectralDegradationFiles["Lh"])
    Lm = fits.getdata(SpectralDegradationFiles["Lm"])
    # TODO sanity check on the dimension of the images
    # TODO sanity check on the dimensions of Lm and Lh
    if not isinstance(YnirSpec, np.ndarray):
        raise TypeError(f'The hyperspectral image stored in {config["hyper_image"]} could not be stored in a numpy '
                        f'array')
    if not isinstance(YnirCam, np.ndarray):
        raise TypeError(f'The multi-spectral image stored in {config["multi_image"]} could not be stored in a numpy '
                        f'array')
    l_h, pix1_h, pix2_h = YnirSpec.shape
    l_m, pix1_m, pix2_m = YnirCam.shape
    lacp = config["lacp"]
    fact_pad = config["fact_pad"]
    downsampling = config["downsampling"] # sous echantillonage
    fluxConv     = config["FLUXCONV_NC"]
    PSF_HS = config["PSF_HS"]
    PSF_MS = config["PSF_MS"]


    cubeHyperspectral = CubeHyperSpectral(YnirSpec, YnirCam, fact_pad,  downsampling, fluxConv, PSF_HS, lacp)
    cubeHyperspectral(Lh)

    cubeMultiSpectral = CubeMultiSpectral(YnirCam, fact_pad, PSF_MS)
    cubeMultiSpectral(cubeHyperspectral, Lm)

    ##############################################
    #              Data Fusion                   #
    ##############################################
    nr, nc = config["nr"], config["nc"]

    myFusion = Weighted_Sobolev_Reg(cubeMultiSpectral, cubeHyperspectral, Lm, Lh, PSF_MS, PSF_HS, nc, nr)
    myFusion()

if __name__ == "__main__":
    config = load_config('config.yaml')
    image = main(config)
