import warnings
import numpy as np

from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

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
"Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs – Application to High Dimensional Infrared Astronomical Imaging"
IEEE Transactions on Computatonal Imaging, vol.6, Sept. 2020.
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

    ####################################################################################################################
    #                                               Data Loading & Preprocessing                                       #
    ####################################################################################################################

    datafiles                = {"hyper_image": config["hyper_image"],
                                "multi_image": config["multi_image"]}

    SpectralDegradationFiles = {"Lh" : config["LH"],
                                "Lm" : config["LM"]}

    PSF_files                = {"PSF_HS" : config["PSF_HS"],
                                "PSF_MS" : config["PSF_MS"]}

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

    if not Lm.shape[1] == Lh.shape[0] :
        raise TypeError(f'The given spectral operators  do not have matching size. The number of bands in the operator '
                        f'for the hyperspectral image ({Lh.shape[0]}) does not match that of the multi-spectral image '
                        f'({Lm.shape[1]}).')

    if not Lm.shape[0] == YnirCam.shape[0] or Lh.shape[0] == YnirSpec.shape[0]:
        raise TypeError(f'Either the number of bands in the multi-spectral image ({YnirCam.shape[0]}) does not '
                        f'correspond to that in the associated operator Lm ({Lm.shape[0]}) or the number of bands in'
                        f' the hyperspectral image ({YnirSpec.shape[0]}) is not in adequacy with that in the '
                        f'corresponding operator Lh ({Lh.shape[0]}).')




    l_h, pix1_h, pix2_h = YnirSpec.shape
    l_m, pix1_m, pix2_m = YnirCam.shape

    lacp         = config["lacp"]
    fact_pad     = config["fact_pad"]
    downsampling = config["downsampling"] # sous echantillonage
    fluxConv     = config["FLUXCONV_NC"]

    PSF_HS = PSF_files["PSF_HS"] # path to PSF_HS file
    PSF_MS = PSF_files["PSF_MS"] # path to PSF_MS file



    #if not PSF_HS_data.shape[2] == PSF_HS_data.shape[3] and PSF_MS_data.shape[2] == PSF_MS_data.shape[3]:
    #    raise TypeError(f' The given PSF a ')
    #if not PSF_HS_data.shape[2] == PSF_MS_data.shape[2]:
    #    raise TypeError

    if not config["nr"] * config["nc"] == PSF_MS_data.shape[2] * PSF_MS_data.shape[3]:
        raise ValueError(f'There is a discrepancy between the value of (nr, nc) and that of the spatial dimensions of '
                         f'the PSF that is {PSF_MS_data.shape[2]} x {PSF_MS_data.shape[3]}')


    cubeHyperspectral = CubeHyperSpectral(YnirSpec, YnirCam,
                                          fact_pad,  downsampling,
                                          fluxConv, PSF_HS, lacp)

    cubeHyperspectral(Lh)

    #np.savez(config["OutputDir"] + 'Z.npz', format=type(cubeHyperspectral.Z), data=cubeHyperspectral.Z)

    cubeMultiSpectral = CubeMultiSpectral(YnirCam, fact_pad, PSF_MS)

    cubeMultiSpectral(cubeHyperspectral, Lm)

    ####################################################################################################################
    #                                                       Data Fusion                                                #
    ####################################################################################################################

    mu        = config['mu']
    nr, nc    = config["nr"], config["nc"]
    outputDir = config["OutputDir"]

    myFusion = Weighted_Sobolev_Regularisation(cubeMultiSpectral, cubeHyperspectral,
                                    Lm, Lh,
                                    PSF_MS, PSF_HS,
                                    nc, nr,
                                    outputDir,
                                    mu, first_run=False)

    myFusion(save_it=False)


"""
    if first_run == False :
        answer = input("Are you using exactly the same images and the same PSFs of NirCam and NirSpec instruments as "
                       "for the first run? Answer YES or NO")
        if answer == "YES" :
            print("Great !")
            # TODO write the fusion call with A, B and C being loaded instead of computed

        if answer == "NO" :
            print("Then you should set first_run as True")
        else:
            print('The answer should be YES or NO')
"""

if __name__ == "__main__":
    config = load_config('config.yaml')
    image = main(config)
