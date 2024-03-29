#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday December 13th
@author: Lina Issa
lina.issa@irap.omp.eu

This python script creates a configuration file with the help of PyYAML python library. Once a configuration file
created,you only need to load it before launching the fusion algorithm using the load_config() function defined in
main.py.

The configuration parameters are split into two files : fusion_config.yaml and observation_config.yaml.

In fusion_config.yaml are gathered the parameters that control the fusion algorithm itself:

- OutputDir             : The directory in which the outputfiles are saved
- OutputFileName        : name of the outputfile name
- Regularisation_Method : type of the regularisation for the fusion algorithm
- Spectral_Scope        : the spectral window in which the fusion is performed
- lacp            : The spectral dimension of the reduced hyperspectral image as retrieved by a PCA.
- mu              : the regularisation strength

In observation_config.yaml are gathered the parameter related to the context of acquisition of the data images:

- InputDir        : The directory in which the data files are stored
- PSF_HS          : ?? A .fits file used to get the g band    --> self-consistency problem
- PSF_MS          : ?? A .fits file used to get the h band    --> self-consistency problem
- multi_image     : The path to the NIRCam image stored as a .fits file
- hyper_image     : The path to the NIRSpec image stored as a .fits file
- TableWave       : The spectral resolution to interpolate the multispectral image into. Defined from the hyperspectral
 image.
- LM              : Spectral degradation operator for the multispectral. The shape should be (number of bands in the
multispectral image, number of bands in the hyperspectral image). Used in get_weights function. In the rows of LM are
stored the transmission functions of the different filters.
- LH              : Spectral degradation operator for the hyperspectral image. The shape should be (number of bands in
the hyperspectral image, number of bands in the hyperspectral image). It is a diagonal matrix with the spectral
transmission function of NirSpec. Used in the choose_subspace routine and in get_weights function
- FLUXCONV_NC     : conversion parameter to convert the number of electrons into a physical unit (to be updated)
-exp_time         : exposure time in seconds for NIRSpec
-ConvConst        : constant for the pce conversion used to get the spectral response in Spectral_Responses_filters.py


************************************************* From Constants.py *****************************************************
- nr              : number of pixels in the PSFs
- nc              : number of pixels in the PSFs
- fact_pad        : padding factor
- downsampling    : downsampling factor of the downsizing operator involved in the spatial deformation of the images
 (sous-échantillonnage). Should be an integer
*       High spatial resolution image size with padding and padding factor                                              *
*       Examples :                                                                                                      *
*       Original MS band size : 300x300                                                                                 *
*       fact_pad : 41
*       d        : 3
*       nr = 300 + 2*fact_pad + 2                                                                                       *
*       nc = 300 + 2*fact_pad + 2

*       Original MS band size : 90x90                                                                                   *
*       fact_pad : 44
*       d        : 6
*       nr = 90 + 2*fact_pad + 2                                                                                        *
*       nc = 90 + 2*fact_pad + 2
*       Conditions (to chose fact_pad) : nr % d = 0; nc % d = 0; (2*fact_pad+2) % d = 0  and (fact_pad//d + 1)%2 =0     *
*************************************************************************************************************************

"""

from yaml import dump
import os.path as opath

DataDir = '/Users/lina/Documents/Thesis/MyFusionProject/Input/Data_1'

    ######################################################################################
    #    Creating a dictionary of parameters regarding the context of the observation    #
    ######################################################################################

context_of_observation = {
    "InputDir"     : DataDir ,
    "PSF_HS"       : opath.join(DataDir, 'PSF_HS.fits'),
    "PSF_MS"       : opath.join(DataDir, 'PSF_MS.fits'),
    "multi_image"  : opath.join(DataDir, 'NIRCam_cube.fits'),
    "hyper_image"  : opath.join(DataDir, 'NIRSpec_subcube.fits'),
    "TableWave"  :  opath.join(DataDir, 'tabwave.fits'),
    "NIRCam_Filters"  : ['f115w', 'f140m', 'f150w', 'f150w2', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f210m', 'f212n'],
    "NIRSpec_Filters" : 'f100lp',
    "FLUXCONV_NC"  : 0.031**2,
    "exp_time"     :  128.841,
    "ConvConst"    : 205000
     }
# Parameters discarded : "downsampling": 3 , "nr": 204, "nc": 204, "fact_pad" : 41

    ######################################################################################
    #        Creating a dictionary of parameters regarding the fusion algorithm          #
    ######################################################################################

context_of_fusion  = {
    "OutputDir"             : '/Users/lina/Documents/MyFusion/Output',
    "OutputFileName"        : 'Zfusion_SobolevReg',
    "Regularisation_Method" : 'SobolevReg',
    "lacp"                  : 10,
    "mu"                    : 100,
    "Spectral_Scope"        : []
     }

    #################################################
    #    Dumping the dico into a config.yaml file   #
    #################################################

with open('fusion_config.yaml', 'w+') as f:
    f.write(dump(context_of_fusion))

with open('observation_config.yaml', 'w+') as f:
    f.write(dump(context_of_observation))