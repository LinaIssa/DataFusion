#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday December 13th
@author: Lina Issa
lina.issa@irap.omp.eu

This python script creates a configuration file with the help of PyYAML python library. Once a configuration file
 created,you only need to load it before launching the fusion algorithm.
Meaning of the parameters:

- DataDir         : The directory in which the data files are stored
- OutputDir       : The directory in which the outputfiles are saved
- PSF_HS          : ?? A .fits file used to get the g band    --> self-consistency problem
- PSF_MS          : ?? A .fits file used to get the h band    --> self-consistency problem
- multi_image     : The path to the NIRCam image stored as a .fits file
- hyper_image     : The path to the NIRSpec image stored as a .fits file
- LM              : Spectral degradation operator for the multispectral. The shape should be (number of bands in the
multispectral image, number of bands in the hyperspectral image). Used in get_weights function. In the rows of LM are
stored the transmission functions of the different filters.
- LH              : Spectral degradationoperator for the hyperspectral image. The shape should be (number of bands in
the hyperspectral image, number of bands in the hyperspectral image). It is a diagonal matrix with the spectral
transmission function of NirSpec. Used in the choose_subspace routine and in get_weights function
- nr              : number of pixels in the PSFs
- nc              : number of pixels in the PSFs
- fact_pad        : padding factor
- FLUXCONV_NC     : ??
- downsampling    : downsampling factor of the downsizing operator involved in the spatial deformation of the images (sous-échantillonnage). Should be an integer
- lacp            : The spectral dimension of the reduced hyperspectral image as retrieved by a PCA.
- mu              : the regularisation strength

************************************************* From Constants.py *****************************************************
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

DataDir = '/Users/lina/Documents/MyFusion/Input/simulated_data'

    ###########################################
    #    Creating a dictionary of parameters  #
    ###########################################

data = {
    "DataDir"         : DataDir ,
    "OutputDir"       : '/Users/lina/Documents/MyFusion/Output',
    "PSF_HS"          : opath.join(DataDir, 'H_fft.fits'),
    "PSF_MS"          : opath.join(DataDir, 'G_fft.fits'),
    "multi_image"     : opath.join(DataDir, 'NIRCam_multi.fits'),
    "hyper_image"     : opath.join(DataDir, 'NIRSpec_subcube.fits'),
    "LM"              : opath.join(DataDir, 'Lm.fits'),
    "LH"              : opath.join(DataDir, 'Lh.fits'),
    "nr"              : 204,
    "nc"              : 204,
    "fact_pad"        : 41,
    "FLUXCONV_NC"     : 0.031**2,
    "downsampling"    : 3 ,
    "lacp"            : 10,
    "mu"              : 10
     }


    #################################################
    #    Dumping the dico into a config.yaml file   #
    #################################################

with open('fusion_config.yaml', 'w+') as f:
    f.write(dump(data))

