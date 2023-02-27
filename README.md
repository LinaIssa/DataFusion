[![Python](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://python.org)

# DataFusion 

## Context 

This fusion algorithm provides tools to perform data fusion for astronomical images in the context of JWST new data releases. 
More specifically it combines hyperspectral image from NirSpec with the multispectral image from NirCam instrument to produce a new image both spatially and spectrally resolved.
This algorithm is largery inspired by the fusion code developed by [Claire Guilloteau](https://github.com/cguilloteau/Fast-fusion-of-astronomical-images). 
The details og the fusion procedure can be found in [[1]](https://arxiv.org/abs/1912.11868) and in [[2]](https://ui.adsabs.harvard.edu/abs/2020AJ....160...28G/abstract).

In the repository you will find : 
- `main.py` : launch the fusion algorithm 
- `Cube.py` : pre-process the data images before the fusion 
- `Fusion.py` : gather the regularisation-dependant methods of fusion 
- `tools.py`  : inventory of small help functions. Imported in `Fusion.py` and `Cube.py` modules
- `create_ConfigFile.py` : create the config.yaml, a yaml configuration file with all the parameters. 


## How to use  

### Main code in `main.py` 
Before running `main.py`, you should create or update a configuration file config.yaml in which all the important 
parameters used in this code are stored. This configuration file can be created with the module `create_ConfigFile.py`
where we give the description of the parameters.
Then, the datacubes should be instantiated as Cube objects from `Cube.py` in which all the needed preprocessing takes 
place. 
 
By construction, regularisation methods are class objects with a set of attributes and methods. They can be found in 
`Fusion.py` . The chosen regularisation method needs to be instantiated so that the fusion code is performed with the 
embedded call method.

By default, this main will launch the weighted sobolev regularisation.  


### Data preprocessing in `Cube.py` 
Imported in `main.py` by:

```python
from Cube  import CubeHyperSpectral, CubeMultiSpectral
cubeHyperspectral = CubeHyperSpectral(YnirSpec, YnirCam,fact_pad,  downsampling,fluxConv, PSF_HS, lacp)
cubeHyperspectral(Lh)

cubeMultiSpectral = CubeMultiSpectral(YnirCam, fact_pad, PSF_MS)
cubeMultiSpectral(cubeHyperspectral, Lm)
```

### Fusion procedure in `Fusion.py` 
Imported in `main.py` by:

```python
from Fusion import Weighted_Sobolev_Regularisation
myFusion = Weighted_Sobolev_Regularisation(cubeMultiSpectral, cubeHyperspectral,
                                           Lm, Lh,
                                           PSF_MS, PSF_HS,
                                           nc, nr,
                                           outputDir,
                                           mu)

myFusion()
```

### Parameters created in `creat_CongigFile.py` 

This python script creates a configuration file with the help of PyYAML python library. Once a configuration file
 created,you only need to load it before launching the fusion algorithm.
 
## References:  

[1] : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon “Simulated JWST datasets for multispectral and hyperspectral image fusion” The Astronomical Journal, vol. 160, no. 1, p. 28, Jun. 2020.

[2] : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon "Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs – Application to High Dimensional Infrared Astronomical Imaging" IEEE Transactions on Computatonal Imaging, vol.6, Sept. 2020.
