import numpy as np
import os.path as opath
from main       import load_config
from astropy.io import fits
from pandeia.engine.instrument_factory import InstrumentFactory
from yaml import load, safe_dump
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
'''
@author: Lina Issa, adapted from Claire Guilloteau's FRHOMAGE algorithm.

According to the instrumental filters selected in the configuration file, this script generates the spectral responses 
for NIRCam and NIRSpec using pandeia engine. 

'''
# ----------------------------------------------------------------------------------------------------------------------
#                                Get pce from pandeia engine for NIRSpec & NIRCam
#----------------------------------------------------------------------------------------------------------------------
def get_NIRCam_filters(filters_list : list, spectral_resampling : np.array, instrument_config: dict):
    '''
    Computes spectral degradation operator Lm of the instrument with unit convestion (mjy/arcsec^2 to e-)
    :param filters_list : the list of the NIRCam filters selected for the fusion. Defined in the fusion_config.yaml file
    :param spectral_resampling : the spectral resolution from NIRSpec on which the NIRCam spectral response is resampled
    :param instrument_config : the loaded instrument config file in which the parameters related to the context of the
    observation are stored.
    return : the operator Lm
    '''
    FLUXCONV_NC = instrument_config['FLUXCONV_NC']
    exp_time    = instrument_config['exp_time']
    cste_conv   = instrument_config['ConvConst']

    spectral_resampling = np.reshape(spectral_resampling, (spectral_resampling.shape[0]))

    short_wave_list = ['f115w', 'f140m', 'f150w', 'f150w2', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f210m',
                       'f212n']
    long_wave_list  = ['f250m', 'f277w', 'f300m', 'f322w2', 'f323n', 'f335m', 'f356w', 'f360m', 'f405n', 'f410m', 'f430m',
                  'f444w', 'f460m', 'f466n', 'f470n', 'f480m']

    Lm = np.zeros((len(filters_list), len(spectral_resampling)))
    i  = 0

    for filter in filters_list:

        if filter in short_wave_list:
            mode = 'sw_imaging'
            aper = 'sw'

        elif filter in long_wave_list:
            mode = 'lw_imaging'
            aper = 'lw'

        else :
            raise ValueError(f'The given filter {filter} has to be either in the long or the short wavelength list')


        config = {'filter': filter, 'aperture': aper, 'disperser': None}
        print(type(filter), type(aper))
        Lm[i]  = get_pce(mode, config, spectral_resampling)
        i += 1

    ##########################################################################
    #        Resampling  the spectral response of NIRCam to that of NIRSpec
    ##########################################################################

    delt_lamb = np.linspace(
        spectral_resampling[1]-spectral_resampling[0], spectral_resampling[-1]-spectral_resampling[-2],
                            num=len(spectral_resampling)
    )

    ##################################
    #        Computing Lm
    ##################################

    Lm = cste_conv * Lm * spectral_resampling**(-1) * 1.5091905 * delt_lamb * exp_time * FLUXCONV_NC

    return Lm
def get_NIRSpec_filters(filter : str, tablewave : np.ndarray) :
    """
    @author Lina Issa, inspired from the paper : "PDRs4all: NIRSpec simulation of integral field unit
    spectroscopy of the Orion Bar photodissociation region" by Canin et al and submitted in 2022

    Compute the combined throughput of entire telescope/instrument/detector system for NIRSpec instrument

    : param filter    : the NIRSpec filter to consider for the fusion
    : param tablewave : Wavelength vector to interpolate throughput onto. As defined in pandeia's instrument.py
    : return          : system throughput as a function of wave

    """

    disperser_filter_combinations = {'f100lp': 'g140h', 'f170lp':'g235h', 'f290lp': 'g395h'}

    if filter not in disperser_filter_combinations.keys():
        raise ValueError(f'The provided filter {filter} is unknown. These are the expected filters: {disperser_filter_combinations.keys()}.')

    disperser  = disperser_filter_combinations[filter]

    obsmode  = {'instrument': 'nirspec',
               'mode': 'ifu',
               'disperser': disperser,
               'filter': filter}

    detector = {'readout_pattern': 'nrsrapid',
                'nint': 1,
                'ngroup': 5}

    conf     = {'instrument': obsmode,
                'detector': detector}

    i = InstrumentFactory(config=conf)
    pce = i.get_total_eff(tablewave)

    return pce


def get_pce(mode : str, config : dict, wave_resampling : np.array):
    '''
    This function is tailored for NIRCam instrument

    :param mode : depends on the filter
    :param config :  a dict containing the context of the observation
    :param wave_resampling : the spectral resolution used for the upsampling of the NirCam spectral responses
    '''
    # Get photon conversion efficiency of the instrument, from pandeia engine
    obsmode = {
               'instrument': 'nircam',
               'mode': mode,
               'filter': config['filter'],
               'aperture': config['aperture'],
               'disperser': config['disperser']
               }

    conf = {'instrument': obsmode}

    i = InstrumentFactory(config=conf)
    pce = i.get_total_eff(wave_resampling)

    return pce
# ----------------------------------------------------------------------------------------------------------------------
#                                          MAIN
#----------------------------------------------------------------------------------------------------------------------
def main(observation_config : dict) :

    DataDir         = observation_config["InputDir"]
    NIRCam_Filters  = observation_config['NIRCam_Filters']
    NIRSpec_Filters = observation_config['NIRSpec_Filters']
    tablewave       = fits.getdata(observation_config['TableWave'])
    #if not isinstance(spectral_scope, list):
    #    raise TypeError(f'The spectral scope provided is {type(spectral_scope)} while it should be a list')

    #if spectral_scope != [] :
    #    wave1, wave2 = spectral_scope[0], spectral_scope[1]
    #    if wave1 < spectral_sampling[1] and wave2 > spectral_sampling[1]:
    #        a, b = np.where(spectral_sampling>wave1)[0][0], np.where(spectral_sampling<wave2[0][-1])
    #        spectral_sampling = spectral_sampling[a:b]


    Lm_pce  = get_NIRCam_filters(NIRCam_Filters, tablewave, observation_config)
    Lh_pce  = get_NIRSpec_filters(NIRSpec_Filters,tablewave)

    Lh      = np.diag(
        np.reshape(
            Lh_pce,
            (len(tablewave),)
        )
    )


    hdu = fits.PrimaryHDU(Lm_pce)
    hdu.writeto(opath.join(DataDir, "NirCam_SpectralResponse.fits"), overwrite=True)

    hdu = fits.PrimaryHDU(Lh)
    hdu.writeto(opath.join(DataDir, f'NirSpec_Throughput_{NIRSpec_Filters}.fits'), overwrite=True)

    SpectralTransmissionFiles = {'NIRCam_SpectralResponses'  : opath.join(DataDir, 'NirCam_SpectralResponse.fits'),
                                 'NIRSpec_SpectralResponses' : opath.join(DataDir, f'NirSpec_Throughput_{NIRSpec_Filters}.fits')
                                 }

    with open('observation_config.yaml', 'r') as yamlfile:
        stream = load(yamlfile,  Loader=Loader)
        stream.update(SpectralTransmissionFiles)

    with open('observation_config.yaml', 'w') as yamlfile:
        safe_dump(stream, yamlfile)  # Also note the safe_dump
# ----------------------------------------------------------------------------------------------------------------------
#                                           Parameters loading
#----------------------------------------------------------------------------------------------------------------------

observation_config = load_config('observation_config.yaml', type = 'observation')

main(observation_config)

