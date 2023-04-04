import numpy as np
from main       import load_config
from astropy.io import fits
from pandeia.engine.instrument_factory import InstrumentFactory

"""
In t
"""
def get_filters(filters_list : list, spectral_resampling : np.array, instrument_config: dict, ):
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

def get_pce(mode : str, config : dict, wave_resampling : np.array):
    '''
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

def main(fusion_config : dict, instrument_config : dict) :

    NIRSpecData       = fits.getdata(fusion_config['hyper_image'])
    spectral_sampling = np.sort(NIRSpecData[:,0,0])
    filters_list      = fusion_config['NIRCam_Filters']


    Lm  = get_filters(filters_list, spectral_sampling, instrument_config)
    hdu = fits.PrimaryHDU(Lm)
    hdu.writeto("NirCam_SpectralResponse.fits", overwrite=True)

fusion_config     = load_config('fusion_config.yaml')
instrument_config = load_config('observation_config.yaml', type = 'observation')

main(fusion_config, instrument_config)

