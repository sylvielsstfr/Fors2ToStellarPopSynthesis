"""Module providing SPS fit utilities with jaxopt
Please note that data files from DSPS must be donwloaded in data folder:
The file must be downloaded from 
https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/
refer to https://dsps.readthedocs.io/en/latest/quickstart.html

"""
#pylint: disable-all

import copy
import os

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import jaxopt
import numpy as np
import optax
from interpax import interp1d
from jax import grad, hessian, jacfwd, jacrev, jit, vmap

jax.config.update("jax_enable_x64", True)

from diffstar import sfh_singlegal
from dsps import (calc_obs_mag, calc_rest_mag,
                  calc_rest_sed_sfh_table_lognormal_mdf,
                  calc_rest_sed_sfh_table_met_table, load_ssp_templates)
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.dust.att_curves import (RV_C00, _frac_transmission_from_k_lambda,
                                  sbl18_k_lambda)


def _get_package_dir()-> str:
    """get the path of this fitters package
    """
    dirname = os.path.dirname(__file__)
    return dirname

FILENAME_SSP_DATA = 'data/tempdata.h5'
FULLFILENAME_SSP_DATA = os.path.join(_get_package_dir(),FILENAME_SSP_DATA)
SSP_DATA = load_ssp_templates(fn=FULLFILENAME_SSP_DATA)


TODAY_GYR = 13.8


@jit
def mean_spectrum(wls, params,z_obs):
    """ Model of spectrum

    :param wls: wavelengths of the spectrum in rest frame
    :type wls: float

    :param params: parameters for the fit
    :type params: dictionnary of parameters

    :param z_obs: redshift at which the model SSP should be calculated
    :type z_obs: float
    :return: the spectrum
    :rtype: float

    """

    # decode the parameters
    MAH_lgmO = params["MAH_lgmO"]
    MAH_logtc = params["MAH_logtc"]
    MAH_early_index = params["MAH_early_index"]
    MAH_late_index = params["MAH_late_index"]
    list_param_mah = [MAH_lgmO,MAH_logtc,MAH_early_index,MAH_late_index]

    MS_lgmcrit = params["MS_lgmcrit"]
    MS_lgy_at_mcrit = params["MS_lgy_at_mcrit"]
    MS_indx_lo = params["MS_indx_lo"]
    MS_indx_hi = params["MS_indx_hi"]
    MS_tau_dep = params["MS_tau_dep"]
    list_param_ms = [MS_lgmcrit,MS_lgy_at_mcrit,MS_indx_lo,MS_indx_hi,MS_tau_dep]

    Q_lg_qt = params["Q_lg_qt"]
    Q_qlglgdt = params["Q_qlglgdt"]
    Q_lg_drop = params["Q_lg_drop"]
    Q_lg_rejuv = params["Q_lg_rejuv"]
    list_param_q = [Q_lg_qt, Q_qlglgdt,Q_lg_drop,Q_lg_rejuv]

    Av = params["Av"]
    uv_bump = params["uv_bump"]
    plaw_slope = params["plaw_slope"]
    list_param_dust = [Av,uv_bump,plaw_slope]

    # compute SFR
    tarr = np.linspace(0.1, TODAY_GYR, 100)
    sfh_gal = sfh_singlegal(
    tarr, list_param_mah , list_param_ms, list_param_q)

    # metallicity
    gal_lgmet = -2.0 # log10(Z)
    gal_lgmet_scatter = 0.2 # lognormal scatter in the metallicity distribution function

    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY) # age of the universe in Gyr at z_obs
    t_obs = t_obs[0] # age_at_z function returns an array, but SED functions accept a float for this argument

    # clear sfh in future
    sfh_gal = jnp.where(tarr<t_obs, sfh_gal, 0)

    # compute the SED_info object
    gal_t_table = tarr
    gal_sfr_table = sfh_gal
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(
    gal_t_table, gal_sfr_table, gal_lgmet, gal_lgmet_scatter,
    SSP_DATA.ssp_lgmet, SSP_DATA.ssp_lg_age_gyr, SSP_DATA.ssp_flux, t_obs)

    # compute dust attenuation
    wave_spec_micron = SSP_DATA.ssp_wave/10_000
    k = sbl18_k_lambda(wave_spec_micron,uv_bump,plaw_slope)
    dsps_flux_ratio = _frac_transmission_from_k_lambda(k,Av)

    sed_attenuated = dsps_flux_ratio * sed_info.rest_sed

    # interpolate with interpax which is differentiable
    #Fobs = jnp.interp(wls, ssp_data.ssp_wave, sed_attenuated)


    Fobs = interp1d(wls, SSP_DATA.ssp_wave, sed_attenuated,method='cubic')

    return Fobs





@jit
def mean_mags(X, params,z_obs):
    """ Model of photometry

    Compute the magnitudes in Filters

    :param X: List of to be used (Galex, sdss, vircam)
    :type X: a list of tuples of two arrays (one array with wavelength and one array of corresponding transmission)

    :param params: Model parameters
    :type params: Dictionnary of parameters


    :return: array the magnitude for the SED spectrum model
    :rtype: float

    """

    # decode the parameters
    MAH_lgmO = params["MAH_lgmO"]
    MAH_logtc = params["MAH_logtc"]
    MAH_early_index = params["MAH_early_index"]
    MAH_late_index = params["MAH_late_index"]
    list_param_mah = [MAH_lgmO,MAH_logtc,MAH_early_index,MAH_late_index]

    MS_lgmcrit = params["MS_lgmcrit"]
    MS_lgy_at_mcrit = params["MS_lgy_at_mcrit"]
    MS_indx_lo = params["MS_indx_lo"]
    MS_indx_hi = params["MS_indx_hi"]
    MS_tau_dep = params["MS_tau_dep"]
    list_param_ms = [MS_lgmcrit,MS_lgy_at_mcrit,MS_indx_lo,MS_indx_hi,MS_tau_dep]

    Q_lg_qt = params["Q_lg_qt"]
    Q_qlglgdt = params["Q_qlglgdt"]
    Q_lg_drop = params["Q_lg_drop"]
    Q_lg_rejuv = params["Q_lg_rejuv"]
    list_param_q = [Q_lg_qt, Q_qlglgdt,Q_lg_drop,Q_lg_rejuv]

    Av = params["Av"]
    uv_bump = params["uv_bump"]
    plaw_slope = params["plaw_slope"]
    list_param_dust = [Av,uv_bump,plaw_slope]

    # compute SFR
    tarr = np.linspace(0.1, TODAY_GYR, 100)
    sfh_gal = sfh_singlegal(
    tarr, list_param_mah , list_param_ms, list_param_q)

    # metallicity
    gal_lgmet = -2.0 # log10(Z)
    gal_lgmet_scatter = 0.2 # lognormal scatter in the metallicity distribution function

    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY) # age of the universe in Gyr at z_obs
    t_obs = t_obs[0] # age_at_z function returns an array, but SED functions accept a float for this argument

    # clear sfh in future
    sfh_gal = jnp.where(tarr<t_obs, sfh_gal, 0)

    # compute SED
    gal_t_table = tarr
    gal_sfr_table = sfh_gal

    # create the sed object
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(
    gal_t_table, gal_sfr_table, gal_lgmet, gal_lgmet_scatter,
    SSP_DATA.ssp_lgmet, SSP_DATA.ssp_lg_age_gyr, SSP_DATA.ssp_flux, t_obs)

    # compute dust attenuation
    wave_spec_micron = SSP_DATA.ssp_wave/10_000
    k = sbl18_k_lambda(wave_spec_micron,uv_bump,plaw_slope)
    dsps_flux_ratio = _frac_transmission_from_k_lambda(k,Av)

    sed_attenuated = dsps_flux_ratio * sed_info.rest_sed

    # calculate magnitudes in observation frame
    mags_predictions = []

    #decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    #def vect_obs_mag(x,y):
    #    obs_mag = calc_obs_mag(ssp_data.ssp_wave, sed_attenuated,x,y,z_obs, *DEFAULT_COSMOLOGY)

    mags_predictions = jax.tree_map(lambda x,y : calc_obs_mag(SSP_DATA.ssp_wave, sed_attenuated,x,y,z_obs, *DEFAULT_COSMOLOGY),
                                    list_wls_filters,
                                    list_transm_filters)
    mags_predictions = jnp.array(mags_predictions)

    return mags_predictions

@jit
def mean_sfr(params,z_obs):
    """ Model of the SFR

    :param params: Fitted parameter dictionnary
    :type params: float as a dictionnary

    :return: array of the star formation rate
    :rtype: float

    """

    # decode the parameters
    MAH_lgmO = params["MAH_lgmO"]
    MAH_logtc = params["MAH_logtc"]
    MAH_early_index = params["MAH_early_index"]
    MAH_late_index = params["MAH_late_index"]
    list_param_mah = [MAH_lgmO,MAH_logtc,MAH_early_index,MAH_late_index]

    MS_lgmcrit = params["MS_lgmcrit"]
    MS_lgy_at_mcrit = params["MS_lgy_at_mcrit"]
    MS_indx_lo = params["MS_indx_lo"]
    MS_indx_hi = params["MS_indx_hi"]
    MS_tau_dep = params["MS_tau_dep"]
    list_param_ms = [MS_lgmcrit,MS_lgy_at_mcrit,MS_indx_lo,MS_indx_hi,MS_tau_dep]

    Q_lg_qt = params["Q_lg_qt"]
    Q_qlglgdt = params["Q_qlglgdt"]
    Q_lg_drop = params["Q_lg_drop"]
    Q_lg_rejuv = params["Q_lg_rejuv"]
    list_param_q = [Q_lg_qt, Q_qlglgdt,Q_lg_drop,Q_lg_rejuv]

    Av = params["Av"]
    uv_bump = params["uv_bump"]
    plaw_slope = params["plaw_slope"]
    list_param_dust = [Av,uv_bump,plaw_slope]


    # compute SFR
    tarr = np.linspace(0.1, TODAY_GYR, 100)
    sfh_gal = sfh_singlegal(
    tarr, list_param_mah , list_param_ms, list_param_q)

    # metallicity
    gal_lgmet = -2.0 # log10(Z)
    gal_lgmet_scatter = 0.2 # lognormal scatter in the metallicity distribution function

    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY) # age of the universe in Gyr at z_obs
    t_obs = t_obs[0] # age_at_z function returns an array, but SED functions accept a float for this argument

    # clear sfh in future
    sfh_gal = jnp.where(tarr<t_obs, sfh_gal, 0)


    return tarr,sfh_gal

@jit
def lik_spec(p,wls,F, sigma_obs,z_obs):
    """
    neg loglikelihood(parameters,x,y,sigmas) for the spectrum
    """

    params = {"MAH_lgmO":p[0],
              "MAH_logtc":p[1],
              "MAH_early_index":p[2],
              "MAH_late_index": p[3],

              "MS_lgmcrit":p[4],
              "MS_lgy_at_mcrit":p[5],
              "MS_indx_lo":p[6],
              "MS_indx_hi":p[7],
              "MS_tau_dep":p[8],

              "Q_lg_qt":p[9],
              "Q_qlglgdt":p[10],
              "Q_lg_drop":p[11],
              "Q_lg_rejuv":p[12],

              "Av":p[13],
              "uv_bump":p[14],
              "plaw_slope":p[15],
              "scaleF":p[16]
             }
    scaleF =  params["scaleF"]
    # residuals
    resid = mean_spectrum(wls, params,z_obs) - F*scaleF

    return 0.5*jnp.sum((resid/(sigma_obs*jnp.sqrt(scaleF)))** 2)


@jit
def lik_mag(p,xf,mags_measured, sigma_mag_obs,z_obs):
    """
    neg loglikelihood(parameters,x,y,sigmas) for the photometry
    """

    params = {"MAH_lgmO":p[0],
              "MAH_logtc":p[1],
              "MAH_early_index":p[2],
              "MAH_late_index": p[3],

              "MS_lgmcrit":p[4],
              "MS_lgy_at_mcrit":p[5],
              "MS_indx_lo":p[6],
              "MS_indx_hi":p[7],
              "MS_tau_dep":p[8],

              "Q_lg_qt":p[9],
              "Q_qlglgdt":p[10],
              "Q_lg_drop":p[11],
              "Q_lg_rejuv":p[12],

              "Av":p[13],
              "uv_bump":p[14],
              "plaw_slope":p[15],
              "scaleF":p[16]
             }
    #scaleF =  params["scaleF"]

    all_mags_redictions = mean_mags(xf, params,z_obs)
    resid = mags_measured - all_mags_redictions

    return 0.5*jnp.sum((resid/sigma_mag_obs)** 2)


@jit
def lik_comb(p,xc,datac, sigmac, z_obs,weight= 0.5):
    """
    neg loglikelihood(parameters,xc,yc,sigmasc) combining the spectroscopy and the photometry

    Xc = [Xspec_data, Xf_sel]
    Yc = [Yspec_data, mags_measured ]
    EYc = [EYspec_data, data_selected_magserr]

    weight must be between 0 and 1
    """

    resid_spec = lik_spec(p,xc[0],datac[0], sigmac[0],z_obs)
    resid_phot = lik_mag(p,xc[1],datac[1], sigmac[1],z_obs)

    return weight*resid_spec + (1-weight)*resid_phot


def get_infos_spec(res, model, wls,F, eF):
    """_summary_

    :param res: _description_
    :type res: _type_
    :param model: _description_
    :type model: _type_
    :param wls: _description_
    :type wls: _type_
    :param F: _description_
    :type F: _type_
    :param eF: _description_
    :type eF: _type_
    :return: _description_
    :rtype: _type_
    """
    params    = res.params
    fun_min   = model(params,wls,F,eF)
    jacob_min =jax.jacfwd(model)(params, wls,F,eF)
    #covariance matrix of parameters
    inv_hessian_min =jax.scipy.linalg.inv(jax.hessian(model)(params, wls,F,eF))
    return params,fun_min,jacob_min,inv_hessian_min


def get_infos_mag(res, model, xf, mgs, mgse):
    """_summary_

    :param res: _description_
    :type res: _type_
    :param model: _description_
    :type model: _type_
    :param xf: _description_
    :type xf: _type_
    :param mgs: _description_
    :type mgs: _type_
    :param mgse: _description_
    :type mgse: _type_
    :return: _description_
    :rtype: _type_
    """
    params    = res.params
    fun_min   = model(params,xf,mgs,mgse)
    jacob_min =jax.jacfwd(model)(params, xf, mgs, mgse)
    #covariance matrix of parameters
    inv_hessian_min =jax.scipy.linalg.inv(jax.hessian(model)(params, xf, mgs , mgse))
    return params,fun_min,jacob_min,inv_hessian_min


def get_infos_comb(res, model, xc, datac, sigmac,weight):
    """_summary_

    :param res: _description_
    :type res: _type_
    :param model: _description_
    :type model: _type_
    :param xc: _description_
    :type xc: _type_
    :param datac: _description_
    :type datac: _type_
    :param sigmac: _description_
    :type sigmac: _type_
    :param weight: _description_
    :type weight: _type_
    :return: _description_
    :rtype: _type_
    """
    params    = res.params
    fun_min   = model(params,xc,datac,sigmac,weight=weight)
    jacob_min =jax.jacfwd(model)(params, xc,datac,sigmac,weight=weight)
    #covariance matrix of parameters
    inv_hessian_min =jax.scipy.linalg.inv(jax.hessian(model)(params,xc,datac,sigmac,weight=weight))
    return params,fun_min,jacob_min,inv_hessian_min


