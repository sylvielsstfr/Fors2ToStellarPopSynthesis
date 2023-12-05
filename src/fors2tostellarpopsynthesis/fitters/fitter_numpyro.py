"""Module to fit SPS with numpyro.
    Please note that data files from DSPS must be donwloaded in data folder:
The file must be downloaded from
https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/
refer to https://dsps.readthedocs.io/en/latest/quickstart.html

 """

#pylint: disable-all

from collections import OrderedDict

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from interpax import interp1d
from jax import vmap
from jax.lax import concatenate, cond, fori_loop, select
from numpyro import optim
from numpyro.diagnostics import print_summary
from numpyro.distributions import constraints
from numpyro.handlers import condition, seed, trace
from numpyro.infer import HMC, MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoBNAFNormal, AutoMultivariateNormal
from numpyro.infer.reparam import NeuTraReparam

from fors2tostellarpopsynthesis.fitters.fitter_jaxopt import (
    SSP_DATA, mean_mags, mean_sfr, mean_spectrum, ssp_spectrum_fromparam)
from fors2tostellarpopsynthesis.parameters import (SSPParametersFit,
                                                   paramslist_to_dict)

jax.config.update("jax_enable_x64", True)
p = SSPParametersFit()


PARAM_SIMLAW_NODUST = np.array(["uniform","normal","normal","normal",
                "normal","normal","normal","normal","normal",
                "normal","normal","normal","normal",
                "fixed","fixed","fixed","fixed"])

PARAM_SIMLAW_WITHDUST = np.array(["uniform","normal","normal","normal",
                "normal","normal","normal","normal","normal",
                "normal","normal","normal","normal",
                "uniform","uniform","uniform","fixed"])


PARAM_NAMES = np.array(p.PARAM_NAMES_FLAT)

# increase the range of the DSPS parameters
FLAG_INCREASE_RANGE_MAH = True
if FLAG_INCREASE_RANGE_MAH:
    # MAH_logtc
    p.PARAMS_MIN = p.PARAMS_MIN.at[1].set(0.01)
    p.PARAMS_MAX = p.PARAMS_MAX.at[1].set(0.15)
    # MAH_early_index
    p.PARAMS_MIN = p.PARAMS_MIN.at[2].set(0.1)
    p.PARAMS_MAX = p.PARAMS_MAX.at[2].set(10.)
    # MAH_late_index
    p.PARAMS_MIN = p.PARAMS_MIN.at[3].set(0.1)
    p.PARAMS_MAX = p.PARAMS_MAX.at[3].set(10.)

FLAG_INCREASE_RANGE_MS = True
if FLAG_INCREASE_RANGE_MS:
    # MS_lgmcrit  12
    p.PARAMS_MIN = p.PARAMS_MIN.at[4].set(9.)
    p.PARAMS_MAX = p.PARAMS_MAX.at[4].set(13.)
    # MS_lgy_at_mcrit : -1
    p.PARAMS_MIN = p.PARAMS_MIN.at[5].set(-2.)
    p.PARAMS_MAX = p.PARAMS_MAX.at[5].set(-0.7)
    #MS_indx_lo : 1
    p.PARAMS_MIN = p.PARAMS_MIN.at[6].set(0.7)
    p.PARAMS_MAX = p.PARAMS_MAX.at[6].set(2.)
    #MS_indx_hi : -1
    p.PARAMS_MIN = p.PARAMS_MIN.at[7].set(-2.)
    p.PARAMS_MAX = p.PARAMS_MAX.at[7].set(-0.7)
    #MS_tau_dep : 2
    p.PARAMS_MIN = p.PARAMS_MIN.at[8].set(0.7)
    p.PARAMS_MAX = p.PARAMS_MAX.at[8].set(3.)

FLAG_INCREASE_RANGE_Q = True
if FLAG_INCREASE_RANGE_Q:
    #'Q_lg_qt', 1.0),
    p.PARAMS_MIN = p.PARAMS_MIN.at[9].set(0.5)
    p.PARAMS_MAX = p.PARAMS_MAX.at[9].set(2.)
    #('Q_qlglgdt', -0.50725),
    p.PARAMS_MIN = p.PARAMS_MIN.at[10].set(-2.)
    p.PARAMS_MAX = p.PARAMS_MAX.at[10].set(-0.2)
    # ('Q_lg_drop', -1.01773),
    p.PARAMS_MIN = p.PARAMS_MIN.at[11].set(-2.)
    p.PARAMS_MAX = p.PARAMS_MAX.at[11].set(-0.5)
    #('Q_lg_rejuv', -0.212307),
    p.PARAMS_MIN = p.PARAMS_MIN.at[12].set(-2.)
    p.PARAMS_MAX = p.PARAMS_MAX.at[12].set(-0.1)


PARAM_VAL = p.INIT_PARAMS
PARAM_MIN = p.PARAMS_MIN
PARAM_MAX = p.PARAMS_MAX
PARAM_SIGMA = jnp.sqrt(0.5*((PARAM_VAL-PARAM_MIN)**2 + (PARAM_VAL-PARAM_MAX)**2))


z_obs_0 = 0.01
sigma_obs_0 = 1e-11
sigmarel_obs_0 = 0.1

condlist_fix = jnp.where(PARAM_SIMLAW_NODUST == "fixed",True,False)
condlist_uniform = jnp.where(PARAM_SIMLAW_NODUST == "uniform",True,False)


def galaxymodel_nodust_av(wlsin,Fobs=None,
                       initparamval = PARAM_VAL, minparamval = PARAM_MIN,maxparamval = PARAM_MAX,
                       sigmaparamval = PARAM_SIGMA,paramnames = PARAM_NAMES,z_obs= z_obs_0, sigmarel = sigmarel_obs_0):

    """
    Average Model of Galaxy spectrum at rest without dust. The goal is to provide
    the SED flux errors on the fiducial model.

    :param wlsin: array of input spectrum wavelength
    :type wlsin: float in Angstrom
    :param Fobs: SED Flux returned by SPS
    :type Fobs: jax array of floats
    :param initparamval: initialisation of fit parameter values
    :type initparamval: jnp array of floats
    :param minparamval: minimum values of fit parameters
    :type minparamval: jnp array of floats
    :param maxparamval: maximum values of fit parameters
    :type maxparamval: jnp array of floats
    :param sigmaparamval: standard deviation of fit parameters
    :type sigmaparamval: jnp array of floats
    :param paramnames: Names of parameters
    :type paramnames: array of strings
    :param z_obs:  redshift of observation
    :type z_obs: float
    :param sigmarel: relative error on flux in all wavelength bin
    :type sigmarel: float
    :return: tuple of SED flux and its errors
    :rtype: two jnp arrays
    """


    dict_params = OrderedDict()

    # MAH_lgmO
    idx = 0
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # MAH_logtc
    idx = 1
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    #MAH_early_index
    idx = 2
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    #MAH_late_index
    idx = 3
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_lgmcrit
    idx = 4
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_lgy_at_mcrit
    idx = 5
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_indx_lo
    idx = 6
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_indx_hi
    idx = 7
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_tau_dep
    idx = 8
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_qt
    idx = 9
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_qlglgdt
    idx = 10
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_drop
    idx = 11
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_rejuv
    idx = 12
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # AV
    idx = 13
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.0,0.1))
    dict_params[name] = valmean

    # UV_BUMP
    idx = 14
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.0,0.1))
    dict_params[name] = valmean

    # PLAW_SLOPE
    idx = 15
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.,0.1))
    dict_params[name] = valmean

    # SCALEF
    idx = 16
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = 1.


    wls,sed_notattenuated,sed_attenuated = ssp_spectrum_fromparam(dict_params,z_obs)

    # interpolate measured
    Fobs = interp1d(wlsin,wls,sed_notattenuated)
    sigmaF = Fobs*sigmarel

    return Fobs,sigmaF


def galaxymodel_nodust(wlsin,Fobs=None,
                       initparamval = PARAM_VAL, minparamval = PARAM_MIN,maxparamval = PARAM_MAX,
                       sigmaparamval = PARAM_SIGMA,paramnames = PARAM_NAMES,z_obs= z_obs_0, sigma = sigma_obs_0):

    """
    Full bayesian model of Galaxy spectrum at rest without dust
    Feed numpyro with the SED flux sample

    :param wlsin: array of input spectrum wavelength
    :type wlsin: float in Angstrom
    :param Fobs: SED Flux returned by SPS
    :type Fobs: jax array of floats
    :param initparamval: initialisation of fit parameter values
    :type initparamval: jnp array of floats
    :param minparamval: minimum values of fit parameters
    :type minparamval: jnp array of floats
    :param maxparamval: maximum values of fit parameters
    :type maxparamval: jnp array of floats
    :param sigmaparamval: standard deviation of fit parameters
    :type sigmaparamval: jnp array of floats
    :param paramnames: Names of parameters
    :type paramnames: array of strings
    :param z_obs:  redshift of observation
    :type z_obs: float
    :param sigma: error on flux in all wavelength bin
    :type sigma: float


    """


    dict_params = OrderedDict()

    # MAH_lgmO
    idx = 0
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # MAH_logtc
    idx = 1
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    #MAH_early_index
    idx = 2
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    #MAH_late_index
    idx = 3
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_lgmcrit
    idx = 4
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_lgy_at_mcrit
    idx = 5
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_indx_lo
    idx = 6
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_indx_hi
    idx = 7
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_tau_dep
    idx = 8
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_qt
    idx = 9
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_qlglgdt
    idx = 10
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_drop
    idx = 11
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_rejuv
    idx = 12
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # AV
    idx = 13
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.0,0.1))
    dict_params[name] = valmean

    # UV_BUMP
    idx = 14
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.0,0.1))
    dict_params[name] = valmean

    # PLAW_SLOPE
    idx = 15
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.,0.1))
    dict_params[name] = valmean

    # SCALEF
    idx = 16
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = 1.


    wls,sed_notattenuated,sed_attenuated = ssp_spectrum_fromparam(dict_params,z_obs)

    # interpolate measured
    Fsim = interp1d(wlsin,wls,sed_notattenuated)

    with numpyro.plate("obs", wlsin.shape[0]):  # les observables sont indépendantes
        numpyro.sample('F', dist.Normal(Fsim, sigma), obs=Fobs)




def galaxymodel_withdust_av(wlsin,Fobs=None,
                         initparamval = PARAM_VAL, minparamval = PARAM_MIN,maxparamval = PARAM_MAX,
                         sigmaparamval = PARAM_SIGMA,paramnames = PARAM_NAMES,z_obs= z_obs_0, sigmarel = sigmarel_obs_0):
    """
    Average Model of Galaxy spectrum at rest with dust. The goal is to provide
    the SED flux errors on the fiducial model.

    :param wlsin: array of input spectrum wavelength
    :type wlsin: float in Angstrom
    :param Fobs: SED Flux returned by SPS
    :type Fobs: jax array of floats
    :param initparamval: initialisation of fit parameter values
    :type initparamval: jnp array of floats
    :param minparamval: minimum values of fit parameters
    :type minparamval: jnp array of floats
    :param maxparamval: maximum values of fit parameters
    :type maxparamval: jnp array of floats
    :param sigmaparamval: standard deviation of fit parameters
    :type sigmaparamval: jnp array of floats
    :param paramnames: Names of parameters
    :type paramnames: array of strings
    :param z_obs:  redshift of observation
    :type z_obs: float
    :param sigmarel: relative error on flux in all wavelength bin
    :type sigmarel: float
    :return: tuple of SED flux and its errors
    :rtype: two jnp arrays

    """


    dict_params = OrderedDict()

    # MAH_lgmO
    idx = 0
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # MAH_logtc
    idx = 1
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    #MAH_early_index
    idx = 2
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    #MAH_late_index
    idx = 3
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_lgmcrit
    idx = 4
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_lgy_at_mcrit
    idx = 5
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_indx_lo
    idx = 6
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_indx_hi
    idx = 7
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_tau_dep
    idx = 8
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_qt
    idx = 9
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_qlglgdt
    idx = 10
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_drop
    idx = 11
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_rejuv
    idx = 12
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # AV
    idx = 13
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # UV_BUMP
    idx = 14
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # PLAW_SLOPE
    idx = 15
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # SCALEF
    idx = 16
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = 1.0



    wls,sed_notattenuated,sed_attenuated = ssp_spectrum_fromparam(dict_params,z_obs)

    # interpolate measured
    Fobs = interp1d(wlsin,wls,sed_attenuated)
    sigmaF = Fobs*sigmarel

    return Fobs,sigmaF



def galaxymodel_withdust(wlsin,Fobs=None,
                         initparamval = PARAM_VAL, minparamval = PARAM_MIN,maxparamval = PARAM_MAX,
                         sigmaparamval = PARAM_SIGMA,paramnames = PARAM_NAMES,z_obs= z_obs_0, sigma = sigma_obs_0):
    """
    Full bayesian model of Galaxy spectrum at rest with dust
    Feed numpyro with the SED flux sample

    :param wlsin: array of input spectrum wavelength
    :type wlsin: float in Angstrom
    :param Fobs: SED Flux returned by SPS
    :type Fobs: jax array of floats
    :param initparamval: initialisation of fit parameter values
    :type initparamval: jnp array of floats
    :param minparamval: minimum values of fit parameters
    :type minparamval: jnp array of floats
    :param maxparamval: maximum values of fit parameters
    :type maxparamval: jnp array of floats
    :param sigmaparamval: standard deviation of fit parameters
    :type sigmaparamval: jnp array of floats
    :param paramnames: Names of parameters
    :type paramnames: array of strings
    :param z_obs:  redshift of observation
    :type z_obs: float
    :param sigma: error on flux in all wavelength bin
    :type sigma: float


    """


    dict_params = OrderedDict()

    # MAH_lgmO
    idx = 0
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # MAH_logtc
    idx = 1
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    #MAH_early_index
    idx = 2
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    #MAH_late_index
    idx = 3
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_lgmcrit
    idx = 4
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_lgy_at_mcrit
    idx = 5
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_indx_lo
    idx = 6
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_indx_hi
    idx = 7
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_tau_dep
    idx = 8
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_qt
    idx = 9
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_qlglgdt
    idx = 10
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_drop
    idx = 11
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_rejuv
    idx = 12
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # AV
    idx = 13
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # UV_BUMP
    idx = 14
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # PLAW_SLOPE
    idx = 15
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # SCALEF
    idx = 16
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = 1.0


    wls,sed_notattenuated,sed_attenuated = ssp_spectrum_fromparam(dict_params,z_obs)

    # interpolate measured
    Fsim = interp1d(wlsin,wls,sed_attenuated)

    with numpyro.plate("obs", wlsin.shape[0]):  # les observables sont indépendantes
        numpyro.sample('F', dist.Normal(Fsim, sigma), obs=Fobs)


def galaxymodel_withdust_try(wlsin,Fobs=None,
                       initparamval = PARAM_VAL, minparamval = PARAM_MIN,maxparamval = PARAM_MAX,
                       sigmaparamval = PARAM_SIGMA,paramnames = PARAM_NAMES,z_obs= z_obs_0, sigma = sigma_obs_0):

    """
    Full bayesian model of Galaxy spectrum at rest without dust
    Feed numpyro with the SED flux sample

    :param wlsin: array of input spectrum wavelength
    :type wlsin: float in Angstrom
    :param Fobs: SED Flux returned by SPS
    :type Fobs: jax array of floats
    :param initparamval: initialisation of fit parameter values
    :type initparamval: jnp array of floats
    :param minparamval: minimum values of fit parameters
    :type minparamval: jnp array of floats
    :param maxparamval: maximum values of fit parameters
    :type maxparamval: jnp array of floats
    :param sigmaparamval: standard deviation of fit parameters
    :type sigmaparamval: jnp array of floats
    :param paramnames: Names of parameters
    :type paramnames: array of strings
    :param z_obs:  redshift of observation
    :type z_obs: float
    :param sigma: error on flux in all wavelength bin
    :type sigma: float


    """


    dict_params = OrderedDict()

    # MAH_lgmO
    idx = 0
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Uniform(valmin,valmax))
    dict_params[name] = val

    # MAH_logtc
    idx = 1
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    #MAH_early_index
    idx = 2
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    #MAH_late_index
    idx = 3
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_lgmcrit
    idx = 4
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_lgy_at_mcrit
    idx = 5
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val


    # MS_indx_lo
    idx = 6
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_indx_hi
    idx = 7
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # MS_tau_dep
    idx = 8
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_qt
    idx = 9
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_qlglgdt
    idx = 10
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_drop
    idx = 11
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # Q_lg_rejuv
    idx = 12
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = val

    # AV
    idx = 13
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.0,0.1))
    dict_params[name] = valmean

    # UV_BUMP
    idx = 14
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.0,0.1))
    dict_params[name] = valmean

    # PLAW_SLOPE
    idx = 15
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(0.,0.1))
    dict_params[name] = valmean

    # SCALEF
    idx = 16
    name = paramnames[idx]
    valmean = initparamval[idx]
    valmin = minparamval[idx]
    valmax = maxparamval[idx]
    scale = sigmaparamval[idx]
    #val = numpyro.sample(name,dist.Normal(valmean,scale))
    dict_params[name] = 1.


    wls,sed_notattenuated,sed_attenuated = ssp_spectrum_fromparam(dict_params,z_obs)

    # interpolate measured
    Fsim = interp1d(wlsin,wls,sed_notattenuated)

    with numpyro.plate("obs", wlsin.shape[0]):  # les observables sont indépendantes
        numpyro.sample('F', dist.Normal(Fsim, sigma), obs=Fobs)