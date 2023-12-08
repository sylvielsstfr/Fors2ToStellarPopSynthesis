"""Fit a single spectrum with numpyro
"""
# pylint: disable=unused-import
# pylint: disable=line-too-long

import collections
import copy
import os
import pickle
import re
from collections import OrderedDict

import arviz as az
import h5py
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pandas as pd
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
from fors2tostellarpopsynthesis.fitters.fitter_numpyro import (
    PARAM_MAX, PARAM_MIN, PARAM_NAMES, PARAM_SIGMA, PARAM_SIMLAW_NODUST,
    PARAM_SIMLAW_WITHDUST, PARAM_VAL, galaxymodel_nodust,
    galaxymodel_nodust_av, galaxymodel_withdust, galaxymodel_withdust_av)
from fors2tostellarpopsynthesis.parameters import (SSPParametersFit,
                                                   paramslist_to_dict)

# initialisation
jax.config.update("jax_enable_x64", True)


print("============= Running fit_singlespectrum_numpyro ============================")
print(" >>>> Device  :", jax.devices())


# Steering parameters

## Steering MCMC
#NUM_SAMPLES = 5_000
#N_CHAINS    = 4
#NUM_WARMUP  = 1_000
NUM_SAMPLES = 1_000
N_CHAINS    = 4
NUM_WARMUP  = 500
df_mcmc = pd.Series({"num_samples":NUM_SAMPLES, "n_chains":N_CHAINS, "num_warmup":NUM_WARMUP})


print("=========== Start MCMC  ============= :")
print(df_mcmc)

## Steering execution

FLAG_NODUST = True
FLAG_DUST = True

## Defining MCMC output files

#fileout_nodust_pickle = f"DSPS_nodust_mcmc_params_{N_CHAINS}_{NUM_WARMUP}_{NUM_SAMPLES}.pickle"
#fileout_nodust_csv = f"DSPS_nodust_mcmc_params_{N_CHAINS}_{NUM_WARMUP}_{NUM_SAMPLES}.csv"
fileout_nodust_hdf = f"DSPS_nodust_mcmc_params_{N_CHAINS}_{NUM_WARMUP}_{NUM_SAMPLES}.hdf"

#fileout_dust_pickle = f"DSPS_dust_mcmc_params_{N_CHAINS}_{NUM_WARMUP}_{NUM_SAMPLES}.pickle"
#fileout_dust_csv = f"DSPS_dust_mcmc_params_{N_CHAINS}_{NUM_WARMUP}_{NUM_SAMPLES}.csv"
fileout_dust_hdf = f"DSPS_dust_mcmc_params_{N_CHAINS}_{NUM_WARMUP}_{NUM_SAMPLES}.hdf"



# Select observation
# - choose the redshift of the observation
# - choose the relative error on flux at all wavelength. Here choose 10%
# - choose the absolute error. This value will be overwritten
#   after recalculating the absolute error for each wavelength (provided an an array)
Z_OBS = 0.5
SIGMAREL_OBS = 0.1
SIGMA_OBS = 1e-11

df_info = pd.Series({"z_obs":Z_OBS,"sigmarel_obs":SIGMAREL_OBS})

print("=========== Start Observations  ============= :")
print(df_info)

# initialisation of parameters
p = SSPParametersFit()

# select parameter true values and change it wrt default value
# select parameter true values and change it wrt default value
dict_sel_params_true = copy.deepcopy(p.DICT_PARAMS_true)
dict_sel_params_true['MAH_lgmO'] = 10.0
dict_sel_params_true['MAH_logtc'] = 0.8
dict_sel_params_true['MAH_early_index'] = 3.0
dict_sel_params_true['MAH_late_index'] = 0.5
dict_sel_params_true['AV'] = 0.5
dict_sel_params_true['UV_BUMP'] = 2.5
dict_sel_params_true['PLAW_SLOPE'] = -0.1

list_sel_params_true = list(dict_sel_params_true.values())

df_params = pd.DataFrame({"name":PARAM_NAMES,
                          "min": PARAM_MIN,
                          "val": PARAM_VAL,
                          "max": PARAM_MAX,
                          "sig":PARAM_SIGMA,
                          "true":list_sel_params_true})

df_params  = df_params.round(decimals=3)

print("=========== DSPS Parameters to fit ============= :")
print(df_params)

# generate spectrum from true selected values
# - it provide the wlsamm wavelength array
# - it provide the suposedly measured spectrum from true parameter
wlsall,spec_rest_noatt_true,spec_rest_att_true = ssp_spectrum_fromparam(dict_sel_params_true,Z_OBS)


# calculate the array of errors on the spectrum by using the average models in numpyro
# the goal is to obtain
# - sigmanodust_obs to replace sigma_obs
# - sigmadust_obs to replace sigma_obs
with seed(rng_seed=42):
    spec_nodust,sigmanodust_obs = galaxymodel_nodust_av(wlsall,Fobs=None,
                       initparamval = PARAM_VAL, minparamval = PARAM_MIN,maxparamval = PARAM_MAX,
                       sigmaparamval = PARAM_SIGMA,paramnames = PARAM_NAMES,z_obs= Z_OBS, sigmarel = SIGMAREL_OBS)



with seed(rng_seed=42):
    spec_withdust,sigmadust_obs = galaxymodel_withdust_av(wlsall,Fobs=None,
                       initparamval = PARAM_VAL, minparamval = PARAM_MIN,maxparamval = PARAM_MAX,
                       sigmaparamval = PARAM_SIGMA,paramnames = PARAM_NAMES,z_obs= Z_OBS, sigmarel = SIGMAREL_OBS)

# The problem is the above is that the parameters are drawn randomly
# Thus redefine the errors properly

sigmanodust_obs_true = SIGMAREL_OBS*spec_rest_noatt_true
sigmadust_obs_true = SIGMAREL_OBS*spec_rest_att_true


# Run MCMC

if FLAG_NODUST:

    print(f"===========  MCMC simulation : No Dust , num_samples = {NUM_SAMPLES}, n_chains = {N_CHAINS}, num_warmup = {NUM_WARMUP} ========")
    print(f" >>> output file {fileout_nodust_hdf}")
    # Run NUTS.
    rng_key = jax.random.PRNGKey(42)
    rng_key, rng_key0, rng_key1, rng_key2 = jax.random.split(rng_key, 4)


    kernel = NUTS(galaxymodel_nodust, dense_mass=True, target_accept_prob=0.9,init_strategy=numpyro.infer.init_to_median())


    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
            num_chains = N_CHAINS,
            chain_method ='vectorized',
            progress_bar = True)
    # see https://forum.pyro.ai/t/cannot-find-valid-initial-parameters-when-using-nuts-for-simple-gaussian-mixture-model-in-numpyro/2181/2
    with numpyro.validation_enabled():
        mcmc.run(rng_key, wlsin=wlsall, Fobs=spec_rest_noatt_true,
                       initparamval = PARAM_VAL,
                       minparamval = PARAM_MIN,
                       maxparamval = PARAM_MAX,
                       sigmaparamval = PARAM_SIGMA,
                       paramnames = PARAM_NAMES,
                       z_obs = Z_OBS,
                       sigma = sigmanodust_obs_true)
                       #extra_fields=('potential_energy',))
        mcmc.print_summary()
        samples_nuts = mcmc.get_samples()

    az.ess(samples_nuts, relative=True)  # efficacité relative

    df_nodust = pd.DataFrame(samples_nuts)
    df_nodust.to_hdf(fileout_nodust_hdf, key="dsps_mcmc_nodust",mode='a', complevel=9)
    df_info.to_hdf(fileout_nodust_hdf,key="obs",mode='a')
    df_params.to_hdf(fileout_nodust_hdf,key="params",mode='a')
    df_mcmc.to_hdf(fileout_nodust_hdf,key="mcmc",mode='a')

    data = az.from_numpyro(mcmc)




if FLAG_DUST:
    print(f"===========  MCMC simulation : With Dust , num_samples = {NUM_SAMPLES}, n_chains = {N_CHAINS}, num_warmup = {NUM_WARMUP} ========")
    print(f" >>> output file {fileout_dust_hdf}")
    # Run NUTS.
    rng_key = jax.random.PRNGKey(42)
    rng_key, rng_key0, rng_key1, rng_key2 = jax.random.split(rng_key, 4)


    kernel = NUTS(galaxymodel_withdust, dense_mass=True, target_accept_prob=0.9,
              init_strategy=numpyro.infer.init_to_median())

    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
            num_chains=N_CHAINS,
            chain_method='vectorized',
            progress_bar=True)

    # see https://forum.pyro.ai/t/cannot-find-valid-initial-parameters-when-using-nuts-for-simple-gaussian-mixture-model-in-numpyro/2181/2
    with numpyro.validation_enabled():
        mcmc.run(rng_key, wlsin=wlsall, Fobs=spec_rest_att_true,
                       initparamval = PARAM_VAL,
                       minparamval = PARAM_MIN,
                       maxparamval = PARAM_MAX,
                       sigmaparamval = PARAM_SIGMA,
                       paramnames = PARAM_NAMES,
                       z_obs = Z_OBS,
                       sigma = sigmadust_obs_true)
                       #extra_fields=('potential_energy',))
        mcmc.print_summary()
        samples_nuts = mcmc.get_samples()

    az.ess(samples_nuts, relative=True)  # efficacité relative

    df_dust = pd.DataFrame(samples_nuts)
    df_dust.to_hdf(fileout_dust_hdf, key="dsps_mcmc_dust",mode='a', complevel=9)
    df_info.to_hdf(fileout_dust_hdf,key="obs",mode='a')
    df_params.to_hdf(fileout_dust_hdf,key="params",mode='a')
    df_mcmc.to_hdf(fileout_dust_hdf,key="mcmc",mode='a')

    data = az.from_numpyro(mcmc)
