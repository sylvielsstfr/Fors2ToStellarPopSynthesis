"""Module to fit a series of Fors2 spectra
"""
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=trailing-newlines
# pylint: disable=unused-import


import pickle
import re
from collections import OrderedDict

import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# Filters
from fors2tostellarpopsynthesis.filters import FilterInfo
# Fitter jaxopt
from fors2tostellarpopsynthesis.fitters.fitter_jaxopt import (get_infos_comb,
                                                              get_infos_mag,
                                                              get_infos_spec,
                                                              lik_comb,
                                                              lik_mag,
                                                              lik_spec)
from fors2tostellarpopsynthesis.fitters.fitter_util import (
    plot_fit_ssp_photometry, plot_fit_ssp_spectrophotometry,
    plot_fit_ssp_spectrophotometry_sl, plot_fit_ssp_spectroscopy, plot_SFH,
    rescale_photometry, rescale_spectroscopy, rescale_starlight_inrangefors2)
# Fors2 and StarLight
from fors2tostellarpopsynthesis.fors2starlightio import (
    Fors2DataAcess, SLDataAcess, convert_flux_torestframe)
# parameters
from fors2tostellarpopsynthesis.parameters import (SSPParametersFit,
                                                   paramslist_to_dict)

jax.config.update("jax_enable_x64", True)

plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams["axes.labelsize"] = 'xx-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['xtick.labelsize']= 'xx-large'
plt.rcParams['ytick.labelsize']= 'xx-large'
plt.rcParams['legend.fontsize']=  16

kernel = kernels.RBF(0.5, (6000, 20000.0))
gpr = GaussianProcessRegressor(kernel=kernel ,random_state=0)

FLAG_REMOVE_GALEX = True
FLAG_REMOVE_GALEX_FUV = True
FLAG_PLOT = False

if __name__ == '__main__':

    # Create the filters
    ps = FilterInfo()

    # Fors2 spectra
    fors2 = Fors2DataAcess()
    fors2_tags = fors2.get_list_of_groupkeys()
    list_of_fors2_attributes = fors2.get_list_subgroup_keys()

    # Starlight spectra
    sl = SLDataAcess()
    sl_tags = sl.get_list_of_groupkeys()

    ###########################################################################
    # here select range for the spectra to be fitted
    ###########################################################################
    # this code may crash, probably due to bad some bad garbage collection ?
    #
    file_indexes_selected = np.arange(96 ,550)
    all_selected_spectrum_tags = fors2_tags[file_indexes_selected]


    rank = file_indexes_selected[0]

    for selected_spectrum_tag in all_selected_spectrum_tags:
        selected_spectrum_number = int(re.findall("^SPEC(.*)", selected_spectrum_tag)[0])


        fors2_attr =fors2.getattribdata_fromgroup(selected_spectrum_tag)
        z_obs = fors2_attr['redshift']

        title_spec = selected_spectrum_tag + f" z= {z_obs:.2f}"

        print( "##################################################################")
        print(f"#       FIT {rank} spectrum :   {title_spec}")
        print( "##################################################################")
        rank += 1

        #retrieve magnitude data
        data_mags, data_magserr = fors2.get_photmagnitudes(selected_spectrum_tag)

        # get the Fors2 spectrum
        spec_obs = fors2.getspectrumcleanedemissionlines_fromgroup(selected_spectrum_tag)

        Xs = spec_obs['wl']
        Ys = spec_obs['fnu']
        EYs = spec_obs['bg']
        EYs_med = spec_obs['bg_med']

        #convert to restframe
        Xspec_data, Yspec_data = convert_flux_torestframe(Xs,Ys,z_obs)
        EYspec_data = EYs*(1+z_obs)
        EYspec_data_med = EYs_med*(1+z_obs)
        # smooth the error over the spectrum
        fit_res=gpr.fit(Xspec_data[:, None], EYspec_data )
        EYspec_data_sm = gpr.predict(Xspec_data[:, None], return_std=False)
        # need to increase error to decrease chi2 error
        EYspec_data_sm *=2

        #plot the selected spectrum (IO)
        #fors2.plot_spectro_photom_rescaling(selected_spectrum_tag)

        # parameters for fit
        p = SSPParametersFit()

        init_params = p.INIT_PARAMS
        params_min = p.PARAMS_MIN
        params_max = p.PARAMS_MAX

        # Choose filters with mags without Nan
        NoNaN_mags = np.intersect1d(np.argwhere(~np.isnan(data_mags)).flatten(),np.argwhere(~np.isnan(data_magserr)).flatten())
        # selected indexes for filters
        index_selected_filters = NoNaN_mags

        if FLAG_REMOVE_GALEX:
            galex_indexes = np.array([0,1])
            index_selected_filters = np.setdiff1d(index_selected_filters,galex_indexes)
        elif FLAG_REMOVE_GALEX_FUV:
            galex_indexes = np.array([0])
            index_selected_filters = np.setdiff1d(index_selected_filters,galex_indexes)


        # Select filters
        XF = ps.get_2lists()
        NF = len(XF[0])
        list_wls_f_sel = []
        list_trans_f_sel = []

        list_name_f_sel = []
        list_wlmean_f_sel = []

        for index in index_selected_filters:
            list_wls_f_sel.append(XF[0][index])
            list_trans_f_sel.append(XF[1][index])
            the_filt = ps.filters_transmissionlist[index]
            the_wlmean = the_filt.wave_mean
            list_wlmean_f_sel.append(the_wlmean)
            list_name_f_sel.append(ps.filters_namelist[index])

        list_wlmean_f_sel = jnp.array(list_wlmean_f_sel)
        Xf_sel = (list_wls_f_sel,list_trans_f_sel)

        # get the magnitudes and magnitude errors
        data_selected_mags =  jnp.array(data_mags[index_selected_filters])
        data_selected_magserr = jnp.array(data_magserr[index_selected_filters])


        if len(data_selected_mags) == 0:
            print(f">>>>>>>> No magnitude for spectrum {selected_spectrum_tag} ==> SKIP")
            continue

        #fit with magnitudes only
        lbfgsb = jaxopt.ScipyBoundedMinimize(fun=lik_mag, method="L-BFGS-B")

        res_m = lbfgsb.run(init_params, bounds=(params_min ,params_max ), xf = Xf_sel, mags_measured = data_selected_mags, sigma_mag_obs = data_selected_magserr,z_obs=z_obs)
        params_m,fun_min_m,jacob_min_m,inv_hessian_min_m = get_infos_mag(res_m, lik_mag,  xf = Xf_sel, mgs = data_selected_mags, mgse = data_selected_magserr,z_obs=z_obs)
        print("params:",params_m,"\nfun@min:",fun_min_m,"\njacob@min:",jacob_min_m)


        # Convert fitted parameters into a dictionnary
        dict_params_m = paramslist_to_dict( params_m,p.PARAM_NAMES_FLAT)

        # rescale photometry datapoints
        xphot_rest,yphot_rest,eyphot_rest,factor = rescale_photometry(dict_params_m,list_wlmean_f_sel,data_selected_mags,data_selected_magserr,z_obs)


        # plot model with photometry
        #plot_fit_ssp_photometry(dict_params_m,list_wlmean_f_sel,data_selected_mags,data_selected_magserr,z_obs, subtit = title_spec ,ax=None)

        #rescale Fors2 spectroscopy
        Xspec_data_rest,Yspec_data_rest,EYspec_data_rest,factor = rescale_spectroscopy(dict_params_m,Xspec_data,Yspec_data,EYspec_data,z_obs)

        # fit spectroscopy
        lbfgsb = jaxopt.ScipyBoundedMinimize(fun=lik_spec, method="L-BFGS-B")
        res_s = lbfgsb.run(init_params, bounds=(params_min ,params_max ), wls=Xspec_data_rest, F=Yspec_data_rest, sigma_obs = EYspec_data_rest,z_obs=z_obs)
        params_s,fun_min_s,jacob_min_s,inv_hessian_min_s = get_infos_spec(res_s, lik_spec, wls=Xspec_data, F=Yspec_data,eF=EYspec_data,z_obs=z_obs)
        print("params:",params_s,"\nfun@min:",fun_min_s,"\njacob@min:",jacob_min_s)


        # Convert fitted parameters with spectroscopy into a dictionnary
        dict_params_s = paramslist_to_dict( params_s,p.PARAM_NAMES_FLAT)

        # plot fit for spectroscopy only
        #plot_fit_ssp_spectroscopy(dict_params_s,Xspec_data_rest,Yspec_data_rest,EYspec_data_rest,z_obs,subtit = title_spec)


        # combining spectro and photometry
        Xc = [Xspec_data_rest, Xf_sel]
        Yc = [Yspec_data_rest,  data_selected_mags ]
        EYc = [EYspec_data_rest, data_selected_magserr]
        weight_spec = 0.5
        Ns = len(Yspec_data_rest)
        Nm = len(data_selected_mags)
        Nc = Ns+Nm

        # do the combined fit
        lbfgsb = jaxopt.ScipyBoundedMinimize(fun=lik_comb, method="L-BFGS-B")
        res_c = lbfgsb.run(init_params, bounds=(params_min ,params_max ), xc=Xc, datac=Yc,sigmac=EYc,z_obs=z_obs,weight=weight_spec)
        params_c,fun_min_c,jacob_min_c,inv_hessian_min_c = get_infos_comb(res_c, lik_comb, xc=Xc, datac=Yc,sigmac=EYc,z_obs=z_obs,weight=weight_spec)
        print("params_c:",params_c,"\nfun@min:",fun_min_c,"\njacob@min:",jacob_min_c)
        #      ,"\n invH@min:",inv_hessian_min_c)
        params_cm,fun_min_cm,jacob_min_cm,inv_hessian_min_cm  = get_infos_mag(res_c, lik_mag,  xf = Xf_sel, mgs = data_selected_mags, mgse = data_selected_magserr,z_obs=z_obs)
        print("params_cm:",params_cm,"\nfun@min:",fun_min_cm,"\njacob@min:",jacob_min_cm)
        params_cs,fun_min_cs,jacob_min_cs,inv_hessian_min_cs = get_infos_spec(res_c, lik_spec, wls=Xspec_data_rest, F=Yspec_data_rest,eF=EYspec_data_rest,z_obs=z_obs)
        print("params_cs:",params_cs,"\nfun@min:",fun_min_cs,"\njacob@min:",jacob_min_cs)

        #dict_to_save
        dict_out = OrderedDict()
        dict_out["fors2name"] = selected_spectrum_tag
        dict_out["zobs"] = z_obs
        dict_out["Nc"]   = Nc
        dict_out["Ns"]   = Ns
        dict_out["Nm"]   = Nm
        dict_out["funcmin_c"] = fun_min_c
        dict_out["funcmin_m"] = fun_min_cm
        dict_out["funcmin_s"] = fun_min_cs

        # convert into a dictionnary
        dict_params_c = paramslist_to_dict( params_c,p.PARAM_NAMES_FLAT)

        dict_out.update(dict_params_c)

        # plot the combined fit
        #plot_fit_ssp_spectrophotometry(dict_params_c ,Xspec_data_rest,Yspec_data_rest,EYspec_data_rest,xphot_rest,yphot_rest,eyphot_rest,z_obs=z_obs,subtit = title_spec )

        #load starlight spectrum
        dict_sl = sl.getspectrum_fromgroup(selected_spectrum_tag)
        # rescale starlight spectrum
        w_sl ,fnu_sl , _ = rescale_starlight_inrangefors2(dict_sl["wl"],dict_sl["fnu"],Xspec_data_rest,Yspec_data_rest )

        # plot starlight
        if FLAG_PLOT:
            plot_fit_ssp_spectrophotometry_sl(dict_params_c ,Xspec_data_rest,Yspec_data_rest,EYspec_data_rest,xphot_rest,yphot_rest,eyphot_rest,w_sl,fnu_sl,z_obs=z_obs,subtit = title_spec)

        # plot SFR
        #plot_SFH(dict_params_c,z_obs,subtit = title_spec , ax=None)

        #save parameters
        filename_params = f"fitparams_{selected_spectrum_tag}.pickle"
        with open(filename_params, 'wb') as f:
            print(dict_out)
            pickle.dump(dict_out, f)

        if FLAG_PLOT:
            plt.show()
