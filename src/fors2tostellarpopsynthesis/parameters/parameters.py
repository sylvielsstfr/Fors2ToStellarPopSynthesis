"""Module to handle parameters
"""
# pylint: disable=trailing-newlines

import itertools
from collections import OrderedDict

import jax.numpy as jnp
import numpy as np
from diffstar.defaults import (DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS,
                               DEFAULT_Q_PARAMS)

# DSPS parameters
MAH_PARAMNAMES = ["MAH_lgmO","MAH_logtc","MAH_early_index","MAH_late_index"]
MS_PARAMNAMES = ["MS_lgmcrit", "MS_lgy_at_mcrit", "MS_indx_lo", "MS_indx_hi", "MS_tau_dep"]
Q_PARAMNAMES = ["Q_lg_qt", "Q_qlglgdt", "Q_lg_drop", "Q_lg_rejuv"]

DEFAULT_MAH_PARAMS_MIN = DEFAULT_MAH_PARAMS + np.array([-3., -0.01, -1.5,-0.5])
DEFAULT_MAH_PARAMS_MAX = DEFAULT_MAH_PARAMS + np.array([2., +0.01, +1.5,+0.5])

DEFAULT_MS_PARAMS_MIN = DEFAULT_MS_PARAMS - 0.25*np.ones((5))
DEFAULT_MS_PARAMS_MAX = DEFAULT_MS_PARAMS + 0.25*np.ones((5))

DEFAULT_Q_PARAMS_MIN = DEFAULT_Q_PARAMS - 0.1*np.ones((4,))
DEFAULT_Q_PARAMS_MAX = DEFAULT_Q_PARAMS + 0.1*np.ones((4,))


# Dust parameters
AV = 1
UV_BUMP_AMPL = 2.0
PLAW_SLOPE = -0.25
DEFAULT_DUST_PARAMS = [AV, UV_BUMP_AMPL, PLAW_SLOPE]
DUST_PARAMNAMES = ["AV", "UV_BUMP_AMPL", "PLAW_SLOPE"]
DEFAULT_DUST_PARAMS_MIN = DEFAULT_DUST_PARAMS + np.array([-1.,-1.,-0.1])
DEFAULT_DUST_PARAMS_MAX = DEFAULT_DUST_PARAMS + np.array([2.,1.,0.25])


#Scaling parameters
SCALEF = 1.0
DEFAULT_SCALEF_PARAMS = np.array([SCALEF])
SCALEF_PARAMNAMES = ["SCALEF"]
DEFAULT_SCALEF_PARAMS_MIN =  np.array([1.])
DEFAULT_SCALEF_PARAMS_MAX = np.array([1.])

# bound parameters together
DEFAULT_PARAMS = [DEFAULT_MAH_PARAMS,DEFAULT_MS_PARAMS,DEFAULT_Q_PARAMS,
                  DEFAULT_DUST_PARAMS, DEFAULT_SCALEF_PARAMS ]

PARAMS_MIN = np.concatenate(([DEFAULT_MAH_PARAMS_MIN,DEFAULT_MS_PARAMS_MIN,DEFAULT_Q_PARAMS_MIN,
                              DEFAULT_DUST_PARAMS_MIN,DEFAULT_SCALEF_PARAMS_MIN]))
PARAMS_MAX = np.concatenate(([DEFAULT_MAH_PARAMS_MAX,DEFAULT_MS_PARAMS_MAX,DEFAULT_Q_PARAMS_MAX,
                              DEFAULT_DUST_PARAMS_MAX,DEFAULT_SCALEF_PARAMS_MAX]))

INIT_PARAMS = np.concatenate(DEFAULT_PARAMS)
INIT_PARAMS = jnp.array(INIT_PARAMS)

PARAM_NAMES = [MAH_PARAMNAMES,MS_PARAMNAMES,Q_PARAMNAMES,DUST_PARAMNAMES,SCALEF_PARAMNAMES]
PARAM_NAMES_FLAT = list(itertools.chain(*PARAM_NAMES))

DICT_PARAM_MAH_true = OrderedDict([(MAH_PARAMNAMES[0],DEFAULT_MAH_PARAMS[0]),
                                         (MAH_PARAMNAMES[1],DEFAULT_MAH_PARAMS[1]),
                                         (MAH_PARAMNAMES[2],DEFAULT_MAH_PARAMS[2]),
                                         (MAH_PARAMNAMES[3],DEFAULT_MAH_PARAMS[3])
                                         ])

DICT_PARAM_MAH_true_selected = OrderedDict([(MAH_PARAMNAMES[0],DEFAULT_MAH_PARAMS[0]),
                                         ])

DICT_PARAM_MS_true = OrderedDict([(MS_PARAMNAMES[0],DEFAULT_MS_PARAMS[0]),
                                         (MS_PARAMNAMES[1],DEFAULT_MS_PARAMS[1]),
                                         (MS_PARAMNAMES[2],DEFAULT_MS_PARAMS[2]),
                                         (MS_PARAMNAMES[3],DEFAULT_MS_PARAMS[3]),
                                         (MS_PARAMNAMES[4],DEFAULT_MS_PARAMS[4])])

DICT_PARAM_Q_true = OrderedDict([(Q_PARAMNAMES[0],DEFAULT_Q_PARAMS[0]),
                                         (Q_PARAMNAMES[1],DEFAULT_Q_PARAMS[1]),
                                         (Q_PARAMNAMES[2],DEFAULT_Q_PARAMS[2]),
                                         (Q_PARAMNAMES[3],DEFAULT_Q_PARAMS[3])])

DICT_PARAM_DUST_true = OrderedDict([(DUST_PARAMNAMES[0],DEFAULT_DUST_PARAMS[0]),
                                         (DUST_PARAMNAMES[1],DEFAULT_DUST_PARAMS[1]),
                                         (DUST_PARAMNAMES[2],DEFAULT_DUST_PARAMS[2])])
DICT_PARAM_DUST_true_selected = OrderedDict([(DUST_PARAMNAMES[0],DEFAULT_DUST_PARAMS[0])])

DICT_PARAM_SCALEF_true = OrderedDict([(SCALEF_PARAMNAMES[0],DEFAULT_SCALEF_PARAMS[0]) ])

DICT_PARAMS_true = DICT_PARAM_MAH_true
DICT_PARAMS_true.update(DICT_PARAM_MS_true)
DICT_PARAMS_true.update(DICT_PARAM_Q_true)
DICT_PARAMS_true.update(DICT_PARAM_DUST_true)
DICT_PARAMS_true.update(DICT_PARAM_SCALEF_true)



def paramslist_to_dict(params_list,param_names):
    """
    Convert the list of parameters into a dictionnary
    :param params_list: list of params values
    :type params_list: float in an array

    :param param_names: list of parameter names
    :type params_names: strings in an array

    :return: dictionnary of parameters
    :rtype: dictionnary
    """
    list_of_tuples = list(zip(param_names,params_list))
    dict_params = OrderedDict(list_of_tuples )
    return dict_params

