"""Module to handle parameters
"""
# pylint: disable=trailing-newlines
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long
# pylint: disable=W1309

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
PARAMS_MIN= jnp.array(PARAMS_MIN)
PARAMS_MAX= jnp.array(PARAMS_MAX)

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

class SSPParametersFit():
    """Contain all necessary parameters to fit SSP
    """
    def __init__(self):
        """Book space for initialisation constants
        """
        # DSPS parameters

        self.DEFAULT_MAH_PARAMS = DEFAULT_MAH_PARAMS
        self.DEFAULT_MS_PARAMS = DEFAULT_MS_PARAMS
        self.DEFAULT_Q_PARAMS = DEFAULT_Q_PARAMS

        self.MAH_PARAMNAMES = ["MAH_lgmO","MAH_logtc","MAH_early_index","MAH_late_index"]
        self.MS_PARAMNAMES = ["MS_lgmcrit", "MS_lgy_at_mcrit", "MS_indx_lo", "MS_indx_hi", "MS_tau_dep"]
        self.Q_PARAMNAMES = ["Q_lg_qt", "Q_qlglgdt", "Q_lg_drop", "Q_lg_rejuv"]

        self.DEFAULT_MAH_PARAMS_MIN = self.DEFAULT_MAH_PARAMS + np.array([-3., -0.01, -1.5,-0.5])
        self.DEFAULT_MAH_PARAMS_MAX = self.DEFAULT_MAH_PARAMS + np.array([2., +0.01, +1.5,+0.5])

        self.DEFAULT_MS_PARAMS_MIN = self.DEFAULT_MS_PARAMS - 0.25*np.ones((5))
        self.DEFAULT_MS_PARAMS_MAX = self.DEFAULT_MS_PARAMS + 0.25*np.ones((5))

        self.DEFAULT_Q_PARAMS_MIN = self.DEFAULT_Q_PARAMS - 0.1*np.ones((4,))
        self.DEFAULT_Q_PARAMS_MAX = self.DEFAULT_Q_PARAMS + 0.1*np.ones((4,))


        # Dust parameters
        self.AV = 1.0
        self.UV_BUMP = 2.0
        self.PLAW_SLOPE = -0.25
        self.DEFAULT_DUST_PARAMS = [self.AV, self.UV_BUMP, self.PLAW_SLOPE]
        self.DUST_PARAMNAMES = ["AV", "UV_BUMP", "PLAW_SLOPE"]
        self.DEFAULT_DUST_PARAMS_MIN = self.DEFAULT_DUST_PARAMS + np.array([-1.,-1.,-0.1])
        self.DEFAULT_DUST_PARAMS_MAX = self.DEFAULT_DUST_PARAMS + np.array([2.,1.,0.25])


        #Scaling parameters
        self.SCALEF = 1.0
        self.DEFAULT_SCALEF_PARAMS = np.array([self.SCALEF])
        self.SCALEF_PARAMNAMES = ["SCALEF"]
        self.DEFAULT_SCALEF_PARAMS_MIN =  np.array([1.])
        self.DEFAULT_SCALEF_PARAMS_MAX = np.array([1.])

        # bound parameters together
        self.DEFAULT_PARAMS = [self.DEFAULT_MAH_PARAMS,self.DEFAULT_MS_PARAMS,self.DEFAULT_Q_PARAMS,
                  self.DEFAULT_DUST_PARAMS, self.DEFAULT_SCALEF_PARAMS ]

        self.PARAMS_MIN = np.concatenate(([self.DEFAULT_MAH_PARAMS_MIN,self.DEFAULT_MS_PARAMS_MIN,self.DEFAULT_Q_PARAMS_MIN,
                              self.DEFAULT_DUST_PARAMS_MIN,self.DEFAULT_SCALEF_PARAMS_MIN]))
        self.PARAMS_MAX = np.concatenate(([self.DEFAULT_MAH_PARAMS_MAX,self.DEFAULT_MS_PARAMS_MAX,self.DEFAULT_Q_PARAMS_MAX,
                              self.DEFAULT_DUST_PARAMS_MAX,self.DEFAULT_SCALEF_PARAMS_MAX]))

        self.PARAMS_MIN= jnp.array(self.PARAMS_MIN)
        self.PARAMS_MAX= jnp.array(self.PARAMS_MAX)

        self.INIT_PARAMS = np.concatenate(self.DEFAULT_PARAMS)
        self.INIT_PARAMS = jnp.array(INIT_PARAMS)

        self.PARAM_NAMES = [self.MAH_PARAMNAMES,self.MS_PARAMNAMES,self.Q_PARAMNAMES,self.DUST_PARAMNAMES,self.SCALEF_PARAMNAMES]
        self.PARAM_NAMES_FLAT = list(itertools.chain(*self.PARAM_NAMES))

        self.DICT_PARAM_MAH_true = OrderedDict([(self.MAH_PARAMNAMES[0],self.DEFAULT_MAH_PARAMS[0]),
                                         (self.MAH_PARAMNAMES[1],self.DEFAULT_MAH_PARAMS[1]),
                                         (self.MAH_PARAMNAMES[2],self.DEFAULT_MAH_PARAMS[2]),
                                         (self.MAH_PARAMNAMES[3],self.DEFAULT_MAH_PARAMS[3])
                                         ])


        self.DICT_PARAM_MS_true = OrderedDict([(self.MS_PARAMNAMES[0],self.DEFAULT_MS_PARAMS[0]),
                                         (self.MS_PARAMNAMES[1],self.DEFAULT_MS_PARAMS[1]),
                                         (self.MS_PARAMNAMES[2],self.DEFAULT_MS_PARAMS[2]),
                                         (self.MS_PARAMNAMES[3],self.DEFAULT_MS_PARAMS[3]),
                                         (self.MS_PARAMNAMES[4],self.DEFAULT_MS_PARAMS[4])])

        self.DICT_PARAM_Q_true = OrderedDict([(self.Q_PARAMNAMES[0],self.DEFAULT_Q_PARAMS[0]),
                                         (self.Q_PARAMNAMES[1],self.DEFAULT_Q_PARAMS[1]),
                                         (self.Q_PARAMNAMES[2],self.DEFAULT_Q_PARAMS[2]),
                                         (self.Q_PARAMNAMES[3],self.DEFAULT_Q_PARAMS[3])])

        self.DICT_PARAM_DUST_true = OrderedDict([(self.DUST_PARAMNAMES[0],self.DEFAULT_DUST_PARAMS[0]),
                                         (self.DUST_PARAMNAMES[1],self.DEFAULT_DUST_PARAMS[1]),
                                         (self.DUST_PARAMNAMES[2],self.DEFAULT_DUST_PARAMS[2])])


        self.DICT_PARAM_SCALEF_true = OrderedDict([(self.SCALEF_PARAMNAMES[0],self.DEFAULT_SCALEF_PARAMS[0]) ])

        self.DICT_PARAMS_true = self.DICT_PARAM_MAH_true
        self.DICT_PARAMS_true.update(self.DICT_PARAM_MS_true)
        self.DICT_PARAMS_true.update(self.DICT_PARAM_Q_true)
        self.DICT_PARAMS_true.update(self.DICT_PARAM_DUST_true)
        self.DICT_PARAMS_true.update(self.DICT_PARAM_SCALEF_true)

    def __repr__(self) -> str:
        all_str = []
        all_str.append(f"Class SSPParametersFit : ")
        all_str.append(" - Name     : " + str(self.PARAM_NAMES_FLAT))
        all_str.append(" - Mean Val : " + str(self.INIT_PARAMS))
        all_str.append(" - Min Val  : " + str(self.PARAMS_MIN))
        all_str.append(" - Max Val  : " + str(self.PARAMS_MAX))

        return "\n".join(all_str)
