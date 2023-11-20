"""Module to test parameters module"""

# pylint: disable=line-too-long
# pylint: disable=missing-final-newline
# pylint: disable=W0612
# pylint: disable=unused-import

# python -m unittest test_parameters.py

import unittest

import numpy as np

from fors2tostellarpopsynthesis.parameters import (INIT_PARAMS,
                                                   PARAM_NAMES_FLAT,
                                                   DICT_PARAMS_true)


class ParametersTestCase(unittest.TestCase):
    """A test case for the fors2tostellarpopsynthesis parameters package."""

    def test_init_params(self):
        """Check the values of INIT_PARAMS
        """
        self.assertTrue(np.allclose(INIT_PARAMS,
                                    np.array([12.,0.05,2.5,1., 12.,
                                              -1.,1., -1.,2.,1.,
                                              -0.50725,-1.01773,-0.212307,
                                              1.,2., -0.25, 1.]),rtol=1e-02, atol=1e-02))
    def test_param_names_flat(self):
        """Check the values of the list PARAM_NAMES_FLAT
        """
        self.assertSequenceEqual(PARAM_NAMES_FLAT,['MAH_lgmO','MAH_logtc',
                                                            'MAH_early_index','MAH_late_index',
                                                            'MS_lgmcrit','MS_lgy_at_mcrit','MS_indx_lo','MS_indx_hi','MS_tau_dep',
                                                            'Q_lg_qt','Q_qlglgdt','Q_lg_drop','Q_lg_rejuv','AV','UV_BUMP_AMPL','PLAW_SLOPE','SCALEF'])

if __name__ == "__main__":
    unittest.main()