"""Module to test filters module"""

# pylint: disable=line-too-long
# pylint: disable=missing-final-newline
# pylint: disable=W0612


import unittest

import numpy as np

from fors2tostellarpopsynthesis.filters import FilterInfo


class FiltersTestCase(unittest.TestCase):
    """A test case for the fors2tostellarpopsynthesis filters package."""

    def test_filterconfig(self):
        """test if we have the right config of FilterInfo
        """
        ps = FilterInfo()
        index_list = ps.filters_indexlist
        name_list = ps.filters_namelist
        survey_list = ps.filters_surveylist

        self.assertTrue(np.allclose(index_list,[0,1,2,3,4,5,6,7,8,9,10]),msg="fors2tostellarpopsynthesis.filters.FilterInfo error")

if __name__ == "__main__":
    unittest.main()