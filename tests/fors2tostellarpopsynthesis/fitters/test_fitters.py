"""Module to test fitters module"""

# pylint: disable=line-too-long
# pylint: disable=missing-final-newline
# pylint: disable=W0612
# pylint: disable=unused-import

# python -m unittest test_fitters.py

import pathlib as pl
import unittest

from dsps import load_ssp_templates
from dsps.data_loaders.defaults import SSPData

from fors2tostellarpopsynthesis.fitters.fitter_jaxopt import (
    FILENAME_SSP_DATA, FULLFILENAME_SSP_DATA, _get_package_dir)


class TestPathsAndFilesCase(unittest.TestCase):
    """Test suite for check if input files for fors2 and Starlight ecists

    :param unittest: class to apply unit test
    :type unittest: TestCase
    """

    def test_filename_ssp(self):
        """Check ssp filename
        """
        self.assertEqual(FILENAME_SSP_DATA,'data/tempdata.h5')



    def test_iopath(self):
        """Test if directory containing io tools exists
        """
        path = _get_package_dir()
        self.assertTrue(pl.Path(path).is_dir())

    def test_sspfile(self):
        """Test if ssp file exists
        The file must be downloaded from 
        https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/
        please refer to https://dsps.readthedocs.io/en/latest/quickstart.html
        """

        self.assertTrue(pl.Path(FULLFILENAME_SSP_DATA).is_file())




class TestFors2IOCase(unittest.TestCase):
    """
    Test the IO for SSP data
    """

    def test_open_file(self):
        """test if can create an instance of SSP templates
        """
        ssp_templates =  load_ssp_templates(FULLFILENAME_SSP_DATA)
        self.assertTrue(isinstance(ssp_templates,SSPData))





if __name__ == "__main__":
    unittest.main()
