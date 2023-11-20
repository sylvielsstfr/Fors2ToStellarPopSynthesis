"""Module to test forst2 and starlight IO module"""

# pylint: disable=line-too-long
# pylint: disable=missing-final-newline
# pylint: disable=W0612
# pylint: disable=unused-import

# python -m unittest test_fors2starlightio.py

import pathlib as pl
import unittest

import numpy as np

from fors2tostellarpopsynthesis.fors2starlightio import (
    FILENAME_FORS2PHOTOM, FILENAME_STARLIGHT, FULL_FILENAME_FORS2PHOTOM,
    FULL_FILENAME_STARLIGHT, Fors2DataAcess, SLDataAcess, _getPackageDir,
    convertflambda_to_fnu, flux_norm, ordered_keys)


class TestPathsAndFilesCase(unittest.TestCase):
    """Test suite for check if input files for fors2 and Starlight ecists

    :param unittest: class to apply unit test
    :type unittest: TestCase
    """

    def test_filename_fors2(self):
        """Check fors2 filename
        """
        self.assertEqual(FILENAME_FORS2PHOTOM,'data/FORS2spectraGalexKidsPhotom.h5')

    def test_filename_sl(self):
        """Check StarLight filename
        """
        self.assertEqual(FILENAME_STARLIGHT,'data/SLspectra.h5')

    def test_iopath(self):
        """Test if directory containing io tools exists
        """
        path = _getPackageDir()
        self.assertTrue(pl.Path(path).is_dir())

    def test_fors2file(self):
        """Test if fors2 file exists
        """
        self.assertTrue(pl.Path(FULL_FILENAME_FORS2PHOTOM).is_file())

    def test_slfile(self):
        """Test if starlight file exists
        """
        self.assertTrue(pl.Path(FULL_FILENAME_STARLIGHT).is_file())


class TestFors2IOCase(unittest.TestCase):
    """
    Test the IO for Fors2DataAcess class
    """

    def test_open_file(self):
        """test if can create an instance of Fors2DataAcess
        """
        fors2 = Fors2DataAcess()
        self.assertTrue(isinstance(fors2,Fors2DataAcess))


class TestSLIOCase(unittest.TestCase):
    """
    Test the IO for SLDataAcess class
    """

    def test_open_file(self):
        """test if can create an instance of SLDataAcess
        """
        sl = SLDataAcess()
        self.assertTrue(isinstance(sl,SLDataAcess))


if __name__ == "__main__":
    unittest.main()