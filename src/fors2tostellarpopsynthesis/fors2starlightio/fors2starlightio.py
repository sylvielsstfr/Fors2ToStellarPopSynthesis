""" Module to handle io of fors2, stalight spectra and photometric data"""

# pylint: disable=line-too-long
# pylint: disable=trailing-newlines
# pylint: disable=redundant-u-string-prefix
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=unused-import

import os
from collections import OrderedDict

import h5py
import numpy as np
from astropy import constants as const
from astropy import units as u
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

U_FNU = u.def_unit(u'fnu',  u.erg / (u.cm**2 * u.s * u.Hz))
U_FL = u.def_unit(u'fl',  u.erg / (u.cm**2 * u.s * u.AA))

# ---------------
ordered_keys = ['name','num','ra','dec', 'redshift','Rmag','RT', 'RV','eRV','Nsp','lines',
                'ra_galex','dec_galex','fuv_mag', 'fuv_magerr','nuv_mag', 'nuv_magerr',
                'fuv_flux', 'fuv_fluxerr','nuv_flux', 'nuv_fluxerr','asep_galex',
                'ID', 'KIDS_TILE','RAJ2000','DECJ2000','Z_ML', 'Z_B','asep_kids','CLASS_STAR',
                'MAG_GAAP_u','MAG_GAAP_g','MAG_GAAP_r','MAG_GAAP_i','MAG_GAAP_Z','MAG_GAAP_Y','MAG_GAAP_J', 'MAG_GAAP_H','MAG_GAAP_Ks',
                'MAGERR_GAAP_u','MAGERR_GAAP_g','MAGERR_GAAP_r','MAGERR_GAAP_i','MAGERR_GAAP_Z','MAGERR_GAAP_Y','MAGERR_GAAP_J','MAGERR_GAAP_H','MAGERR_GAAP_Ks',
                'FLUX_GAAP_u','FLUX_GAAP_g','FLUX_GAAP_r','FLUX_GAAP_i','FLUX_GAAP_Z','FLUX_GAAP_Y','FLUX_GAAP_J', 'FLUX_GAAP_H','FLUX_GAAP_Ks',
                'FLUXERR_GAAP_u','FLUXERR_GAAP_g','FLUXERR_GAAP_r','FLUXERR_GAAP_i','FLUXERR_GAAP_Z','FLUXERR_GAAP_Y','FLUXERR_GAAP_J','FLUXERR_GAAP_H','FLUXERR_GAAP_Ks',
                'FLUX_RADIUS', 'EXTINCTION_u','EXTINCTION_g','EXTINCTION_r', 'EXTINCTION_i',]


FILENAME_FORS2PHOTOM = "data/FORS2spectraGalexKidsPhotom.h5"
FILENAME_STARLIGHT = "data/SLspectra.h5"

FULL_FILENAME_FORS2PHOTOM = os.path.join(__path__,FILENAME_FORS2PHOTOM)
FULL_FILENAME_STARLIGHT = os.path.join(__path__,FILENAME_STARLIGHT)
#---------------
def convertflambda_to_fnu(wl:np.array, flambda:np.array) -> np.array:
    """
    Convert spectra density flambda to fnu.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param flambda: flux density in erg/s/cm2 /AA or W/cm2/AA
    :type flambda: float

    :return: fnu, flux density in erg/s/cm2/Hz or W/cm2/Hz
    :rtype: float

    Compute Fnu = wl**2/c Flambda
    check the conversion units with astropy units and constants

    """

    fnu = (flambda*U_FL*(wl*u.AA)**2/const.c).to(U_FNU)/(1*U_FNU)
    #fnu = (flambda* (u.erg / (u.cm**2 * u.s * u.AA)) *(wl*u.AA)**2/const.c).to( u.erg / (u.cm**2 * u.s * u.Hz))/(u.erg / (u.cm**2 * u.s * u.Hz))

    return fnu

def flux_norm(
    wl: np.array,
    fl: np.array,
    wlcenter: float = 6231.,
    wlwdt: float = 50.
) -> float:
    """Normalize the flux at a given wavelength

    :param wl: wavelength array of the spectrum
    :type wl: np.array of floats

    :param fl: flux rray of the spectrum
    :type fl: np.array of float

    :param wlcenter: wavelength where the normalisation is done, defaults to 6231.
    :type wlcenter: float, optional

    :param wlwdt: wavelength width where the normalisation factor si calculated, defaults to 50.
    :type wlwdt: float, optional

    :return: the normalisation factor
    :rtype: float
    """
    lambda_red = wlcenter
    lambda_width = wlwdt
    lambda_sel_min = lambda_red-lambda_width /2.
    lambda_sel_max = lambda_red+lambda_width /2.

    idx_wl_sel = np.where(np.logical_and(wl>= lambda_sel_min,wl<= lambda_sel_max))[0]
    flarr_norm = fl[idx_wl_sel]
    return np.median(flarr_norm)



class Fors2DataAcess():
    """IO class to read  Fors2 h5 file
    """
    def __init__(self,filename:str = FULL_FILENAME_FORS2PHOTOM):
        """read the input filename

        :param filename:name of fors2 file
        :type filename: str
        """
        if os.path.isfile(filename):
            self.hf = h5py.File(filename, 'r')
            self.list_of_groupkeys = list(self.hf.keys())
             # pick one key
            key_sel =  self.list_of_groupkeys[0]
            # pick one group
            group = self.hf.get(key_sel)
            #pickup all attribute names
            self.list_of_subgroup_keys = []
            for k in group.attrs.keys():
                self.list_of_subgroup_keys.append(k)
        else:
            self.hf = None
            self.list_of_groupkeys = []
            self.list_of_subgroup_keys = []

    def close_file(self):
        """Close hdf5 file
        """
        self.hf.close()

    def get_list_of_groupkeys(self):
        """Provides the list of fors2 spectra identifiers
        :return: return list of keys of the h5 file where each key match a Fors2 spectrum tag id
        :rtype: list of str
        """
        return self.list_of_groupkeys

    def get_list_subgroup_keys(self):
        """Provide the list of parameters linked to the fors2 spectrum,
        example like redshift,...

        :return: return list of sub-keys related to more parametes linked to the fors2 spectrum
        :rtype: list of str
        """
        return self.list_of_subgroup_keys

    def getattribdata_fromgroup(self,groupname:str)->OrderedDict:
        """provide the parameters of a fors2 spectrum

        :param groupname: identifier of the fors2 spectrum
        :type groupname: string

        :return: values of all parameters of the fors2 spectrum parameters
        :rtype: OrderedDict
        """
        attr_dict = OrderedDict()
        if groupname in self.list_of_groupkeys:
            group = self.hf.get(groupname)
            for  nameval in self.list_of_subgroup_keys:
                attr_dict[nameval] = group.attrs[nameval]
        else:
            print(f'getattribdata_fromgroup : No group {groupname}')
        return attr_dict


    def getspectrum_fromgroup(self,groupname:str) -> dict:
        """return the fors2 spectrum

        :param groupname: the fors2 spectrum identification tag name
        :type groupname: str

        :return: returns the spectrum with a wavelength array, fnu array and flambda array
        :rtype: dict
        """
        spec_dict = {}
        if groupname in self.list_of_groupkeys:
            group = self.hf.get(groupname)
            wl = np.array(group.get("wl"))
            fl = np.array(group.get("fl"))
            spec_dict["wl"] = wl
            spec_dict["fl"] = fl

            #convert to fnu
            fnu = convertflambda_to_fnu(wl, fl)
            fnorm = flux_norm(wl,fnu)
            spec_dict["fnu"] = fnu/fnorm

        else:
            print(f'getspectrum_fromgroup : No group {groupname}')
        return spec_dict

    #kernel = kernels.RBF(0.5, (8000, 10000.0))
    #gp = GaussianProcessRegressor(kernel=kernel ,random_state=0)

    def getspectrumcleanedemissionlines_fromgroup(self,groupname:str,gp:GaussianProcessRegressor,nsigs:float=8.) ->dict:
        """Clean the spectrum from any emission line or any defects i the spectrum

        :param groupname: identifier tag name of the spectrum
        :type groupname: str
        :param gp: The guassian process regerssor to be used
        :type gp: GaussianProcessRegressor

        kernel = kernels.RBF(0.5, (8000, 10000.0))
        gp = GaussianProcessRegressor(kernel=kernel ,random_state=0)

        :param nsigs: the number of std dev, defaults to 8.
        :type nsigs: float, optional
        :return: the spectrum cleaned
        :rtype: dict
        """
        spec_dict = {}
        if groupname in self.list_of_groupkeys:
            group = self.hf.get(groupname)
            wl = np.array(group.get("wl"))
            fl = np.array(group.get("fl"))

            #convert to fnu
            fnu = convertflambda_to_fnu(wl, fl)
            fnorm = flux_norm(wl,fnu)

            # fit gaussian process
            X = wl
            Y = fnu/fnorm

            DeltaY,Z = Y - gp.predict(X[:, None], return_std=True)
            background = np.sqrt(np.median(DeltaY**2))
            #indexes_toremove = np.where(np.abs(DeltaY)> nsigs * background)[0]
            indexes_toremove = np.where(np.logical_or(np.abs(DeltaY)> nsigs * background,Y<=0))[0]

            Xclean = np.delete(X,indexes_toremove)
            Yclean  = np.delete(Y,indexes_toremove)
            Zclean  = np.delete(Z,indexes_toremove)

            spec_dict["wl"] = Xclean
            spec_dict["fnu"] = Yclean
            spec_dict["bg"] = np.abs(Zclean) # strange scikit learn bug returning negative std.
            spec_dict["bg_med"] = background # overestimated median error

        else:
            print(f'getspectrum_fromgroup : No group {groupname}')
        return spec_dict


class SLDataAcess():
    """Handle Access to starlight spectra
    """
    def __init__(self,filename:str = FULL_FILENAME_STARLIGHT):
        """Check StarLight spectra file exists and can be open

        :param filename: filename of starlight spectra
        :type filename: str
        """
        if os.path.isfile(filename):
            self.hf = h5py.File(filename, 'r')
            self.list_of_groupkeys = list(self.hf.keys())
             # pick one key
            key_sel =  self.list_of_groupkeys[0]
            # pick one group
            group = self.hf.get(key_sel)
            #pickup all attribute names
            self.list_of_subgroup_keys = []
            for k in group.attrs.keys():
                self.list_of_subgroup_keys.append(k)
        else:
            self.hf = None
            self.list_of_groupkeys = []
            self.list_of_subgroup_keys = []

    def close_file(self):
        """Close h5 file
        """
        self.hf.close()

    def get_list_of_groupkeys(self) -> list:
        """Get list of StarLight spectrum tag names

        :return: The list of available StarLight spectrum tag names
        :rtype: list
        """
        return self.list_of_groupkeys


    def get_list_subgroup_keys(self) ->list:
        """List of parameters associated with each StarLight spectrum

        :return: The list of parameters
        :rtype: list
        """
        return self.list_of_subgroup_keys

    def getattribdata_fromgroup(self,groupname:str) -> OrderedDict:
        """The values associated to a given StarLight spectrum

        :param groupname: tag name of the StarLight specrum
        :type groupname: str
        :return: The values associated to the StarLight spectrum
        :rtype: OrderedDict
        """
        attr_dict = OrderedDict()
        if groupname in self.list_of_groupkeys:
            group = self.hf.get(groupname)
            for  nameval in self.list_of_subgroup_keys:
                attr_dict[nameval] = group.attrs[nameval]
        else:
            print(f'getattribdata_fromgroup : No group {groupname}')
        return attr_dict


    def getspectrum_fromgroup(self,groupname:str) ->dict:
        """The spectrum associated to a given StarLight spectrum tag name

        :param groupname:  tag name of the StarLight specrum
        :type groupname: str
        :return: the spectra (wavelength, fnu and flambda)
        :rtype: dict
        """
        spec_dict = {}
        if groupname in self.list_of_groupkeys:
            group = self.hf.get(groupname)
            wl = np.array(group.get("wl"))
            fl = np.array(group.get("fl"))
            spec_dict["wl"] = wl
            spec_dict["fl"] = fl

            #convert to fnu
            fnu = convertflambda_to_fnu(wl, fl)
            fnorm = flux_norm(wl,fnu)
            spec_dict["fnu"] = fnu/fnorm


        else:
            print(f'getspectrum_fromgroup : No group {groupname}')
        return spec_dict

