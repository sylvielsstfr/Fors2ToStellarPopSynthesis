""" Module to handle io of fors2, stalight spectra and photometric data"""

# pylint: disable=line-too-long
# pylint: disable=trailing-newlines
# pylint: disable=redundant-u-string-prefix
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=unused-import
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=too-many-locals
# pylint: disable=broad-exception-caught
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace

import os
import re
from collections import OrderedDict

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

kernel = kernels.RBF(0.5, (6000, 10000.0))
gpr = GaussianProcessRegressor(kernel=kernel ,random_state=0)

U_FNU = u.def_unit(u'fnu',  u.erg / (u.cm**2 * u.s * u.Hz))
U_FL = u.def_unit(u'fl',  u.erg / (u.cm**2 * u.s * u.AA))

lambda_FUV = 1528.
lambda_NUV = 2271.
lambda_U = 3650.
lambda_B = 4450.
lambda_G = 4640.
lambda_R = 5580.
lambda_I = 8060.
lambda_Z = 9000.
lambda_Y = 10200.
lambda_J = 12200.
lambda_H = 16300.
lambda_K = 21900.
lambda_L = 34500.

PHOTWL = np.array([lambda_FUV, lambda_NUV, lambda_B, lambda_G, lambda_R ,lambda_I, lambda_Z, lambda_Y, lambda_J, lambda_H, lambda_K ])
PHOTFilterTag = ['FUV','NUV','B','G','R','I','Z','Y','J','H','Ks']

def plot_filter_tag(ax,fluxlist):
    """Show the name of the photometric filter

    :param ax: the axis
    :type ax: axis
    :param fluxlist: _description_
    :type fluxlist: _type_
    """
    goodfl = fluxlist[np.isfinite(fluxlist)]
    ymin = np.mean(goodfl)
    dy=ymin/5

    for idx,flux in enumerate(fluxlist):
        if np.isfinite(flux):
            #ax.text(WL[idx],flux, FilterTag[idx],fontsize=10,ha='center', va='bottom')

            fl = flux - dy
            if fl <0:
                fl += 2*dy
            ax.text(PHOTWL[idx],fl, PHOTFilterTag[idx],fontsize=16,color="g",weight='bold',ha='center', va='bottom')

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



def _getPackageDir():
    """This method must live in the top level of this package, so if this
    moves to a utils file then the returned path will need to account for that.
    """
    dirname = os.path.dirname(__file__)
    return dirname



FULL_FILENAME_FORS2PHOTOM = os.path.join(_getPackageDir(),FILENAME_FORS2PHOTOM)
FULL_FILENAME_STARLIGHT = os.path.join(_getPackageDir(),FILENAME_STARLIGHT)




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

def flux_norm(wl: np.array,fl: np.array,wlcenter: float = 6231.,wlwdt: float = 50.) -> float:
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

def convert_flux_torestframe(wl: np.array,fl: np.array,redshift:float=0.) -> tuple[np.array, np.array]:
    """Transform the current flux into restframe

    :param wl: wavelength
    :type wl: np.array
    :param fl: flux
    :type fl: np.array
    :param redshift: redshift of the object, defaults to 0
    :type redshift: float, optional
    :return: the spectrum blueshifted in restframe
    :rtype: tuple[np.array, np.array]
    """
    factor = 1.+redshift
    return wl/factor,fl #*factor

def convert_flux_toobsframe(wl: np.array,fl: np.array,redshift:float=0.) -> tuple[np.array, np.array] :
    """convert flux to observed frame

    :param wl: wavelength
    :type wl: np.array
    :param fl: flux
    :type fl: np.array
    :param redshift: redshift of the object, defaults to 0
    :type redshift: int, optional
    :return: the spectrum redshifted
    :rtype: tuple[np.array, np.array]
    """
    factor = 1.+redshift
    return wl*factor,fl #/factor




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
        """Provides the list of fors2 spectra identifiers sorted
        :return: return list of keys of the h5 file where each key match a Fors2 spectrum tag id
        :rtype: list of str
        """
        list_of_specnames = self.list_of_groupkeys
        nums = np.array([int(re.findall("^SPEC(.*)",name)[0])  for name in list_of_specnames])
        indexes_sorted = np.argsort(nums)
        names_sorted = np.array(list_of_specnames)[indexes_sorted]
        return names_sorted

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



    def getspectrumcleanedemissionlines_fromgroup(self,groupname:str,nsigs:float=8.) ->dict:
        """Clean the spectrum from any emission line or any defects i the spectrum

        :param groupname: identifier tag name of the spectrum
        :type groupname: str

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
            gpr.fit(X[:, None], Y)

            Z = Y - gpr.predict(X[:, None], return_std=False)
            DeltaY = np.abs(Z)
            bkg = np.sqrt(np.median(DeltaY**2))

            #indexes_toremove = np.where(np.abs(DeltaY)> nsigs * background)[0]
            indexes_toremove = np.where(np.logical_or(DeltaY> nsigs * bkg,Y<=0))[0]

            Xclean = np.delete(X,indexes_toremove)
            Yclean  = np.delete(Y,indexes_toremove)
            Zclean  = np.delete(Z,indexes_toremove)

            spec_dict["wl"] = Xclean
            spec_dict["fnu"] = Yclean
            spec_dict["bg"] = np.abs(Zclean) # strange scikit learn bug returning negative std.
            spec_dict["bg_med"] = bkg # overestimated median error

        else:
            print(f'getspectrum_fromgroup : No group {groupname}')
        return spec_dict

    def get_photmagnitudes(self,specname:str) -> tuple[np.array,np.array]:
        """get magnitudes and errors from phtometric surveys

        :param specname: spectrum name
        :type specname: str
        :return: magnitude and errors on magnitudes
        :rtype: tuple[np.array,np.array]
        """
        attrs = self.getattribdata_fromgroup(specname)

        mags = np.array([attrs["fuv_mag"], attrs["nuv_mag"],\
                         attrs['MAG_GAAP_u'] + attrs['EXTINCTION_u'], attrs['MAG_GAAP_g'] + attrs['EXTINCTION_g'],\
                         attrs['MAG_GAAP_r'] + attrs['EXTINCTION_r'], attrs['MAG_GAAP_i'] + attrs['EXTINCTION_i'],\
                         attrs['MAG_GAAP_Z'], attrs['MAG_GAAP_Y'],\
                         attrs['MAG_GAAP_J'], attrs['MAG_GAAP_H'], attrs['MAG_GAAP_Ks']\
                        ])

        magserr = np.array([attrs["fuv_magerr"], attrs["nuv_magerr"],\
                            attrs['MAGERR_GAAP_u'], attrs['MAGERR_GAAP_g'],\
                            attrs['MAGERR_GAAP_r'], attrs['MAGERR_GAAP_i'],\
                            attrs['MAGERR_GAAP_Z'], attrs['MAGERR_GAAP_Y'],\
                            attrs['MAGERR_GAAP_J'], attrs['MAGERR_GAAP_H'], attrs['MAGERR_GAAP_Ks']\
                           ])

        return mags, magserr

    def get_ugrimagnitudes_corrected(self, specname:str) -> tuple[np.array,np.array]:
        """get magnitudes and errors from phtometric surveys

        :param specname: spectrum name
        :type specname: str
        :return: magnitude and errors on magnitudes
        :rtype: tuple[np.array,np.array]
        """
        attrs = self.getattribdata_fromgroup(specname)

        mags = np.array([attrs['MAG_GAAP_u'], attrs['MAG_GAAP_g'],\
                         attrs['MAG_GAAP_r'], attrs['MAG_GAAP_i']\
                        ])

        magserr = np.array([attrs['MAGERR_GAAP_u'], attrs['MAGERR_GAAP_g'],\
                            attrs['MAGERR_GAAP_r'], attrs['MAGERR_GAAP_i']\
                           ])

        return mags, magserr

    def get_photfluxes(self,specname:str) -> tuple[np.array,np.array]:
        """get fluxes and errors from photometric surveys

        :param specname: spectrum name
        :type specname: str
        :return: magnitude and errors on magnitudes
        :rtype: tuple[np.array,np.array]
        """
        attrs = self.getattribdata_fromgroup(specname)

        mags = np.array([ attrs["fuv_mag"], attrs["nuv_mag"], attrs['MAG_GAAP_u'], attrs['MAG_GAAP_g'], attrs['MAG_GAAP_r'], attrs['MAG_GAAP_i'], attrs['MAG_GAAP_Z'], attrs['MAG_GAAP_Y'],
            attrs['MAG_GAAP_J'], attrs['MAG_GAAP_H'],attrs['MAG_GAAP_Ks'] ])

        magserr = np.array([ attrs["fuv_magerr"], attrs["nuv_magerr"], attrs['MAGERR_GAAP_u'], attrs['MAGERR_GAAP_g'], attrs['MAGERR_GAAP_r'], attrs['MAGERR_GAAP_i'], attrs['MAGERR_GAAP_Z'], attrs['MAGERR_GAAP_Y'],
            attrs['MAGERR_GAAP_J'], attrs['MAGERR_GAAP_H'],attrs['MAGERR_GAAP_Ks'] ])

        mfluxes = [ 10**(-0.4*m) for m in mags ]
        mfluxeserr = []
        for f,em in zip(mfluxes,magserr):
            ferr = 0.4*np.log(10)*em*f
            mfluxeserr.append(ferr)

        mfluxes = np.array(mfluxes)
        mfluxeserr = np.array(mfluxeserr)
        return mfluxes,mfluxeserr


    def plot_spectro_photom_noscaling(self,specname:str,ax=None,figsize=(12,6)) -> None:
        """plot spectrometry and photometry

        :param specname: name of the spectrum ex: 'SPEC100'
        :type specname: str
        :param ax: _description_, defaults to None
        :type ax: _type_, optional
        :param figsize: figure size, defaults to (12,6)
        :type figsize: tuple, optional
        """

        spec = self.getspectrumcleanedemissionlines_fromgroup(specname)
        attrs = self.getattribdata_fromgroup(specname)
        z_obs = attrs["redshift"]
        asep_galex = attrs['asep_galex']
        asep_kids = attrs['asep_kids']
        speclabel = f"specname : z={z_obs:.2f} sep = ({asep_galex:.3f}, {asep_kids:.3f}) arscec"

        title = specname + f" redshift = {z_obs:.3f}" + " spectrum Not rescaled"

        mags = np.array([ attrs["fuv_mag"], attrs["nuv_mag"], attrs['MAG_GAAP_u'], attrs['MAG_GAAP_g'], attrs['MAG_GAAP_r'], attrs['MAG_GAAP_i'], attrs['MAG_GAAP_Z'], attrs['MAG_GAAP_Y'],
            attrs['MAG_GAAP_J'], attrs['MAG_GAAP_H'],attrs['MAG_GAAP_Ks'] ])

        magserr = np.array([ attrs["fuv_magerr"], attrs["nuv_magerr"], attrs['MAGERR_GAAP_u'], attrs['MAGERR_GAAP_g'], attrs['MAGERR_GAAP_r'], attrs['MAGERR_GAAP_i'], attrs['MAGERR_GAAP_Z'], attrs['MAGERR_GAAP_Y'],
            attrs['MAGERR_GAAP_J'], attrs['MAGERR_GAAP_H'],attrs['MAGERR_GAAP_Ks'] ])

        mfluxes = [ 10**(-0.4*m) for m in mags ]

        mfluxeserr = []
        for f,em in zip(mfluxes,magserr):
            ferr = 0.4*np.log(10)*em*f
            mfluxeserr.append(ferr)

        mfluxes = np.array(mfluxes)
        mfluxeserr = np.array(mfluxeserr)

        #fluxes =  [ attrs["fuv_flux"], attrs["nuv_flux"], attrs['FLUX_GAAP_u'], attrs['FLUX_GAAP_g'], attrs['FLUX_GAAP_r'], attrs['FLUX_GAAP_i'], attrs['FLUX_GAAP_Z'], attrs['FLUX_GAAP_Y'],
        #    attrs['FLUX_GAAP_J'], attrs['FLUX_GAAP_H'],attrs['FLUX_GAAP_Ks'] ]

        #fluxeserr =  [ attrs["fuv_fluxerr"], attrs["nuv_fluxerr"], attrs['FLUXERR_GAAP_u'], attrs['FLUXERR_GAAP_g'], attrs['FLUXERR_GAAP_r'], attrs['FLUXERR_GAAP_i'], attrs['FLUXERR_GAAP_Z'], attrs['FLUX_GAAP_Y'],
        #    attrs['FLUXERR_GAAP_J'], attrs['FLUXERR_GAAP_H'],attrs['FLUXERR_GAAP_Ks'] ]

        if ax is None:
            _, ax =plt.subplots(1,1,figsize=figsize)

        # plot the spectrum
        X = spec["wl"]
        Y = spec["fnu"]
        ax.plot(X,Y,'-',color="b",label=speclabel)

        # show the gaussian process fitted
        gpr.fit(X[:, None], Y)
        xfit = np.linspace(X.min(),X.max())
        yfit, yfit_err = gpr.predict(xfit[:, None], return_std=True)
        ax.plot(xfit, yfit, '-', color='cyan')
        ax.fill_between(xfit, yfit -  yfit_err, yfit +  yfit_err, color='gray', alpha=0.3)


        ax.set_xlabel("$\lambda (\\AA)$")
        ylabel = "$f_\\nu$ (a.u)"
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        ax.legend()

        # plot the photpmetry
        ax2 =ax.twinx()
        ax2.errorbar(PHOTWL,mfluxes,yerr=mfluxeserr,fmt='o',color="r",ecolor="r",ms=7,label='Galex (UV) + Kids (optics) + VISTA')
        ax2.set_ylabel("maggies")
        plot_filter_tag(ax2,mfluxes)
        plt.show()

    def plot_spectro_photom_rescaling(self,specname:str,ax=None,figsize=(12,6)) -> None:
        """plot spectrometry and photometry with rescaling spec

        :param specname: name of the spectrum ex: 'SPEC100'
        :type specname: str
        :param ax: _description_, defaults to None
        :type ax: _type_, optional
        :param figsize: figure size, defaults to (12,6)
        :type figsize: tuple, optional
        """

        spec = self.getspectrumcleanedemissionlines_fromgroup(specname)
        attrs = self.getattribdata_fromgroup(specname)
        z_obs = attrs["redshift"]
        asep_galex = attrs['asep_galex']
        asep_kids = attrs['asep_kids']
        speclabel = f"specname : z={z_obs:.2f} sep = ({asep_galex:.3f}, {asep_kids:.3f}) arscec"
        title = specname + f" redshift = {z_obs:.3f}" + " spectrum rescaled"

        mags = np.array([ attrs["fuv_mag"], attrs["nuv_mag"], attrs['MAG_GAAP_u'], attrs['MAG_GAAP_g'], attrs['MAG_GAAP_r'], attrs['MAG_GAAP_i'], attrs['MAG_GAAP_Z'], attrs['MAG_GAAP_Y'],
            attrs['MAG_GAAP_J'], attrs['MAG_GAAP_H'],attrs['MAG_GAAP_Ks'] ])

        magserr = np.array([ attrs["fuv_magerr"], attrs["nuv_magerr"], attrs['MAGERR_GAAP_u'], attrs['MAGERR_GAAP_g'], attrs['MAGERR_GAAP_r'], attrs['MAGERR_GAAP_i'], attrs['MAGERR_GAAP_Z'], attrs['MAGERR_GAAP_Y'],
            attrs['MAGERR_GAAP_J'], attrs['MAGERR_GAAP_H'],attrs['MAGERR_GAAP_Ks'] ])

        mfluxes = [ 10**(-0.4*m) for m in mags ]

        mfluxeserr = []
        for f,em in zip(mfluxes,magserr):
            ferr = 0.4*np.log(10)*em*f
            mfluxeserr.append(ferr)

        mfluxes = np.array(mfluxes)
        mfluxeserr = np.array(mfluxeserr)


        if ax is None:
            _, ax =plt.subplots(1,1,figsize=figsize)

        # plot the spectrum
        Xspec = spec["wl"]
        Yspec = spec["fnu"]

        # show the faussian process fitted
        try:
            gpr.fit(Xspec[:, None], Yspec)
            xfit = np.linspace(Xspec.min(),Xspec.max())
            yfit, yfit_err = gpr.predict(xfit[:, None], return_std=True)

            photom_wl_inrange_indexes = np.where(np.logical_and(PHOTWL>Xspec.min(),PHOTWL<Xspec.max()))[0]

            Xphot = PHOTWL[photom_wl_inrange_indexes]
            Yphot = mfluxes[photom_wl_inrange_indexes]

            yfit_photom = gpr.predict(Xphot[:, None], return_std=False)

            # scaling factor is flux-photom/flux-spec
            scaling_factor = np.mean(Yphot/yfit_photom)

            # correct spectrum
            Yspec *= scaling_factor

            # correcte fitted gaussian
            yfit *= scaling_factor
            yfit_err *= scaling_factor

            ax.plot(Xspec,Yspec,'-',color="b",label=speclabel)
            ax.plot(xfit, yfit, '-', color='cyan')
            ax.fill_between(xfit, yfit -  yfit_err, yfit +  yfit_err, color='gray', alpha=0.3)
        except Exception as e:
            scaling_factor = 0.0
            print(e)



        ax.set_xlabel("$\lambda (\\AA)$")
        ylabel = "$f_\\nu$ (a.u)"
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        ax.legend()

        # plot the photpmetry
        ax.errorbar(PHOTWL,mfluxes,yerr=mfluxeserr,fmt='o',color="r",ecolor="r",ms=7,label='Galex (UV) + Kids (optics) + VISTA')
        ax.set_ylabel("maggies")
        plot_filter_tag(ax,mfluxes)

        if scaling_factor !=0:
            all_y = np.concatenate([Yspec,mfluxes[2:]])
        else:
            all_y = mfluxes[2:]


        ymax = np.max(all_y)

        if ymax>0:
            ax.set_ylim(0.,ymax)

        plt.show()


    def plot_allspectra(self,ax = None, figsize=(12,6),ylim=(1e-1,1e2),mode="fl",frame="obs"):
        """plot all fors2 spectra

        :param ax: matplotlib axe, defaults to None
        :type ax: ax, optional
        :param figsize: figure size, defaults to (12,6)
        :type figsize: tuple, optional
        :param mode: specify if want flambda (fl) or fnu
        :type mode: str, optional
        :param frame: rest frame or obs frame
        :param frame: str, optional
        :param ylim: tuple for y boundary figure vertical scale
        :type ylim: tuple of 2 floats
        """

        if ax is None:
            _, ax =plt.subplots(1,1,figsize=figsize)

        fors2_tags = self.get_list_of_groupkeys()

        for tag in fors2_tags:
            spec = self.getspectrum_fromgroup(tag)

            if frame == "obs":
                title = "Fors2 spectra in obs frame"
                if mode == "fl":
                    x,y = spec["wl"],spec["fl"]
                    ylabel = "$f_\\lambda$ (a.u)"
                else:
                    x,y= spec["wl"],spec["fnu"]
                    ylabel = "$f_\nu$ (a.u)"
            else:
                title = "Fors2 spectra in rest frame"
                if mode == "fl":
                    x,y = convert_flux_torestframe(spec["wl"],spec["fl"])
                    ylabel = "$f_\\lambda$ (a.u)"
                else:
                    x,y = convert_flux_torestframe(spec["wl"],spec["fnu"])
                    ylabel = "$f_\nu$ (a.u)"
            ax.plot(x,y)


        ax.set_yscale("log")
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("$\lambda (\\AA)$")
        ax.set_title(title)
        ax.grid()

        plt.show()

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
        """Get list of StarLight spectrum tag names sorted

        :return: The list of available StarLight spectrum tag names
        :rtype: list
        """
        list_of_specnames = self.list_of_groupkeys
        nums = np.array([int(re.findall("^SPEC(.*)",name)[0])  for name in list_of_specnames])
        indexes_sorted = np.argsort(nums)
        names_sorted = np.array(list_of_specnames)[indexes_sorted]
        return names_sorted


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

    def plot_allspectra(self,ax = None, figsize=(12,6),xlim=(0.,2e4),ylim=(1e-6,1e-3),mode="fl"):
        """plt all starlight spectra

        :param ax: axe, defaults to None
        :type ax: ax, optional
        :param figsize: size of the figure, defaults to (12,6)
        :type figsize: tuple, optional
        :param xlim: tuple for x boundary figure horizontal scale
        :type xlim: tuple of 2 floats
        :param ylim: tuple for y boundary figure vertical scale
        :type ylim: tuple of 2 floats
        :param mode: choose the type of flux, defaults to "fl" or "fnu"
        :type mode: str, optional
        """

        if ax is None:
            _, ax =plt.subplots(1,1,figsize=figsize)

        sl_tags = self.get_list_of_groupkeys()

        for tag in sl_tags:
            spec = self.getspectrum_fromgroup(tag)


            title = "StarLight spectra (rest frame)"
            if mode == "fl":
                x,y = spec["wl"],spec["fl"]
                ylabel = "$f_\\lambda$ (a.u)"
            else:
                x,y= spec["wl"],spec["fnu"]
                ylabel = "$f_\nu$ (a.u)"

            ax.plot(x,y)

        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$\lambda (\\AA)$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()

        plt.show()



