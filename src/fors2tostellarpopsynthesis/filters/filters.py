"""Module Handling telescopes filters transmission curves"""
# pylint: disable=line-too-long
# pylint: disable=unused-variable
# pylint: disable=trailing-newlines
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=C0301
# pylint: disable=C0305
# pylint: disable=R0402
# pylint: disable=E1101
# pylint: disable=R0915
from collections import OrderedDict

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from sedpy import observate


class FilterInfo():
    """Class to encapsulate different tasks to fetch the filters
    transmissions. The survey used are Galex in UV, SDSS for KIDS 
    in visible and VISTA in IR.
    """
    def __init__(self)->None :
        self.filters_galex = np.array(["galex_FUV","galex_NUV"])
        self.filters_sdss = np.array(["sdss_u0","sdss_g0","sdss_r0","sdss_i0"])
        self.filters_vircam = np.array(["vista_vircam_Z","vista_vircam_Y","vista_vircam_J","vista_vircam_H","vista_vircam_Ks"])

        # Galex filters
        self.all_filt_galex = []
        for filtname in self.filters_galex:
            filt = observate.Filter(filtname)
            self.all_filt_galex.append(filt)
        self.n_galex = len(self.all_filt_galex)
        # colors for Galex
        cmap = mpl.cm.PuBu
        cnorm = colors.Normalize(vmin=0, vmax=self.n_galex)
        scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        self.all_colors_galex = scalarmap.to_rgba(np.arange(self.n_galex+1), alpha=1)

        # SDSS filters (for KIDS survey)
        self.all_filt_sdss = []
        for filtname in self.filters_sdss:
            filt = observate.Filter(filtname)
            self.all_filt_sdss.append(filt)
        self.n_sdss = len(self.all_filt_sdss)
        # colors for SDSS
        cmap = mpl.cm.Reds
        cnorm = colors.Normalize(vmin=0, vmax=self.n_sdss)
        scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        self.all_colors_sdss = scalarmap.to_rgba(np.arange(self.n_sdss+1), alpha=1)

        # VIRCAM
        self.all_filt_vircam = []
        for filtname in self.filters_vircam:
            filt = observate.Filter(filtname)
            self.all_filt_vircam.append(filt)
        self.n_vircam = len(self.all_filt_vircam)
        # colors for Vircam
        cmap = mpl.cm.Wistia
        cnorm = colors.Normalize(vmin=0, vmax=self.n_vircam)
        scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        self.all_colors_vircam = scalarmap.to_rgba(np.arange(self.n_vircam+1), alpha=1)

        self.filters_indexlist = []
        self.filters_surveylist = []
        self.filters_namelist = []
        self.filters_transmissionlist = []
        self.filters_transmissionnormlist = []
        self.filters_colorlist = []

        filter_count = 0

        for index in range(self.n_galex):
            self.filters_indexlist.append(filter_count)
            self.filters_surveylist.append("galex")
            self.filters_namelist.append(self.filters_galex[index])
            self.filters_transmissionlist.append(self.all_filt_galex[index])
            self.filters_transmissionnormlist.append(100.0)
            self.filters_colorlist.append(self.all_colors_galex[index+1])
            filter_count+= 1

        for index in range(self.n_sdss):
            self.filters_indexlist.append(filter_count)
            self.filters_surveylist.append("sdss")
            self.filters_namelist.append(self.filters_sdss[index])
            self.filters_transmissionlist.append(self.all_filt_sdss[index])
            self.filters_transmissionnormlist.append(1.0)
            self.filters_colorlist.append(self.all_colors_sdss[index+1])
            filter_count+= 1

        for index in range(self.n_vircam):
            self.filters_indexlist.append(filter_count)
            self.filters_surveylist.append("vircam")
            self.filters_namelist.append(self.filters_vircam[index])
            self.filters_transmissionlist.append(self.all_filt_vircam[index])
            if index==0:
                self.filters_transmissionnormlist.append(100.0)
            else:
                self.filters_transmissionnormlist.append(1.0)
            self.filters_colorlist.append(self.all_colors_vircam[index+1])
            filter_count+= 1

    def get_pytree(self):
        """
        :return: a dict of a tuple of a dict
        :rtype: a dict
        """
        the_dict = OrderedDict()

        for index in self.filters_indexlist:
            the_subdict = OrderedDict()
            the_filt = self.filters_transmissionlist[index]
            the_norm = self.filters_transmissionnormlist[index]

            the_name = self.filters_namelist[index]
            the_wlmean = the_filt.wave_mean
            the_wls = the_filt.wavelength
            the_transm =the_filt.transmission/the_norm

            the_subdict["name"] = the_name
            the_subdict["wlmean"] = the_wlmean
            the_subdict["wls"] = jnp.array(the_wls)
            the_subdict["transm"] = jnp.array(the_transm)
            the_dict[index] = the_subdict

        return the_dict

    def get_2lists(self):
        """
        Provide filter transmissions curves as a tuple of lists for all filters
        :return: a list of wavelengths for each filter and a list of transmission for each filter 
        :rtype: two lists
        """
        the_list1 = []
        the_list2 = []

        for index in self.filters_indexlist:

            the_filt = self.filters_transmissionlist[index]
            the_norm = self.filters_transmissionnormlist[index]

            #the_name = self.filters_namelist[index]
            #the_wlmean = the_filt.wave_mean
            the_wls = the_filt.wavelength
            the_transm =the_filt.transmission/the_norm

            the_list1.append(the_wls)
            the_list2.append(the_transm)

        return the_list1,the_list2

    def get_3lists(self):
        """
        Provide filter transmissions curves as a triplet of lists
        :return: a 3-tuple of array, 1) wavelengths, 2) transmissions, 3) the name
        :rtype: 3 lists
        """
        the_list1 = []
        the_list2 = []
        the_list3 = []

        for index in self.filters_indexlist:
            the_name = self.filters_namelist[index]

            the_filt = self.filters_transmissionlist[index]
            the_norm = self.filters_transmissionnormlist[index]

            the_name = self.filters_namelist[index]
            #the_wlmean = the_filt.wave_mean
            the_wls = the_filt.wavelength
            the_transm =the_filt.transmission/the_norm

            the_list1.append(the_wls)
            the_list2.append(the_transm)
            the_list3.append(the_name)

        return the_list1,the_list2,the_list3

    def plot_transmissions(self,ax = None)-> None:
        """plot transmission of filters
        :param ax: figure axis to plot the image, defaults to None
        :type ax: matplotlib axis, optional
        """

        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(12,6))

        for index in self.filters_indexlist:
            the_name = self.filters_namelist[index]
            the_filt = self.filters_transmissionlist[index]
            the_norm = self.filters_transmissionnormlist[index]
            the_wlmean = the_filt.wave_mean
            the_color = self.filters_colorlist[index]
            the_transmission =the_filt.transmission/the_norm
            ax.plot(the_filt.wavelength,the_transmission,color=the_color)

            if index%2 ==0:
                ax.text(the_wlmean, 0.7, the_name,horizontalalignment='center',verticalalignment='center',color=the_color,fontweight="bold")
            else:
                ax.text(the_wlmean, 0.75, the_name,horizontalalignment='center',verticalalignment='center',color=the_color,fontweight="bold")


        ax.grid()
        ax.set_title("Transmission")
        ax.set_xlabel("$\lambda (\AA)$")
        ax.set_xlim(0.,25000.)

    def dump(self):
        """Dump all the filters info available
        """
        print("filters_indexlist   : \t ", self.filters_indexlist)
        print("filters_surveylist  : \t ", self.filters_surveylist)
        print("filters__namelist   : \t ", self.filters_namelist)




