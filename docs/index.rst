.. fors2tostellarpopsynthesis documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fors2tostellarpopsynthesis's documentation!
========================================================================================


Introduction
------------

Package to fit spectroscopic data and photometric data
on a Stellar Synthesis Population code `DSPS <https://dsps.readthedocs.io/en/latest/index.html>`_.
Please refer to the `scientific publication on DSPS <https://arxiv.org/abs/2112.06830>`_.

The purpose of this package is to provide a set of tools
to fit SSP models on spectral data and photometric data.
From the fitted models, a set Galaxies Spectral Templates
may be extracted from those data such those could be used in SED template fitting for PhotoZ estimation codes.

We apply this code to Fors2 Spectra combined with Photometric data from surveys 
`Galex <http://www.galex.caltech.edu/>`_, `KIDS <https://kids.strw.leidenuniv.nl/>`_ and `VISTA <https://www.eso.org/public/teles-instr/paranal-observatory/surveytelescopes/vista/surveys/>`_.

- the galex data are available through astro-query
- KIDS-VISTA data from `ESO User Portal <https://www.eso.org/sso>`_.
- the photometric survey filters are obtained from `sedpy <https://github.com/bd-j/sedpy>`_. 

Notice to benefit from the implementation of DSPS in Jax, this package is also developped in JAX.


About this project
^^^^^^^^^^^^^^^^^^
This project was automatically generated using the LINCC-Frameworks 
`python-project-template <https://github.com/lincc-frameworks/python-project-template>`_.

Please refer to see :ref:`installation-section` for more details.



.. toctree::
   :hidden:

   Home page <self>
   Introduction <introduction>
   Installation <installation>
   Installation on GPU <installationgpu>
   Quick Start <quickstart>
   Description <description>
   API Reference <autoapi/index>
   Notebooks <notebooks>
