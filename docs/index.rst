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


Dev Guide - Getting Started
---------------------------

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

.. code-block:: bash

   >> conda create env -n <env_name> python=3.10
   >> conda activate <env_name>


Once you have created a new environment, you can install this project for local
development using the following commands:

.. code-block:: bash

   >> pip install -e .'[dev]'
   >> pre-commit install
   >> conda install pandoc


Notes:

1) The single quotes around ``'[dev]'`` may not be required for your operating system.
2) ``pre-commit install`` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   `pre-commit <https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html>`_.
3) Installing ``pandoc`` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   `Sphinx and Python Notebooks <https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks>`_.


.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
