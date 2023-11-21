# fors2tostellarpopsynthesis

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/fors2tostellarpopsynthesis?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/fors2tostellarpopsynthesis/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/fors2tostellarpopsynthesis/smoke-test.yml)](https://github.com/LSSTDESC/fors2tostellarpopsynthesis/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/LSSTDESC/fors2tostellarpopsynthesis/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/fors2tostellarpopsynthesis)
[![Read the Docs](https://img.shields.io/readthedocs/fors2tostellarpopsynthesis)](https://fors2tostellarpopsynthesis.readthedocs.io/)
[![benchmarks](https://img.shields.io/github/actions/workflow/status/LSSTDESC/fors2tostellarpopsynthesis/asv-main.yml?label=benchmarks)](https://LSSTDESC.github.io/fors2tostellarpopsynthesis/)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).


## Introduction

Package to fit spectroscopic data and photometric data
on a Stellar Synthesis Population code [DSPS](https://dsps.readthedocs.io/en/latest/index.html).
Please refer to the [scientific publication on DSPS](https://arxiv.org/abs/2112.06830).

The purpose of this package is to provide a set of tools
to fit SSP models on spectral data and photometric data.
From the fitted models, a set Galaxies Spectral Templates
may be extracted from those data such those could be used in SED template fitting for PhotoZ estimation codes.

We apply this code to Fors2 Spectra combined with Photometric data from surveys [Galex](http://www.galex.caltech.edu/), [KIDS](https://kids.strw.leidenuniv.nl/) and [VISTA](https://www.eso.org/public/teles-instr/paranal-observatory/surveytelescopes/vista/surveys/).

- the galex data are available through astro-query
- KIDS-VISTA data from ESO User Portal at https://www.eso.org/sso
- the photometric survey filters are obtained from [sedpy](https://github.com/bd-j/sedpy) 

Notice to benefit from the implementation of DSPS in Jax, this package is also developped in JAX.


## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1) The single quotes around `'[dev]'` may not be required for your operating system.
2) `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3) Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
