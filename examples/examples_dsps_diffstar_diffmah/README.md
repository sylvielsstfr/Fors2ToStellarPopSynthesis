# README.md

- creation : 2023/10/02
- last update : 2023/10/03


Examples for using the python packages for Stellar population synthesis


## very first test with dsfps

- demo_simple_dfps.ipynb, compare_dfps.ipynb


## demo of diffstar
- demo_diffstar_sfh.ipynb
- demo_dsps_diffstar_sfh.ipynb

## demo for diffmah
- diffmah_halo_populations.ipynb


## work how to use dust attenuation in dfsps

- validate_dsps_attenuation.ipynb : notebook found to check the Calzetti attenuation
- validate_dsps_attenuation-sbl18.ipynb : my adaptation for Salim attenuation
- validate_dsps_attenuation-sbl18-combined.ipynb : my adaptation to vary different parameters for dust attenuation


## Combine everything to generate a spectrum with dust attenuation from disffstar params and dust parameters
- demo_dsps_diffstar_sfh-dustattenuation.ipynb



## Demo to show the use of python-fsps used to generate tempdata.h5
- demo_python-fsps.ipynb


## Input data

The data used by DSPS were generated from FSPS and dowloaded from https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/:
- **tempdata.h5**, 
- **tempdata_v2.h5**, 
- **tempfilter.h5**,