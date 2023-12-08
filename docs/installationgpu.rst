Installation on GPU
===================

.. highlight:: sh

Create a conda environnement
----------------------------

.. code-block:: bash
   conda create -n conda_jaxgpuv100_dsps_py310 python=3.10 numpy scipy scikit-learn astropy pandas matplotlib seaborn h5py tables corner



Activate this environnement
---------------------------

.. code-block:: bash
    conda activate conda_jaxgpuv100_dsps_py310



Install Jax
-----------

.. code-block:: bash

    conda install -y jaxlib=*=***cuda*** jax cuda-nvcc -c conda-forge -c nvidia 


Check nvidia version
--------------------

.. code-block:: bash

    nvcc --version 



Check the GPU are active
------------------------

.. code-block:: bash

    python -c "import jax;print(jax.devices())"  



Install other jax dependent libraries
-------------------------------------

.. code-block:: bash

    conda install -c conda-forge numpyro  
    pip install jaxopt  
    pip install optax  
    pip install corner  
    conda install -c conda-forge arviz
    pip install --no-deps equinox
    pip install --no-deps interpax


Install the libraries for this package
--------------------------------------

.. code-block:: bash

   pip install fsps
   pip install diffmah
   pip install diffstar
   pip install dsps
   pip install astro-sedpy



Install this package
--------------------

Ne pas installer  ``Fors2ToStellarPopSynthesis`` avec pip install -e ''[dev]''

.. code-block:: bash

   pip install .


Always check the device:

.. code-block:: bash

    python -c "import jax;print(jax.devices())"



Data
-----

Don't forget to download package data in ``fors2tostellarpopsynthesis/fitters/data/``  :

.. code-block:: bash

    curl https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5 > tempdata.h5


It is possible one have to copy those data by hands at the installation location

.. code-block:: bash

    cp tempdata.h5 /pbs/throng/lsst/users/dagoret/desc/JAX2023/miniconda3/envs/conda_jaxgpuv100_dsps_py310/lib/python3.10/site-packages/fors2tostellarpopsynthesis/fitters/data/.

