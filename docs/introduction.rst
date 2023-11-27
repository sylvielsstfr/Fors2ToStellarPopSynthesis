Introduction
============



In this package ``Fors2ToStellarPopSynthesis`` we propose to fit an SED model derived from the `DSPS <https://dsps.readthedocs.io/en/latest/index.html>`_ model 
to `spectroscopic data obtained from the Fors2 instrument of the ESO UT1 telescope <https://arxiv.org/pdf/1011.1947.pdf>`_ 
within a pilot galaxy survey in the direction of an X cluster ``RXJ0054.0-2823`` at a redshift of ```z = 0.29``.  
We complemented these observations with photometric surveys of `Galex <http://www.galex.caltech.edu/>`_ in UV range, 
`KIDS <https://kids.strw.leidenuniv.nl/>`_  in visible range 


This package is organized around a library of Python utilities that implement this functionality.
As DSPS is based on self-differentiating libraries coded in JAX, the model fitting libraries are also coded in JAX.

The package is structured like a typical python project:

| Fors2ToStellarPopSynthesis>tree -L 1
|
| ├── LICENSE
| ├── README.md
| ├── _readthedocs
| ├── benchmarks
| ├── docs
| ├── examples
| ├── lib
| ├── pyproject.toml
| ├── scripts
| ├── src
| └── tests




src/
----


Folder where all python source libraries are installed. 
Please refer to the section :ref:`installation-section` to install all packages. 


tests/
------ 

Folder where unitary tests are implemented.

In ``tests/fors2tostellarpopsynthesis`` run

.. code-block:: bash

   >> python -m unittest -v


docs/
-----

Folder where are installed the sphinx documentation including the notebooks

The documentation is generated from the docs/ folder as folow:

.. code-block:: bash

   >> make html

Then the documentation is produced in ``_readthedocs`` folder. 


examples/ (old)
--------------

Folder where dependency packages were originaly tested in notebooks. 

lib/
----
Original libaries used by notebooks in ``examples/``








