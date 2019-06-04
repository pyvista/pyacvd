pyacvd Documentation
====================
This module takes a surface mesh and returns a uniformly meshed surface using voronoi clustering.  This approach is loosely based on research by S. Valette, and J. M. Chassery in `ACVD <https://github.com/valette/ACVD>`_.


Installation
------------
Installation is straightforward using pip::

    $ pip install pyacvd
    
You can also visit `GitHub <https://github.com/akaszynski/pyacvd>`_ to download the latest source and install it running the following from the source directory::

    $ pip install .

You will need a working copy of VTK.  This can be obtained by either building for the source or installing it using a Python distribution like `Anaconda <https://www.continuum.io/downloads>`_.  The other dependencies are ``numpy`` and ``scipy``.



