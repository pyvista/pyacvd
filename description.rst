Overview:
ACVD provides the user with a method of uniformly resampling a surface triangular mesh.  The resampled mesh can be of higher or lower density than the original mesh and handles noisy and non-manifold meshes.

This module was created by translating existing source code written in C++ by S. Valette and J.-M Chassery.  Their work can be found in the link below.  Thanks for your work!

This module can also be used to cluster an existing mesh.  As it uniformly clusters a surface mesh, it is similar to a k-means clustering approach, except applied uniformly to a surface mesh.

Please not that this module is in its early stage of development and will likely have many bugs.  Please report them  on GitHub


Requirements:
This module is dependant on vtk, Cython, Numpy, and Scipy


Installation:
    $ pip install ACVD

    - or for a single user -
    $ pip install ACVD --user

    - or download the source code and execute the following from the download folder: -
    $ python setup.py install


License:
Inherits the CECILL-B license


Notes:
Translated from S. Valette, J.-M. Chassery, R. Prost C++ code here:
http://www.creatis.insa-lyon.fr/site/en/acvd
