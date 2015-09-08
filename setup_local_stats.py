#-----------------------------------------------------------------------------#
#                                                                             #
# This code is a Cython setup script, that will compile the calc_local_stats  #
# function. This code needs to be run whenever changes are made to the        #
# function. The function itself calculates local statistics around each pixel #
# in an image, and then produces FITS images of the produced maps.            #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 3/9/2015                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the setup and cythonize modules to compile the Cython code
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

# Run the setup code to compile the Cython code
setup(include_dirs=[np.get_include()], ext_modules =\
 cythonize("calc_local_stats.pyx"))