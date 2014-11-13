#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the skewness, kurtosis, radially averaged        #
# structure function, and quadrupole/monopole ratio of the synchrotron         #
# intensity for various values of gamma. Each of these quantities is plotted   #
# against the sonic and Alfvenic Mach numbers, to see which quantities are     #
# sensitive tracers of the sonic and Alfvenic Mach numbers.                    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 17/10/2014                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the functions that calculate the structure and correlation functions
# using FFT, the function that calculates the radially averaged structure or 
# correlation functions, and the function that calculates multipoles of 2D 
# images
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D