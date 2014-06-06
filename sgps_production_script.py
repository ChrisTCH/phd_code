#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to open the FITS file containing the SGPS#
# data, and produce FITS files and images of the polarisation gradient,       #
# polarisation curvature, and various other diagnostic quantities. The        #
# calculation of these quantities will occur in separate functions. The       #
# quantities will then be saved as FITS files in the same directory as the    #
# SGPS data.                                                                  #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 6/6/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
import aplpy
from astropy.io import fits
from calc_Polar_Grad import calc_Polar_Grad

# Create a string object which stores the directory of the SGPS data
data_loc = '~/Documents/PhD/SGPS_Data/'

# Open the SGPS data FITS file
sgps_fits = fits.open(data_loc + 'sgps_iqup.imcat.fits')

# Print the information about the data file. This should show that there is only
# a primary HDU, which contains all of the image data
sgps_fits.info()

# Obtain the header of the primary HDU
sgps_hdr = sgps_fits[0].header

# Print the header information to the screen (WARNING: The header for the sgps
# data is very long)
# print hdr

# Extract the data from the FITS file, which is held in the primary HDU
sgps_data = sgps_fits[0].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'SGPS data successfully extracted from the FITS file'

# Check the shape of the data array.
# NOTE: The shape of the array should be (1, 4, 553, 1142), with the first 
# dimension being meaningless, the second being the Stokes parameters and
# the polarised intensity, the third dimension corresponds to the y-axis,
# which is Galactic Latitude, and the fourth dimension is Galactic Longitude,
# which goes along the x-axis.
print 'The shape of the data array is: {}'.format(sgps_data.shape) 

# Create a temporary array where the redundant first dimension is removed
temp_arr = sgps_data[0]

# The elements in the second dimension of the sgps data array are Stokes I,
# Stokes Q, polarised intensity, and Stokes U, in that order.
# Extract these slices from the temporary data array
Sto_I = temp_arr[0]
Sto_Q = temp_arr[1]
polar_inten = temp_arr[2]
Sto_U = temp_arr[3]

# Print a message to the screen to show that everything is going smoothly
print 'Stokes parameters successfully extracted from data'

# Pass the Stokes Q and U arrays to the function which calculates the 
# polarisation gradient. This function returns an array of the same size as
# the arrays in Stokes Q and U.
polar_grad = calc_Polar_Grad(Sto_Q, Sto_U)



