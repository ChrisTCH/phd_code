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

from calc_Sto_1Diff import calc_Sto_1Diff
from calc_Sto_2Diff import calc_Sto_2Diff

from calc_Polar_Grad import calc_Polar_Grad
from calc_Ang_Grad import calc_Ang_Grad
from calc_Mod_Div_Grad import calc_Mod_Div_Grad
from calc_Quad_Curv import calc_Quad_Curv

from mat2FITS_Image import mat2FITS_Image
from fits2aplpy import fits2aplpy

# Create a string object which stores the directory of the SGPS data
data_loc = '/Users/chrisherron/Documents/PhD/SGPS_Data/'

# Open the SGPS data FITS file
sgps_fits = fits.open(data_loc + 'sgps_iqup.imcat.fits')

# Print the information about the data file. This should show that there is only
# a primary HDU, which contains all of the image data
sgps_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the primary HDU
sgps_hdr = sgps_fits[0].header

# Delete all of the history cards in the header
del sgps_hdr[41:]
# Delete all of the information about the third and fourth axes of the array,
# which will be irrelevant in produced plots
del sgps_hdr[22:30]

# Print the header information to the screen (WARNING: The header for the sgps
# data is very long if the history cards are not removed)
print sgps_hdr
# Print a blank line to make the script output easier to read
print ''

# Extract the data from the FITS file, which is held in the primary HDU
sgps_data = sgps_fits[0].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'SGPS data successfully extracted from the FITS file.'

# Check the shape of the data array.
# NOTE: The shape of the array should be (1, 4, 553, 1142), with the first 
# dimension being meaningless, the second being the Stokes parameters and
# the polarised intensity, the third dimension corresponds to the y-axis,
# which is Galactic Latitude, and the fourth dimension is Galactic Longitude,
# which goes along the x-axis.
print 'The shape of the data array is: {}.'.format(sgps_data.shape) 

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
print 'Stokes parameters successfully extracted from data.'

# Calculate the first order partial derivatives of Stokes Q and U with respect
# to the y and x axes of the image. This function returns arrays that are the 
# same size as the arrays in Stokes Q and U.
dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(Sto_Q, Sto_U)

# Calculate the second order partial derivatives of Stokes Q and U with
# respect to the y and x axes of the image. This function returns arrays that
# are the same size as the arrays in Stokes Q and U.
d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2 = \
calc_Sto_2Diff(dQ_dy, dQ_dx, dU_dy, dU_dx)

# Print a message to the screen to show that the derivatives have been 
# calculated correctly
print 'Derivatives of Stokes parameters successfully calculated.\n'

#--------------------- POLARISATION GRADIENT MAGNITUDE ------------------------

## Use the first order partial derivatives to calculate the magnitude of the
## polarisation gradient of the image.
#polar_grad = calc_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)
#
## Print a message to the screen to show that the magnitude of the polarisation
## gradient has been calculated successfully.
#print 'Magnitude of the polarisation gradient calculated successfully.'
#
## Convert the matrix of polarisation gradient values into a FITS file, using
## the header information of the SGPS data. Also save the FITS file that is
## produced by the function.
#polar_grad_FITS = mat2FITS_Image(polar_grad, sgps_hdr,\
#data_loc + 'sgps_polar_grad.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the magnitude of the polarisation ' + \
#'gradient.'
#
## Create an image of the magnitude of the polarisation gradient for the SGPS
## data using aplpy and the produced FITS file. This image is automatically
## saved using the given filename.
#fits2aplpy(polar_grad_FITS, data_loc + 'sgps_polar_grad.png', \
#colour = 'hot')
#
## Print a message to the screen to show that the image of the magnitude of the
## polarisation gradient has been successfully produced and saved.
#print 'Image of the magnitude of the polarisation gradient successfully saved.'\
#+ '\n'

#------------------------ POLARISATION GRADIENT ANGLE -------------------------

## Pass the Stokes Q and U arrays to the function which calculates the angle of
## the polarisation gradient with respect to the x axis. This function returns
## an array of the same size as the arrays in Stokes Q and U.
#ang_polar_grad = calc_Ang_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)
#
## Print a message to the screen to show that the angle of the polarisation 
## gradient has been calculates successfully.
#print 'Angle of the polarisation gradient calculated successfully.'
#
## Convert the matrix containing the angles of the polarisation gradient into
## a FITS file, using the header information of the SGPS data. Also save the 
## FITS file that is produced by the function.
#ang_grad_FITS = mat2FITS_Image(ang_polar_grad, sgps_hdr,\
#data_loc + 'sgps_ang_grad.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the angle of the polarisation ' + \
#'gradient.'
#
## Create an image of the angle of the polarisation gradient for the SGPS
## data using aplpy and the produced FITS file. This image is automatically
## saved using the given filename.
#fits2aplpy(ang_grad_FITS, data_loc + 'sgps_ang_grad.png', colour = 'RdBu',\
#vmin = -90.0, vmax = 90.0)
#
## Print a message to the screen to show that the image of the angle of the
## polarisation gradient has been successfully produced and saved.
#print 'Image of the angle of the polarisation gradient successfully saved.\n'

#----------------- MODULUS OF THE DIVERGENCE OF GRADIENT ----------------------

## Here the modulus of the divergence of the polarisation gradient is 
## calculated, which depends only upon second order derivatives of the Stokes
## parameters. The formula is given on page 105 of PhD Logbook 1, and page
## 52 of PhD Logbook 2. This quantity is rotationally and translationally
## invariant in the Q-U plane. 
#
## Pass the second derivative arrays to the function that calculates the
## modulus of the divergence of the polarisation gradient. This function 
## returns an array of the same size as the second derivative arrays.
#mod_div_grad = calc_Mod_Div_Grad(d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2)
#
## Print a message to the screen to show that the modulus of the divergence
## of the gradient has been calculated successfully.
#print 'Modulus of the divergence of the polarisation gradient calculated' +\
#' successfully.'
#
## Convert the matrix containing values of the divergence of the gradient into
## a FITS file, using the header information of the SGPS data. Also save the 
## FITS file that is produced by the function.
#mod_div_grad_FITS = mat2FITS_Image(mod_div_grad, sgps_hdr,\
#data_loc + 'sgps_mod_div_grad.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the modulus of the divergence of the'+\
#' polarisation gradient'
#
## Create an image of the modulus of the divergence of the polarisation gradient
## for the SGPS data using aplpy and the produced FITS file. This image is 
## automatically saved using the given filename.
#fits2aplpy(mod_div_grad_FITS, data_loc + 'sgps_mod_div_grad.png',\
#colour = 'hot')
#
## Print a message to the screen to show that the image of the modulus of the 
## divergence of the polarisation gradient has been successfully produced and
## saved.
#print 'Image of the modulus of the divergence of the polarisation gradient' +\
#' successfully saved.\n'

#------------------------- QUADRATURE OF CURVATURES ---------------------------

# Here the quadrature of the curvatures in the x and y directions is 
# calculated, which depends upon both the first and second order derivatives of
# the Stokes parameters. The formula is given on page 96 of PhD Logbook 1, and
# page 23 of PhD Logbook 2. 

# Pass the partial derivative arrays to the function that calculates the
# quadrature of the curvatures. This function returns an array of the same size
# as the partial derivative arrays.
quad_curv = calc_Quad_Curv(dQ_dy, dQ_dx, dU_dy, dU_dx,\
d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2)

# Print a message to the screen to show that the quadrature of the curvatures
# has been calculated successfully.
print 'Quadrature of the curvatures calculated successfully.'

# Convert the matrix containing values of the quadrature of curvatures into
# a FITS file, using the header information of the SGPS data. Also save the 
# FITS file that is produced by the function.
quad_curv_FITS = mat2FITS_Image(quad_curv, sgps_hdr,\
data_loc + 'sgps_quad_curv.fits')

# Print a message to the screen to show that the FITS file was produced and
# saved successfully.
print 'FITS file successfully saved for the quadrature of curvatures.'

# Create an image of the quadrature of curvatures for the SGPS data using aplpy
# and the produced FITS file. This image is automatically saved using the given
# filename.
fits2aplpy(quad_curv_FITS, data_loc + 'sgps_quad_curv.png',\
colour = 'hot')

# Print a message to the screen to show that the image of the quadrature of the
# curvatures has been successfully produced and saved.
print 'Image of the quadrature of the curvatures successfully saved.\n'

#------------------------------------------------------------------------------

# Print a message to the screen to show that all tasks have been 
# completed successfully.
print 'All files and images produced successfully.'