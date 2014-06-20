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
from calc_Direc_Div import calc_Direc_Div
from calc_Direc_Curv import calc_Direc_Curv
from calc_Rad_Tang_Direc import calc_Rad_Tang_Direc
from calc_Tang_Direc_Amp import calc_Tang_Direc_Amp
from calc_Rad_Direc_Amp import calc_Rad_Direc_Amp

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

# Extract the size of each pixel from the header. This is the length of each 
# side of the pixel (assumed to be square), in degrees. This is then
# converted into radians.
pix_size = np.deg2rad(sgps_hdr['CDELT2'])

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
dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(Sto_Q, Sto_U, pix_size)

# Calculate the second order partial derivatives of Stokes Q and U with
# respect to the y and x axes of the image. This function returns arrays that
# are the same size as the arrays in Stokes Q and U.
d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2 = \
calc_Sto_2Diff(dQ_dy, dQ_dx, dU_dy, dU_dx, pix_size)

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

## Here the quadrature of the curvatures in the x and y directions is 
## calculated, which depends upon both the first and second order derivatives of
## the Stokes parameters. The formula is given on page 96 of PhD Logbook 1, and
## page 23 of PhD Logbook 2. 
#
## Pass the partial derivative arrays to the function that calculates the
## quadrature of the curvatures. This function returns an array of the same size
## as the partial derivative arrays.
#quad_curv = calc_Quad_Curv(dQ_dy, dQ_dx, dU_dy, dU_dx,\
#d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2)
#
## Print a message to the screen to show that the quadrature of the curvatures
## has been calculated successfully.
#print 'Quadrature of the curvatures calculated successfully.'
#
## Convert the matrix containing values of the quadrature of curvatures into
## a FITS file, using the header information of the SGPS data. Also save the 
## FITS file that is produced by the function.
#quad_curv_FITS = mat2FITS_Image(quad_curv, sgps_hdr,\
#data_loc + 'sgps_quad_curv.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the quadrature of curvatures.'
#
## Create an image of the quadrature of curvatures for the SGPS data using aplpy
## and the produced FITS file. This image is automatically saved using the given
## filename.
#fits2aplpy(quad_curv_FITS, data_loc + 'sgps_quad_curv.png',\
#colour = 'hot')
#
## Print a message to the screen to show that the image of the quadrature of the
## curvatures has been successfully produced and saved.
#print 'Image of the quadrature of the curvatures successfully saved.\n'

#--------------------------- DIRECTIONAL DERIVATIVE ---------------------------

## Here the modulus of the directional derivative of the complex polarisation 
## vector is calculated for various directions determined by the angle to the
## horizontal theta. This depends upon the first order derivatives of the Stokes
## parameters. The formula is given on page 60 of PhD Logbook 1, and page 53 of
## PhD Logbook 2. 
#
## Pass the partial derivative arrays to the function that calculates the
## directional derivative data cube. This function returns a data cube with three
## axes. Each slice of the data cube is an image of the directional derivative 
## magnitude for a particular value of theta, and this image is of the same size
## as the partial derivative arrays.
#direc_div, theta = calc_Direc_Div(dQ_dy, dQ_dx, dU_dy, dU_dx, num_theta = 40)
#
## Print a message to the screen to show that the directional derivative data
## cube has been calculated successfully.
#print 'Directional derivative data cube calculated successfully.'
#
## The FITS header for the SGPS data needs to be updated for the directional
## derivative data cube, so that the value of theta for each slice of the data
## cube is known by any program reading the data cube. Create a copy of the SGPS
## header, which will be modified for the directional derivative cube.
#direc_div_hdr = sgps_hdr
#
## Insert a new header keyword into the header for the directional derivative,
## which will record the starting pixel of theta.
#direc_div_hdr.insert(22, ('CRPIX3', 1.00000000000E+00))
#
## Insert a new header keyword into the header for the directional derivative,
## which will record the difference between successive values of theta.
#direc_div_hdr.insert(23, ('CDELT3', theta[1] - theta[0]))
#
## Insert a new header keyword into the header for the directional derivative,
## which will record the starting value of theta.
#direc_div_hdr.insert(24, ('CRVAL3', -1.80000000000E+02))
#
## Insert a new header keyword into the header for the directional derivative,
## which will record that the third axis of the data cube specifies values of
## theta.
#direc_div_hdr.insert(25, ('CTYPE3', 'THETA-DEG'))
#
## Convert the data cube containing values of the directional derivative into
## a FITS file, using the header information of the SGPS data. Also save the 
## FITS file that is produced by the function.
#direc_div_FITS = mat2FITS_Image(direc_div, direc_div_hdr,\
#data_loc + 'sgps_direc_div.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the directional derivative.'

#--------------------------- DIRECTIONAL CURVATURE ----------------------------

## Here the directional curvature of the complex polarisation vector is
## calculated for various directions determined by the angle to the horizontal
## theta. The directional curvature is the curvature of the path in the Q-U
## plane traced out when moving at an angle theta with respect to the 
## horizontal axis. This depends upon both the first and second order derivatives
## of the Stokes parameters. The formula is given on page 22 of PhD Logbook 2.
#
## Pass the partial derivative arrays to the function that calculates the
## directional curvature data cube. This function returns a data cube with three
## axes. Each slice of the data cube is an image of the directional curvature 
## for a particular value of theta, and this image is of the same size as the
## partial derivative arrays.
#direc_curv, theta = calc_Direc_Curv(dQ_dy, dQ_dx, dU_dy, dU_dx,\
#d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2, num_theta = 10)
#
## Print a message to the screen to show that the directional curvature data
## cube has been calculated successfully.
#print 'Directional curvature data cube calculated successfully.'
#
## The FITS header for the SGPS data needs to be updated for the directional
## curvature data cube, so that the value of theta for each slice of the data
## cube is known by any program reading the data cube. Create a copy of the SGPS
## header, which will be modified for the directional curvature cube.
#direc_curv_hdr = sgps_hdr
#
## Insert a new header keyword into the header for the directional curvature,
## which will record the starting pixel of theta.
#direc_curv_hdr.insert(22, ('CRPIX3', 1.00000000000E+00))
#
## Insert a new header keyword into the header for the directional curvature,
## which will record the difference between successive values of theta.
#direc_curv_hdr.insert(23, ('CDELT3', theta[1] - theta[0]))
#
## Insert a new header keyword into the header for the directional curvature,
## which will record the starting value of theta.
#direc_curv_hdr.insert(24, ('CRVAL3', -1.80000000000E+02))
#
## Insert a new header keyword into the header for the directional curvature,
## which will record that the third axis of the data cube specifies values of
## theta.
#direc_curv_hdr.insert(25, ('CTYPE3', 'THETA-DEG'))
#
## Convert the data cube containing values of the directional curvature into
## a FITS file, using the header information of the SGPS data. Also save the 
## FITS file that is produced by the function.
#direc_curv_FITS = mat2FITS_Image(direc_curv, direc_curv_hdr,\
#data_loc + 'sgps_direc_curv.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the directional curvature.'

#--------- RADIAL AND TANGENTIAL COMPONENTS OF DIRECTIONAL DERIVATIVE ---------

## Here the radial and tangential components of the directional derivative of the
## complex polarisation vector are calculated for various directions determined
## by the angle to the horizontal theta. This depends upon the first order
## derivatives of the Stokes parameters, and the Stokes parameters themselves.
## The formulae are given on page 57 of PhD Logbook 2.
#
## Pass the partial derivative arrays to the function that calculates the
## radial and tangential component data cubes. This function returns two data
## cubes, each with three axes. Each slice of a data cube is an image of either
## the radial or tangential component of the directional derivative for a
## particular value of theta, and this image is of the same size as the
## partial derivative arrays.
#rad_comp, tang_comp, theta = calc_Rad_Tang_Direc(Sto_Q, Sto_U, dQ_dy, dQ_dx,\
#dU_dy, dU_dx, num_theta = 40)
#
## Print a message to the screen to show that the radial and tangential 
## component data cubes have been calculated successfully.
#print 'Radial and tangential component data cubes calculated successfully.'
#
## The FITS header for the SGPS data needs to be updated for the radial and
## tangential component data cubes, so that the value of theta for each slice of
## the data cube is known by any program reading the data cube. Create a copy of
## the SGPS header, which will be modified for the radial and tangential 
## component cubes.
#rad_tang_hdr = sgps_hdr
#
## Insert a new header keyword into the header for the radial and tangential
## components of the directional derivative, which will record the starting
## pixel of theta.
#rad_tang_hdr.insert(22, ('CRPIX3', 1.00000000000E+00))
#
## Insert a new header keyword into the header for the radial and tangential
## components of the directional derivative, which will record the difference
## between successive values of theta.
#rad_tang_hdr.insert(23, ('CDELT3', theta[1] - theta[0]))
#
## Insert a new header keyword into the header for the radial and tangential
## components of the directional derivative, which will record the starting
## value of theta.
#rad_tang_hdr.insert(24, ('CRVAL3', -1.80000000000E+02))
#
## Insert a new header keyword into the header for the radial and tangential
## components of the directional derivative, which will record that the third
## axis of the data cube specifies values of theta.
#rad_tang_hdr.insert(25, ('CTYPE3', 'THETA-DEG'))
#
## Convert the data cube containing values of the radial component of the
## directional derivative into a FITS file, using the header information of the
## SGPS data. Also save the FITS file that is produced by the function.
#rad_comp_direc_div_FITS = mat2FITS_Image(rad_comp, rad_tang_hdr,\
#data_loc + 'sgps_rad_comp_direc_div_40.fits')
#
## Convert the data cube containing values of the tangential component of the
## directional derivative into a FITS file, using the header information of the
## SGPS data. Also save the FITS file that is produced by the function.
#tang_comp_direc_div_FITS = mat2FITS_Image(tang_comp, rad_tang_hdr,\
#data_loc + 'sgps_tang_comp_direc_div_40.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS files successfully saved for the radial and tangential components'\
#+ ' of the directional derivative.'

#--------- AMPLITUDE OF TANGENTIAL COMPONENT OF DIRECTIONAL DERIVATIVE ---------

## Here the amplitude of the tangential component of the directional derivative
## of the complex polarisation vector is calculated. This depends upon the first
## order derivatives of the Stokes parameters, and the Stokes parameters
## themselves. The formula is given on page 57 of PhD Logbook 2.
#
## Pass the partial derivative arrays to the function that calculates the
## amplitude of the tangential component. This function returns an array of the
## amplitude values (essentially an image), which is the same shape as the 
## input arrays.
#tang_comp_amp = calc_Tang_Direc_Amp(Sto_Q, Sto_U, dQ_dy, dQ_dx,\
#dU_dy, dU_dx, num_theta = 40)
#
## Print a message to the screen to show that the amplitude of the tangential 
## component array has been calculated successfully.
#print 'Amplitude of the tangential component calculated successfully.'
#
## Convert the array containing values of the amplitude of the tangential
## component of the directional derivative into a FITS file, using the header
## information of the SGPS data. Also save the FITS file that is produced by the
## function.
#tang_comp_direc_div_amp_FITS = mat2FITS_Image(tang_comp_amp, sgps_hdr,\
#data_loc + 'sgps_tang_comp_direc_div_amp_40.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS files successfully saved for the amplitude of the tangential'\
#+ ' component of the directional derivative.'
#
## Create an image of the amplitude of the tangential component of the 
## directional derivative for the SGPS data using aplpy and the produced FITS
## file. This image is automatically saved using the given filename.
#fits2aplpy(tang_comp_direc_div_amp_FITS, data_loc + 'sgps_tang_comp_amp.png',\
#colour = 'hot')
#
## Print a message to the screen to show that the image of the amplitude of the
## tangential component of the directional derivative has been successfully
## produced and saved.
#print 'Image of the amplitude of the tangential component of the directional'\
#+ ' derivative successfully saved.\n'

#----------- AMPLITUDE OF RADIAL COMPONENT OF DIRECTIONAL DERIVATIVE ----------

# Here the amplitude of the radial component of the directional derivative
# of the complex polarisation vector is calculated. This depends upon the first
# order derivatives of the Stokes parameters, and the Stokes parameters
# themselves. The formula is given on page 57 of PhD Logbook 2.

# Pass the partial derivative arrays to the function that calculates the
# amplitude of the radial component. This function returns an array of the
# amplitude values (essentially an image), which is the same shape as the 
# input arrays.
rad_comp_amp = calc_Rad_Direc_Amp(Sto_Q, Sto_U, dQ_dy, dQ_dx,\
dU_dy, dU_dx, num_theta = 40)

# Print a message to the screen to show that the amplitude of the radial
# component array has been calculated successfully.
print 'Amplitude of the radial component calculated successfully.'

# Convert the array containing values of the amplitude of the radial
# component of the directional derivative into a FITS file, using the header
# information of the SGPS data. Also save the FITS file that is produced by the
# function.
rad_comp_direc_div_amp_FITS = mat2FITS_Image(rad_comp_amp, sgps_hdr,\
data_loc + 'sgps_rad_comp_direc_div_amp_40.fits')

# Print a message to the screen to show that the FITS file was produced and
# saved successfully.
print 'FITS files successfully saved for the amplitude of the radial'\
+ ' component of the directional derivative.'

# Create an image of the amplitude of the radial component of the 
# directional derivative for the SGPS data using aplpy and the produced FITS
# file. This image is automatically saved using the given filename.
fits2aplpy(rad_comp_direc_div_amp_FITS, data_loc + 'sgps_rad_comp_amp.png',\
colour = 'hot')

# Print a message to the screen to show that the image of the amplitude of the
# radial component of the directional derivative has been successfully
# produced and saved.
print 'Image of the amplitude of the radial component of the directional'\
+ ' derivative successfully saved.\n'

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# Print a message to the screen to show that all tasks have been 
# completed successfully.
print 'All files and images produced successfully.'