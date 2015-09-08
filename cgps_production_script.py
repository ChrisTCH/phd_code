#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to open the FITS files containing the    #
# CGPS data, and produce FITS files and images of the polarisation gradient,  #
# polarisation curvature, and various other diagnostic quantities. The        #
# calculation of these quantities will occur in separate functions. The       #
# quantities will then be saved as FITS files in the same directory as the    #
# CGPS data.                                                                  #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 19/8/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
import aplpy
from astropy.io import fits

# Import any matplotlib related things here
from matplotlib.colors import LogNorm

# Import functions that calculate the spatial derivatives of Stokes parameters
from calc_Sto_1Diff import calc_Sto_1Diff
from calc_Sto_2Diff import calc_Sto_2Diff

# Import functions that calculate more complicated quantities involving
# the Stokes parameters.
from calc_Polar_Inten import calc_Polar_Inten
from calc_Polar_Angle import calc_Polar_Angle
from calc_Polar_Grad import calc_Polar_Grad
from calc_Ang_Grad import calc_Ang_Grad
from calc_Mod_Div_Grad import calc_Mod_Div_Grad
from calc_Quad_Curv import calc_Quad_Curv
from calc_Direc_Div import calc_Direc_Div
from calc_Direc_Curv import calc_Direc_Curv
from calc_Direc_Accel import calc_Direc_Accel
from calc_Rad_Tang_Direc import calc_Rad_Tang_Direc
from calc_Tang_Direc_Amp import calc_Tang_Direc_Amp
from calc_Rad_Direc_Amp import calc_Rad_Direc_Amp
from calc_Grad_Polar_Grad import calc_Grad_Polar_Grad
from calc_Ang_Grad_Polar_Grad import calc_Ang_Grad_Polar_Grad
from calc_Ang_Betw_DP_DModDP import calc_Ang_Betw_DP_DModDP
from calc_FFT2 import calc_FFT2

# Import utility functions
from mat2FITS_Image import mat2FITS_Image
from fits2aplpy import fits2aplpy
from hist_plot import hist_plot
from mat_plot import mat_plot
from bayes_block import bayes_block

# Create a string object which stores the directory of the CGPS data
data_loc = '/Users/chrisherron/Documents/PhD/CGPS_2015/'

# Create a string that will be used to control what Q and U FITS files are used
# to perform calculations, and that will be appended into the filename of 
# anything produced in this script
save_append = 'high_lat'

# Open the CGPS Stokes Q data FITS file
cgps_Q_fits = fits.open(data_loc + 'Sto_Q_{}.fits'.format(save_append))

# Print the information about the Stokes Q file. This should show that there is
# only a primary HDU, which contains all of the image data
cgps_Q_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the primary HDU for the Stokes Q data
cgps_Q_hdr = cgps_Q_fits[0].header

# Print the header information to the screen for Stokes Q
print cgps_Q_hdr
# Print a blank line to make the script output easier to read
print ''

# Open the CGPS Stokes U data FITS file
cgps_U_fits = fits.open(data_loc + 'Sto_U_{}.fits'.format(save_append))

# Print the information about the Stokes U file. This should show that there is
# only a primary HDU, which contains all of the image data
cgps_U_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the primary HDU for the Stokes U data
cgps_U_hdr = cgps_U_fits[0].header

# Print the header information to the screen for Stokes U
print cgps_U_hdr
# Print a blank line to make the script output easier to read
print ''

# Open the CGPS Stokes I data FITS file
cgps_I_fits = fits.open(data_loc + 'Sto_I_{}.fits'.format(save_append))

# Print the information about the Stokes I file. This should show that there is
# only a primary HDU, which contains all of the image data
cgps_I_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the primary HDU for the Stokes I data
cgps_I_hdr = cgps_I_fits[0].header

# Extract the size of each pixel from the header. This is the length of each 
# side of the pixel (assumed to be square), in degrees. 
pix_size_deg = cgps_Q_hdr['CDELT2']

# Extract the Stokes Q data from the FITS file, which is held in the primary HDU
Sto_Q = cgps_Q_fits[0].data

# Extract the Stokes U data from the FITS file, which is held in the primary HDU
Sto_U = cgps_U_fits[0].data

# Extract the Stokes I data from the FITS file, which is held in the primary HDU
Sto_I = cgps_I_fits[0].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'CGPS data successfully extracted from the FITS file.'

# Check the shape of the data array.
# NOTE: The shape of the array should be (4224, 4224), with the first 
# dimension corresponding to the y-axis, which is Galactic Latitude, and the
# second dimension is Galactic Longitude, which goes along the x-axis.
print 'The shape of the data array is: {}.'.format(Sto_Q.shape) 

# Print a message to the screen to show that everything is going smoothly
print 'Stokes parameters successfully extracted from data.'

# Calculate the first order partial derivatives of Stokes Q and U with respect
# to the y and x axes of the image. This function returns arrays that are the 
# same size as the arrays in Stokes Q and U.
# UNITS ARE KELVIN PER DEGREE
dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(Sto_Q, Sto_U, pix_size_deg)

# Calculate the second order partial derivatives of Stokes Q and U with
# respect to the y and x axes of the image. This function returns arrays that
# are the same size as the arrays in Stokes Q and U.
# UNITS ARE KELVIN PER SQUARE DEGREE
d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2 = \
calc_Sto_2Diff(dQ_dy, dQ_dx, dU_dy, dU_dx, pix_size_deg)

# Print a message to the screen to show that the derivatives have been 
# calculated correctly
print 'Derivatives of Stokes parameters successfully calculated.\n'

#------------------------ POLARISATION INTENSITY -------------------------------

# Use the Stokes parameters to calculate the observed polarisation intensity at
# each pixel of the image.
polar_inten = calc_Polar_Inten(Sto_Q, Sto_U)

# Print a message to the screen to show that the observed polarisation intensity
# has been calculated successfully.
print 'Observed polarisation intensity calculated successfully.'

# Convert the matrix of polarisation intensity values into a FITS file, using
# the header information of the CGPS data. Also save the FITS file that is
# produced by the function.
polar_inten_FITS = mat2FITS_Image(polar_inten, cgps_Q_hdr,\
data_loc + 'Polar_Inten_{}.fits'.format(save_append))

# Print a message to the screen to show that the FITS file was produced and
# saved successfully.
print 'FITS file successfully saved for the polarisation intensity.'

# # Create an image of the observed polarisation intensity for the CGPS
# # data using aplpy and the produced FITS file. This image is automatically
# # saved using the given filename.
# fits2aplpy(polar_inten_FITS, data_loc+'Polar_Inten_{}.png'.format(save_append),\
# colour = 'hot', convention = 'wells')

# # Print a message to the screen to show that the image of the observed
# # polarisation intensity has been successfully produced and saved.
# print 'Image of the observed polarisation intensity successfully saved.\n'

#------------------------- POLARISATION FRACTION -------------------------------

# Use the Stokes parameters to calculate the observed polarisation fraction at
# each pixel of the image.
polar_frac = polar_inten / Sto_I

# Print a message to the screen to show that the observed polarisation fraction
# has been calculated successfully.
print 'Observed polarisation fraction calculated successfully.'

# Convert the matrix of polarisation fraction values into a FITS file, using
# the header information of the CGPS data. Also save the FITS file that is
# produced by the function.
polar_frac_FITS = mat2FITS_Image(polar_frac, cgps_Q_hdr,\
data_loc + 'Polar_Frac_{}.fits'.format(save_append))

# Print a message to the screen to show that the FITS file was produced and
# saved successfully.
print 'FITS file successfully saved for the polarisation intensity.'

#-------------------------- POLARISATION ANGLE --------------------------------

# Use the Stokes parameters to calculate the observed polarisation angle at
# each pixel of the image.
polar_angle = calc_Polar_Angle(Sto_Q, Sto_U)

# Print a message to the screen to show that the observed polarisation angle
# has been calculated successfully.
print 'Observed polarisation angle calculated successfully.'

# Convert the matrix of polarisation angle values into a FITS file, using
# the header information of the CGPS data. Also save the FITS file that is
# produced by the function.
polar_angle_FITS = mat2FITS_Image(polar_angle, cgps_Q_hdr,\
data_loc + 'Polar_Angle_{}.fits'.format(save_append))

# Print a message to the screen to show that the FITS file was produced and
# saved successfully.
print 'FITS file successfully saved for the polarisation angle.'

# # Create an image of the observed polarisation angle for the CGPS
# # data using aplpy and the produced FITS file. This image is automatically
# # saved using the given filename.
# fits2aplpy(polar_angle_FITS, data_loc+'Polar_Angle_{}.png'.format(save_append),\
# colour = 'RdBu', vmin = -90.0, vmax = 90.0, convention = 'wells')

# # Print a message to the screen to show that the image of the observed
# # polarisation angle has been successfully produced and saved.
# print 'Image of the observed polarisation angle successfully saved.\n'

#--------------------- POLARISATION GRADIENT MAGNITUDE ------------------------

# Use the first order partial derivatives to calculate the magnitude of the
# polarisation gradient of the image.
polar_grad = calc_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)

# Print a message to the screen to show that the magnitude of the polarisation
# gradient has been calculated successfully.
print 'Magnitude of the polarisation gradient calculated successfully.'

# Convert the matrix of polarisation gradient values into a FITS file, using
# the header information of the CGPS data. Also save the FITS file that is
# produced by the function.
polar_grad_FITS = mat2FITS_Image(polar_grad, cgps_Q_hdr,\
data_loc + 'Polar_Grad_{}.fits'.format(save_append))

# Print a message to the screen to show that the FITS file was produced and
# saved successfully.
print 'FITS file successfully saved for the magnitude of the polarisation ' + \
'gradient.'

# # Create an image of the magnitude of the polarisation gradient for the CGPS
# # data using aplpy and the produced FITS file. This image is automatically
# # saved using the given filename.
# fits2aplpy(polar_grad_FITS, data_loc + 'Polar_Grad_{}.png'.format(save_append),\
# colour = 'hot', convention = 'wells')

# # Print a message to the screen to show that the image of the magnitude of the
# # polarisation gradient has been successfully produced and saved.
# print 'Image of the magnitude of the polarisation gradient successfully saved.'\
# + '\n'

#------------------------ POLARISATION GRADIENT ANGLE -------------------------

# # Pass the Stokes Q and U arrays to the function which calculates the angle of
# # the polarisation gradient with respect to the x axis. This function returns
# # an array of the same size as the arrays in Stokes Q and U.
# ang_polar_grad = calc_Ang_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)

# # Print a message to the screen to show that the angle of the polarisation 
# # gradient has been calculates successfully.
# print 'Angle of the polarisation gradient calculated successfully.'

# # Convert the matrix containing the angles of the polarisation gradient into
# # a FITS file, using the header information of the CGPS data. Also save the 
# # FITS file that is produced by the function.
# ang_grad_FITS = mat2FITS_Image(ang_polar_grad, cgps_Q_hdr,\
# data_loc + 'cgps_ang_grad.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS file successfully saved for the angle of the polarisation ' + \
# 'gradient.'

# # Create an image of the angle of the polarisation gradient for the CGPS
# # data using aplpy and the produced FITS file. This image is automatically
# # saved using the given filename.
# fits2aplpy(ang_grad_FITS, data_loc + 'cgps_ang_grad.png', colour = 'RdBu',\
# vmin = -90.0, vmax = 90.0, convention = 'wells')

# # Print a message to the screen to show that the image of the angle of the
# # polarisation gradient has been successfully produced and saved.
# print 'Image of the angle of the polarisation gradient successfully saved.\n'

#----------------- MODULUS OF THE DIVERGENCE OF GRADIENT ----------------------

# # Here the modulus of the divergence of the polarisation gradient is 
# # calculated, which depends only upon second order derivatives of the Stokes
# # parameters. The formula is given on page 105 of PhD Logbook 1, and page
# # 52 of PhD Logbook 2. This quantity is rotationally and translationally
# # invariant in the Q-U plane. 

# # Pass the second derivative arrays to the function that calculates the
# # modulus of the divergence of the polarisation gradient. This function 
# # returns an array of the same size as the second derivative arrays.
# mod_div_grad = calc_Mod_Div_Grad(d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2)

# # Print a message to the screen to show that the modulus of the divergence
# # of the gradient has been calculated successfully.
# print 'Modulus of the divergence of the polarisation gradient calculated' +\
# ' successfully.'

# # Convert the matrix containing values of the divergence of the gradient into
# # a FITS file, using the header information of the CGPS data. Also save the 
# # FITS file that is produced by the function.
# mod_div_grad_FITS = mat2FITS_Image(mod_div_grad, cgps_Q_hdr,\
# data_loc + 'cgps_mod_div_grad.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS file successfully saved for the modulus of the divergence of the'+\
# ' polarisation gradient'

# # Create an image of the modulus of the divergence of the polarisation gradient
# # for the CGPS data using aplpy and the produced FITS file. This image is 
# # automatically saved using the given filename.
# fits2aplpy(mod_div_grad_FITS, data_loc + 'cgps_mod_div_grad.png',\
# colour = 'hot', convention = 'wells')

# # Print a message to the screen to show that the image of the modulus of the 
# # divergence of the polarisation gradient has been successfully produced and
# # saved.
# print 'Image of the modulus of the divergence of the polarisation gradient' +\
# ' successfully saved.\n'

#------------------------- QUADRATURE OF CURVATURES ---------------------------

# # Here the quadrature of the curvatures in the x and y directions is 
# # calculated, which depends upon both the first and second order derivatives of
# # the Stokes parameters. The formula is given on page 96 of PhD Logbook 1, and
# # page 23 of PhD Logbook 2. 

# # Pass the partial derivative arrays to the function that calculates the
# # quadrature of the curvatures. This function returns an array of the same size
# # as the partial derivative arrays.
# quad_curv = calc_Quad_Curv(dQ_dy, dQ_dx, dU_dy, dU_dx,\
# d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2)

# # Print a message to the screen to show that the quadrature of the curvatures
# # has been calculated successfully.
# print 'Quadrature of the curvatures calculated successfully.'

# # Convert the matrix containing values of the quadrature of curvatures into
# # a FITS file, using the header information of the CGPS data. Also save the 
# # FITS file that is produced by the function.
# quad_curv_FITS = mat2FITS_Image(quad_curv, cgps_Q_hdr,\
# data_loc + 'cgps_quad_curv.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS file successfully saved for the quadrature of curvatures.'

# # Create an image of the quadrature of curvatures for the CGPS data using aplpy
# # and the produced FITS file. This image is automatically saved using the given
# # filename.
# fits2aplpy(quad_curv_FITS, data_loc + 'cgps_quad_curv.png',\
# colour = 'hot', convention = 'wells')

# # Print a message to the screen to show that the image of the quadrature of the
# # curvatures has been successfully produced and saved.
# print 'Image of the quadrature of the curvatures successfully saved.\n'

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
## The FITS header for the CGPS data needs to be updated for the directional
## derivative data cube, so that the value of theta for each slice of the data
## cube is known by any program reading the data cube. Create a copy of the CGPS
## header, which will be modified for the directional derivative cube.
#direc_div_hdr = cgps_hdr
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
## a FITS file, using the header information of the CGPS data. Also save the 
## FITS file that is produced by the function.
#direc_div_FITS = mat2FITS_Image(direc_div, direc_div_hdr,\
#data_loc + 'cgps_direc_div.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the directional derivative.\n'

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
#d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2, num_theta = 40)
#
## Print a message to the screen to show that the directional curvature data
## cube has been calculated successfully.
#print 'Directional curvature data cube calculated successfully.'
#
## The FITS header for the CGPS data needs to be updated for the directional
## curvature data cube, so that the value of theta for each slice of the data
## cube is known by any program reading the data cube. Create a copy of the CGPS
## header, which will be modified for the directional curvature cube.
#direc_curv_hdr = cgps_hdr
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
## a FITS file, using the header information of the CGPS data. Also save the 
## FITS file that is produced by the function.
#direc_curv_FITS = mat2FITS_Image(direc_curv, direc_curv_hdr,\
#data_loc + 'cgps_direc_curv_no_denom.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the directional curvature.\n'

#------------------------- DIRECTIONAL ACCELERATION ---------------------------

## Here the directional acceleration of the complex polarisation vector is
## calculated for various directions determined by the angle to the horizontal
## theta. The directional acceleration is the magnitude of the acceleration 
## vector that results from moving along the path in the Q-U plane traced out
## when moving at an angle theta with respect to the horizontal axis. This 
## depends upon the second order derivatives of the Stokes parameters. The
## formula is given on page 76 of PhD Logbook 2.
#
## Pass the partial derivative arrays to the function that calculates the
## directional acceleration data cube. This function returns a data cube with
## three axes. Each slice of the data cube is an image of the directional
## acceleration for a particular value of theta, and this image is of the same
## size as the partial derivative arrays.
#direc_accel, theta = calc_Direc_Accel(d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2,\
#num_theta = 40)
#
## Print a message to the screen to show that the directional acceleration data
## cube has been calculated successfully.
#print 'Directional acceleration data cube calculated successfully.'
#
## The FITS header for the CGPS data needs to be updated for the directional
## acceleration data cube, so that the value of theta for each slice of the data
## cube is known by any program reading the data cube. Create a copy of the CGPS
## header, which will be modified for the directional acceleration cube.
#direc_acc_hdr = cgps_hdr
#
## Insert a new header keyword into the header for the directional acceleration,
## which will record the starting pixel of theta.
#direc_acc_hdr.insert(22, ('CRPIX3', 1.00000000000E+00))
#
## Insert a new header keyword into the header for the directional acceleration,
## which will record the difference between successive values of theta.
#direc_acc_hdr.insert(23, ('CDELT3', theta[1] - theta[0]))
#
## Insert a new header keyword into the header for the directional acceleration,
## which will record the starting value of theta.
#direc_acc_hdr.insert(24, ('CRVAL3', -1.80000000000E+02))
#
## Insert a new header keyword into the header for the directional acceleration,
## which will record that the third axis of the data cube specifies values of
## theta.
#direc_acc_hdr.insert(25, ('CTYPE3', 'THETA-DEG'))
#
## Convert the data cube containing values of the directional acceleration into
## a FITS file, using the header information of the CGPS data. Also save the 
## FITS file that is produced by the function.
#direc_acc_FITS = mat2FITS_Image(direc_accel, direc_acc_hdr,\
#data_loc + 'cgps_direc_accel.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS file successfully saved for the directional acceleration.\n'

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
## The FITS header for the CGPS data needs to be updated for the radial and
## tangential component data cubes, so that the value of theta for each slice of
## the data cube is known by any program reading the data cube. Create a copy of
## the CGPS header, which will be modified for the radial and tangential 
## component cubes.
#rad_tang_hdr = cgps_hdr
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
## CGPS data. Also save the FITS file that is produced by the function.
#rad_comp_direc_div_FITS = mat2FITS_Image(rad_comp, rad_tang_hdr,\
#data_loc + 'cgps_rad_comp_direc_div_40.fits')
#
## Convert the data cube containing values of the tangential component of the
## directional derivative into a FITS file, using the header information of the
## CGPS data. Also save the FITS file that is produced by the function.
#tang_comp_direc_div_FITS = mat2FITS_Image(tang_comp, rad_tang_hdr,\
#data_loc + 'cgps_tang_comp_direc_div_40.fits')
#
## Print a message to the screen to show that the FITS file was produced and
## saved successfully.
#print 'FITS files successfully saved for the radial and tangential components'\
#+ ' of the directional derivative.\n'

#--------- AMPLITUDE OF TANGENTIAL COMPONENT OF DIRECTIONAL DERIVATIVE ---------

# # Here the amplitude of the tangential component of the directional derivative
# # of the complex polarisation vector is calculated. This depends upon the first
# # order derivatives of the Stokes parameters, and the Stokes parameters
# # themselves. The formula is given on page 57 of PhD Logbook 2.

# # Pass the partial derivative arrays to the function that calculates the
# # amplitude of the tangential component. This function returns an array of the
# # amplitude values (essentially an image), which is the same shape as the 
# # input arrays.
# tang_comp_amp = calc_Tang_Direc_Amp(Sto_Q, Sto_U, dQ_dy, dQ_dx,\
# dU_dy, dU_dx, num_theta = 40)

# # Print a message to the screen to show that the amplitude of the tangential 
# # component array has been calculated successfully.
# print 'Amplitude of the tangential component calculated successfully.'

# # Convert the array containing values of the amplitude of the tangential
# # component of the directional derivative into a FITS file, using the header
# # information of the CGPS data. Also save the FITS file that is produced by the
# # function.
# tang_comp_direc_div_amp_FITS = mat2FITS_Image(tang_comp_amp, cgps_Q_hdr,\
# data_loc + 'cgps_tang_comp_direc_div_amp_40.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS files successfully saved for the amplitude of the tangential'\
# + ' component of the directional derivative.'

# # Create an image of the amplitude of the tangential component of the 
# # directional derivative for the CGPS data using aplpy and the produced FITS
# # file. This image is automatically saved using the given filename.
# fits2aplpy(tang_comp_direc_div_amp_FITS, data_loc + 'cgps_tang_comp_amp.png',\
# colour = 'hot', convention = 'wells')

# # Print a message to the screen to show that the image of the amplitude of the
# # tangential component of the directional derivative has been successfully
# # produced and saved.
# print 'Image of the amplitude of the tangential component of the directional'\
# + ' derivative successfully saved.\n'

#----------- AMPLITUDE OF RADIAL COMPONENT OF DIRECTIONAL DERIVATIVE ----------

# # Here the amplitude of the radial component of the directional derivative
# # of the complex polarisation vector is calculated. This depends upon the first
# # order derivatives of the Stokes parameters, and the Stokes parameters
# # themselves. The formula is given on page 57 of PhD Logbook 2.

# # Pass the partial derivative arrays to the function that calculates the
# # amplitude of the radial component. This function returns an array of the
# # amplitude values (essentially an image), which is the same shape as the 
# # input arrays.
# rad_comp_amp = calc_Rad_Direc_Amp(Sto_Q, Sto_U, dQ_dy, dQ_dx,\
# dU_dy, dU_dx, num_theta = 40)

# # Print a message to the screen to show that the amplitude of the radial
# # component array has been calculated successfully.
# print 'Amplitude of the radial component calculated successfully.'

# # Convert the array containing values of the amplitude of the radial
# # component of the directional derivative into a FITS file, using the header
# # information of the CGPS data. Also save the FITS file that is produced by the
# # function.
# rad_comp_direc_div_amp_FITS = mat2FITS_Image(rad_comp_amp, cgps_Q_hdr,\
# data_loc + 'cgps_rad_comp_direc_div_amp_40.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS files successfully saved for the amplitude of the radial'\
# + ' component of the directional derivative.'

# # Create an image of the amplitude of the radial component of the 
# # directional derivative for the CGPS data using aplpy and the produced FITS
# # file. This image is automatically saved using the given filename.
# fits2aplpy(rad_comp_direc_div_amp_FITS, data_loc + 'cgps_rad_comp_amp.png',\
# colour = 'hot', convention = 'wells')

# # Print a message to the screen to show that the image of the amplitude of the
# # radial component of the directional derivative has been successfully
# # produced and saved.
# print 'Image of the amplitude of the radial component of the directional'\
# + ' derivative successfully saved.\n'

#----------- MAGNITUDE OF GRADIENT OF POLARISATION GRADIENT MAGNITUDE ----------

# # Here the magnitude of the gradient of the polarisation gradient magnitude is
# # calculated. This depends upon the first and second order derivatives of the
# # Stokes parameters. The formula is given on page 108 of PhD Logbook 1.

# # Pass the partial derivative arrays to the function that calculates the
# # magnitude of the gradient of the polarisation gradient. This function returns
# # an array of the magnitude values (essentially an image), which is the same
# # shape as the input arrays.
# grad_polar_grad = calc_Grad_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx, d2Q_dy2,\
# d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2)

# # Print a message to the screen to show that the magnitude of the gradient of
# # the polarisation gradient magnitude array has been calculated successfully.
# print 'Magnitude of the gradient of the polarisation gradient calculated' +\
# ' successfully.'

# # Convert the array containing values of the magnitude of the gradient of the
# # polarisation gradient magnitude into a FITS file, using the header
# # information of the CGPS data. Also save the FITS file that is produced by the
# # function.
# grad_polar_grad_FITS = mat2FITS_Image(grad_polar_grad, cgps_Q_hdr,\
# data_loc + 'cgps_grad_polar_grad.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS files successfully saved for the magnitude of the gradient of the'\
# + ' polarisation gradient magnitude.'

# # Create an image of the magnitude of the gradient of the polarisation  
# # gradient magnitude for the CGPS data using aplpy and the produced FITS
# # file. This image is automatically saved using the given filename.
# fits2aplpy(grad_polar_grad_FITS, data_loc + 'cgps_grad_polar_grad.png',\
# colour = 'hot', convention = 'wells')

# # Print a message to the screen to show that the image of the magnitude of the
# # gradient of the polarisation gradient magnitude has been successfully
# # produced and saved.
# print 'Image of the magnitude of the gradient of the polarisation gradient'\
# + ' magnitude successfully saved.\n'

#------------- ANGLE OF GRADIENT OF POLARISATION GRADIENT MAGNITUDE -----------

# # Here the angle of the gradient of the polarisation gradient magnitude with
# # respect to the x axis of the image is calculated. This depends upon the first
# # and second order derivatives of the Stokes parameters. The formula is given 
# # on page 108 of PhD Logbook 1.

# # Pass the partial derivative arrays to the function that calculates the
# # angle of the gradient of the polarisation gradient. This function returns
# # an array of the angle values in degrees (essentially an image), which is the
# # same shape as the input arrays.
# ang_grad_polar_grad = calc_Ang_Grad_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx,\
# d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2)

# # Print a message to the screen to show that the angle of the gradient of
# # the polarisation gradient magnitude array has been calculated successfully.
# print 'Angle of the gradient of the polarisation gradient calculated' +\
# ' successfully.'

# # Convert the array containing values of the angle of the gradient of the
# # polarisation gradient magnitude into a FITS file, using the header
# # information of the CGPS data. Also save the FITS file that is produced by the
# # function.
# ang_grad_polar_grad_FITS = mat2FITS_Image(ang_grad_polar_grad, cgps_Q_hdr,\
# data_loc + 'cgps_ang_grad_polar_grad.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS files successfully saved for the angle of the gradient of the'\
# + ' polarisation gradient magnitude.'

# # Create an image of the angle of the gradient of the polarisation  
# # gradient magnitude for the CGPS data using aplpy and the produced FITS
# # file. This image is automatically saved using the given filename.
# fits2aplpy(ang_grad_polar_grad_FITS, data_loc + 'cgps_ang_grad_polar_grad.png',\
# colour = 'RdBu', vmin = -180.0, vmax = 180.0, convention = 'wells')

# # Print a message to the screen to show that the image of the angle of the
# # gradient of the polarisation gradient magnitude has been successfully
# # produced and saved.
# print 'Image of the angle of the gradient of the polarisation gradient'\
# + ' magnitude successfully saved.\n'

#--------------- ANGLE BETWEEN GRADIENT AND GRADIENT OF GRADIENT --------------

# # Here the angle between the polarisation gradient and the gradient of the
# # polarisation gradient magnitude is calculated. This depends upon the first
# # and second order derivatives of the Stokes parameters. The formula is given 
# # on page 114 of PhD Logbook 1. (This is the Hermitian angle between complex
# # valued vectors).

# # Pass the partial derivative arrays to the function that calculates the
# # angle between the polarisation gradient and the gradient of the polarisation
# # gradient magnitude. This function returns an array of the angle values in
# # degrees (essentially an image), which is the same shape as the input arrays.
# ang_betw_DP_DModDP = calc_Ang_Betw_DP_DModDP(dQ_dy, dQ_dx, dU_dy, dU_dx,\
# d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2)

# # Print a message to the screen to show that the angle between the polarisation
# # gradient and the gradient of the polarisation gradient magnitude array has 
# # been calculated successfully.
# print 'Angle between the polarisation gradient and the gradient of the'\
# + ' polarisation gradient magnitude calculated successfully.'

# # Create a Bayesian Block histogram of the values for the angle between the 
# # polarisation gradient and the gradient of the polarisation gradient 
# # magnitude. This histogram is saved as an image.
# hist_plot(ang_betw_DP_DModDP, data_loc + 'cgps_ang_betw_DP_DModDP_hist.png',\
# 'png', x_label = 'Angle Difference [deg]', title ='Angle between DP and DModDP')

# # Convert the array containing values of the angle between the polarisation 
# # gradient and the gradient of the polarisation gradient magnitude into a 
# # FITS file, using the header information of the CGPS data. Also save the FITS
# # file that is produced by the function.
# ang_betw_DP_DModDP_FITS = mat2FITS_Image(ang_betw_DP_DModDP, cgps_Q_hdr,\
# data_loc + 'cgps_ang_betw_DP_DModDP.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS files successfully saved for the angle between the polarisation'\
# + ' gradient and the gradient of the polarisation gradient magnitude.'

# # Create an image of the angle between the polarisation gradient and the 
# # gradient of the polarisation gradient magnitude for the CGPS data using aplpy
# # and the produced FITS file. This image is automatically saved using the given
# # filename.
# fits2aplpy(ang_betw_DP_DModDP_FITS, data_loc + 'cgps_ang_betw_DP_DModDP.png',\
# colour = 'hot', vmin = 0.0, vmax = 90.0, convention = 'wells')

# # Print a message to the screen to show that the image of the angle between
# # the polarisation gradient and the gradient of the polarisation gradient
# # magnitude has been successfully produced and saved.
# print 'Image of the angle between the polarisation gradient and the gradient'\
# + ' of the polarisation gradient magnitude successfully saved.\n'

#------------------ FOURIER TRANSFORM OF POLARISATION GRADIENT -----------------

## Here the Fourier transform of the image of the magnitude of the polarisation
## gradient is calculated. The idea is to see if there are any spatial
## frequencies which are predominant. The amplitude and phase spectra of the 
## image are both calculated, and images of these spectra produced.
#
## Pass the image of the magnitude of the polarisation gradient to the function
## that calculates the fourier transform of the image. This function returns 
## four arrays, which are the amplitude spectrum, the phase spectrum, and the
## frequencies for the y and x axes respectively.
#grad_amp_spec, grad_phase_spec, grad_y_freq, grad_x_freq =\
#calc_FFT2(polar_grad, pix_size_deg)
#
## Print a message to the screen to show that the fourier transform has been
## calculated successfully.
#print 'Fourier transform of the polarisation gradient magnitude calculated'\
#+ ' successfully.'
#
#print 'Length of y axis: {}'.format(len(grad_y_freq))
#print 'Length of x axis: {}'.format(len(grad_x_freq))
#print 'Shape of amplitude matrix: {}'.format(np.shape(grad_amp_spec))
#print 'Shape of phase matrix: {}'.format(np.shape(grad_phase_spec))
#
## Produce an image of the amplitude spectrum obtained from the Fourier transform
#mat_plot(grad_amp_spec, data_loc + 'cgps_grad_FFT_amp.png', format = 'png',\
#x_ticks = grad_x_freq, y_ticks = grad_y_freq, cmap = 'hot', norm = LogNorm(),\
#aspect = 'equal', origin = 'lower', xlabel = \
#'Horizontal frequency [cycles/degree]', ylabel =\
#'Vertical frequency [cycles/degree]', title =\
#'Amplitude Spectrum Polarisation Gradient')
#
## Produce an image of the phase spectrum obtained from the Fourier transform
#mat_plot(grad_phase_spec, data_loc + 'cgps_grad_FFT_phase.png', format = 'png',\
#x_ticks = grad_x_freq, y_ticks = grad_y_freq, cmap = 'hot', aspect = 'equal',\
#origin = 'lower', xlabel = 'Horizontal frequency [cycles/degree]',\
#ylabel = 'Vertical frequency [cycles/degree]', title =\
#'Phase Spectrum Polarisation Gradient')
#
## Create a Bayesian Block histogram of the values for the amplitude spectrum.
## This histogram is saved as an image.
#hist_plot(grad_amp_spec, data_loc + 'cgps_grad_FFT_amp_hist.png',\
#'png', x_label = 'Amplitude of Frequency Component', title =\
#'Histogram Amplitude Spectrum Polarisation Gradient', bins = 30, log_x = True)
#
## Create a Bayesian Block histogram of the values for the phase spectrum. This
## histogram is saved as an image.
#hist_plot(grad_phase_spec, data_loc + 'cgps_grad_FFT_phase_hist.png',\
#'png', x_label = 'Phase Angle of Frequency Component', title =\
#'Histogram Phase Spectrum Polarisation Gradient')

#------------------ FOURIER TRANSFORM OF TANGENTIAL COMPONENT -----------------

## Here the Fourier transform of the image of the amplitude of the tangential 
## component of the directional derivative is calculated. The idea is to see if
## there are any spatial frequencies which are predominant. The amplitude and 
## phase spectra of the image are both calculated, and images of these spectra
## produced.
#
## Pass the image of the amplitude of the tangential component of the directional
## derivative to the function that calculates the fourier transform of the image.
## This function returns four arrays, which are the amplitude spectrum, the phase
## spectrum, and the frequencies for the y and x axes respectively.
#tang_amp_spec, tang_phase_spec, tang_y_freq, tang_x_freq =\
#calc_FFT2(tang_comp_amp, pix_size_deg)
#
## Print a message to the screen to show that the fourier transform has been
## calculated successfully.
#print 'Fourier transform of the amplitude of the tangential component of the'\
#+ ' directional derivative calculated successfully.'
#
## Produce an image of the amplitude spectrum obtained from the Fourier transform
#mat_plot(tang_amp_spec, data_loc + 'cgps_tang_FFT_amp.png', format = 'png',\
#x_ticks = tang_x_freq, y_ticks = tang_y_freq, cmap = 'hot', norm = LogNorm(),\
#aspect = 'equal', origin = 'lower', xlabel = \
#'Horizontal frequency [cycles/degree]', ylabel =\
#'Vertical frequency [cycles/degree]', title =\
#'Amplitude Spectrum Tangential Component')
#
## Produce an image of the phase spectrum obtained from the Fourier transform
#mat_plot(tang_phase_spec, data_loc + 'cgps_tang_FFT_phase.png', format = 'png',\
#x_ticks = tang_x_freq, y_ticks = tang_y_freq, cmap = 'hot', aspect = 'equal',\
#origin = 'lower', xlabel = 'Horizontal frequency [cycles/degree]',\
#ylabel = 'Vertical frequency [cycles/degree]', title =\
#'Phase Spectrum Tangential Component')
#
## Create a Bayesian Block histogram of the values for the amplitude spectrum. 
## This histogram is saved as an image.
#hist_plot(tang_amp_spec, data_loc + 'cgps_tang_FFT_amp_hist.png',\
#'png', x_label = 'Amplitude of Frequency Component', title =\
#'Histogram Amplitude Spectrum Tangential Component', bins = 30, log_x = True)
#
## Create a Bayesian Block histogram of the values for the phase spectrum. 
## This histogram is saved as an image.
#hist_plot(tang_phase_spec, data_loc + 'cgps_tang_FFT_phase_hist.png',\
#'png', x_label = 'Phase Angle of Frequency Component', title =\
#'Histogram Phase Spectrum Tangential Component')

##------------------- FOURIER TRANSFORM OF RADIAL COMPONENT --------------------
#
## Here the Fourier transform of the image of the amplitude of the radial 
## component of the directional derivative is calculated. The idea is to see if
## there are any spatial frequencies which are predominant. The amplitude and 
## phase spectra of the image are both calculated, and images of these spectra
## produced.
#
## Pass the image of the amplitude of the radial component of the directional
## derivative to the function that calculates the fourier transform of the image.
## This function returns four arrays, which are the amplitude spectrum, the phase
## spectrum, and the frequencies for the y and x axes respectively.
#rad_amp_spec, rad_phase_spec, rad_y_freq, rad_x_freq =\
#calc_FFT2(rad_comp_amp, pix_size_deg)
#
## Print a message to the screen to show that the fourier transform has been
## calculated successfully.
#print 'Fourier transform of the amplitude of the radial component of the'\
#+ ' directional derivative calculated successfully.'
#
## Produce an image of the amplitude spectrum obtained from the Fourier transform
#mat_plot(rad_amp_spec, data_loc + 'cgps_rad_FFT_amp.png', format = 'png',\
#x_ticks = rad_x_freq, y_ticks = rad_y_freq, cmap = 'hot', norm = LogNorm(),\
#aspect = 'equal', origin = 'lower', xlabel = \
#'Horizontal frequency [cycles/degree]', ylabel =\
#'Vertical frequency [cycles/degree]', title =\
#'Amplitude Spectrum Radial Component')
#
## Produce an image of the phase spectrum obtained from the Fourier transform
#mat_plot(rad_phase_spec, data_loc + 'cgps_rad_FFT_phase.png', format = 'png',\
#x_ticks = rad_x_freq, y_ticks = rad_y_freq, cmap = 'hot', aspect = 'equal',\
#origin = 'lower', xlabel = 'Horizontal frequency [cycles/degree]',\
#ylabel = 'Vertical frequency [cycles/degree]', title =\
#'Phase Spectrum Radial Component')
#
## Create a Bayesian Block histogram of the values for the amplitude spectrum. 
## This histogram is saved as an image.
#hist_plot(rad_amp_spec, data_loc + 'cgps_rad_FFT_amp_hist.png',\
#'png', x_label = 'Amplitude of Frequency Component', title =\
#'Histogram Amplitude Spectrum Radial Component', bins = 30, log_x = True)
#
## Create a Bayesian Block histogram of the values for the phase spectrum. 
## This histogram is saved as an image.
#hist_plot(rad_phase_spec, data_loc + 'cgps_rad_FFT_phase_hist.png',\
#'png', x_label = 'Phase Angle of Frequency Component', title =\
#'Histogram Phase Spectrum Radial Component')

#---- Angle Between Polarisation Gradient and Observed Polarisation Angle -----

# # Here the angle between the polarisation gradient and the observed 
# # polarisation angle is calculated. This depends upon the first
# # order derivatives of the Stokes parameters.

# # Calculate the angle between the observed polarisation angle and the 
# # polarisation gradient, by subtracting one from the other, and taking the 
# # absolute value.
# ang_betw_DP_polar = np.abs(ang_polar_grad - polar_angle)

# # There are some situations where the angle calculated above will be over 90
# # degrees, but I am defining the angle between the polarisation gradient and
# # the observed polarisation angle to be the acute angle between them. Thus, for
# # pixels where the angular separation is above 90 degrees, we need to calculate
# # the acute angle from the obtuse angle.

# # First find the pixel locations where the angular separation is above 90 
# # degrees
# ang_above_90 = ang_betw_DP_polar > 90.0

# # For the pixels that have angular separation above 90 degrees, replace this 
# # value by the acute angle.
# ang_betw_DP_polar[ang_above_90] = 180.0 - ang_betw_DP_polar[ang_above_90]

# # Print a message to the screen to show that the angle between the polarisation
# # gradient and the observed polarisation angle has been calculated successfully.
# print 'Angle between the polarisation gradient and the observed polarisation'\
# + ' angle calculated successfully.'

# # Create a Bayesian Block histogram of the values for the angle between the 
# # polarisation gradient and the observed polarisation angle. This histogram is
# # saved as an image.
# hist_plot(ang_betw_DP_polar, data_loc + 'cgps_ang_betw_DP_polar_hist.png',\
# 'png', x_label = 'Angle Difference [deg]', title ='Angle between DP and Polar'\
# ' Angle')

# # Convert the array containing values of the angle between the polarisation 
# # gradient and the observed polarisation angle into a FITS file, using the 
# # header information of the CGPS data. Also save the FITS file that is produced
# # by the function.
# ang_betw_DP_polar_FITS = mat2FITS_Image(ang_betw_DP_polar, cgps_Q_hdr,\
# data_loc + 'cgps_ang_betw_DP_polar.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS files successfully saved for the angle between the polarisation'\
# + ' gradient and the observed polarisation angle.'

# # Create an image of the angle between the polarisation gradient and the 
# # observed polarisation angle for the CGPS data using aplpy and the produced 
# # FITS file. This image is automatically saved using the given filename.
# fits2aplpy(ang_betw_DP_polar_FITS, data_loc + 'cgps_ang_betw_DP_polar.png',\
# colour = 'hot', vmin = 0.0, vmax = 90.0, convention = 'wells')

# # Print a message to the screen to show that the image of the angle between
# # the polarisation gradient and the observed polarisation angle
# # has been successfully produced and saved.
# print 'Image of the angle between the polarisation gradient and the observed'\
# + ' polarisation angle successfully saved.\n'

#---- Angle Between Gradient of Gradient and Observed Polarisation Angle -----

# # Here the angle between the gradient of the polarisation gradient magnitude
# # and the observed polarisation angle is calculated. This depends upon the first
# # and second order derivatives of the Stokes parameters.

# # Calculate the angle between the observed polarisation angle and the 
# # gradient of the polarisation gradient magnitude, by subtracting one from the
# # other, and taking the absolute value.
# ang_betw_DDP_polar = np.abs(ang_grad_polar_grad - polar_angle)

# # There are some situations where the angle calculated above will be over 90
# # degrees, and other situations where the angle calculated will be over 180
# # degrees, but I am defining the angle between the polarisation gradient and
# # the observed polarisation angle to be the acute angle between them. Thus, for
# # pixels where the angular separation is above 90 degrees, we need to calculate
# # the acute angle from the obtuse angle.

# # First find the pixel locations where the angular difference is greater than or
# # equal to 180 degrees.
# ang_above_180 = ang_betw_DDP_polar >= 180.0

# # For the pixels that have angular separation above 180 degrees, replace this 
# # value by the acute angle.
# ang_betw_DDP_polar[ang_above_180] = ang_betw_DDP_polar[ang_above_180] - 180.0

# # Now find the pixel locations where the angular separation is above 90 
# # degrees
# ang_above_90 = ang_betw_DDP_polar > 90.0

# # For the pixels that have angular separation above 90 degrees, replace this 
# # value by the acute angle.
# ang_betw_DDP_polar[ang_above_90] = 180.0 - ang_betw_DDP_polar[ang_above_90]

# # Print a message to the screen to show that the angle between the gradient of
# # the polarisation gradient magnitude and the observed polarisation angle has
# # been calculated successfully.
# print 'Angle between the gradient of the polarisation gradient magnitude and'\
# + ' the observed polarisation angle calculated successfully.'

# # Create a histogram of the values for the angle between the gradient of the
# # polarisation gradient magnitude and the observed polarisation angle. This
# # histogram is saved as an image.
# hist_plot(ang_betw_DDP_polar, data_loc + 'cgps_ang_betw_DDP_polar_hist.png',\
# 'png', x_label = 'Angle Difference [deg]', title ='Angle between DDP and Polar'\
# ' Angle')

# # Convert the array containing values of the angle between the gradient of the
# # polarisation gradient magnitude and the observed polarisation angle into a
# # FITS file, using the header information of the CGPS data. Also save the FITS
# # file that is produced by the function.
# ang_betw_DDP_polar_FITS = mat2FITS_Image(ang_betw_DDP_polar, cgps_Q_hdr,\
# data_loc + 'cgps_ang_betw_DDP_polar.fits')

# # Print a message to the screen to show that the FITS file was produced and
# # saved successfully.
# print 'FITS file successfully saved for the angle between the gradient of the'\
# + ' polarisation gradient magnitude and the observed polarisation angle.'

# # Create an image of the angle between the gradient of the polarisation gradient
# # magnitude and the observed polarisation angle for the CGPS data using aplpy
# # and the produced FITS file. This image is automatically saved using the given
# # filename.
# fits2aplpy(ang_betw_DDP_polar_FITS, data_loc + 'cgps_ang_betw_DDP_polar.png',\
# colour = 'hot', vmin = 0.0, vmax = 90.0, convention = 'wells')

# # Print a message to the screen to show that the image of the angle between
# # the gradient of the polarisation gradient magnitude and the observed
# # polarisation angle has been successfully produced and saved.
# print 'Image of the angle between the gradient of the polarisation gradient'\
# ' magnitude and the observed polarisation angle successfully saved.\n'

#------------------------------------------------------------------------------

# Print a message to the screen to show that all tasks have been 
# completed successfully.
print 'All files and images produced successfully.'