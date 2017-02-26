#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to read in the Stokes Q and U data for   #
# S-PASS and Urumqi data at 2.3 and 4.8 GHz respectively for a region of the  #
# Galactic plane, provided by Xiaohui. The curvature in the direction that    #
# maximises the directional derivative is calculated for these images, so     #
# we can try to determine whether we have backlit or internal emission in     #
# each image, and compare the result to what Xiaohui found.                   #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 24/1/2017                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
from astropy.io import fits

# Import mat2FITS_Image to convert arrays to FITS format
from mat2FITS_Image import mat2FITS_Image

# Import the functions that calculate derivatives of Stokes Q and U
from calc_Sto_1Diff import calc_Sto_1Diff
from calc_Sto_2Diff import calc_Sto_2Diff

# Import the function that calculates the curvature in the direction that
# maximises the directional derivative
from calc_Curv4MaxDirecDeriv import calc_Curv4MaxDirecDeriv

# Create a string object which stores the directory of the spass data
data_loc = '/Users/chrisherron/Documents/PhD/SGPS_Data/'

# Create a string object which stores the directory to save into
save_loc = '/Users/chrisherron/Documents/PhD/Pol_2016/Xiaohui_Curvature/'

#--------------------- EXTRACT DATA FOR S-PASS --------------------------------

# Open the S-PASS data FITS file
spass_fits = fits.open(data_loc + 'g22.spass13cm.sf.iuqpipa.fits')

# Print the information about the data file. This should show that there are
# five HDUs, which contains all of the image data
spass_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the Stokes U HDU
spass_hdr = spass_fits[2].header

# Extract the Stokes U data from the FITS file, which is held in the second HDU
spass_U_data = spass_fits[2].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'S-PASS Stokes U data successfully extracted from the FITS file.'

# Extract the Stokes Q data from the FITS file, which is held in the third HDU
spass_Q_data = spass_fits[3].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'S-PASS Stokes Q data successfully extracted from the FITS file.'

#--------------------- EXTRACT DATA FOR URUMQI --------------------------------

# Open the Urumqi data FITS file
urum_fits = fits.open(data_loc + 'g22.uru6cm.sf.iuqpi.fits')

# Print the information about the data file. This should show that there are
# five HDUs, which contains all of the image data
urum_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the Stokes U HDU
urum_hdr = urum_fits[2].header

# Extract the Stokes U data from the FITS file, which is held in the second HDU
urum_U_data = urum_fits[2].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'Urumqi Stokes U data successfully extracted from the FITS file.'

# Extract the Stokes Q data from the FITS file, which is held in the third HDU
urum_Q_data = urum_fits[3].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'Urumqi Stokes Q data successfully extracted from the FITS file.'

#-------------------------------------------------------------------------------

# Print a message to the screen to show that everything is going smoothly
print 'Stokes parameters successfully extracted from data.'

# Extract values from the S-PASS header that specify the separation between 
# pixels along the x and y directions in degrees
dx = np.absolute(spass_hdr['CDELT1'])
dy = np.absolute(spass_hdr['CDELT2'])

# Calculate the polarisation intensity images for both data sets
spass_pol_inten = np.sqrt(np.power(spass_Q_data,2.0)\
 + np.power(spass_U_data,2.0))
urum_pol_inten = np.sqrt(np.power(urum_Q_data,2.0)\
 + np.power(urum_U_data,2.0))

# Save the polarisation intensity images for both data sets
mat2FITS_Image(spass_pol_inten, spass_hdr, save_loc +\
 'spass_pol_inten.fits', clobber = True)
mat2FITS_Image(urum_pol_inten, urum_hdr, save_loc +\
 'urum_pol_inten.fits', clobber = True)

#---------------------- Curvature Calculations SPASS ---------------------------

# Calculate the first order spatial derivatives of Stokes Q and U for the 
# spass data
sp_dQ_dy, sp_dQ_dx, sp_dU_dy, sp_dU_dx =\
 calc_Sto_1Diff(spass_Q_data, spass_U_data, pix_sep = dy)

# Calculate all of the second order spatial derivatives of Stokes Q and U for
# the spass data
sp_d2Q_dy2, sp_d2Q_dydx, sp_d2Q_dx2, sp_d2U_dy2, sp_d2U_dydx, sp_d2U_dx2 =\
 calc_Sto_2Diff(sp_dQ_dy, sp_dQ_dx, sp_dU_dy, sp_dU_dx, pix_sep = dy)

# Calculate the curvature in the direction that maximises the directional 
# derivative for the spass data
sp_curv4max_direc_deriv = calc_Curv4MaxDirecDeriv(sp_dQ_dy, sp_dQ_dx,\
sp_dU_dy, sp_dU_dx, sp_d2Q_dy2, sp_d2Q_dx2, sp_d2U_dy2, sp_d2U_dx2,\
sp_d2Q_dydx, sp_d2U_dydx)

# Save the curvature in the direction that maximises the directional
# derivative array
mat2FITS_Image(sp_curv4max_direc_deriv, spass_hdr, save_loc +\
 'spass_Curv4MaxDirecDeriv.fits', clobber = True)

# Save the result of the curvature multiplied by the polarisation intensity,
# as this quantity makes it easier to see if emission is backlit or internal
mat2FITS_Image(sp_curv4max_direc_deriv * spass_pol_inten, spass_hdr, save_loc +\
 'spass_Curv4MaxDirecDeriv_times_pol_inten.fits', clobber = True)

# Print a message to say the diagnostic has been calculated
print 'Curvature in the direction that maximises the directional ' +\
 'derivative calculated for spass data'

#---------------------- Curvature Calculations Urumqi --------------------------

# Calculate the first order spatial derivatives of Stokes Q and U for the 
# urumqi data
ur_dQ_dy, ur_dQ_dx, ur_dU_dy, ur_dU_dx =\
 calc_Sto_1Diff(urum_Q_data, urum_U_data, pix_sep = dy)

# Calculate all of the second order spatial derivatives of Stokes Q and U for
# the urumqi data
ur_d2Q_dy2, ur_d2Q_dydx, ur_d2Q_dx2, ur_d2U_dy2, ur_d2U_dydx, ur_d2U_dx2 =\
 calc_Sto_2Diff(ur_dQ_dy, ur_dQ_dx, ur_dU_dy, ur_dU_dx, pix_sep = dy)

# Calculate the curvature in the direction that maximises the directional 
# derivative for the urumqi data
ur_curv4max_direc_deriv = calc_Curv4MaxDirecDeriv(ur_dQ_dy, ur_dQ_dx,\
ur_dU_dy, ur_dU_dx, ur_d2Q_dy2, ur_d2Q_dx2, ur_d2U_dy2, ur_d2U_dx2,\
ur_d2Q_dydx, ur_d2U_dydx)

# Save the curvature in the direction that maximises the directional
# derivative array
mat2FITS_Image(ur_curv4max_direc_deriv, urum_hdr, save_loc +\
 'urumqi_Curv4MaxDirecDeriv.fits', clobber = True)

# Save the result of the curvature multiplied by the polarisation intensity,
# as this quantity makes it easier to see if emission is backlit or internal
mat2FITS_Image(ur_curv4max_direc_deriv * urum_pol_inten, urum_hdr, save_loc +\
 'urumqi_Curv4MaxDirecDeriv_times_pol_inten.fits', clobber = True)

# Print a message to say the diagnostic has been calculated
print 'Curvature in the direction that maximises the directional ' +\
 'derivative calculated for urumqi data'