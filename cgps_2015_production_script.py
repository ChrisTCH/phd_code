#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to open the FITS files containing the    #
# smoothed CGPS data, and produce FITS files and images of the polarisation   #
# intensity and polarisation gradient, as well as various other diagnostic    #
# quantities. This is performed for each final angular resolution that was    #
# used to smooth the data. The calculation of these quantities will occur in  #
# separate functions. The quantities will then be saved as FITS files in the  #
# same directory as the CGPS data.                                            #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 20/8/2015                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
import aplpy
from astropy.io import fits

# Import functions that calculate the spatial derivatives of Stokes parameters
from calc_Sto_1Diff import calc_Sto_1Diff

# Import functions that calculate more complicated quantities involving
# the Stokes parameters.
from calc_Polar_Inten import calc_Polar_Inten
from calc_Polar_Grad import calc_Polar_Grad
from calc_Rad_Direc_Amp import calc_Rad_Direc_Amp
from calc_Tang_Direc_Amp import calc_Tang_Direc_Amp
from calc_Direc_Amp_Max import calc_Direc_Amp_Max

# Import utility functions
from mat2FITS_Image import mat2FITS_Image
from fits2aplpy import fits2aplpy
from hist_plot import hist_plot
from mat_plot import mat_plot

# Create a string object which stores the directory of the CGPS data
data_loc = '/Volumes/CAH_ExtHD/CGPS_2015/'

# Create a string that will be used to control what Q and U FITS files are used
# to perform calculations, and that will be appended into the filename of 
# anything produced in this script. This is either 'high_lat' or 'plane'
save_append = 'plane'

# Create an array that specifies all of the final resolution values that were 
# used to create mosaics. This code will calculate quantities for each of the
# resolutions given in this array
final_res_array = np.array([75, 90, 105, 120, 135, 150, 165, 180, 195, 210,\
 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 450,\
 480, 510, 540, 570, 600, 630, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140,\
 1200])

# Loop over the values for the final resolution of the survey, as we will 
# calculate quantities for each resolution
for i in range(len(final_res_array)):
	# Print a message to show that calculations are starting for the current
	# final resolution
	print 'Calculations starting for final resolution of {} arcseconds'\
	.format(final_res_array[i])

	# Open the CGPS Stokes Q FITS file for the current resolution
	cgps_Q_fits = fits.open(data_loc + 'Sto_Q_{}_smoothed/'.format(save_append)\
	 + 'Sto_Q_{}_smooth2_{}.fits'.format(save_append, final_res_array[i]))

	# Obtain the header of the primary HDU for the Stokes Q data
	cgps_Q_hdr = cgps_Q_fits[0].header

	# Open the CGPS Stokes U data FITS file
	cgps_U_fits = fits.open(data_loc + 'Sto_U_{}_smoothed/'.format(save_append)\
	 + 'Sto_U_{}_smooth2_{}.fits'.format(save_append, final_res_array[i]))

	# Obtain the header of the primary HDU for the Stokes U data
	cgps_U_hdr = cgps_U_fits[0].header

	# Extract the size of each pixel from the header. This is the length of each 
	# side of the pixel (assumed to be square), in degrees. 
	pix_size_deg = cgps_Q_hdr['CDELT2']

	# Extract the Stokes Q data from the FITS file, which is held in the
	# primary HDU
	Sto_Q = cgps_Q_fits[0].data

	# Extract the Stokes U data from the FITS file, which is held in the
	# primary HDU
	Sto_U = cgps_U_fits[0].data

	# Print a message to the screen saying that the data was successfully 
	# extracted
	print 'CGPS data successfully extracted from the FITS file.'

	# Calculate the first order partial derivatives of Stokes Q and U with 
	# respect to the y and x axes of the image. This function returns arrays 
	# that are the same size as the arrays in Stokes Q and U.
	# UNITS ARE KELVIN PER DEGREE
	dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(Sto_Q, Sto_U, pix_size_deg)

	#------------------------ POLARISATION INTENSITY ---------------------------

	# Use the Stokes parameters to calculate the observed polarisation intensity
	# at each pixel of the image.
	polar_inten = calc_Polar_Inten(Sto_Q, Sto_U)

	# Print a message to the screen to show that the observed polarisation 
	# intensity has been calculated successfully.
	print 'Observed polarisation intensity calculated successfully.'

	# # Convert the matrix of polarisation intensity values into a FITS file, 
	# # using the header information of the CGPS data. Also save the FITS file 
	# # that is produced by the function.
	# polar_inten_FITS = mat2FITS_Image(polar_inten, cgps_Q_hdr,\
	# data_loc + 'Polar_Inten_{}_smooth2_{}.fits'.format(save_append,\
	#  final_res_array[i]))

	# # Print a message to the screen to show that the FITS file was produced and
	# # saved successfully.
	# print 'FITS file successfully saved for the polarisation intensity.'

	# # If we are calculating quantities for the high latitude extension of the 
	# # CGPS survey, then produce an image of the polarisation intensity
	# if save_append == 'high_lat':
	# 	# Create an image of the observed polarisation intensity for the CGPS
	# 	# data using aplpy and the produced FITS file. This image is 
	# 	# automatically saved using the given filename.
	# 	fits2aplpy(polar_inten_FITS, data_loc+'Polar_Inten_{}_smooth2_{}.png'.\
	# 		format(save_append, final_res_array[i]), colour = 'hot',\
	# 		 convention = 'wells')

	# 	# Print a message to the screen to show that the image of the observed
	# 	# polarisation intensity has been successfully produced and saved.
	# 	print 'Image of the observed polarisation intensity successfully saved.\n'

	#--------------------- POLARISATION GRADIENT MAGNITUDE ---------------------

	# Use the first order partial derivatives to calculate the magnitude of the
	# polarisation gradient of the image.
	polar_grad = calc_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)

	# Print a message to the screen to show that the magnitude of the 
	# polarisation gradient has been calculated successfully.
	print 'Magnitude of the polarisation gradient calculated successfully.'

	# # Convert the matrix of polarisation gradient values into a FITS file, using
	# # the header information of the CGPS data. Also save the FITS file that is
	# # produced by the function.
	# polar_grad_FITS = mat2FITS_Image(polar_grad, cgps_Q_hdr,\
	# data_loc + 'Polar_Grad_{}_smooth2_{}.fits'.format(save_append,\
	#  final_res_array[i]))

	# # Print a message to the screen to show that the FITS file was produced and
	# # saved successfully.
	# print 'FITS file successfully saved for the magnitude of the polarisation '\
	#  + 'gradient.'

	# # If we are calculating quantities for the high latitude extension of the 
	# # CGPS survey, then produce an image of the polarisation gradient
	# if save_append == 'high_lat':
	# 	# Create an image of the observed polarisation gradient for the CGPS
	# 	# data using aplpy and the produced FITS file. This image is 
	# 	# automatically saved using the given filename.
	# 	fits2aplpy(polar_grad_FITS, data_loc+'Polar_Grad_{}_smooth2_{}.png'.\
	# 		format(save_append, final_res_array[i]), colour = 'hot',\
	# 		 convention = 'wells')

	# 	# Print a message to the screen to show that the image of the observed
	# 	# polarisation intensity has been successfully produced and saved.
	# 	print 'Image of the observed polarisation gradient successfully saved.\n'

	#------------------- Normalised Polarisation Gradient ----------------------

	# # Use the calculated polarisation gradient and polarised intensity to 
	# # calculate the normalised polarisation gradient
	# norm_polar_grad = polar_grad / polar_inten

	# # Print a message to the screen to show that the 
	# # normalised polarisation gradient has been calculated successfully.
	# print 'Normalised polarisation gradient calculated successfully.'

	# # Convert the matrix of normalised polarisation gradient values into a FITS 
	# # file, using the header information of the CGPS data. Also save the FITS 
	# # file that is produced by the function.
	# norm_polar_grad_FITS = mat2FITS_Image(norm_polar_grad, cgps_Q_hdr,\
	# data_loc + 'Norm_Polar_Grad_{}_smooth2_{}.fits'.format(save_append,\
	#  final_res_array[i]))

	# # Print a message to the screen to show that the FITS file was produced and
	# # saved successfully.
	# print 'FITS file successfully saved for the normalised polarisation '\
	#  + 'gradient.'

	#------------- Maximum Radial Component Directional Derivative -------------

	# Calculate the maximum value of the radial component of the directional
	# derivative at each pixel of the image, using the Stokes Q and U values
	# and their derivatives
	rad_direc_amp = calc_Rad_Direc_Amp(Sto_Q, Sto_U, dQ_dy, dQ_dx, dU_dy, dU_dx)

	# Print a message to the screen to show that the maximum of the radial
	# component of the directional derivative has been calculated successfully.
	print 'Maximum Radial Component Direc Deriv calculated successfully.'

	# Convert the matrix of maximum radial component values into a FITS file, 
	# using the header information of the CGPS data. Also save the FITS file 
	# that is produced by the function.
	rad_direc_amp_FITS = mat2FITS_Image(rad_direc_amp, cgps_Q_hdr,\
	data_loc + 'Rad_Direc_Amp_{}_smooth2_{}.fits'.format(save_append,\
	 final_res_array[i]))

	# Print a message to the screen to show that the FITS file was produced and
	# saved successfully.
	print 'FITS file successfully saved for the radial component of the '\
	 + 'directional derivative.'

	#----------- Maximum Tangential Component Directional Derivative -----------

	# Calculate the maximum value of the tangential component of the directional
	# derivative at each pixel of the image, using the Stokes Q and U values
	# and their derivatives
	tang_direc_amp = calc_Tang_Direc_Amp(Sto_Q, Sto_U, dQ_dy, dQ_dx, dU_dy, dU_dx)

	# Print a message to the screen to show that the maximum of the tangential
	# component of the directional derivative has been calculated successfully.
	print 'Maximum Tangential Component Direc Deriv calculated successfully.'

	# Convert the matrix of maximum tangential component values into a FITS file
	# using the header information of the CGPS data. Also save the FITS file 
	# that is produced by the function.
	tang_direc_amp_FITS = mat2FITS_Image(tang_direc_amp, cgps_Q_hdr,\
	data_loc + 'Tang_Direc_Amp_{}_smooth2_{}.fits'.format(save_append,\
	 final_res_array[i]))

	# Print a message to the screen to show that the FITS file was produced and
	# saved successfully.
	print 'FITS file successfully saved for the tangential component of the '\
	 + 'directional derivative.'

	#---- Difference Radial and Tangential Components Directional Derivative ---

	# Calculate the difference between the maximum values of the radial and 
	# tangential components of the directional derivative
	diff_rad_tang_direc = rad_direc_amp - tang_direc_amp

	# Print a message to the screen to show that the maximum of the tangential
	# component of the directional derivative has been calculated successfully.
	print 'Difference Radial - Tangential Component Direc Deriv calculated successfully.'

	# Convert the matrix of difference values into a FITS file
	# using the header information of the CGPS data. Also save the FITS file 
	# that is produced by the function.
	diff_rad_tang_direc_FITS = mat2FITS_Image(diff_rad_tang_direc, cgps_Q_hdr,\
	data_loc + 'Diff_Rad_Tang_Direc_{}_smooth2_{}.fits'.format(save_append,\
	 final_res_array[i]))

	# Print a message to the screen to show that the FITS file was produced and
	# saved successfully.
	print 'FITS file successfully saved for the difference between the radial '\
	 + 'tangential components of the directional derivative.'

	#---------------- Maximum Amplitude Directional Derivative -----------------

	# # Calculate the maximum value of the amplitude of the directional
	# # derivative at each pixel of the image, using the Stokes Q and U values
	# # and their derivatives
	# direc_amp_max = calc_Direc_Amp_Max(Sto_Q, Sto_U, dQ_dy, dQ_dx, dU_dy, dU_dx)

	# # Print a message to the screen to show that the maximum amplitude of the
	# # directional derivative has been calculated successfully.
	# print 'Maximum Amplitude Direc Deriv calculated successfully.'

	# # Convert the matrix of maximum amplitude values into a FITS file
	# # using the header information of the CGPS data. Also save the FITS file 
	# # that is produced by the function.
	# direc_amp_max_FITS = mat2FITS_Image(direc_amp_max, cgps_Q_hdr,\
	# data_loc + 'Direc_Amp_Max_{}_smooth2_{}.fits'.format(save_append,\
	#  final_res_array[i]))

	# # Print a message to the screen to show that the FITS file was produced and
	# # saved successfully.
	# print 'FITS file successfully saved for the maximum amplitude of the '\
	#  + 'directional derivative.'

	#---------------------------------------------------------------------------

	# Print a message to say that everything has been calculated for this 
	# simulations
	print 'Calculations complete for resolution of {} arcseconds'\
	.format(final_res_array[i])

	# Close the Stokes Q FITS file, so that it is no longer in memory
	cgps_Q_fits.close()

	# Close the Stokes U FITS file, so that it is no longer in memory
	cgps_U_fits.close()

# When the code reaches here, everything that needs to be calculated has been
# calculated. Print a message to inform the user that the script has finished
print 'All calculations completed successfully'