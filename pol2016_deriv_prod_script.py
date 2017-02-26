#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths and density and calculates projected quantities related to   #
# the simulation cube, such as the projected density, magnetic field strength, #
# synchrotron intensity, and Faraday depth. These quantities can then be used  #
# to better understand what polarisation diagnostics are telling us.           #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 24/10/2016                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import mat2FITS_Image to convert arrays to FITS format
from mat2FITS_Image import mat2FITS_Image

# Import the functions that calculate derivatives of Stokes Q and U
from calc_Sto_1Diff import calc_Sto_1Diff

# Create a string for the directory that contains the simulated maps to use
simul_loc = '/Volumes/CAH_ExtHD/Pol_2016/'

# Create a string for the specific simulated data set to use in calculations.
# The directories end in:
# b.1p.1_Oct_Burk
# b.1p.01_Oct_Burk
# b.1p2_Aug_Burk
# b1p.1_Oct_Burk
# b1p.01_Oct_Burk
# b1p2_Aug_Burk
# c512b.1p.0049
# c512b.1p.0077
# c512b.1p.025
# c512b.1p.05
# c512b.1p.7
# c512b.5p.0049
# c512b.5p.0077
# c512b.5p.01
# c512b.5p.025
# c512b.5p.05
# c512b.5p.1
# c512b.5p.7
# c512b.5p2
# c512b1p.0049
# c512b1p.0077
# c512b1p.025
# c512b1p.05
# c512b1p.7

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
# simul_arr = ['b1p2/']
simul_arr = ['b.1p.0049/', 'b.1p.0077/', 'b.1p.01/','b.1p.025/', 'b.1p.05/',\
'b.1p.1/', 'b.1p.7/', 'b.1p2/', 'b1p.0049/',\
'b1p.0077/', 'b1p.01/', 'b1p.025/', 'b1p.05/', 'b1p.1/', 'b1p.7/', 'b1p2/']
# 'b.5p.0049/', 'b.5p.0077/', 'b.5p.01/',\
# 'b.5p.025/', 'b.5p.05/', 'b.5p.1/', 'b.5p.7/', 'b.5p2/',

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the images.
# This can be 'x', 'y', or 'z'. The mean magnetic field is along the x axis.
line_o_sight = 'z'

# Create a string that selects the folder to store results in based on the LOS
los_loc = line_o_sight + '_los/'

# Create a variable that controls whether the polarisation diagnostic maps/cubes
# are calculated for the case where the simulation cube is backlit by polarised
# emission, or when polarised emission is generated from within the cube.
# This can be 'backlit' or 'internal'.
emis_mech = ['backlit']

# Create a variable that specifies the range of wavelengths at which mock
# observeations were performed in the case where the observed emission 
# is generated within the simulation cube. These values are for the wavelength
# squared, to give observing frequencies between 0.5 - 2 GHz.
lambda_sq_arr = np.linspace(0.0225, 0.36, 50, endpoint = True) 

# Calculate the spacing in wavelength squared between adjacent frequency
# evaluations
wave_sq_space = lambda_sq_arr[1] - lambda_sq_arr[0]

# Create a variable that specifies the size of a pixel in parsecs
dl = 0.15

# Iterate over the simulations, to produce projected quantities for each
# simulation
for i in range(len(simul_arr)): 
	# Print a message to show what simulation calculations are being performed
	# for
	print 'Calculations starting for {}'.format(simul_arr[i])

	# Create a string for the full directory path to use in calculations
	data_loc =  simul_loc + simul_arr[i] + los_loc

	#----------------------- Gradient of Faraday Depth -------------------------

	# # Open the file that contains the Faraday depth for the simulation
	# Fara_fits = fits.open(data_loc + 'FaradayDepth.fits')

	# # Extract the data for the simulated Faraday Depth
	# Fara_data = Fara_fits[0].data

	# # Extract the header for the simulated Faraday Depth
	# Fara_hdr = Fara_fits[0].header

	# # Calculate the derivatives of the Faraday depth
	# dF_dy, dF_dx = np.gradient(Fara_data, dl)

	# # Calculate the amplitude of the gradient
	# Fara_grad = np.sqrt(np.power(dF_dx,2.0) + np.power(dF_dy,2.0))

	# # Save the produced gradient image as a FITS file
	# mat2FITS_Image(Fara_grad, Fara_hdr, data_loc +\
	#  'FaradayDepthGrad.fits', clobber = True)

	# # Calculate the angle that the gradient makes with the horizontal
	# Fara_grad_ang = np.rad2deg(np.arctan(dF_dy/dF_dx))

	# # Save the produced angle of the gradient image as a FITS file
	# mat2FITS_Image(Fara_grad_ang, Fara_hdr, data_loc +\
	#  'FaradayDepthGradAng.fits', clobber = True)

	# # Close the FITS file
	# Fara_fits.close()

	#--------------------- Gradient of Perp Magnetic Field ---------------------

 #    # Open the file that contains the projected perpendicular component of the
 #    # magnetic field for the simulation
	# BPerp_fits = fits.open(data_loc + 'B_perp_projected.fits')

	# # Extract the data for the simulated perpendicular component of the magnetic
	# # field
	# BPerp_data = BPerp_fits[0].data

	# # Extract the header for the simulated perpendicular component of the
	# # magnetic field
	# BPerp_hdr = BPerp_fits[0].header

	# # Calculate the derivatives of the perpendicular component of the magnetic
	# # field
	# dBP_dy, dBP_dx = np.gradient(BPerp_data, dl)

	# # Calculate the amplitude of the gradient
	# BPerp_grad = np.sqrt(np.power(dBP_dx,2.0) + np.power(dBP_dy,2.0))

	# # Save the produced gradient image as a FITS file
	# mat2FITS_Image(BPerp_grad, BPerp_hdr, data_loc +\
	#  'B_perp_grad.fits', clobber = True)

	# # Calculate the angle that the gradient makes with the horizontal
	# BPerp_grad_ang = np.rad2deg(np.arctan(dBP_dy/dBP_dx))

	# # Save the produced angle of the gradient image as a FITS file
	# mat2FITS_Image(BPerp_grad_ang, BPerp_hdr, data_loc +\
	#  'B_perp_grad_ang.fits', clobber = True)

	# # Close the FITS file
	# BPerp_fits.close()

	#------------ Cross Product Gradients of Faraday Depth and Perp B ----------

	# # Calculate the amplitude of the cross product between the gradients of the 
	# # Faraday Depth and the perpendicular component of the magnetic field
	# cross_prod = np.absolute(dBP_dy * dF_dx - dBP_dx * dF_dy)

	# # Save the produced image of the cross product between the gradients of the 
	# # Faraday Depth and the perpendicular component of the magnetic field as a 
	# # FITS file
	# mat2FITS_Image(cross_prod, Fara_hdr, data_loc +\
	#  'cross_prod_gradFara_gradBPerp.fits', clobber = True)

	# #------------ Angle Between Gradients of Faraday Depth and Perp B ----------

	# # Calculate the dot product between the gradients of the Faraday Depth and
	# # the perpendicular component of the magnetic field
	# dot_prod = dF_dx * dBP_dx + dF_dy * dBP_dy

	# # Calculate the angle between the gradients of the Faraday Depth and the
	# # perpendicular component of the magnetic field
	# ang_betw_grads = np.rad2deg(np.arccos(dot_prod / (Fara_grad * BPerp_grad)))

	# # Save the produced image of the angle between the gradients of the Faraday
	# # Depth and the perpendicular component of the magnetic field as a FITS file
	# mat2FITS_Image(ang_betw_grads, Fara_hdr, data_loc +\
	#  'ang_betw_gradFara_gradBPerp.fits', clobber = True)

	# #-------------------- Gradient Polarisation Intensity ----------------------

	# # Open the file that contains the polarisation intensity of the simulation
	# pol_inten_fits = fits.open(data_loc + 'PolarInten_internal.fits')

	# # Extract the data for the simulated polarisation intensity
	# pol_inten_data = pol_inten_fits[0].data

	# # Extract the header for the simulated polarisation intensity
	# pol_inten_hdr = pol_inten_fits[0].header

	# # Calculate the derivatives of the polarisation intensity
	# dP_dy = np.gradient(pol_inten_data, dl, axis = 1)
	# dP_dx = np.gradient(pol_inten_data, dl, axis = 2)

	# # Calculate the amplitude of the gradient
	# pol_inten_grad = np.sqrt(np.power(dP_dx,2.0) + np.power(dP_dy,2.0))

	# # Save the produced gradient image as a FITS file
	# mat2FITS_Image(pol_inten_grad, pol_inten_hdr, data_loc +\
	#  'PolarIntenGrad.fits', clobber = True)

	# # Calculate the angle that the gradient makes with the horizontal
	# pol_inten_grad_ang = np.rad2deg(np.arctan2(dP_dy,dP_dx))

	# # Save the produced angle of the gradient image as a FITS file
	# mat2FITS_Image(pol_inten_grad_ang, pol_inten_hdr, data_loc +\
	#  'PolarIntenGradAng.fits', clobber = True)

	# #---------------- Wavelength Gradient Polarisation Intensity ---------------

	# # Calculate the derivative of the polarisation intensity with respect to
	# # wavelength squared
	# pol_inten_wav_grad = np.gradient(pol_inten_data, wave_sq_space, axis = 0)

	# # Save the produced gradient image as a FITS file
	# mat2FITS_Image(pol_inten_wav_grad, pol_inten_hdr, data_loc +\
	#  'PolarIntenWavGrad.fits', clobber = True)

	# # Close the FITS file
	# pol_inten_fits.close()

	#---------------------- Gradient Polarisation Angle ------------------------

	# # Loop over the possible emission mechanisms
	# for emis in emis_mech:
	# 	# Open the file that contains the polarisation angle of the simulation
	# 	pol_angle_fits = fits.open(data_loc + 'PolarAngle_{}.fits'.format(emis))

	# 	# Extract the header for the simulated polarisation angle
	# 	pol_angle_hdr = pol_angle_fits[0].header

	# 	# Open the FITS file that contains Stokes Q for the simulation, for this
	# 	# line of sight, and this emission mechanism
	# 	StoQ_fits = fits.open(data_loc + 'StoQ_{}.fits'.format(emis))

	# 	# Extract the data for Stokes Q
	# 	StoQ = StoQ_fits[0].data

	# 	# Open the FITS file that contains Stokes U for the simulation, for this
	# 	# line of sight, and this emission mechanism
	# 	StoU_fits = fits.open(data_loc + 'StoU_{}.fits'.format(emis))

	# 	# Extract the data for Stokes U
	# 	StoU = StoU_fits[0].data

	# 	# Calculate all of the first order spatial derivatives that we need
	# 	dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ, StoU, pix_sep = dl)

	# 	# Calculate the derivative of the polarisation angle with respect to x
	# 	dAng_dx = 0.5 * (StoU * dQ_dx - StoQ * dU_dx) / (np.power(StoU,2.0) +\
	# 	 np.power(StoQ,2.0))

	# 	# Calculate the derivative of the polarisation angle with respect to y
	# 	dAng_dy = 0.5 * (StoU * dQ_dy - StoQ * dU_dy) / (np.power(StoU,2.0) +\
	# 	 np.power(StoQ,2.0))

	# 	# Calculate the amplitude of the gradient
	# 	pol_angle_grad = np.sqrt(np.power(dAng_dx,2.0) + np.power(dAng_dy,2.0))

	# 	# Calculate the polarisation intensity times the amplitude of the 
	# 	# gradient
	# 	pol_angle_grad_times_inten = 2.0 * np.sqrt(np.power(StoQ,2.0) +\
	# 	 np.power(StoU,2.0)) * pol_angle_grad

	# 	# Save the produced gradient image as a FITS file
	# 	mat2FITS_Image(pol_angle_grad_times_inten, pol_angle_hdr, data_loc +\
	# 	 'PolarAngleGradTimesInten_{}.fits'.format(emis), clobber = True)

	# 	# # Calculate the angle that the gradient makes with the horizontal
	# 	# pol_angle_grad_ang = np.rad2deg(np.arctan2(dAng_dy,dAng_dx))

	# 	# # Save the produced angle of the gradient image as a FITS file
	# 	# mat2FITS_Image(pol_angle_grad_ang, pol_angle_hdr, data_loc +\
	# 	#  'PolarAngleGradAng_{}.fits'.format(emis), clobber = True)

	# 	# Close the FITS file
	# 	pol_angle_fits.close()
	# 	StoQ_fits.close()
	# 	StoU_fits.close()

	#------------------- Rotation Measure Times Pol Inten ----------------------

	# # Open the file that contains the polarisation intensity of the simulation
	# pol_inten_fits = fits.open(data_loc + 'PolarInten_internal.fits')

	# # Extract the data for the simulated polarisation intensity
	# pol_inten_data = pol_inten_fits[0].data

	# # Extract the header for the simulated polarisation intensity
	# pol_inten_hdr = pol_inten_fits[0].header

	# # Open the FITS file that contains rotation measure for the simulation, for 
	# # this line of sight
	# RM_fits = fits.open(data_loc + 'RotMeas_internal.fits')

	# # Extract the data for the rotation measure
	# RM = RM_fits[0].data

	# # Calculate the polarisation intensity times the rotation measure
	# RM_times_inten = pol_inten_data * RM

	# # Save the image of RM times polarisation intensity
	# mat2FITS_Image(RM_times_inten, pol_inten_hdr, data_loc +\
	#  'RotMeasTimesInten_internal.fits', clobber = True)

	# # Close the FITS file
	# pol_inten_fits.close()
	# RM_fits.close()

	# #--------------------- Mixed Derivative Max Amplitude ----------------------

	# # NOTE: ONLY CALCULATE THIS FOR THE CASE OF BACKLIT EMISSION

	# if 'backlit' in emis_mech:
	# 	# Open the file that contains the Faraday depth for the simulation
	# 	Fara_fits = fits.open(data_loc + 'FaradayDepth.fits')

	# 	# Extract the data for the simulated Faraday Depth
	# 	Fara_data = Fara_fits[0].data

	# 	# Extract the header for the simulated Faraday Depth
	# 	Fara_hdr = Fara_fits[0].header

	# 	# Calculate the derivatives of the Faraday depth
	# 	dF_dy, dF_dx = np.gradient(Fara_data, dl)

	# 	# Open the file that contains the polarisation angle of the simulation
	# 	pol_angle_fits = fits.open(data_loc + 'PolarAngle_backlit.fits')

	# 	# Extract the header for the simulated polarisation angle
	# 	pol_angle_hdr = pol_angle_fits[0].header

	# 	# Open the FITS file that contains Stokes Q for the simulation, for this
	# 	# line of sight, and this emission mechanism
	# 	StoQ_fits = fits.open(data_loc + 'StoQ_backlit.fits')

	# 	# Extract the data for Stokes Q
	# 	StoQ = StoQ_fits[0].data

	# 	# Open the FITS file that contains Stokes U for the simulation, for this
	# 	# line of sight, and this emission mechanism
	# 	StoU_fits = fits.open(data_loc + 'StoU_backlit.fits')

	# 	# Extract the data for Stokes U
	# 	StoU = StoU_fits[0].data

	# 	# Calculate all of the first order spatial derivatives that we need
	# 	dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ, StoU, pix_sep = dl)

	# 	# Calculate the derivative of the polarisation angle with respect to x
	# 	dAng_dx = 0.5 * (StoU * dQ_dx - StoQ * dU_dx) / (np.power(StoU,2.0) +\
	# 	 np.power(StoQ,2.0))

	# 	# Calculate the derivative of the polarisation angle with respect to y
	# 	dAng_dy = 0.5 * (StoU * dQ_dy - StoQ * dU_dy) / (np.power(StoU,2.0) +\
	# 	 np.power(StoQ,2.0))

	# 	# Calculate the first group of terms required for the calculation
	# 	first_group = 4 * (np.power(dF_dx,2.0) + np.power(dF_dy,2.0)) +\
	# 	16 * np.power(Fara_data,2.0) * (np.power(dAng_dx,2.0) +\
	# 	np.power(dAng_dy,2.0))

	# 	# Calculate the second group of terms required for the calculation
	# 	second_group = np.power(8 * Fara_data * (dAng_dy * dF_dx - dAng_dx *\
	# 	 dF_dy),2.0)

	# 	# Calculate the maximum amplitude of the mixed derivative, squared
	# 	mix_deriv_max_sq = 0.5 * (first_group + np.sqrt(\
	# 		np.power(first_group,2.0) - 4 * second_group))

	# 	# Calculate the maximum amplitude of the mixed derivative
	# 	mix_deriv_max = np.sqrt(mix_deriv_max_sq)

	# 	# Save the produced gradient image as a FITS file
	# 	mat2FITS_Image(mix_deriv_max, pol_angle_hdr, data_loc +\
	# 	 'MixDerivMax_backlit.fits', clobber = True)

	#----------------- Angle that Maximises the Mixed Derivative ---------------

	# NOTE: ONLY CALCULATE THIS FOR THE CASE OF BACKLIT EMISSION

	if 'backlit' in emis_mech:
		# Open the file that contains the Faraday depth for the simulation
		Fara_fits = fits.open(data_loc + 'FaradayDepth.fits')

		# Extract the data for the simulated Faraday Depth
		Fara_data = Fara_fits[0].data

		# Extract the header for the simulated Faraday Depth
		Fara_hdr = Fara_fits[0].header

		# Calculate the derivatives of the Faraday depth
		dF_dy, dF_dx = np.gradient(Fara_data, dl)

		# Open the file that contains the polarisation angle of the simulation
		pol_angle_fits = fits.open(data_loc + 'PolarAngle_backlit.fits')

		# Extract the header for the simulated polarisation angle
		pol_angle_hdr = pol_angle_fits[0].header

		# Open the FITS file that contains Stokes Q for the simulation, for this
		# line of sight, and this emission mechanism
		StoQ_fits = fits.open(data_loc + 'StoQ_backlit.fits')

		# Extract the data for Stokes Q
		StoQ = StoQ_fits[0].data

		# Open the FITS file that contains Stokes U for the simulation, for this
		# line of sight, and this emission mechanism
		StoU_fits = fits.open(data_loc + 'StoU_backlit.fits')

		# Extract the data for Stokes U
		StoU = StoU_fits[0].data

		# Calculate all of the first order spatial derivatives that we need
		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ, StoU, pix_sep = dl)

		# Calculate the derivative of the polarisation angle with respect to x
		dAng_dx = 0.5 * (StoU * dQ_dx - StoQ * dU_dx) / (np.power(StoU,2.0) +\
		 np.power(StoQ,2.0))

		# Calculate the derivative of the polarisation angle with respect to y
		dAng_dy = 0.5 * (StoU * dQ_dy - StoQ * dU_dy) / (np.power(StoU,2.0) +\
		 np.power(StoQ,2.0))

		# Calculate the first group of terms required for the calculation
		first_group = np.power(dF_dx,2.0) + np.power(dF_dy,2.0) +\
		4 * np.power(Fara_data,2.0) * (np.power(dAng_dx,2.0) +\
		np.power(dAng_dy,2.0))

		# Calculate the second group of terms required for the calculation
		second_group = 16 * np.power(Fara_data * (dAng_dy * dF_dx - dAng_dx *\
		 dF_dy),2.0)

		# Calculate the denominator used to calculate the angle
		denom = np.sqrt(np.power(first_group,2.0) - second_group)

		# Calculate the numerator of the sin formula
		numer_sin = dF_dx * dF_dy + 4 * np.power(Fara_data,2.0) *dAng_dx*dAng_dy

		# Calculate the numerator of the cos formula
		numer_cos = np.power(dF_dy,2.0) - np.power(dF_dx,2.0) +\
		4 *np.power(Fara_data,2.0)*(np.power(dAng_dy,2.0)-np.power(dAng_dx,2.0))

		# Calculate the quantities that we will calculate the inverse sin and cos
		# of, so that we can check all of the values are valid
		inv_sin = numer_sin/denom
		inv_cos = numer_cos/denom

		# Find entries where the numerator for the sin calculation is larger than
		# the denominator
		fix_sin_one = inv_sin > 1.0

		# Convert the improper values, so that they won't break the inverse sin
		# function. 
		inv_sin[fix_sin_one] = 1.0

		# Now find entries where the numerator for the sin calculation has an 
		# absolute value larger than the denominator, and is negative
		fix_sin_neg_one = inv_sin < -1.0

		# Convert the improper values, so they won't break the inverse sin function
		inv_sin[fix_sin_neg_one] = -1.0

		# Find entries where the numerator for the cos calculation is larger than
		# the denominator
		fix_cos_one = inv_cos > 1.0

		# Convert the improper values, so that they won't break the inverse cos
		# function. 
		inv_cos[fix_cos_one] = 1.0

		# Now find entries where the numerator for the cos calculation has an 
		# absolute value larger than the denominator, and is negative
		fix_cos_neg_one = inv_cos < -1.0

		# Convert the improper values, so they won't break the inverse cos function
		inv_cos[fix_cos_neg_one] = -1.0

		# Calculate double the angle for which the mixed derivative is 
		# maximised, at each pixel, using inverse sin
		double_theta_sin = np.arcsin(inv_sin)

		# Calculate double the angle for which the mixed derivative is 
		# maximised, at each pixel, using inverse cos
		double_theta_cos = np.arccos(inv_cos)

		# Find the entries in the array where the angle returned by inverse cos
		# is more than pi/2, as in this case the value for double the angle lies
		# in the second or third quadrant, so we need to adjust the angle that
		# is measured by inverse sin
		theta_cos_entries = double_theta_cos > np.pi/2.0

		# Find the entries of the array where the measured angle was in the first
		# quadrant, but it is supposed to be in the second quadrant
		second_quad = np.logical_and(theta_cos_entries, double_theta_sin >= 0)

		# Find the entries of the array where the measured angle was in the fourth
		# quadrant, but it is supposed to be in the third quadrant
		third_quad = np.logical_and(theta_cos_entries, double_theta_sin < 0)

		# For entries that are supposed to be in the second quadrant, adjust the
		# value of the measured angle
		double_theta_sin[second_quad] = np.pi - double_theta_sin[second_quad]

		# For entries that are supposed to be in the third quadrant, adjust the
		# value of the measured angle
		double_theta_sin[third_quad] = -1.0 * np.pi - double_theta_sin[third_quad]

		# Calculate the angle for which the mixed derivative is maximised,
		# at each pixel
		mix_max_ang = np.rad2deg(0.5 * double_theta_sin)

		# Save the produced angle image as a FITS file
		mat2FITS_Image(mix_max_ang, pol_angle_hdr, data_loc +\
		 'MixMaxAng_backlit.fits', clobber = True)

	#---------------------------------------------------------------------------

	# Print a message to state that the FITS files were saved successfully
	print 'FITS files of projected quantities saved successfully {}'.\
	format(simul_arr[i])

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All projected diagnostic maps calculated successfully'