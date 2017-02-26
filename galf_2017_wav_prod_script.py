#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of the observed Stokes Q   #
# and U images for band averaged images of sections of the GALFACTS survey,    #
# and produces images of spatial polarisation diagnostics.                     #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 8/2/2017                                                         #
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
from calc_Sto_2Diff import calc_Sto_2Diff
from calc_Sto_Wav_1Diff import calc_Sto_Wav_1Diff
from calc_Sto_Wav_2Diff import calc_Sto_Wav_2Diff

# Import the functions that calculate all of the polarisation diagnostics
from calc_Polar_Inten import calc_Polar_Inten
from calc_Polar_Angle import calc_Polar_Angle
from calc_Rot_Meas import calc_Rot_Meas
from calc_Polar_Grad import calc_Polar_Grad
from calc_Direc_Amp_Max import calc_Direc_Amp_Max
from calc_Direc_Amp_Min import calc_Direc_Amp_Min
from calc_Direc_Max_Ang import calc_Direc_Max_Ang
from calc_Rad_Direc_Amp import calc_Rad_Direc_Amp
from calc_Tang_Direc_Amp import calc_Tang_Direc_Amp
from calc_Rad_Direc_Ang import calc_Rad_Direc_Ang
from calc_Tang_Direc_Ang import calc_Tang_Direc_Ang
from calc_Wav_Grad import calc_Wav_Grad
from calc_Rad_Wav_Grad import calc_Rad_Wav_Grad
from calc_Tang_Wav_Grad import calc_Tang_Wav_Grad
from calc_Direc_Curv import calc_Direc_Curv
from calc_Quad_Curv import calc_Quad_Curv
from calc_Curv4MaxDirecDeriv import calc_Curv4MaxDirecDeriv
from calc_Wav_Curv import calc_Wav_Curv
from calc_Mix_Deriv_Max import calc_Mix_Deriv_Max
from calc_Mix_Max_Ang import calc_Mix_Max_Ang

# Create a list of polarisation diagnostics to calculate. Only the diagnostics
# that are on this list will be calculated
diag_list = ['Inten', 'Angle', 'RM', 'Grad', 'Direc_Amp_Max', 'Direc_Amp_Min',\
'Direc_Max_Ang', 'Rad_Direc_Amp', 'Tang_Direc_Amp', 'Rad_Direc_Ang',\
'Tang_Direc_Ang', 'Wav_Grad', 'Rad_Wav_Grad', 'Tang_Wav_Grad',\
'Curv4MaxDirecDeriv', 'Wav_Curv', 'Mix_Deriv_Max', 'Mix_Max_Ang']

# Create a string for the directory that contains the Stokes Q and U maps to use
data_loc = '/Volumes/CAH_ExtHD/GALFACTS_2017/'

# Open the FITS file that contains Stokes Q, as a memory map, so that we don't
# load it all in at once.
StoQ_fits = fits.open(data_loc + 'GALFACTS_S1_chanavg_subset_Q.fits',\
	memmap = True)

# Extract the data for Stokes Q, as a memory map
StoQ = StoQ_fits[0].data

# Extract the header for Stokes Q
StoQ_hdr = StoQ_fits[0].header

# Open the FITS file that contains Stokes U, as a memory map, so that we don't
# load it all in at once.
StoU_fits = fits.open(data_loc + 'GALFACTS_S1_chanavg_subset_U.fits',\
	memmap = True)

# Extract the data for Stokes U, as a memory map
StoU = StoU_fits[0].data

# Print a message to the screen to show that the data has been loaded
print 'All required data loaded successfully'

# Determine the spacing between adjacent pixels, to be used in calculating
# derivatives of quantities. This is in degrees.
pix_size_deg = np.absolute(StoQ_hdr['CDELT1'])

# Get the number of frequency slices
num_slice = StoQ_hdr['NAXIS3']

# Get the starting frequency
start_freq = StoQ_hdr['CRVAL3']

# Get the frequency interval between adjacent slices
freq_int = StoQ_hdr['CDELT3']

# Create an array that specifies the frequency of each slice of the Q and U
# cubes, in Hz
freq_arr = np.linspace(start_freq, start_freq + (num_slice-1) * freq_int,\
 num = num_slice)

# Convert these frequencies into wavelengths, in metres
wav_arr = 3.0 * np.power(10.0,8.0) / freq_arr

# Calculate the wavelength squared of each slice
wav_sq_arr = np.power(wav_arr,2.0)

# Calculate the difference in wavelength squared between adjacent slices
# This has the same length as the number of frequency slices, so we can
# associate a value of the separation in wavelength squared to each slice
wave_sq_space = np.gradient(wav_sq_arr)

# Create an array that will be used to store all of the quantities that we 
# calculate, one at a time. After each quantity has been calculated in the 
# array, the array will be saved, and then the next quantity put into the array,
# overwriting the previous values that were there
storage_arr = np.zeros(np.shape(StoQ), dtype = np.float32)

# #---------------------- Polarisation Intensity -------------------------

# # Check to see if we need to calculate the polarisation intensity
# if 'Inten' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate the polarisation intensity for this Stokes Q and U
# 		polar_inten = calc_Polar_Inten(StoQ_slice, StoU_slice)

# 		# Put this slice into the storage array
# 		storage_arr[i] = polar_inten

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Polar Inten'

# 	# Save the polarisation intensity
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'PolarInten_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Polarisation intensity calculated'

# #------------------------- Polarisation Angle ------------------------------

# # Check to see if we need to calculate the polarisation angle
# if 'Angle' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate the observed polarisation angle for this Stokes Q and U
# 		polar_angle = calc_Polar_Angle(StoQ_slice, StoU_slice)

# 		# Put this slice into the storage array
# 		storage_arr[i] = polar_angle

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Polar Angle'

# 	# Save the polarisation angle array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'PolarAngle_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Polarisation angle calculated'

# #----------------------- Polarisation Gradient -----------------------------

# # Check to see if we need to calculate the polarisation gradient
# if 'Grad' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed polarisation gradient for this Stokes Q and U
# 		polar_grad = calc_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = polar_grad

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Polar Grad'

# 	# Save the polarisation gradient array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'PolarGrad_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Polarisation gradient calculated'

# #------------ Maximum Amplitude of the Directional Derivative --------------

# # Check to see if we need to calculate the maximum amplitude of the 
# # directional derivative
# if 'Direc_Amp_Max' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed maximum amplitude of the directional
# 		# derivative for this Stokes Q and U
# 		direc_amp_max = calc_Direc_Amp_Max(dQ_dy, dQ_dx, dU_dy, dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = direc_amp_max

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Direc Amp Max'

# 	# Save the maximum amplitude of the directional derivative array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'DirecAmpMax_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Maximum amplitude of the directional derivative calculated'

# #------------ Minimum Amplitude of the Directional Derivative --------------

# # Check to see if we need to calculate the minimum amplitude of the 
# # directional derivative
# if 'Direc_Amp_Min' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed minimum amplitude of the directional
# 		# derivative for this Stokes Q and U
# 		direc_amp_min = calc_Direc_Amp_Min(dQ_dy, dQ_dx, dU_dy, dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = direc_amp_min

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Direc Amp Min'

# 	# Save the minimum amplitude of the directional derivative array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'DirecAmpMin_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Minimum amplitude of the directional derivative calculated'

# #------------ Angle that Maximises the Directional Derivative --------------

# # Check to see if we need to calculate the angle that maximises the 
# # directional derivative
# if 'Direc_Max_Ang' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed angle that maximises the directional
# 		# derivative for this Stokes Q and U
# 		direc_max_ang = calc_Direc_Max_Ang(dQ_dy, dQ_dx, dU_dy, dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = direc_max_ang

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Direc Max Ang'

# 	# Save the angle that maximises the directional derivative array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'DirecMaxAng_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Angle that maximises the directional derivative calculated'

# #------------ Maximum Radial Component Directional Derivative --------------

# # Check to see if we need to calculate the maximum radial component of the 
# # directional derivative
# if 'Rad_Direc_Amp' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed maximum radial component of the directional
# 		# derivative for this Stokes Q and U
# 		rad_direc_amp = calc_Rad_Direc_Amp(StoQ_slice, StoU_slice,\
# 		 dQ_dy,dQ_dx,dU_dy, dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = rad_direc_amp

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Rad Direc Amp'

# 	# Save the maximum radial component of the directional derivative array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'RadDirecAmp_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print'Maximum radial component of the directional derivative calculated'

# #---------- Maximum Tangential Component Directional Derivative ------------

# # Check to see if we need to calculate the maximum tangential component of 
# # the directional derivative
# if 'Tang_Direc_Amp' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed maximum tangential component of the directional
# 		# derivative for this Stokes Q and U
# 		tang_direc_amp = calc_Tang_Direc_Amp(StoQ_slice,StoU_slice,\
# 			dQ_dy,dQ_dx,dU_dy,dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = tang_direc_amp

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Tang Direc Amp'

# 	# Save the maximum tangential component of the directional derivative 
# 	# array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'TangDirecAmp_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Maximum tangential component of the directional derivative '+\
# 	 'calculated'

# #-------- Angle Maximises Radial Component Directional Derivative ----------

# # Check to see if we need to calculate the angle that maximises the radial
# # component of the directional derivative
# if 'Rad_Direc_Ang' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed angle that maximises the radial component of 
# 		# the directional derivative for this Stokes Q and U
# 		rad_direc_ang = calc_Rad_Direc_Ang(StoQ_slice,StoU_slice,\
# 			dQ_dy,dQ_dx,dU_dy,dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = rad_direc_ang

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Rad Direc Ang'

# 	# Save the angle that maximises the radial component of the directional 
# 	# derivative array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'RadDirecAng_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Angle that maximises the radial component of the directional '+\
# 	 'derivative calculated'

# #-------- Angle Maximises Tangential Component Directional Derivative ------

# # Check to see if we need to calculate the angle that maximises the 
# # tangential component of the directional derivative
# if 'Tang_Direc_Ang' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate the observed angle that maximises the tangential component 
# 		# of the directional derivative for this Stokes Q and U
# 		tang_direc_ang = calc_Tang_Direc_Ang(StoQ_slice,StoU_slice,\
# 			dQ_dy,dQ_dx,dU_dy,dU_dx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = tang_direc_ang

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Tang Direc Ang'

# 	# Save the angle that maximises the tangential component of the 
# 	# directional derivative array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'TangDirecAng_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Angle that maximises the tangential component of the ' +\
# 	'directional derivative calculated'

# #------- Curvature in Direction that Maximises Directional Derivative ------

# # Check to see if we need to calculate the curvature in the direction that
# # maximises the directional derivative
# if 'Curv4MaxDirecDeriv' in diag_list:
# 	# Iterate over the slices of the Stokes Q and U arrays
# 	for i in range(num_slice):
# 		# Extract the current Stokes Q slice from the full array
# 		StoQ_slice = StoQ[i]

# 		# Extract the current Stokes U slice from the full array
# 		StoU_slice = StoU[i]

# 		# Calculate all of the first order spatial derivatives that we need
# 		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
# 		 pix_sep = pix_size_deg)

# 		# Calculate all of the second order spatial derivatives that we need
# 		d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2 =\
# 		 calc_Sto_2Diff(dQ_dy, dQ_dx, dU_dy, dU_dx, pix_sep = pix_size_deg)

# 		# Calculate the curvature in the direction that maximises the 
# 		# directional derivative for this Stokes Q and U
# 		curv4max_direc_deriv = calc_Curv4MaxDirecDeriv(dQ_dy, dQ_dx,\
# 		dU_dy, dU_dx, d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2, d2Q_dydx,d2U_dydx)

# 		# Put this slice into the storage array
# 		storage_arr[i] = curv4max_direc_deriv

# 	# Change the header to reflect what is being saved
# 	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Curv 4 Max Direc Deriv'

# 	# Save the curvature in the direction that maximises the directional
# 	# derivative array
# 	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
# 	 'Curv4MaxDirecDeriv_S1_chanavg_subset.fits', clobber = True)

# 	# Print a message to say the diagnostic has been calculated
# 	print 'Curvature in the direction that maximises the directional ' +\
# 	 'derivative calculated'

#------------------------ Rotation Measure -----------------------------

# Check to see if we need to calculate the rotation measure
if 'RM' in diag_list:
	# Iterate over the slices of the Stokes Q and U arrays
	for i in range(num_slice):
		# Extract the current Stokes Q slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoQ_slice = StoQ[max(0,i-1):min(i+2,num_slice)]

		# Extract the current Stokes U slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoU_slice = StoU[max(0,i-1):min(i+2,num_slice)]

		# Calculate the rotation measure for this Stokes Q and U
		rot_meas, unwound_angle = calc_Rot_Meas(StoQ_slice, StoU_slice,\
		 wave_sq_space[i])

		# Determine the slice of the array to return
		if i == 0:
			# Set the return slice to be 0
			return_slice = 0
		else:
			# For all other cases, we need to return the second slice of the
			# array
			return_slice = 1

		# Put this slice into the storage array
		storage_arr[i] = rot_meas[return_slice]

	# Change the header to reflect what is being saved
	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Rot Meas'

	# Save the rotation measure
	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
	 'RotMeas_S1_chanavg_subset.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Rotation measure calculated'

#----------------------- Wavelength Gradient ---------------------------

# Check to see if we need to calculate the wavelength gradient
if 'Wav_Grad' in diag_list:
	# Iterate over the slices of the Stokes Q and U arrays
	for i in range(num_slice):
		# Extract the current Stokes Q slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoQ_slice = StoQ[max(0,i-1):min(i+2,num_slice)]

		# Extract the current Stokes U slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoU_slice = StoU[max(0,i-1):min(i+2,num_slice)]

		# Calculate all of the first order wavelength derivatives that we need
		dQ_dl, dU_dl = calc_Sto_Wav_1Diff(StoQ_slice, StoU_slice,\
		 wave_sq_space[i])

		# Calculate the wavelength gradient for this Stokes Q and U
		wav_grad = calc_Wav_Grad(dQ_dl, dU_dl)

		# Determine the slice of the array to return
		if i == 0:
			# Set the return slice to be 0
			return_slice = 0
		else:
			# For all other cases, we need to return the second slice of the
			# array
			return_slice = 1

		# Put this slice into the storage array
		storage_arr[i] = wav_grad[return_slice]

	# Change the header to reflect what is being saved
	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Wav Grad'

	# Save the wavelength gradient
	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
	 'WavGrad_S1_chanavg_subset.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Wavelength gradient calculated'

#-------------- Radial Component of the Wavelength Gradient ------------

# Check to see if we need to calculate the radial component of the
# wavelength gradient
if 'Rad_Wav_Grad' in diag_list:
	# Iterate over the slices of the Stokes Q and U arrays
	for i in range(num_slice):
		# Extract the current Stokes Q slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoQ_slice = StoQ[max(0,i-1):min(i+2,num_slice)]

		# Extract the current Stokes U slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoU_slice = StoU[max(0,i-1):min(i+2,num_slice)]

		# Calculate all of the first order wavelength derivatives that we need
		dQ_dl, dU_dl = calc_Sto_Wav_1Diff(StoQ_slice, StoU_slice,\
		 wave_sq_space[i])

		# Calculate the radial component of the wavelength gradient for this
		# Stokes Q and U
		rad_wav_grad = calc_Rad_Wav_Grad(StoQ_slice, StoU_slice, dQ_dl, dU_dl)

		# Determine the slice of the array to return
		if i == 0:
			# Set the return slice to be 0
			return_slice = 0
		else:
			# For all other cases, we need to return the second slice of the
			# array
			return_slice = 1

		# Put this slice into the storage array
		storage_arr[i] = rad_wav_grad[return_slice]

	# Change the header to reflect what is being saved
	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Rad Wav Grad'

	# Save the radial component of the wavelength gradient
	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
	 'RadWavGrad_S1_chanavg_subset.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Radial component of the wavelength gradient calculated'

#------------ Tangential Component of the Wavelength Gradient ----------

# Check to see if we need to calculate the tangential component of the
# wavelength gradient
if 'Tang_Wav_Grad' in diag_list:
	# Iterate over the slices of the Stokes Q and U arrays
	for i in range(num_slice):
		# Extract the current Stokes Q slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoQ_slice = StoQ[max(0,i-1):min(i+2,num_slice)]

		# Extract the current Stokes U slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoU_slice = StoU[max(0,i-1):min(i+2,num_slice)]

		# Calculate all of the first order wavelength derivatives that we need
		dQ_dl, dU_dl = calc_Sto_Wav_1Diff(StoQ_slice, StoU_slice,\
		 wave_sq_space[i])

		# Calculate the tangential component of the wavelength gradient for 
		# this Stokes Q and U
		tang_wav_grad = calc_Tang_Wav_Grad(StoQ_slice, StoU_slice, dQ_dl, dU_dl)

		# Determine the slice of the array to return
		if i == 0:
			# Set the return slice to be 0
			return_slice = 0
		else:
			# For all other cases, we need to return the second slice of the
			# array
			return_slice = 1

		# Put this slice into the storage array
		storage_arr[i] = tang_wav_grad[return_slice]

	# Change the header to reflect what is being saved
	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Tang Wav Grad'

	# Save the tangential component of the wavelength gradient
	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
	 'TangWavGrad_S1_chanavg_subset.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Tangential component of the wavelength gradient calculated'

#----------------------- Wavelength Curvature --------------------------

# Check to see if we need to calculate the wavelength curvature
if 'Wav_Curv' in diag_list:
	# Iterate over the slices of the Stokes Q and U arrays
	for i in range(num_slice):
		# Extract the current Stokes Q slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoQ_slice = StoQ[max(0,i-2):min(i+3,num_slice)]

		# Extract the current Stokes U slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoU_slice = StoU[max(0,i-2):min(i+3,num_slice)]

		# Calculate all of the first order wavelength derivatives that we need
		dQ_dl, dU_dl = calc_Sto_Wav_1Diff(StoQ_slice, StoU_slice,\
		 wave_sq_space[i])

		# Calculate all of the second order wavelength derivatives that we need
		d2Q_dl2, d2U_dl2 = calc_Sto_Wav_2Diff(dQ_dl, dU_dl, wave_sq_space[i])

		# Calculate the wavelength curvature for this Stokes Q and U
		wav_curv = calc_Wav_Curv(dQ_dl, dU_dl, d2Q_dl2, d2U_dl2)

		# Determine the slice of the array to return
		if i == 0:
			# Set the return slice to be 0
			return_slice = 0
		elif i == 1:
			# Set the return slice to be 1
			return_slice = 1
		else:
			# For all other cases, we need to return the second slice of the
			# array
			return_slice = 2

		# Put this slice into the storage array
		storage_arr[i] = wav_curv[return_slice]

	# Change the header to reflect what is being saved
	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Wav Curv'

	# Save the wavelength curvature
	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
	 'WavCurv_S1_chanavg_subset.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Wavelength curvature calculated'

#---------------- Maximum Amplitude of Mixed Derivative ----------------

# Check to see if we need to calculate the maximum amplitude of the 
# mixed derivative
if 'Mix_Deriv_Max' in diag_list:
	# Iterate over the slices of the Stokes Q and U arrays
	for i in range(num_slice):
		# Extract the current Stokes Q slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoQ_slice = StoQ[max(0,i-1):min(i+2,num_slice)]

		# Extract the current Stokes U slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoU_slice = StoU[max(0,i-1):min(i+2,num_slice)]

		# Calculate all of the first order spatial derivatives that we need
		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
		 pix_sep = pix_size_deg)

		# Calculate the maximum amplitude of the mixed derivative for this 
		# Stokes Q and U
		mix_deriv_max = calc_Mix_Deriv_Max(dQ_dy, dQ_dx, dU_dy, dU_dx,\
			wave_sq_space[i])

		# Determine the slice of the array to return
		if i == 0:
			# Set the return slice to be 0
			return_slice = 0
		else:
			# For all other cases, we need to return the second slice of the
			# array
			return_slice = 1

		# Put this slice into the storage array
		storage_arr[i] = mix_deriv_max[return_slice]

	# Change the header to reflect what is being saved
	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Mix Deriv Max'

	# Save the maximum amplitude of the mixed derivative
	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
	 'MixDerivMax_S1_chanavg_subset.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Maximum amplitude of the mixed derivative calculated'

#--------- Angle that Maximises the Amplitude of Mixed Derivative ------

# Check to see if we need to calculate the angle that maximises the 
# amplitude of the mixed derivative
if 'Mix_Max_Ang' in diag_list:
	# Iterate over the slices of the Stokes Q and U arrays
	for i in range(num_slice):
		# Extract the current Stokes Q slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoQ_slice = StoQ[max(0,i-1):min(i+2,num_slice)]

		# Extract the current Stokes U slice from the full array, as well as
		# the slices immediately before and after this, if they exist
		StoU_slice = StoU[max(0,i-1):min(i+2,num_slice)]

		# Calculate all of the first order spatial derivatives that we need
		dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ_slice, StoU_slice,\
		 pix_sep = pix_size_deg)

		# Calculate the angle that maximises the amplitude of the mixed 
		# derivative for this Stokes Q and U
		mix_max_ang = calc_Mix_Max_Ang(dQ_dy, dQ_dx, dU_dy, dU_dx,\
			wave_sq_space[i])

		# Determine the slice of the array to return
		if i == 0:
			# Set the return slice to be 0
			return_slice = 0
		else:
			# For all other cases, we need to return the second slice of the
			# array
			return_slice = 1

		# Put this slice into the storage array
		storage_arr[i] = mix_max_ang[return_slice]

	# Change the header to reflect what is being saved
	StoQ_hdr['OBJECT'] = 'GALFACTS_S1 Mix Max Ang'

	# Save the angle that maximises the amplitude of the mixed derivative
	mat2FITS_Image(storage_arr, StoQ_hdr, data_loc +\
	 'MixMaxAng_S1_chanavg_subset.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Angle that maximises the mixed derivative calculated'

# Close FITS files, and delete all large arrays
StoQ_fits.close()
StoU_fits.close()
del StoQ
del StoU
del storage_arr