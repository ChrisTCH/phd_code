#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of the observed Stokes Q   #
# and U images for simulations for any line of sight, and produces FITS files  #
# of polarisation diagnostics. These calculations can be made for the case     #
# where emission comes from within the cube, or when the cube is backlit by    #
# polarised emission.                                                          #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 1/11/2016                                                        #
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
diag_list = ['Curv4MaxDirecDeriv']
#['Inten', 'Angle', 'RM', 'Grad', 'Direc_Amp_Max', 'Direc_Amp_Min',\
#'Direc_Max_Ang', 'Rad_Direc_Amp', 'Tang_Direc_Amp', 'Rad_Direc_Ang',\
#'Tang_Direc_Ang', 'Wav_Grad', 'Rad_Wav_Grad', 'Tang_Wav_Grad', 'Direc_Curv',\
#'Quad_Curv', 'Curv4MaxDirecDeriv', 'Wav_Curv', 'Mix_Deriv_Max', 'Mix_Max_Ang']

# Create a string for the directory that contains the simulated Stokes Q and U
# maps to use
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
# along the x, y or z axis of the data cube when constructing the images or 
# cubes of polarisation diagnostics. This can be 'x', 'y', or 'z'. The mean 
# magnetic field is along the x axis.
line_o_sight = 'z'

# Create a string that selects the folder to store Q and U in based on the LOS
los_loc = line_o_sight + '_los/'

# Create a variable that controls whether the polarisation diagnostic maps/cubes
# are calculated for the case where the simulation cube is backlit by polarised
# emission, or when polarised emission is generated from within the cube.
# This can be 'backlit' or 'internal'.
emis_mech = 'internal'

# Create a variable that holds the number of theta values to use in the 
# calculation of the directional curvature
num_theta = 36

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

# Iterate over the simulations, to produce polarisation diagnostics for each
# simulation
for i in range(len(simul_arr)): 
	# Print a message to show what simulation calculations are being performed
	# for
	print 'Simulations starting for {}'.format(simul_arr[i])

	# Create a string for the full directory path to use in calculations
	data_loc =  simul_loc + simul_arr[i] + los_loc

	# Open the FITS file that contains Stokes Q for the simulation, for this
	# line of sight, and this emission mechanism
	StoQ_fits = fits.open(data_loc + 'StoQ_{}.fits'.format(emis_mech))

	# Extract the data for Stokes Q
	StoQ = StoQ_fits[0].data

	# Extract the header for Stokes Q
	StoQ_hdr = StoQ_fits[0].header 

	# Open the FITS file that contains Stokes U for the simulation, for this
	# line of sight, and this emission mechanism
	StoU_fits = fits.open(data_loc + 'StoU_{}.fits'.format(emis_mech))

	# Extract the data for Stokes U
	StoU = StoU_fits[0].data

	# Print a message to the screen to show that the data has been loaded
	print 'All required data loaded successfully'

	# Now that Stokes Q and U have been loaded, calculate all polarisation
	# diagnostics that can be calculated for either propagation mechanism

	# Calculate all of the first order spatial derivatives that we need
	dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ, StoU, pix_sep = dl)

	# Calculate all of the second order spatial derivatives that we need
	d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2 =\
	 calc_Sto_2Diff(dQ_dy, dQ_dx, dU_dy, dU_dx, pix_sep = dl)

	#------------------------- Polarisation Angle ------------------------------

	# Check to see if we need to calculate the polarisation angle
	if 'Angle' in diag_list:
		# Calculate the observed polarisation angle for this Stokes Q and U
		polar_angle = calc_Polar_Angle(StoQ, StoU)

		# Save the polarisation angle array
		mat2FITS_Image(polar_angle, StoQ_hdr, data_loc + 'PolarAngle_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Polarisation angle calculated'

	#----------------------- Polarisation Gradient -----------------------------

	# Check to see if we need to calculate the polarisation gradient
	if 'Grad' in diag_list:
		# Calculate the observed polarisation gradient for this Stokes Q and U
		polar_grad = calc_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)

		# Save the polarisation gradient array
		mat2FITS_Image(polar_grad, StoQ_hdr, data_loc + 'PolarGrad_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Polarisation gradient calculated'

	#------------ Maximum Amplitude of the Directional Derivative --------------

	# Check to see if we need to calculate the maximum amplitude of the 
	# directional derivative
	if 'Direc_Amp_Max' in diag_list:
		# Calculate the observed maximum amplitude of the directional
		# derivative for this Stokes Q and U
		direc_amp_max = calc_Direc_Amp_Max(dQ_dy, dQ_dx, dU_dy, dU_dx)

		# Save the maximum amplitude of the directional derivative array
		mat2FITS_Image(direc_amp_max, StoQ_hdr, data_loc + 'DirecAmpMax_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Maximum amplitude of the directional derivative calculated'

	#------------ Minimum Amplitude of the Directional Derivative --------------

	# Check to see if we need to calculate the minimum amplitude of the 
	# directional derivative
	if 'Direc_Amp_Min' in diag_list:
		# Calculate the observed minimum amplitude of the directional
		# derivative for this Stokes Q and U
		direc_amp_min = calc_Direc_Amp_Min(dQ_dy, dQ_dx, dU_dy, dU_dx)

		# Save the minimum amplitude of the directional derivative array
		mat2FITS_Image(direc_amp_min, StoQ_hdr, data_loc + 'DirecAmpMin_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Minimum amplitude of the directional derivative calculated'

	#------------ Angle that Maximises the Directional Derivative --------------

	# Check to see if we need to calculate the angle that maximises the 
	# directional derivative
	if 'Direc_Max_Ang' in diag_list:
		# Calculate the observed angle that maximises the directional
		# derivative for this Stokes Q and U
		direc_max_ang = calc_Direc_Max_Ang(dQ_dy, dQ_dx, dU_dy, dU_dx)

		# Save the angle that maximises the directional derivative array
		mat2FITS_Image(direc_max_ang, StoQ_hdr, data_loc + 'DirecMaxAng_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Angle that maximises the directional derivative calculated'

	#------------ Maximum Radial Component Directional Derivative --------------

	# Check to see if we need to calculate the maximum radial component of the 
	# directional derivative
	if 'Rad_Direc_Amp' in diag_list:
		# Calculate the observed maximum radial component of the directional
		# derivative for this Stokes Q and U
		rad_direc_amp = calc_Rad_Direc_Amp(StoQ, StoU, dQ_dy,dQ_dx,dU_dy, dU_dx)

		# Save the maximum radial component of the directional derivative array
		mat2FITS_Image(rad_direc_amp, StoQ_hdr, data_loc + 'RadDirecAmp_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print'Maximum radial component of the directional derivative calculated'

	#---------- Maximum Tangential Component Directional Derivative ------------

	# Check to see if we need to calculate the maximum tangential component of 
	# the directional derivative
	if 'Tang_Direc_Amp' in diag_list:
		# Calculate the observed maximum tangential component of the directional
		# derivative for this Stokes Q and U
		tang_direc_amp = calc_Tang_Direc_Amp(StoQ,StoU,dQ_dy,dQ_dx,dU_dy,dU_dx)

		# Save the maximum tangential component of the directional derivative 
		# array
		mat2FITS_Image(tang_direc_amp, StoQ_hdr, data_loc + 'TangDirecAmp_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Maximum tangential component of the directional derivative '+\
		 'calculated'

	#-------- Angle Maximises Radial Component Directional Derivative ----------

	# Check to see if we need to calculate the angle that maximises the radial
	# component of the directional derivative
	if 'Rad_Direc_Ang' in diag_list:
		# Calculate the observed angle that maximises the radial component of 
		# the directional derivative for this Stokes Q and U
		rad_direc_ang = calc_Rad_Direc_Ang(StoQ,StoU,dQ_dy,dQ_dx,dU_dy,dU_dx)

		# Save the angle that maximises the radial component of the directional 
		# derivative array
		mat2FITS_Image(rad_direc_ang, StoQ_hdr, data_loc + 'RadDirecAng_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Angle that maximises the radial component of the directional '+\
		 'derivative calculated'

	#-------- Angle Maximises Tangential Component Directional Derivative ------

	# Check to see if we need to calculate the angle that maximises the 
	# tangential component of the directional derivative
	if 'Tang_Direc_Ang' in diag_list:
		# Calculate the observed angle that maximises the tangential component 
		# of the directional derivative for this Stokes Q and U
		tang_direc_ang = calc_Tang_Direc_Ang(StoQ,StoU,dQ_dy,dQ_dx,dU_dy,dU_dx)

		# Save the angle that maximises the tangential component of the 
		# directional derivative array
		mat2FITS_Image(tang_direc_ang, StoQ_hdr, data_loc + 'TangDirecAng_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Angle that maximises the tangential component of the ' +\
		'directional derivative calculated'

	#----------------------- Quadrature of Curvature ---------------------------

	# Check to see if we need to calculate the quadrature of the curvature
	if 'Quad_Curv' in diag_list:
		# Calculate the quadrature of the curvature for this Stokes Q and U
		quad_curv = calc_Quad_Curv(dQ_dy, dQ_dx, dU_dy, dU_dx, d2Q_dy2,\
		 d2Q_dx2, d2U_dy2, d2U_dx2)

		# Save the quadrature of the curvature array
		mat2FITS_Image(quad_curv, StoQ_hdr, data_loc + 'QuadCurv_' +\
		 emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Quadrature of the curvature calculated'

	#------- Curvature in Direction that Maximises Directional Derivative ------

	# Check to see if we need to calculate the curvature in the direction that
	# maximises the directional derivative
	if 'Curv4MaxDirecDeriv' in diag_list:
		# Calculate the curvature in the direction that maximises the 
		# directional derivative for this Stokes Q and U
		curv4max_direc_deriv = calc_Curv4MaxDirecDeriv(dQ_dy, dQ_dx,\
			dU_dy, dU_dx, d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2, d2Q_dydx,d2U_dydx)

		# Save the curvature in the direction that maximises the directional
		# derivative array
		mat2FITS_Image(curv4max_direc_deriv, StoQ_hdr, data_loc +\
		 'Curv4MaxDirecDeriv_' + emis_mech + '.fits', clobber = True)

		# Print a message to say the diagnostic has been calculated
		print 'Curvature in the direction that maximises the directional ' +\
		 'derivative calculated'

	#---------------------------------------------------------------------------

	# Now calculate polarisation diagnostics that are specific to the case of
	# backlit emission. We may only be able to calculate these diagnostics for
	# this case, or the calculation may differ between the backlit and internal
	# cases.
	if emis_mech == 'backlit':
		#----------------------- Directional Curvature -------------------------

		# Check to see if we need to calculate the directional curvature
		if 'Direc_Curv' in diag_list:
			# Calculate the directional curvature for this Stokes Q and U, and
			# the number of theta values that were selected
			direc_curv, theta_arr = calc_Direc_Curv(dQ_dy, dQ_dx, dU_dy, dU_dx,\
				d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2, d2Q_dydx,d2U_dydx,num_theta)

			# Create a new header for the directional curvature
			direc_curv_hdr = StoQ_hdr

			# Add header keywords to describe the theta axis of the directional
			# curvature array
			# Specify the reference pixel along the theta axis
			direc_curv_hdr['CRPIX3'] = 1

			# Specify the value of theta at the reference pixel
			direc_curv_hdr['CRVAL3'] = theta_arr[0]

			# Specify the increment in theta along each slice of the array
			direc_curv_hdr['CDELT3'] = theta_arr[1] - theta_arr[0]

			# Specify what the third axis is
			direc_curv_hdr['CTYPE3'] = 'Theta (degrees)'

			# Save the directional curvature array
			mat2FITS_Image(direc_curv, direc_curv_hdr, data_loc +\
			 'DirecCurv_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Directional curvature calculated'

	elif emis_mech == 'internal':
		# In this case we need to produce polarisation diagnostics for the case
		# when the emission is generated within the simulation cube, over a
		# range of frequencies

		# Calculate all of the first order wavelength derivatives that we need
		dQ_dl, dU_dl = calc_Sto_Wav_1Diff(StoQ, StoU, wave_sq_space)

		# Calculate all of the second order wavelength derivatives that we need
		d2Q_dl2, d2U_dl2 = calc_Sto_Wav_2Diff(dQ_dl, dU_dl, wave_sq_space)

		#---------------------- Polarisation Intensity -------------------------

		# Check to see if we need to calculate the polarisation intensity
		if 'Inten' in diag_list:
			# Calculate the polarisation intensity for this Stokes Q and U
			polar_inten = calc_Polar_Inten(StoQ, StoU)

			# Save the polarisation intensity
			mat2FITS_Image(polar_inten, StoQ_hdr, data_loc +\
			 'PolarInten_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Polarisation intensity calculated'

		#------------------------ Rotation Measure -----------------------------

		# Check to see if we need to calculate the rotation measure
		if 'RM' in diag_list:
			# Calculate the rotation measure for this Stokes Q and U
			rot_meas, unwound_angle = calc_Rot_Meas(StoQ, StoU, wave_sq_space)

			# Save the rotation measure
			mat2FITS_Image(rot_meas, StoQ_hdr, data_loc +\
			 'RotMeas_' + emis_mech + '.fits', clobber = True)

			# Save the unwound polarisation angle array
			mat2FITS_Image(unwound_angle, StoQ_hdr, data_loc +\
			 'UnwoundAngle_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Rotation measure calculated'

		#----------------------- Wavelength Gradient ---------------------------

		# Check to see if we need to calculate the wavelength gradient
		if 'Wav_Grad' in diag_list:
			# Calculate the wavelength gradient for this Stokes Q and U
			wav_grad = calc_Wav_Grad(dQ_dl, dU_dl)

			# Save the wavelength gradient
			mat2FITS_Image(wav_grad, StoQ_hdr, data_loc +\
			 'WavGrad_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Wavelength gradient calculated'

		#-------------- Radial Component of the Wavelength Gradient ------------

		# Check to see if we need to calculate the radial component of the
		# wavelength gradient
		if 'Rad_Wav_Grad' in diag_list:
			# Calculate the radial component of the wavelength gradient for this
			# Stokes Q and U
			rad_wav_grad = calc_Rad_Wav_Grad(StoQ, StoU, dQ_dl, dU_dl)

			# Save the radial component of the wavelength gradient
			mat2FITS_Image(rad_wav_grad, StoQ_hdr, data_loc +\
			 'RadWavGrad_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Radial component of the wavelength gradient calculated'

		#------------ Tangential Component of the Wavelength Gradient ----------

		# Check to see if we need to calculate the tangential component of the
		# wavelength gradient
		if 'Tang_Wav_Grad' in diag_list:
			# Calculate the tangential component of the wavelength gradient for 
			# this Stokes Q and U
			tang_wav_grad = calc_Tang_Wav_Grad(StoQ, StoU, dQ_dl, dU_dl)

			# Save the tangential component of the wavelength gradient
			mat2FITS_Image(tang_wav_grad, StoQ_hdr, data_loc +\
			 'TangWavGrad_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Tangential component of the wavelength gradient calculated'

		#----------------------- Wavelength Curvature --------------------------

		# Check to see if we need to calculate the wavelength curvature
		if 'Wav_Curv' in diag_list:
			# Calculate the wavelength curvature for this Stokes Q and U
			wav_curv = calc_Wav_Curv(dQ_dl, dU_dl, d2Q_dl2, d2U_dl2)

			# Save the wavelength curvature
			mat2FITS_Image(wav_curv, StoQ_hdr, data_loc +\
			 'WavCurv_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Wavelength curvature calculated'

		#---------------- Maximum Amplitude of Mixed Derivative ----------------

		# Check to see if we need to calculate the maximum amplitude of the 
		# mixed derivative
		if 'Mix_Deriv_Max' in diag_list:
			# Calculate the maximum amplitude of the mixed derivative for this 
			# Stokes Q and U
			mix_deriv_max = calc_Mix_Deriv_Max(dQ_dy, dQ_dx, dU_dy, dU_dx,\
				wave_sq_space)

			# Save the maximum amplitude of the mixed derivative
			mat2FITS_Image(mix_deriv_max, StoQ_hdr, data_loc +\
			 'MixDerivMax_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Maximum amplitude of the mixed derivative calculated'

		#--------- Angle that Maximises the Amplitude of Mixed Derivative ------

		# Check to see if we need to calculate the angle that maximises the 
		# amplitude of the mixed derivative
		if 'Mix_Max_Ang' in diag_list:
			# Calculate the angle that maximises the amplitude of the mixed 
			# derivative for this Stokes Q and U
			mix_max_ang = calc_Mix_Max_Ang(dQ_dy, dQ_dx, dU_dy, dU_dx,\
				wave_sq_space)

			# Save the angle that maximises the amplitude of the mixed 
			# derivative
			mat2FITS_Image(mix_max_ang, StoQ_hdr, data_loc +\
			 'MixMaxAng_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Angle that maximises the mixed derivative calculated'

		#---------------------- Directional Curvature --------------------------

		# Check to see if we need to calculate the directional curvature
		if 'Direc_Curv' in diag_list:
			# Unpack the tuple representing the size of the provided partial 
			# derivative arrays, so that an array which will hold all of the 
			# directional curvature values can be produced.
			lambda_length, y_length, x_length = np.shape(dQ_dy)

			# Create an empty four-dimensional array, that will hold the
			# values of the directional derivative calculated at each
			# frequency. The first axis is frequency, second is theta, third 
			# is y, and the fourth is x
			direc_curv = np.zeros((lambda_length, num_theta, y_length,x_length)\
				, dtype = np.float32)

			# Iterate over the wavelength array, and calculate the directional
			# curvature for different values of theta, for each wavelength
			for j in range(lambda_length):
				# Calculate the directional curvature for this Stokes Q and U
				direc_curv[j], theta_arr = calc_Direc_Curv(dQ_dy[j], dQ_dx[j],\
				 dU_dy[j], dU_dx[j], d2Q_dy2[j], d2Q_dx2[j], d2U_dy2[j],\
				 d2U_dx2[j], d2Q_dydx[j], d2U_dydx[j], num_theta)

			# Create a new header for the directional curvature
			direc_curv_hdr = StoQ_hdr

			# Delete all of the keywords that are currently present for the 
			# third axis of the array
			del direc_curv_hdr['CRPIX3']
			del direc_curv_hdr['CRVAL3']
			del direc_curv_hdr['CDELT3']
			del direc_curv_hdr['CTYPE3']

			# Add header keywords to describe the theta axis of the directional
			# curvature array
			# Specify the reference pixel along the theta axis
			direc_curv_hdr['CRPIX3'] = 1

			# Specify the value of theta at the reference pixel
			direc_curv_hdr['CRVAL3'] = theta_arr[0]

			# Specify the increment in theta along each slice of the array
			direc_curv_hdr['CDELT3'] = theta_arr[1] - theta_arr[0]

			# Specify what the third axis is
			direc_curv_hdr['CTYPE3'] = 'Theta (degrees)'

			# Add header keywords to describe the wavelength squared axis of the 
			# Stokes Q array
			# Specify the reference pixel along the wavelength squared axis
			direc_curv_hdr['CRPIX4'] = 1

			# Specify the wavelength squared at the reference pixel
			direc_curv_hdr['CRVAL4'] = lambda_sq_arr[0]

			# Specify the increment in wavelength squared along each slice of the 
			# array
			direc_curv_hdr['CDELT4'] = lambda_sq_arr[1] - lambda_sq_arr[0]

			# Specify what the third axis is
			direc_curv_hdr['CTYPE4'] = 'Wavelength Squared (m^2)'

			# Save the tangential component of the wavelength gradient
			mat2FITS_Image(direc_curv, direc_curv_hdr, data_loc +\
			 'DirecCurv_' + emis_mech + '.fits', clobber = True)

			# Print a message to say the diagnostic has been calculated
			print 'Directional curvature calculated'
	
	# Close all of the fits files, to save memory
	StoQ_fits.close()
	StoU_fits.close()

	# Print a message to state that the FITS files were saved successfully
	print 'FITS files of polarisation diagnostics saved successfully {}'.\
	format(simul_arr[i])

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All Stokes Q and U maps calculated successfully'