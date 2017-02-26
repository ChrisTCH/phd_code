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

# Create a string for the directory that contains the simulated magnetic field
# and density cubes to use. 
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

# Create a string for the directory that will be used to save all of the output
save_loc = '/Volumes/CAH_ExtHD/Pol_2016/'

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
simul_arr = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', \
'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/']
# 'c512b.5p.0049/', 'c512b.5p.0077/', 'c512b.5p.01/',\
# 'c512b.5p.025/', 'c512b.5p.05/', 'c512b.5p.1/', 'c512b.5p.7/', 'c512b.5p2/',\

# Create a list, where each entry is the name of the folder that will be used to
# store the results for each simulation
save_sim_arr = ['b.1p.0049/', 'b.1p.0077/', 'b.1p.01/','b.1p.025/', 'b.1p.05/',\
'b.1p.1/', 'b.1p.7/', 'b.1p2/',  'b1p.0049/',\
'b1p.0077/', 'b1p.01/', 'b1p.025/', 'b1p.05/', 'b1p.1/', 'b1p.7/', 'b1p2/']
# 'b.5p.0049/', 'b.5p.0077/', 'b.5p.01/',\
# 'b.5p.025/', 'b.5p.05/', 'b.5p.1/', 'b.5p.7/', 'b.5p2/',

# Create an array, where each entry specifies the pressure of the corresponding
# simulation in the list of simulation directories
press_arr = np.array([0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0])
	# 0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\

# Create an array, where each entry specifies the initial mean magnetic field of
# the corresponding simulation to study 
mag_arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, \
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# 0.5, 0.5, 0.5, 0.5,\
#  0.5, 0.5, 0.5, 0.5,

# Create an array, where each entry specifies the initial sound speed of the
# corresponding simulation to study. This should only be used when the sound
# speed is being calculated from the initial condition of the simulation, so
# that an assumption is made that the sound speed at each pixel does not change
# as the simulation evolves. If the assumption being made is that the 
# simulations are isobaric, so that the pressure at each pixel is the same as 
# the simulations evolve, then the sound speed needs to be calculated at each 
# pixel.
# Calculate the initial sound speed (assumed to be equal to the sound speed
# at any later time) for each simulation, assuming a uniform density of 1
c_s_arr = np.sqrt(press_arr)

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the images.
# This can be 'x', 'y', or 'z'. The mean magnetic field is along the x axis.
line_o_sight = 'z'

# Create a string that selects the folder to store results in based on the LOS
los_loc = line_o_sight + '_los/'

# Depending on the line of sight, the projected quantities will be different
if line_o_sight == 'z':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate projected quantities. Since the line of sight
	# is the z axis, we need to integrate along axis 0. (Numpy convention is 
	# that axes are ordered as (z, y, x))
	int_axis = 0
elif line_o_sight == 'y':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate projected quantities. Since the line of sight
	# is the y axis, we need to integrate along axis 1.
	int_axis = 1
elif line_o_sight == 'x':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate projected quantities. Since the line of sight
	# is the x axis, we need to integrate along axis 2.
	int_axis = 2

# Create a variable that specifies the frequency at which mock observations 
# should be performed, when calculating the synchrotron intensity. 
# This needs to be in Hz.
freq = 1.4 * np.power(10.0, 9.0)

# Create a variable that specifies the size of a pixel in parsecs
dl = 0.15

# Create a variable that specifies the density scaling in cm^-3
n_e = 0.2

# Create a variable that specifies the mass density scaling in kg m^-3
rho_0 = n_e * 1.67 * np.power(10.0, -21.0)

# Create a variable that specifies the permeability of free space
mu_0 = 4.0 * np.pi * np.power(10.0, -7.0)

# Create a variable that specifies the sound speed in gas with a temperature of
# 8000 K, in m s^-1
c_s = 10.15 * np.power(10.0,3.0)

# Create a variable that specifies the spectral index of the synchrotron 
# emission (physically reasonable values are between 0 and -3)
spec_ind = -1.0

# Iterate over the simulations, to produce projected quantities for each
# simulation
for i in range(len(simul_arr)): 
	# Print a message to show what simulation calculations are being performed
	# for
	print 'Calculations starting for {}'.format(simul_arr[i])

	# Create a string for the full directory path to use in calculations
	data_loc =  simul_loc + simul_arr[i]

	# Open the density file that contains the number density for the simulation
	dens_fits = fits.open(data_loc + 'dens.fits')

	# Extract the data for the simulated density
	dens_data = dens_fits[0].data

	# Find any index values where the density array is less than or equal to 
	# zero
	invalid_dens = dens_data <= 0.0

	# Set the invalid values to a very small, positive number
	dens_data[invalid_dens] = 1.0e-4

	# Scale the density to units of cm^-3
	dens_data = n_e * dens_data

	# Open the files that contain the three components of the magnetic field
	magx_fits = fits.open(data_loc + 'magx.fits')
	magy_fits = fits.open(data_loc + 'magy.fits')
	magz_fits = fits.open(data_loc + 'magz.fits')

	# Extract the data for the three components of the magnetic field
	magx_data = magx_fits[0].data
	magy_data = magy_fits[0].data
	magz_data = magz_fits[0].data
	
	# Create a variable that specifies the velocity scaling in m s^-1
	v_0 = c_s / np.sqrt(press_arr[i])

	# Calculate the magnetic field scaling for this simulation in micro Gauss
	B_0 = np.sqrt(mu_0 * rho_0 * np.power(v_0,2.0)) / np.power(10.0,-10.0)

	# Scale the magnetic field components to physical units of micro Gauss
	magx_data = B_0 * magx_data
	magy_data = B_0 * magy_data
	magz_data = B_0 * magz_data

	# Open the files that contain the three components of the velocity field
	velx_fits = fits.open(data_loc + 'velx.fits')
	vely_fits = fits.open(data_loc + 'vely.fits')
	velz_fits = fits.open(data_loc + 'velz.fits')

	# Extract the data for the three components of the velocity field
	velx_data = velx_fits[0].data
	vely_data = vely_fits[0].data
	velz_data = velz_fits[0].data

	# Calculate the velocity of the gas at each point within the simulation
	vel_amp = np.sqrt( np.power(velx_data,2.0) + np.power(vely_data,2.0)\
	+ np.power(velz_data,2.0) )

	# Print a message to the screen to show that the data has been loaded
	print 'All required data loaded successfully'

	# Depending on the line of sight, the strength of the magnetic field 
	# perpendicular to the line of sight is calculated in different ways
	if line_o_sight == 'z':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(magx_data, 2.0) + np.power(magy_data, 2.0) )

		# The remaining component of the magnetic field is the component 
		# parallel to the line of sight
		mag_para = magz_data

		# Calculate the magnitude of the velocity field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared. Units are m s^-1
		vel_perp = v_0 * np.sqrt( np.power(velx_data, 2.0) + np.power(vely_data, 2.0) )

		# The remaining component of the velocity field is the component 
		# parallel to the line of sight. Units are m s^-1
		vel_para = v_0 * velz_data
		
	elif line_o_sight == 'y':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(magx_data, 2.0) + np.power(magz_data, 2.0) )

		# The remaining component of the magnetic field is the component 
		# parallel to the line of sight
		mag_para = magy_data

		# Calculate the magnitude of the velocity field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared. Units are m s^-1
		vel_perp = v_0 * np.sqrt( np.power(velx_data, 2.0) + np.power(velz_data, 2.0) )

		# The remaining component of the velocity field is the component 
		# parallel to the line of sight. Units are m s^-1
		vel_para = v_0 * vely_data
		
	elif line_o_sight == 'x':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the y and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(magy_data, 2.0) + np.power(magz_data, 2.0) )

		# The remaining component of the magnetic field is the component 
		# parallel to the line of sight
		mag_para = magx_data

		# Calculate the magnitude of the velocity field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared. Units are m s^-1
		vel_perp = v_0 * np.sqrt( np.power(vely_data, 2.0) + np.power(velz_data, 2.0) )

		# The remaining component of the velocity field is the component 
		# parallel to the line of sight. Units are m s^-1
		vel_para = v_0 * velx_data

	#----------------------- Synchrotron Intensity -----------------------------

	# Calculate the synchrotron emissivity throughout the cube
	sync_emis = np.power(mag_perp, 1.0 - spec_ind) * np.power(freq,spec_ind)

    # Integrate the synchrotron emissivity along the line of sight, to calculate
    # the synchrotron intensity.
	sync_inten = np.trapz(sync_emis, dx = dl, axis = int_axis)

	# Now create a FITS header for synchrotron intensity, so that it can be
	# saved as FITS file

	# Create a primary HDU to contain the synchrotron intensity data
	pri_hdu_I = fits.PrimaryHDU(sync_inten)

	# Add header keywords to describe the line of sight, frequency,
	# spectral index, pixel size, density, velocity scale, and magnetic
	# field scale for synchrotron intensity
	pri_hdu_I.header['SIM'] = (save_sim_arr[i], 'simulation used')
	pri_hdu_I.header['LOS'] = (int_axis, '0-z, 1-y, 2-x')
	pri_hdu_I.header['FREQ'] = (freq, 'observing frequency (Hz)')
	pri_hdu_I.header['SPEC-IND'] = (spec_ind, 'spectral index')
	pri_hdu_I.header['PIX-SIZE'] = (dl, 'size of each pixel (pc)')
	pri_hdu_I.header['DENSITY'] = (n_e, 'density scaling (cm^-3)')
	pri_hdu_I.header['VELSCALE'] = (v_0, 'velocity scaling (m s^-1)')
	pri_hdu_I.header['MAGSCALE'] = (B_0, 'magnetic field scaling (mu G)')

	# # Save the produced synchrotron intensity image as a FITS file
	# mat2FITS_Image(sync_inten, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'SyncInten.fits', clobber = True)

	# #---------------------------- Faraday Depth --------------------------------

	# # Calculate the Faraday depth for the cube, along the line of sight
	# Fara_depth = 0.81 * np.trapz(dens_data * mag_para, dx = dl, axis = int_axis)

	# # Save the produced Faraday depth image as a FITS file, using the
	# # same header as was created for synchrotron intensity
	# mat2FITS_Image(Fara_depth, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'FaradayDepth.fits', clobber = True)

	# #------------------- Standard Deviation Faraday Depth ----------------------

	# # Calculate the standard deviation of the Faraday depth elements along
	# # the line of sight for the cube
	# St_Dev_Fara_depth = np.std(dens_data * mag_para * dl, axis = int_axis,\
	#  dtype = np.float64)

	# # Save the produced standard deviation of the Faraday depth image as a FITS 
	# # file, using the same header as was created for synchrotron intensity
	# mat2FITS_Image(St_Dev_Fara_depth, pri_hdu_I.header, save_loc + \
	# 	save_sim_arr[i] + los_loc + 'StDevFaradayDepth.fits', clobber = True)

	# #---------------------- Projected B Field Amplitude ------------------------

	# # Calculate the magnetic field amplitude at each pixel
	# B_amp = np.sqrt(np.power(mag_para,2.0) + np.power(mag_perp,2.0))

	# # Average the amplitude of the magnetic field along the line of sight
	# B_amp_proj = np.mean(B_amp, axis = int_axis, dtype = np.float64)

	# # Save the projected magnetic field image as a FITS file, using the
	# # same header as was created for synchrotron intensity
	# mat2FITS_Image(B_amp_proj, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'B_amp_projected.fits', clobber = True)

	# #-------------------------- Projected Density ------------------------------

	# # Integrate the density along the line of sight
	# dens_proj = np.trapz(dens_data, dx = dl, axis = int_axis)

	# # Save the projected density image as a FITS file, using the
	# # same header as was created for synchrotron intensity
	# mat2FITS_Image(dens_proj, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'dens_projected.fits', clobber = True)

	# #------------------------- Projected B Parallel ----------------------------

	# # Integrate the magnetic field parallel to the line of sight along the
	# # line of sight
	# B_para_proj = np.mean(mag_para, axis = int_axis, dtype = np.float64)

	# # Save the projected parallel magnetic field image as a FITS file, using the
	# # same header as was created for synchrotron intensity
	# mat2FITS_Image(B_para_proj, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'B_para_projected.fits', clobber = True)

	# #--------------------- Projected B Perpendicular ---------------------------

	# # Integrate the magnetic field perpendicular to the line of sight along the
	# # line of sight
	# B_perp_proj = np.mean(mag_perp, axis = int_axis, dtype = np.float64)

	# # Save the projected perpendicular magnetic field image as a FITS file, 
	# # using the same header as was created for synchrotron intensity
	# mat2FITS_Image(B_perp_proj, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'B_perp_projected.fits', clobber = True)

	# #-------------------- Projected Alfvenic Mach Number -----------------------

	# # Calculate the Alfven speed at each point within the simulation
	# v_alf = B_amp / np.sqrt(dens_data)

	# # Calculate the Alfvenic Mach number for the simulation
	# alf_mach = vel_amp / v_alf

	# # Calculate the average Alfvenic mach number along the line of sight
	# los_alf = np.mean(alf_mach, axis = int_axis, dtype = np.float64)

	# # Save the projected Alfvenic Mach number image as a FITS file, 
	# # using the same header as was created for synchrotron intensity
	# mat2FITS_Image(los_alf, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'Mean_Alf_Mach.fits', clobber = True)

	# #---------------------- Projected Sonic Mach Number ------------------------

	# # Calculate the sonic Mach number for the simulation (when c_s calculated 
	# # once for each simulation, assuming c_s is not time dependent, and is
	# # uniform)
	# sonic_mach = vel_amp / c_s_arr[i]

	# # Calculate the average sonic Mach number along the line of sight
	# los_sonic = np.mean(sonic_mach, axis = int_axis, dtype = np.float64)

	# # Save the projected Alfvenic Mach number image as a FITS file, 
	# # using the same header as was created for synchrotron intensity
	# mat2FITS_Image(los_sonic, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	#  los_loc + 'Mean_Sonic_Mach.fits', clobber = True)

	#---------------------- Projected V Field Amplitude ------------------------

	# Calculate the velocity field amplitude at each pixel
	V_amp = np.sqrt(np.power(vel_para,2.0) + np.power(vel_perp,2.0))

	# Average the amplitude of the velocity field along the line of sight
	V_amp_proj = np.mean(V_amp, axis = int_axis, dtype = np.float64)

	# Save the projected velocity field image as a FITS file, using the
	# same header as was created for synchrotron intensity
	mat2FITS_Image(V_amp_proj, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	 los_loc + 'V_amp_projected.fits', clobber = True)

	#------------------------- Projected V Parallel ----------------------------

	# Integrate the velocity field parallel to the line of sight along the
	# line of sight
	V_para_proj = np.mean(vel_para, axis = int_axis, dtype = np.float64)

	# Save the projected parallel velocity field image as a FITS file, using the
	# same header as was created for synchrotron intensity
	mat2FITS_Image(V_para_proj, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	 los_loc + 'V_para_projected.fits', clobber = True)

	#--------------------- Projected V Perpendicular ---------------------------

	# Integrate the velocity field perpendicular to the line of sight along the
	# line of sight
	V_perp_proj = np.mean(vel_perp, axis = int_axis, dtype = np.float64)

	# Save the projected perpendicular velocity field image as a FITS file, 
	# using the same header as was created for synchrotron intensity
	mat2FITS_Image(V_perp_proj, pri_hdu_I.header, save_loc + save_sim_arr[i] + \
	 los_loc + 'V_perp_projected.fits', clobber = True)

	#---------------------------------------------------------------------------

	# Close all of the fits files, to save memory
	dens_fits.close()
	magx_fits.close()
	magy_fits.close()
	magz_fits.close()
	velx_fits.close()
	vely_fits.close()
	velz_fits.close()

	# Print a message to state that the FITS files were saved successfully
	print 'FITS files of projected quantities saved successfully {}'.\
	format(save_sim_arr[i])

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All projected diagnostic maps calculated successfully'