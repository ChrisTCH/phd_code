#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in an array of simulated synchrotron #
# intensities, and calculates the normalised structure functions of the        #
# synchrotron intensity for a fixed value of gamma. The multipoles of the      #
# normalised structure function are then calculated, to try and determine      #
# which MHD modes are contributing to the observed synchrotron emission. The   #
# plots produced are for a fixed value of gamma, to see how changing the       #
# magnetic field and pressure influence the multipoles.                        #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 27/10/2014                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the functions that calculate the structure and correlation functions
# using FFT, and the function that calculates multipoles of 2D images
from sf_fft import sf_fft
from cf_fft import cf_fft
from calc_multipole_2D import calc_multipole_2D

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 25

# Create a string variable that will control the magnetic field values of the
# simulations being used to produce the plots. Options are:
# '.1'
# '1'
# 'range'
mag_choose = 'range'

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create an array of strings for the specific simulated data sets to use in 
# calculations.
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
# c512b1p.0049
# c512b1p.0077
# c512b1p.025
# c512b1p.05
# c512b1p.7
# c512b3p.01
# c512b5p.01
# c512b5p2

# Select which simulations to use in the calculation, based on the mag_choose
# variable
if mag_choose == '.1':
	spec_loc = ['c512b.1p.0049/', 'c512b.1p.0077', 'b.1p.01_Oct_Burk/',\
	 'c512b.1p.025', 'c512b.1p.05/',\
	 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', 'b.1p2_Aug_Burk/']
elif mag_choose == '1':
	spec_loc = ['c512b1p.0049/', 'c512b1p.0077', 'b1p.01_Oct_Burk/',\
	 'c512b1p.025', 'c512b1p.05/',\
	 'b1p.1_Oct_Burk/', 'c512b1p.7/', 'b1p2_Aug_Burk/']
elif mag_choose == 'range':
	spec_loc = ['c512b3p.01/', 'c512b5p.01/']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an array for the pressure values that are being studied in the cases
# where mag_choose is .1 or 1.
press_arr = np.array([0.0049, 0.01, 0.05, 0.1, 0.7, 2])

# We want to produce one plot for each value of gamma, so loop over the values
# of gamma 
for gam_index in range(len(gamma_arr)):
	# Print a message to show what value of gamma is being used in the 
	# current calculations
	print 'Starting calculation for gamma = {}'.format(gamma_arr[gam_index])

	# Create an array of zeroes, which will hold the values of the quadrupole
	# ratios calculated for the synchrotron data. This array is 2 dimensional, 
	# with the same number of rows as simulations, the number of columns is equal
	# to the number of bins being used to calculate the multipole ratios.
	# One array is for the multipole ratios calculated when the line of sight
	# is along the z axis, and the other is for when the line of sight is along
	# the x axis.
	quad_mat_z = np.zeros((len(spec_loc), num_bins))
	quad_mat_x = np.zeros((len(spec_loc), num_bins))

	# Create an array of zeroes, which will hold the radius values used to calculate
	# each multipole ratio. This array has the same shape as the array holding
	# the values of the multipole ratios.
	# One array is for the multipole ratios calculated when the line of sight
	# is along the z axis, and the other is for when the line of sight is along
	# the x axis.
	rad_arr_z = np.zeros((len(spec_loc), num_bins))
	rad_arr_x = np.zeros((len(spec_loc), num_bins))

	# We need to loop over the simulations that are being used in the plot,
	# so that we can calculate the structure function for each simulation
	for sim_index in range(len(spec_loc)):
		# Create a string for the full directory path to use in calculations
		data_loc =  simul_loc + spec_loc[sim_index]

		# Open the FITS file that contains the simulated synchrotron intensity maps
		# Add 'x' to the end of the file to use the synchrotron maps calculated
		# with the line of sight along the x-axis.
		sync_fits_x = fits.open(data_loc + 'synint_p1-4x.fits')
		sync_fits_z = fits.open(data_loc + 'synint_p1-4.fits')

		# Extract the data for the simulated synchrotron intensities
		# This is a 3D data cube, where the slices along the third axis are the
		# synchrotron intensities observed for different values of gamma, the power law 
		# index of the cosmic ray electrons emitting the synchrotron emission.
		sync_data_x = sync_fits_x[0].data
		sync_data_z = sync_fits_z[0].data

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised structure
		# function, which only takes values between 0 and 2.
		strfn_x = sf_fft(sync_data_x[gam_index], no_fluct = True, normalise = True)
		strfn_z = sf_fft(sync_data_z[gam_index], no_fluct = True, normalise = True)

		# Shift the 2D structure functions so that the zero radial separation
		# entry is in the centre of the image.
		strfn_x = np.fft.fftshift(strfn_x)
		strfn_z = np.fft.fftshift(strfn_z)

		# Calculate the monopole for the normalised structure function, for the
		# cases where the line of sight is along the x or z axis.
		monopole_arr_x, mono_rad_arr_x = calc_multipole_2D(strfn_x, order = 0,\
		 num_bins = num_bins)
		monopole_arr_z, mono_rad_arr_z = calc_multipole_2D(strfn_z, order = 0,\
		 num_bins = num_bins)

		# Calculate the quadrupole for the normalised structure function. for 
		# for the cases where the line of sight is along the x or z axis
		quadpole_arr_x, quad_rad_arr_x = calc_multipole_2D(strfn_x, order = 2,\
		 num_bins = num_bins)
		quadpole_arr_z, quad_rad_arr_z = calc_multipole_2D(strfn_z, order = 2,\
		 num_bins = num_bins)

		# Insert the calculated multipole ratios into the matrix that stores all
		# of the calculated multipole
		quad_mat_x[sim_index] = quadpole_arr_x / monopole_arr_x
		quad_mat_z[sim_index] = quadpole_arr_z / monopole_arr_z

		# Insert the radius values used to calculate this multipole ratio
		# into the matrix that stores the radius values
		rad_arr_x[sim_index] = mono_rad_arr_x
		rad_arr_z[sim_index] = mono_rad_arr_z

		# Print a message to show that the multipole ratio has been calculated
		print 'Multipoles calculated for {}'.format(spec_loc[sim_index])

	# Now that the multipole ratios have been calculated, start plotting them
	# all on the same plot 

	# Create a figure to display a plot comparing the multipole ratios for all
	# of the synchrotron maps, for a line of sight along the x axis.
	fig1 = plt.figure()

	# Create an axis for this figure
	ax1 = fig1.add_subplot(111)

	# Check to see if we are plotting multipole ratios for a fixed value
	# of the magnetic field, or if we are plotting for a range of values.
	if mag_choose != 'range':
		# Plot all of the multipole ratios, for the chosen
		# magnetic field value
		plt.plot(rad_arr_x[0], quad_mat_x[0], 'b-o', label ='b'+mag_choose+'p{}'.format(.0049))
		plt.plot(rad_arr_x[1], quad_mat_x[1], 'r-o', label ='b'+mag_choose+'p{}'.format(.01))
		plt.plot(rad_arr_x[2], quad_mat_x[2], 'g-o', label ='b'+mag_choose+'p{}'.format(.05))
		plt.plot(rad_arr_x[3], quad_mat_x[3], 'c-o', label ='b'+mag_choose+'p{}'.format(.1))
		plt.plot(rad_arr_x[4], quad_mat_x[4], 'm-o', label ='b'+mag_choose+'p{}'.format(.7))
		plt.plot(rad_arr_x[5], quad_mat_x[5], 'y-o', label ='b'+mag_choose+'p{}'.format(2))
	else:
		# Plot all of the multipole ratios that have been calculated.
		plt.plot(rad_arr_x[0], quad_mat_x[0], 'b-o', label ='b{}p{}'.format(3,.01))
		plt.plot(rad_arr_x[1], quad_mat_x[1], 'r-o', label ='b{}p{}'.format(5,.01))
		plt.plot(rad_arr_x[2], quad_mat_x[2], 'g-o', label ='b{}p{}'.format(5,2))

	# Make the x axis of the plot logarithmic
	ax1.set_xscale('log')

	# # Make the y axis of the plot logarithmic
	# ax1.set_yscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation R', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Quad/Mono', fontsize = 20)

	# Add a title to the plot
	plt.title('Quad/Monopole-x Gamma = {}; B = {}'.\
		format(gamma_arr[gam_index],mag_choose), fontsize = 18)

	# Force the legend to appear on the plot
	plt.legend(loc = 4)

	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Quad_Mono_Comp_x_Gam{}Mag{}.png'.\
		format(gamma_arr[gam_index],mag_choose), format = 'png')

	# Print a message to the screen to show that the plot of all of the 
	# multipole ratios has been saved
	print 'Plot of the multipole ratios'\
	+ ' for synchrotron intensity x saved, for gamma = {}'.format(gamma_arr[gam_index])

	# Create a figure to display a plot comparing the multipole ratios for all
	# of the synchrotron maps, for a line of sight along the z axis.
	fig2 = plt.figure()

	# Create an axis for this figure
	ax2 = fig2.add_subplot(111)

	# Check to see if we are plotting multipole ratios for a fixed value
	# of the magnetic field, or if we are plotting for a range of values.
	if mag_choose != 'range':
		# Plot all of the multipole ratios, for the chosen
		# magnetic field value
		plt.plot(rad_arr_z[0], quad_mat_z[0], 'b-o', label ='b'+mag_choose+'p{}'.format(.0049))
		plt.plot(rad_arr_z[1], quad_mat_z[1], 'r-o', label ='b'+mag_choose+'p{}'.format(.01))
		plt.plot(rad_arr_z[2], quad_mat_z[2], 'g-o', label ='b'+mag_choose+'p{}'.format(.05))
		plt.plot(rad_arr_z[3], quad_mat_z[3], 'c-o', label ='b'+mag_choose+'p{}'.format(.1))
		plt.plot(rad_arr_z[4], quad_mat_z[4], 'm-o', label ='b'+mag_choose+'p{}'.format(.7))
		plt.plot(rad_arr_z[5], quad_mat_z[5], 'y-o', label ='b'+mag_choose+'p{}'.format(2))
	else:
		# Plot all of the multipole ratios that have been calculated.
		plt.plot(rad_arr_z[0], quad_mat_z[0], 'b-o', label ='b{}p{}'.format(3,.01))
		plt.plot(rad_arr_z[1], quad_mat_z[1], 'r-o', label ='b{}p{}'.format(5,.01))
		plt.plot(rad_arr_z[2], quad_mat_z[2], 'g-o', label ='b{}p{}'.format(5,2))

	# Make the x axis of the plot logarithmic
	ax2.set_xscale('log')

	# # Make the y axis of the plot logarithmic
	# ax2.set_yscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation R', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Quad/Mono', fontsize = 20)

	# Add a title to the plot
	plt.title('Quad/Monopole-z Gamma = {}; B = {}'.\
		format(gamma_arr[gam_index],mag_choose), fontsize = 18)

	# Force the legend to appear on the plot
	plt.legend(loc = 4)

	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Quad_Mono_Comp_z_Gam{}Mag{}.png'.\
		format(gamma_arr[gam_index],mag_choose), format = 'png')

	# Print a message to the screen to show that the plot of all of the 
	# multipole ratios has been saved
	print 'Plot of the multipole ratios'\
	+ ' for synchrotron intensity z saved, for gamma = {}\n'.format(gamma_arr[gam_index])

	# Close the figures so that they don't stay in memory
	plt.close(fig1)
	plt.close(fig2)