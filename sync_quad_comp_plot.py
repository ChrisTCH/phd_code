#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in an array of simulated synchrotron #
# intensities, and calculates the normalised structure functions of the        #
# synchrotron intensity for a fixed value of gamma. The magnitude and argument #
# of the quadrupole ratio of the normalised structure function are then        #
# calculated. The plots produced are for a fixed value of gamma, to see how    #
# changing the magnetic field and pressure influences the quadrupole ratio.    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 2/2/2015                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the functions that calculate the structure and correlation functions
# using FFT, the function that calculates multipoles of 2D images and the 
# function that calculates the magnitude and argument of the quadrupole ratio
from sf_fft import sf_fft
from cf_fft import cf_fft
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio

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
	spec_loc = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/',\
	 'c512b.1p.025/', 'c512b.1p.05/',\
	 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', 'b.1p2_Aug_Burk/']
elif mag_choose == '1':
	spec_loc = ['c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
	 'c512b1p.025/', 'c512b1p.05/',\
	 'b1p.1_Oct_Burk/', 'c512b1p.7/', 'b1p2_Aug_Burk/']
elif mag_choose == 'range':
	spec_loc = ['c512b3p.01/', 'c512b5p.01/']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an array for the pressure values that are being studied in the cases
# where mag_choose is .1 or 1.
press_arr = np.array([0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2])

# We want to produce one plot for each value of gamma, so loop over the values
# of gamma 
for gam_index in range(len(gamma_arr)):
	# Print a message to show what value of gamma is being used in the 
	# current calculations
	print 'Starting calculation for gamma = {}'.format(gamma_arr[gam_index])

	# Create an array of zeroes, which will hold the values of the magnitude of 
	# the quadrupole ratios calculated for the synchrotron data. This array is 2
	# dimensional, with the same number of rows as simulations, the number of 
	# columns is equal to the number of bins being used to calculate the 
	# quadrupole ratios. One array is for the quadrupole ratios calculated when 
	# the line of sight is along the z axis, and the other is for when the line 
	# of sight is along the x axis.
	quad_mag_z = np.zeros((len(spec_loc), num_bins))
	quad_mag_x = np.zeros((len(spec_loc), num_bins))

	# Create an array of zeroes, which will hold the values of the argument of 
	# the quadrupole ratios calculated for the synchrotron data. This array is 2
	# dimensional, with the same number of rows as simulations, the number of 
	# columns is equal to the number of bins being used to calculate the 
	# quadrupole ratios. One array is for the quadrupole ratios calculated when 
	# the line of sight is along the z axis, and the other is for when the line 
	# of sight is along the x axis.
	quad_arg_z = np.zeros((len(spec_loc), num_bins))
	quad_arg_x = np.zeros((len(spec_loc), num_bins))

	# Create an array of zeroes, which will hold the radius values used to 
	# calculate each quadrupole ratio. This array has the same shape as the 
	# array holding the magnitudes of the quadrupole ratios.
	# One array is for the quadrupole ratios calculated when the line of sight
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

		# Calculate the magnitude, argument and radius values of the quadrupole
		# ratio for this simulation, for lines of sight along the x and z axes
		quad_mag_x[sim_index], quad_arg_x[sim_index], rad_arr_x[sim_index] =\
		 calc_quad_ratio(strfn_x, num_bins)
		quad_mag_z[sim_index], quad_arg_z[sim_index], rad_arr_z[sim_index] =\
		 calc_quad_ratio(strfn_z, num_bins)

		# Print a message to show that the quadrupole ratio has been calculated
		print 'Quad ratio calculated for {}'.format(spec_loc[sim_index])

	# Now that the quadrupole ratios have been calculated, start plotting them
	# all on the same plot 

	# Create a figure to display a plot comparing the magnitude of the 
	# quadrupole ratios for all of the synchrotron maps, for a line of sight 
	# along the x axis.
	fig1 = plt.figure(figsize = (10,6))

	# Create an axis for this figure
	ax1 = fig1.add_subplot(111)

	# Check to see if we are plotting the magnitude of the quadrupole ratio for 
	# a fixed value of the magnetic field, or if we are plotting for a range of 
	# values.
	if mag_choose != 'range':
		# Plot the magnitude of the quadrupole ratio, for the chosen
		# magnetic field value
		plt.plot(rad_arr_x[0], quad_mag_x[0], 'b-o', label ='b'+mag_choose+'p{}'.format(.0049))
		plt.plot(rad_arr_x[1], quad_mag_x[1], 'r-o', label ='b'+mag_choose+'p{}'.format(.0077))
		plt.plot(rad_arr_x[2], quad_mag_x[2], 'g-o', label ='b'+mag_choose+'p{}'.format(.01))
		plt.plot(rad_arr_x[3], quad_mag_x[3], 'c-o', label ='b'+mag_choose+'p{}'.format(.025))
		plt.plot(rad_arr_x[4], quad_mag_x[4], 'm-o', label ='b'+mag_choose+'p{}'.format(.05))
		plt.plot(rad_arr_x[5], quad_mag_x[5], 'y-o', label ='b'+mag_choose+'p{}'.format(.1))
		plt.plot(rad_arr_x[6], quad_mag_x[6], 'k-o', label ='b'+mag_choose+'p{}'.format(.7))
		plt.plot(rad_arr_x[7], quad_mag_x[7], '-o', color = (1,0.41,0), label ='b'+mag_choose+'p{}'.format(2))
	else:
		# Plot all of the multipole ratios that have been calculated.
		plt.plot(rad_arr_x[0], quad_mag_x[0], 'b-o', label ='b{}p{}'.format(3,.01))
		plt.plot(rad_arr_x[1], quad_mag_x[1], 'r-o', label ='b{}p{}'.format(5,.01))

	# Make the x axis of the plot logarithmic
	ax1.set_xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation [pixels]', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Mag Quad Ratio', fontsize = 20)

	# Add a title to the plot
	plt.title('Mag Quad Ratio-x Gamma = {}; B = {}'.\
		format(gamma_arr[gam_index],mag_choose), fontsize = 18)

	# Shrink the width of the plot axes
	box1 = ax1.get_position()
	ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])

	# Force the legend to appear on the plot
	ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Quad_Mag_Comp_x_Gam{}Mag{}.png'.\
		format(gamma_arr[gam_index],mag_choose), format = 'png')

	# Print a message to the screen to show that the plot of all of the 
	# quadrupole ratios has been saved
	print 'Plot of the magnitude of the quadrupole ratio'\
	+ ' for synchrotron intensity x saved, for gamma = {}'.format(gamma_arr[gam_index])

	# Close the figure, to remove it from memory
	plt.close()

	# Create a figure to display a plot comparing the magnitude of the 
	# quadrupole ratios for all of the synchrotron maps, for a line of sight 
	# along the z axis.
	fig2 = plt.figure(figsize = (10,6))

	# Create an axis for this figure
	ax2 = fig2.add_subplot(111)

	# Check to see if we are plotting the magnitude of the quadrupole ratio for 
	# a fixed value of the magnetic field, or if we are plotting for a range of 
	# values.
	if mag_choose != 'range':
		# Plot the magnitude of the quadrupole ratio, for the chosen
		# magnetic field value
		plt.plot(rad_arr_z[0], quad_mag_z[0], 'b-o', label ='b'+mag_choose+'p{}'.format(.0049))
		plt.plot(rad_arr_z[1], quad_mag_z[1], 'r-o', label ='b'+mag_choose+'p{}'.format(.0077))
		plt.plot(rad_arr_z[2], quad_mag_z[2], 'g-o', label ='b'+mag_choose+'p{}'.format(.01))
		plt.plot(rad_arr_z[3], quad_mag_z[3], 'c-o', label ='b'+mag_choose+'p{}'.format(.025))
		plt.plot(rad_arr_z[4], quad_mag_z[4], 'm-o', label ='b'+mag_choose+'p{}'.format(.05))
		plt.plot(rad_arr_z[5], quad_mag_z[5], 'y-o', label ='b'+mag_choose+'p{}'.format(.1))
		plt.plot(rad_arr_z[6], quad_mag_z[6], 'k-o', label ='b'+mag_choose+'p{}'.format(.7))
		plt.plot(rad_arr_z[7], quad_mag_z[7], '-o', color = (1,0.41,0), label ='b'+mag_choose+'p{}'.format(2))
	else:
		# Plot all of the multipole ratios that have been calculated.
		plt.plot(rad_arr_z[0], quad_mag_z[0], 'b-o', label ='b{}p{}'.format(3,.01))
		plt.plot(rad_arr_z[1], quad_mag_z[1], 'r-o', label ='b{}p{}'.format(5,.01))

	# Make the x axis of the plot logarithmic
	ax2.set_xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation [pixels]', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Mag Quad Ratio', fontsize = 20)

	# Add a title to the plot
	plt.title('Mag Quad Ratio-z Gamma = {}; B = {}'.\
		format(gamma_arr[gam_index],mag_choose), fontsize = 18)

	# Shrink the width of the plot axes
	box2 = ax2.get_position()
	ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])

	# Force the legend to appear on the plot
	ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Quad_Mag_Comp_z_Gam{}Mag{}.png'.\
		format(gamma_arr[gam_index],mag_choose), format = 'png')

	# Print a message to the screen to show that the plot of all of the 
	# quadrupole ratios has been saved
	print 'Plot of the magnitude of the quadrupole ratio'\
	+ ' for synchrotron intensity z saved, for gamma = {}\n'.format(gamma_arr[gam_index])

	# Close the figure, to remove it from memory
	plt.close()

	# Create a figure to display a plot comparing the argument of the 
	# quadrupole ratios for all of the synchrotron maps, for a line of sight 
	# along the x axis.
	fig3 = plt.figure(figsize = (10,6))

	# Create an axis for this figure
	ax3 = fig3.add_subplot(111)

	# Check to see if we are plotting the argument of the quadrupole ratio for 
	# a fixed value of the magnetic field, or if we are plotting for a range of 
	# values.
	if mag_choose != 'range':
		# Plot the argument of the quadrupole ratio, for the chosen
		# magnetic field value
		plt.plot(rad_arr_x[0], quad_arg_x[0], 'b-o', label ='b'+mag_choose+'p{}'.format(.0049))
		plt.plot(rad_arr_x[1], quad_arg_x[1], 'r-o', label ='b'+mag_choose+'p{}'.format(.0077))
		plt.plot(rad_arr_x[2], quad_arg_x[2], 'g-o', label ='b'+mag_choose+'p{}'.format(.01))
		plt.plot(rad_arr_x[3], quad_arg_x[3], 'c-o', label ='b'+mag_choose+'p{}'.format(.025))
		plt.plot(rad_arr_x[4], quad_arg_x[4], 'm-o', label ='b'+mag_choose+'p{}'.format(.05))
		plt.plot(rad_arr_x[5], quad_arg_x[5], 'y-o', label ='b'+mag_choose+'p{}'.format(.1))
		plt.plot(rad_arr_x[6], quad_arg_x[6], 'k-o', label ='b'+mag_choose+'p{}'.format(.7))
		plt.plot(rad_arr_x[7], quad_arg_x[7], '-o', color = (1,0.41,0), label ='b'+mag_choose+'p{}'.format(2))
	else:
		# Plot the argument of the quadrupole ratios that have been calculated.
		plt.plot(rad_arr_x[0], quad_arg_x[0], 'b-o', label ='b{}p{}'.format(3,.01))
		plt.plot(rad_arr_x[1], quad_arg_x[1], 'r-o', label ='b{}p{}'.format(5,.01))

	# Make the x axis of the plot logarithmic
	ax3.set_xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation [pixels]', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Argument Quad Ratio', fontsize = 20)

	# Add a title to the plot
	plt.title('Argument Quad Ratio-x Gamma = {}; B = {}'.\
		format(gamma_arr[gam_index],mag_choose), fontsize = 18)

	# Shrink the width of the plot axes
	box3 = ax3.get_position()
	ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])

	# Force the legend to appear on the plot
	ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Quad_Arg_Comp_x_Gam{}Mag{}.png'.\
		format(gamma_arr[gam_index],mag_choose), format = 'png')

	# Print a message to the screen to show that the plot of all of the 
	# quadrupole ratios has been saved
	print 'Plot of the argument of the quadrupole ratio'\
	+ ' for synchrotron intensity x saved, for gamma = {}'.format(gamma_arr[gam_index])

	# Close the figure, to remove it from memory
	plt.close()

	# Create a figure to display a plot comparing the argument of the 
	# quadrupole ratios for all of the synchrotron maps, for a line of sight 
	# along the z axis.
	fig4 = plt.figure(figsize = (10,6))

	# Create an axis for this figure
	ax4 = fig4.add_subplot(111)

	# Check to see if we are plotting the argument of the quadrupole ratio for 
	# a fixed value of the magnetic field, or if we are plotting for a range of 
	# values.
	if mag_choose != 'range':
		# Plot the argument of the quadrupole ratio, for the chosen
		# magnetic field value
		plt.plot(rad_arr_z[0], quad_arg_z[0], 'b-o', label ='b'+mag_choose+'p{}'.format(.0049))
		plt.plot(rad_arr_z[1], quad_arg_z[1], 'r-o', label ='b'+mag_choose+'p{}'.format(.0077))
		plt.plot(rad_arr_z[2], quad_arg_z[2], 'g-o', label ='b'+mag_choose+'p{}'.format(.01))
		plt.plot(rad_arr_z[3], quad_arg_z[3], 'c-o', label ='b'+mag_choose+'p{}'.format(.025))
		plt.plot(rad_arr_z[4], quad_arg_z[4], 'm-o', label ='b'+mag_choose+'p{}'.format(.05))
		plt.plot(rad_arr_z[5], quad_arg_z[5], 'y-o', label ='b'+mag_choose+'p{}'.format(.1))
		plt.plot(rad_arr_z[6], quad_arg_z[6], 'k-o', label ='b'+mag_choose+'p{}'.format(.7))
		plt.plot(rad_arr_z[7], quad_arg_z[7], '-o', color = (1,0.41,0), label ='b'+mag_choose+'p{}'.format(2))
	else:
		# Plot the argument of the quadrupole ratios that have been calculated.
		plt.plot(rad_arr_z[0], quad_arg_z[0], 'b-o', label ='b{}p{}'.format(3,.01))
		plt.plot(rad_arr_z[1], quad_arg_z[1], 'r-o', label ='b{}p{}'.format(5,.01))

	# Make the x axis of the plot logarithmic
	ax4.set_xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation [pixels]', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Arg Quad Ratio', fontsize = 20)

	# Add a title to the plot
	plt.title('Arg Quad Ratio-z Gamma = {}; B = {}'.\
		format(gamma_arr[gam_index],mag_choose), fontsize = 18)

	# Shrink the width of the plot axes
	box4 = ax4.get_position()
	ax4.set_position([box4.x0, box4.y0, box4.width * 0.8, box4.height])

	# Force the legend to appear on the plot
	ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))

	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Quad_Arg_Comp_z_Gam{}Mag{}.png'.\
		format(gamma_arr[gam_index],mag_choose), format = 'png')

	# Print a message to the screen to show that the plot of all of the 
	# quadrupole ratios has been saved
	print 'Plot of the argument of the quadrupole ratio'\
	+ ' for synchrotron intensity z saved, for gamma = {}\n'.format(gamma_arr[gam_index])

	# Close the figure, to remove it from memory
	plt.close()