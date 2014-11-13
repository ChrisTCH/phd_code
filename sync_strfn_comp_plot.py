#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the structure functions of synchrotron           #
# intensity. This is to compare the structure functions of synchrotron         #
# intensity for different pressures and magnetic field values, for a fixed     #
# value of gamma.                                                              #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 10/10/2014                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the functions that calculate the structure and correlation functions
# using FFT, as well as the function that calculates the radially averaged 
# structure or correlation functions.
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 25

# Create a string variable that will control the magnetic field values of the
# simulations being used to produce the plots. Options are:
# '.1'
# '1'
# 'range'
mag_choose = '.1'

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
# c512b.1p.05
# c512b.1p.7
# c512b1p.0049
# c512b1p.05
# c512b1p.7
# c512b3p.01
# c512b5p.01
# c512b5p2

# Select which simulations to use in the calculation, based on the mag_choose
# variable
if mag_choose == '.1':
	spec_loc = ['c512b.1p.0049/', 'b.1p.01_Oct_Burk/', 'c512b.1p.05/',\
	 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', 'b.1p2_Aug_Burk/']
elif mag_choose == '1':
	spec_loc = ['c512b1p.0049/', 'b1p.01_Oct_Burk/', 'c512b1p.05/',\
	 'b1p.1_Oct_Burk/', 'c512b1p.7/', 'b1p2_Aug_Burk/']
elif mag_choose == 'range':
	spec_loc = ['c512b3p.01/', 'c512b5p.01/', 'c512b5p2/']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an array for the pressure values that are being studied in the cases
# where mag_choose is .1 or 1.
press_arr = np.array([0.0049, 0.01, 0.05, 0.1, 0.7, 2])

# Create an array that will hold all of the spectral index values calculated
# for the structure functions, for the various pressures being studied (for
# a line of sight along the z axis)
m_arr_z = np.zeros((len(gamma_arr), len(spec_loc)))

# Create an array that will hold all of the spectral index values calculated
# for the structure functions, for the various pressures being studied (for
# a line of sight along the x axis)
m_arr_x = np.zeros((len(gamma_arr), len(spec_loc)))

# We want to produce one plot for each value of gamma, so loop over the values
# of gamma 
for gam_index in range(len(gamma_arr)):
	# Print a message to show what value of gamma is being used in the 
	# current calculations
	print 'Starting calculation for gamma = {}'.format(gamma_arr[gam_index])

	# Create an array of zeroes, which will hold the radially averaged structure
	# functions calculated for the synchrotron data. This array is 2 dimensional, 
	# with the same number of rows as simulations, the number of columns is equal
	# to the number of bins being used to calculate the structure functions.
	# One array is for the structure functions calculated when the line of sight
	# is along the z axis, and the other is for when the line of sight is along
	# the x axis.
	sf_mat_z = np.zeros((len(spec_loc), num_bins))
	sf_mat_x = np.zeros((len(spec_loc), num_bins))

	# Create an array of zeroes, which will hold the radius values used to calculate
	# each structure function. This array has the same shape as the array holding
	# the radially averaged structure functions
	# One array is for the structure functions calculated when the line of sight
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
		# the structure function
		strfn_x = sf_fft(sync_data_x[gam_index], no_fluct = True)
		strfn_z = sf_fft(sync_data_z[gam_index], no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins
		rad_sf_x = sfr(strfn_x, num_bins, verbose = False)
		rad_sf_z = sfr(strfn_z, num_bins, verbose = False)

		# Insert the calculated radially averaged structure function
		# into the matrix that stores all of the calculated structure functions
		sf_mat_x[sim_index] = rad_sf_x[1]
		sf_mat_z[sim_index] = rad_sf_z[1]

		# Insert the radius values used to calculate this structure function
		# into the matrix that stores the radius values
		rad_arr_x[sim_index] = rad_sf_x[0]
		rad_arr_z[sim_index] = rad_sf_z[0]

		# Print a message to show that the structure function has been calculated
		print 'Radially averaged structure function calculated for'\
		+ ' {}'.format(spec_loc[sim_index])

	# Loop over the structure functions, to calculate the spectral index
	# for each structure function of synchrotron emission
	for i in range(len(spec_loc)):
		# Calculate the spectral indices of the structure functions calculated for
		# each simulation. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line.
		spec_ind_data_x = np.polyfit(np.log10(rad_arr_x[i,0:np.ceil(num_bins/3.0)]),\
			np.log10(sf_mat_x[i,0:np.ceil(num_bins/3.0)]), 1, full = True)
		spec_ind_data_z = np.polyfit(np.log10(rad_arr_z[i,0:np.ceil(num_bins/3.0)]),\
			np.log10(sf_mat_z[i,0:np.ceil(num_bins/3.0)]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit
		coeff_x = spec_ind_data_x[0]
		coeff_z = spec_ind_data_z[0]

		# Extract the sum of the residuals from the polynomial fit
		residuals_x = spec_ind_data_x[1]
		residuals_z = spec_ind_data_z[1]

		# Enter the calculated value of m, which is the gradient - 1, into the
		# array of m values
		m_arr_x[gam_index, i] = coeff_x[0] - 1.0
		m_arr_z[gam_index, i] = coeff_z[0] - 1.0

		# Print out the results from the linear fit, namely the gradient and the
		# sum of the residuals
		print 'x LOS: Gamma = {}: Gradient = {}: m = {}: Residuals = {}: Simul = {}'\
		.format(gamma_arr[gam_index], coeff_x[0], coeff_x[0]-1.0, residuals_x, spec_loc[i])

	# Now that the radially averaged structure functions have been
	# calculated, start plotting them all on the same plot 

	# # Create a figure to display a plot comparing the radially
	# # averaged structure functions for all of the synchrotron maps
	# fig1 = plt.figure()

	# # Create an axis for this figure
	# ax1 = fig1.add_subplot(111)

	# # Check to see if we are plotting structure functions for a fixed value
	# # of the magnetic field, or if we are plotting for a range of values.
	# if mag_choose != 'range':
	# 	# Plot all of the radially averaged structure functions, for the chosen
	# 	# magnetic field value
	# 	plt.plot(rad_arr[0], sf_mat[0], 'b-o', label ='b'+mag_choose+'p{}'.format(.0049))
	# 	plt.plot(rad_arr[1], sf_mat[1], 'r-o', label ='b'+mag_choose+'p{}'.format(.01))
	# 	plt.plot(rad_arr[2], sf_mat[2], 'g-o', label ='b'+mag_choose+'p{}'.format(.05))
	# 	plt.plot(rad_arr[3], sf_mat[3], 'c-o', label ='b'+mag_choose+'p{}'.format(.1))
	# 	plt.plot(rad_arr[4], sf_mat[4], 'm-o', label ='b'+mag_choose+'p{}'.format(.7))
	# 	plt.plot(rad_arr[5], sf_mat[5], 'y-o', label ='b'+mag_choose+'p{}'.format(2))
	# else:
	# 	# Plot all of the radially averaged structure functions that have 
	# 	# been calculated.
	# 	plt.plot(rad_arr[0], sf_mat[0], 'b-o', label ='b{}p{}'.format(3,.01))
	# 	plt.plot(rad_arr[1], sf_mat[1], 'r-o', label ='b{}p{}'.format(5,.01))
	# 	plt.plot(rad_arr[2], sf_mat[2], 'g-o', label ='b{}p{}'.format(5,2))

	# # Make the x axis of the plot logarithmic
	# ax1.set_xscale('log')

	# # Make the y axis of the plot logarithmic
	# ax1.set_yscale('log')

	# # Add a label to the x-axis
	# plt.xlabel('Radial Separation R', fontsize = 20)

	# # Add a label to the y-axis
	# plt.ylabel('Structure Function', fontsize = 20)

	# # Add a title to the plot
	# plt.title('Sync Inten-x Str Fun Gamma = {}; B = {}'.\
	# 	format(gamma_arr[gam_index],mag_choose), fontsize = 18)

	# # Force the legend to appear on the plot
	# plt.legend(loc = 4)

	# # Save the figure using the given filename and format
	# plt.savefig(simul_loc + 'Sync_Intx_SF_Gam{}Mag{}.png'.\
	# 	format(gamma_arr[gam_index],mag_choose), format = 'png')

	# # Print a message to the screen to show that the plot of all of the synchrotron
	# # structure functions has been saved
	# print 'Plot of the radially averaged structure functions'\
	# + ' for synchrotron intensity saved, for gamma = {}'.format(gamma_arr[gam_index])

	# # Create a figure to display a plot showing the maximum difference between the
	# # structure functions
	# fig2 = plt.figure()

	# # Create an axis for this figure
	# ax2 = fig2.add_subplot(111)

	# # Plot the maximum difference between the structure functions
	# # What this code does is find the maximum and minimum structure function
	# # values for each radial separation value, calculate the difference between
	# # them, and then plot this.
	# # Add /np.max(np.abs(cf_mat), axis = 0) to plot fractional difference
	# plt.plot(rad_arr[0], (np.max(sf_mat, axis = 0) - np.min(sf_mat, axis = 0))\
	# 	/np.max(np.abs(sf_mat), axis = 0), 'b-o') 

	# # Make the x axis of the plot logarithmic
	# ax2.set_xscale('log')

	# # Make the y axis of the plot logarithmic
	# #ax2.set_yscale('log')

	# # Add a label to the x-axis
	# plt.xlabel('Radial Separation R', fontsize = 20)

	# # Add a label to the y-axis
	# plt.ylabel('Max difference', fontsize = 20)

	# # Add a title to the plot
	# plt.title('Max fractional diff Str Fun Gamma = {}; B = {}'.\
	# 	format(gamma_arr[gam_index],mag_choose), fontsize = 20)

	# # Save the figure using the given filename and format
	# plt.savefig(simul_loc + 'Sync_Intx_SF_MaxDiff_Gam{}Mag{}.png'.\
	# 	format(gamma_arr[gam_index],mag_choose), format = 'png')

	# # Close the figures so that they don't stay in memory
	# plt.close(fig1)
	# plt.close(fig2)

	# # Print a message to the screen to show that the plot of the maximum difference
	# # between the structure functions has been saved
	# print 'Plot of the maximum difference between the structure functions saved\n'

# When the code reaches this point, plot of the structure function have been
# produced for every value of gamma. Next, a plot of the slope of the structure
# function on small scales is made against the pressure of the simulation, if 
# we are calculating structure functions for a fixed value of magnetic field
if mag_choose != 'range':
	# In this case we wish to produce a plot of m vs p, with error bars
	# denoting the range of m values for that particular value of p

	# Calculate the mean value of m at each pressure, to act as the y-axis
	# co-ordinate.
	mean_m_x = np.mean(m_arr_x, axis = 0, dtype = np.float64)
	mean_m_z = np.mean(m_arr_z, axis = 0, dtype = np.float64)

	# Calculate the upper error bar, as the difference between the maximum
	# value of m and the mean
	upper_error_x = np.max(m_arr_x, axis = 0) - mean_m_x
	upper_error_z = np.max(m_arr_z, axis = 0) - mean_m_z

	# Calculate the lower error bar, as the difference between the minimum
	# value of m and the mean
	lower_error_x = mean_m_x - np.min(m_arr_x, axis = 0)
	lower_error_z = mean_m_z - np.min(m_arr_z, axis = 0)

	# Join the lower and upper error bars into a single error array, with the 
	# lower error bars in the first row, and the upper error bars in the
	# second row.
	error_arr_x = np.vstack((lower_error_x, upper_error_x))
	error_arr_z = np.vstack((lower_error_z, upper_error_z))

	# Create a figure to display a plot showing m vs pressure
	fig3 = plt.figure()

	# Create an axis for this figure
	ax3 = fig3.add_subplot(111)

	# Plot the values of m as a function of pressure
	plt.errorbar(press_arr, mean_m_x, error_arr_x, fmt = 'b-o', label = 'x LOS') 
	plt.errorbar(press_arr, mean_m_z, error_arr_z, fmt = 'r-o', label = 'z LOS')

	# Make the x axis of the plot logarithmic
	ax3.set_xscale('log')

	# Make the y axis of the plot logarithmic
	#ax3.set_yscale('log')

	# Add a label to the x-axis
	plt.xlabel('Pressure', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('m (SF gradient - 1)', fontsize = 20)

	# Add a title to the plot
	plt.title('SF Gradient vs Pressure B = {}'.format(mag_choose),\
	 fontsize = 20)

	# Add a legend to the plot
	plt.legend(loc = 4)

	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Sync_Intxz_SF_GradvsP_Mag{}.png'.\
		format(mag_choose), format = 'png')

	# Close the figure, now that it has been saved, to save memory
	plt.close(fig3)

	# Print a message to the screen to show that the plot of the gradient of 
	# the SF vs pressure has been saved
	print 'Plot of the structure function gradient vs pressure saved'