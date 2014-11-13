#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in an array of simulated synchrotron #
# intensities, and calculates the structure functions of the synchrotron       #
# intensity for various values of gamma. This is done to measure the spectral  #
# index of the structure on small scales.                                      #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 7/10/2014                                                        #
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

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
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
spec_loc = 'b1p2_Aug_Burk/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc

# Open the FITS file that contains the simulated synchrotron intensity maps
# Add 'x' to the end of the file to use the synchrotron maps calculated
# with the line of sight along the x-axis.
sync_fits = fits.open(data_loc + 'synint_p1-4z_slow.fits')

# Extract the data for the simulated synchrotron intensities
# This is a 3D data cube, where the slices along the third axis are the
# synchrotron intensities observed for different values of gamma, the power law 
# index of the cosmic ray electrons emitting the synchrotron emission.
sync_data = sync_fits[0].data

# Print a message to the screen to show that the synchrotron data has been 
# loaded successfully
print 'Simulated synchrotron data loaded'

# Calculate the shape of the synchrotron data cube
sync_shape = np.shape(sync_data)

# Print the shape of the synchrotron data matrix, as a check
print 'The shape of the synchrotron data matrix is: {}'.\
format(sync_shape)

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an array of zeroes, which will hold the radially averaged structure
# functions calculated for the synchrotron data. This array is 2 dimensional, 
# with the same number of rows as gamma values, the number of columns is equal
# to the number of bins being used to calculate the correlation functions.
sf_mat = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# each structure function. This array has the same shape as the array holding
# the radially averaged structure functions
rad_arr = np.zeros((sync_shape[0], num_bins))

# Loop over the third axis of the data cube, to calculate the structure 
# function for each map of synchrotron emission
for i in range(sync_shape[0]):
	# Calculate the 2D structure function for this slice of the synchrotron
	# intensity data cube. Note that no_fluct = True is set, because we are
	# not subtracting the mean from the synchrotron maps before calculating
	# the structure function
	strfn = sf_fft(sync_data[i], no_fluct = True)

	# Radially average the calculated 2D structure function, using the 
	# specified number of bins
	rad_sf = sfr(strfn, num_bins)

	# # Calculate the square of the mean of the synchrotron intensity values
	# sync_sq_mean = np.power( np.mean(sync_data[i], dtype = np.float64), 2.0 )

	# # Calculate the mean of the synchrotron intensity values squared
	# sync_mean_sq = np.mean( np.power(sync_data[i], 2.0), dtype = np.float64 )

	# # Calculate the normalised, radially averaged correlation function for
	# # this value of gamma
	# norm_rad_corr = (rad_corr[1] - sync_sq_mean) / (sync_mean_sq - sync_sq_mean)

	# Insert the calculated radially averaged structure function
	# into the matrix that stores all of the calculated structure functions
	sf_mat[i] = rad_sf[1]

	# Insert the radius values used to calculate this structure function
	# into the matrix that stores the radius values
	rad_arr[i] = rad_sf[0]

	# Print a message to show that the structure function has been calculated
	print 'Radially averaged structure function calculated for'\
	+ ' gamma = {}'.format(gamma_arr[i])

# Loop over the third axis of the data cube, to calculate the spectral index
# for each structure function of synchrotron emission
for i in range(sync_shape[0]):
	# Calculate the spectral indices of the structure functions calculated for
	# each value of gamma. Note that only the first third of the structure
	# function is used in the calculation, as this is the part that is 
	# close to a straight line.
	spec_ind_data = np.polyfit(np.log10(rad_arr[i,0:np.ceil(num_bins/3.0)]),\
		np.log10(sf_mat[i,0:np.ceil(num_bins/3.0)]), 1, full = True)

	# Extract the returned coefficients from the polynomial fit
	coeff = spec_ind_data[0]

	# Extract the sum of the residuals from the polynomial fit
	residuals = spec_ind_data[1]

	# Print out the results from the linear fit, namely the gradient and the
	# sum of the residuals
	print 'Gamma = {}: Gradient = {}: m = {}: Residuals = {}'\
	.format(gamma_arr[i], coeff[0], coeff[0]-1.0, residuals)

# Now that the radially averaged structure functions have been
# calculated, start plotting them all on the same plot 

# Create a figure to display a plot comparing the radially
# averaged structure functions for all of the synchrotron maps
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot all of the radially averaged structure functions 
plt.plot(rad_arr[0], sf_mat[0], 'b-o', label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(rad_arr[1], sf_mat[1], 'r-o', label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(rad_arr[2], sf_mat[2], 'g-o', label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(rad_arr[3], sf_mat[3], 'c-o', label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(rad_arr[4], sf_mat[4], 'm-o', label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(rad_arr[5], sf_mat[5], 'y-o', label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(rad_arr[6], sf_mat[6], 'k-o', label ='Gamma = {}'.format(gamma_arr[6]))

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
ax1.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Structure Function', fontsize = 20)

# Add a title to the plot
plt.title('Sync Int Str Fun z-LOS Slow', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 4)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Sync_Intz_SF_Comp_slow.png', format = 'png')

# Print a message to the screen to show that the plot of all of the synchrotron
# structure functions has been saved
print 'Plot of the radially averaged structure functions'\
+ ' for synchrotron intensity saved'

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
# plt.title('Maximum fractional difference between Str Fun', fontsize = 20)

# # Save the figure using the given filename and format
# plt.savefig(data_loc + 'Sync_Intz_SF_MaxDiff_x_Alf.png', format = 'png')

# Close the figures so that they don't stay in memory
plt.close(fig1)
# plt.close(fig2)

# # Print a message to the screen to show that the plot of the maximum difference
# # between the structure functions has been saved
# print 'Plot of the maximum difference between the structure functions saved'