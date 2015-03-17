#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in an array of simulated synchrotron #
# intensities, and calculates the normalised correlation functions of          #
# synchrotron intensity. This is to test if the normalised correlation         #
# functions are insensitive to the spectral index of cosmic ray electrons,     #
# which is a claim of the Lazarian and Pogosyan 2012 paper.                    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 10/9/2014                                                        #
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
# correlation functions
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
spec_loc = 'c512b1p.025/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc

# Open the FITS file that contains the simulated synchrotron intensity maps
sync_fits = fits.open(data_loc + 'synint_p1-4.fits')

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

# Create an array of zeroes, which will hold the normalised, radially averaged 
# correlation functions calculated for the synchrotron data. This array is 2 
# dimensional, with the same number of rows as gamma values, the number of 
# columns is equal to the number of bins being used to calculate the correlation
# functions.
cf_mat = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# each normalised, radially averaged correlation function. This array has the 
# same shape as the array holding the normalised, radially averaged correlation
# functions
rad_arr = np.zeros((sync_shape[0], num_bins))

# Loop over the third axis of the data cube, to calculate the correlation 
# function for each map of synchrotron emission
for i in range(sync_shape[0]):
	# Calculate the 2D correlation function for this slice of the synchrotron
	# intensity data cube. Note that no_fluct = True is set, because we are
	# not subtracting the mean from the synchrotron maps before calculating
	# the correlation function
	corr = cf_fft(sync_data[i], no_fluct = True)

	# Radially average the calculated 2D correlation function, using the 
	# specified number of bins
	rad_corr = sfr(corr, num_bins, verbose = False)

	# Calculate the square of the mean of the synchrotron intensity values
	sync_sq_mean = np.power( np.mean(sync_data[i], dtype = np.float64), 2.0 )

	# Calculate the mean of the synchrotron intensity values squared
	sync_mean_sq = np.mean( np.power(sync_data[i], 2.0), dtype = np.float64 )

	# Calculate the normalised, radially averaged correlation function for
	# this value of gamma
	norm_rad_corr = (rad_corr[1] - sync_sq_mean) / (sync_mean_sq - sync_sq_mean)

	# Insert the calculated normalised, radially averaged correlation function
	# into the matrix that stores all of the calculated correlation functions
	cf_mat[i] = norm_rad_corr

	# Insert the radius values used to calculate this correlation function
	# into the matrix that stores the radius values
	rad_arr[i] = rad_corr[0]

	# Print a message to show that the correlation function has been calculated
	print 'Normalised, radially averaged correlation function calculated for'\
	+ ' gamma = {}'.format(gamma_arr[i])

# Now that the normalised, radially averaged correlation functions have been
# calculated, start plotting them all on the same plot 

# Create a figure to display a plot comparing the normalised, radially
# averaged correlation functions for all of the synchrotron maps
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot all of the normalised, radially averaged correlation functions 
plt.plot(rad_arr[0], cf_mat[0], 'b-o', label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(rad_arr[1], cf_mat[1], 'r-o', label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(rad_arr[2], cf_mat[2], 'g-o', label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(rad_arr[3], cf_mat[3], 'c-o', label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(rad_arr[4], cf_mat[4], 'm-o', label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(rad_arr[5], cf_mat[5], 'y-o', label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(rad_arr[6], cf_mat[6], 'k-o', label ='Gamma = {}'.format(gamma_arr[6]))

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax1.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Normalised Correlation Function', fontsize = 20)

# Add a title to the plot
plt.title('Sync Intensity Corr Fun', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Sync_Int_Corr_Comp_1.png', format = 'png')

# Print a message to the screen to show that the plot of all of the synchrotron
# correlation functions has been saved
print 'Plot of the normalised, radially averaged correlation functions'\
+ ' for synchrotron intensity saved'

# Create a figure to display a plot showing the maximum difference between the
# correlation functions
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the maximum difference between the correlation functions
# What this code does is find the maximum and minimum correlation function
# values for each radial separation value, calculate the difference between
# them, and then plot this.
# Add /np.max(np.abs(cf_mat), axis = 0) to plot fractional difference
plt.plot(rad_arr[0], (np.max(cf_mat, axis = 0) - np.min(cf_mat, axis = 0))\
	/np.max(np.abs(cf_mat), axis = 0), 'b-o') 

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax2.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Max difference', fontsize = 20)

# Add a title to the plot
plt.title('Maximum fractional difference between Corr Fun', fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Sync_Int_Max_Diff_1.png', format = 'png')

# Close the figures so that they don't stay in memory
plt.close(fig1)
plt.close(fig2)

# Print a message to the screen to show that the plot of the maximum difference
# between the correlation functions has been saved
print 'Plot of the maximum difference between the correlation functions saved'