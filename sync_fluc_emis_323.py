#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of the component of the    #
# magnetic field in the x and y directions, and calculates the normalised      #
# correlation functions of synchrotron emissivity. This is to test if the      #
# normalised correlation functions are insensitive to the spectral index of    #
# cosmic ray electrons, which is a claim of the Lazarian and Pogosyan 2012     #
# paper.                                                                       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 6/10/2014                                                        #
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
spec_loc = 'c512b5p.01/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc

# Open the FITS file that contains the x-component of the simulated magnetic
# field
mag_x_fits = fits.open(data_loc + 'magx.fits')

# Extract the data for the simulated x-component of the magnetic field
mag_x_data = mag_x_fits[0].data

# # Extract the first octant of the x-component of the magnetic field data, to
# # greatly speed up the processing time
# mag_x_data = mag_x_data[0:256, 0:256, 0:256]

# Open the FITS file that contains the y-component of the simulated magnetic 
# field
mag_y_fits = fits.open(data_loc + 'magy.fits')

# Extract the data for the simulated y-component of the magnetic field
mag_y_data = mag_y_fits[0].data

# # Extract the first octant of the y-component of the magnetic field data, to
# # greatly speed up the processing time
# mag_y_data = mag_y_data[0:256, 0:256, 0:256]

# Print a message to the screen to show that the data has been loaded
print 'Magnetic field components loaded successfully'

# Calculate the magnitude of the magnetic field perpendicular to the line of 
# sight, which is just the square root of the sum of the x and y component
# magnitudes squared.
mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

# Print a message to show that the perpendicular component of the magnetic
# field has been calculated
print 'Perpendicular component of the magnetic field calculated'

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an array of zeroes, which will hold the normalised, radially averaged 
# correlation functions calculated for the synchrotron data. This array is 2 
# dimensional, with the same number of rows as gamma values, the number of 
# columns is equal to the number of bins being used to calculate the correlation
# functions.
cf_mat = np.zeros((len(gamma_arr), num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# each normalised, radially averaged correlation function. This array has the 
# same shape as the array holding the normalised, radially averaged correlation
# functions
rad_arr = np.zeros((len(gamma_arr), num_bins))

# Loop over the gamma values, to calculate the correlation 
# function for each cube of synchrotron emission
for i in range(len(gamma_arr)):
	# Calculate the result of raising the magnetic field strength perpendicular 
	# to the line of sight to the power of gamma
	mag_perp_gamma = np.power(mag_perp, gamma_arr[i])

	# Calculate the square of the mean of the perpendicular component of the 
	# magnetic field raised to the power of gamma
	mag_sq_mean_gamma = np.power(np.mean(mag_perp_gamma, dtype=np.float64), 2.0)

	# Calculate the mean of the squared perpendicular component of the magnetic
	# field raised to the power of gamma
	mag_mean_sq_gamma = np.mean( np.power(mag_perp_gamma, 2.0),dtype=np.float64)

	# Calculate the correlation function for the perpendicular component of the
	# magnetic field, when raised to the power of gamma
	perp_gamma_corr = cf_fft(mag_perp_gamma, no_fluct = True)

	# Print a message to show that the correlation function of the perpendicular 
	# component of the magnetic field has been calculated for gamma
	print 'Correlation function of the perpendicular component of the magnetic'\
	+ ' field calculated for gamma = {}'.format(gamma_arr[i])

	# Calculate the radially averaged correlation function for the perpendicular
	# component of the magnetic field, raised to the power of gamma
	perp_gamma_rad_corr = sfr(perp_gamma_corr, num_bins, verbose = False)

	# Calculate the normalised correlation function for the magnetic field
	# perpendicular to the line of sight, for gamma. This is the right hand
	# side of equation 15.
	mag_gamma_norm_corr = (perp_gamma_rad_corr[1] - mag_sq_mean_gamma)\
	 / (mag_mean_sq_gamma - mag_sq_mean_gamma)

	# Insert the calculated normalised, radially averaged correlation function
	# into the matrix that stores all of the calculated correlation functions
	cf_mat[i] = mag_gamma_norm_corr

	# Insert the radius values used to calculate this correlation function
	# into the matrix that stores the radius values
	rad_arr[i] = perp_gamma_rad_corr[0]

	# Print a message to show that the correlation function has been calculated
	print 'Normalised, radially averaged correlation function calculated for'\
	+ ' gamma = {}'.format(gamma_arr[i])

# Now that the normalised, radially averaged correlation functions have been
# calculated, start plotting them all on the same plot 

# Create a figure to display a plot comparing the normalised, radially
# averaged correlation functions for all of the synchrotron cubes
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
plt.title('Sync Emissivity Corr Fun', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Sync_Emis_Corr_Comp_1.png', format = 'png')

# Print a message to the screen to show that the plot of all of the synchrotron
# correlation functions has been saved
print 'Plot of the normalised, radially averaged correlation functions'\
+ ' for synchrotron emissivity saved'

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
plt.savefig(data_loc + 'Sync_Emis_Max_Diff_1.png', format = 'png')

# Close the figures so that they don't stay in memory
plt.close(fig1)
plt.close(fig2)

# Print a message to the screen to show that the plot of the maximum difference
# between the correlation functions has been saved
print 'Plot of the maximum difference between the correlation functions saved'