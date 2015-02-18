#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in an array of simulated synchrotron #
# intensities, and calculates the normalised structure functions of the        #
# synchrotron intensity for various values of gamma. The magnitude and         #
# direction of the anisotropy are calculated through the quadrupole ratio, and #
# plots of the magnitude and direction of anisotropy are produced for one      #
# simulation for various values of gamma.                                      #
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
# using FFT, the function that calculates multipoles of 2D images, and the
# function that calculates the quadrupole ratio of 2D images.
from sf_fft import sf_fft
from cf_fft import cf_fft
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio

# Set a variable to hold the number of bins to use in calculating the 
# quadrupole ratio of the normalised structure function
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

# Open the FITS file that contains the simulated synchrotron intensity maps
# Add 'x' to the end of the file to use the synchrotron maps calculated
# with the line of sight along the x-axis.
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

# Create an array of zeroes, which will hold the normalised structure
# functions calculated for the synchrotron data. This array is 3 dimensional, 
# with the same number of slices as gamma values, with each slice being the 
# same size as the synchrotron intensity maps.
sf_mat = np.zeros(sync_shape)

# Create an array of zeroes, which will hold the quadrupole ratio magnitudes 
# calculated for each structure function. This array has one row for each gamma
# value, and a column for each radius value used in the calculation.
quad_mag_arr = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the argument of the quadrupole 
# ratio calculated for each structure function. This array has one row for each 
# gamma value, and a column for each radius value used in the calculation.
quad_arg_arr = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# the quadrupole for each structure function. This array has one row for each 
# gamma value, and a column for each radius value used in the calculation.
quad_rad_arr = np.zeros((sync_shape[0], num_bins))

# Loop over the third axis of the data cube, to calculate the normalised 
# structure function for each map of synchrotron emission
for i in range(sync_shape[0]):
	# Calculate the 2D structure function for this slice of the synchrotron
	# intensity data cube. Note that no_fluct = True is set, because we are
	# not subtracting the mean from the synchrotron maps before calculating
	# the structure function. We are also calculating the normalised structure
	# function, which only takes values between 0 and 2.
	strfn = sf_fft(sync_data[i], no_fluct = True, normalise = True)

	# Store the normalised structure function in the array of structure function
	# This has been shifted so that the zero radial separation entry is in 
	# the centre of the image.
	sf_mat[i] = np.fft.fftshift(strfn)

	# Calculate the magnitude and argument of the quadrupole ratio for the 
	# normalised structure function
	quad_mag_arr[i], quad_arg_arr[i], quad_rad_arr[i] = calc_quad_ratio(\
		sf_mat[i], num_bins = num_bins)

	# Print a message to show that the multipoles have been calculated
	print 'Multipoles calculated for gamma = {}'.format(gamma_arr[i])

# Now that the quadrupole ratios have been calculated, start plotting the 
# magnitude of the quadrupole ratio obtained for different gamma on the same 
# plot 

# Create a figure to display a plot comparing the magnitude of the quadrupole 
# ratio for all of the synchrotron maps, i.e. for all gamma
fig1 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the magnitude of the quadrupole ratio for each gamma
plt.plot(quad_rad_arr[0], quad_mag_arr[0], 'b-o',\
 label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(quad_rad_arr[1], quad_mag_arr[1], 'r-o',\
 label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(quad_rad_arr[2], quad_mag_arr[2], 'g-o',\
 label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(quad_rad_arr[3], quad_mag_arr[3], 'c-o',\
 label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(quad_rad_arr[4], quad_mag_arr[4], 'm-o',\
 label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(quad_rad_arr[5], quad_mag_arr[5], 'y-o',\
 label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(quad_rad_arr[6], quad_mag_arr[6], 'k-o',\
 label ='Gamma = {}'.format(gamma_arr[6]))

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Magnitude Quad Ratio', fontsize = 20)

# Add a title to the plot
plt.title('Magnitude Quad Ratio z {}'.format(spec_loc), fontsize = 20)

# Shrink the width of the plot axes
box1 = ax1.get_position()
ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])

# Force the legend to appear on the plot
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Quad_Mag_Comp_z.png', format = 'png')

# Print a message to the screen to show that the plot of the quadrupole ratio
# magnitude for all synchrotron maps has been saved
print 'Plot of the quadrupole ratio magnitude'\
+ ' of synchrotron intensity saved'

# Close the figure, now that it has been saved.
plt.close()

# Create a figure to display a plot comparing the argument of the quadrupole 
# ratio for all of the synchrotron maps, i.e. for all gamma
fig2 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the argument of the quadrupole ratio for each gamma
plt.plot(quad_rad_arr[0], quad_arg_arr[0], 'b-o',\
 label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(quad_rad_arr[1], quad_arg_arr[1], 'r-o',\
 label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(quad_rad_arr[2], quad_arg_arr[2], 'g-o',\
 label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(quad_rad_arr[3], quad_arg_arr[3], 'c-o',\
 label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(quad_rad_arr[4], quad_arg_arr[4], 'm-o',\
 label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(quad_rad_arr[5], quad_arg_arr[5], 'y-o',\
 label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(quad_rad_arr[6], quad_arg_arr[6], 'k-o',\
 label ='Gamma = {}'.format(gamma_arr[6]))

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Quad Ratio Argument', fontsize = 20)

# Add a title to the plot
plt.title('Quad Ratio Argument z {}'.format(spec_loc), fontsize = 20)

# Shrink the width of the plot axes
box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])

# Force the legend to appear on the plot
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Quad_Arg_Comp_z.png', format = 'png')

# Print a message to the screen to show that the plot of the argument of the 
# quadrupole ratio for all synchrotron maps has been saved
print 'Plot of the argument of the quadrupole ratio'\
+ ' of synchrotron intensity saved'

# Close the figure, now that it has been saved.
plt.close()