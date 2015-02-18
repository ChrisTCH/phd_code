#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in an array of simulated synchrotron #
# intensities, and calculates the normalised structure functions of the        #
# synchrotron intensity for various values of gamma. The multipoles of the     #
# normalised structure function are then calculated, to try and determine      #
# which MHD modes are contributing to the observed synchrotron emission.       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 17/10/2014                                                       #
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
# multipoles of the normalised structure functions
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
spec_loc = 'b.1p.01_Oct_Burk/'

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

# Create an array of zeroes, which will hold the monopole values calculated for
# each structure function. This array has one row for each gamma value, and a
# column for each radius value used in the calculation.
monopole_arr = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the quadrupole values calculated 
# for each structure function. This array has one row for each gamma value, and
# a column for each radius value used in the calculation.
quadpole_arr = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the octopole values calculated for
# each structure function. This array has one row for each gamma value, and a
# column for each radius value used in the calculation.
octopole_arr = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# the monopole for each structure function. This array has one row for each 
# gamma value, and a column for each radius value used in the calculation.
mono_rad_arr = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# the quadrupole for each structure function. This array has one row for each 
# gamma value, and a column for each radius value used in the calculation.
quad_rad_arr = np.zeros((sync_shape[0], num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# the octopole for each structure function. This array has one row for each 
# gamma value, and a column for each radius value used in the calculation.
octo_rad_arr = np.zeros((sync_shape[0], num_bins))

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

	# Calculate the monopole for the normalised structure function
	monopole_arr[i], mono_rad_arr[i] = calc_multipole_2D(sf_mat[i], order = 0,\
	 num_bins = num_bins)

	# Calculate the quadrupole for the normalised structure function
	quadpole_arr[i], quad_rad_arr[i] = calc_multipole_2D(sf_mat[i], order = 2,\
	 num_bins = num_bins)

	# Calculate the octopole for the normalised structure function
	octopole_arr[i], octo_rad_arr[i] = calc_multipole_2D(sf_mat[i], order = 4,\
	 num_bins = num_bins)

	# Print a message to show that the multipoles have been calculated
	print 'Multipoles calculated for gamma = {}'.format(gamma_arr[i])

# Now that the multipoles have been calculated, start plotting the multipoles 
# obtained for different gamma on the same plot 

# # Create a figure to display a plot comparing the quadrupole divided by the
# # monopole for all of the synchrotron maps, i.e. for all gamma
# fig1 = plt.figure()

# # Create an axis for this figure
# ax1 = fig1.add_subplot(111)

# # Plot the quadrupole divided by the monopole for each gamma
# plt.plot(mono_rad_arr[0], quadpole_arr[0] / monopole_arr[0], 'b-o',\
#  label ='Gamma = {}'.format(gamma_arr[0]))
# plt.plot(mono_rad_arr[1], quadpole_arr[1] / monopole_arr[1], 'r-o',\
#  label ='Gamma = {}'.format(gamma_arr[1]))
# plt.plot(mono_rad_arr[2], quadpole_arr[2] / monopole_arr[2], 'g-o',\
#  label ='Gamma = {}'.format(gamma_arr[2]))
# plt.plot(mono_rad_arr[3], quadpole_arr[3] / monopole_arr[3], 'c-o',\
#  label ='Gamma = {}'.format(gamma_arr[3]))
# plt.plot(mono_rad_arr[4], quadpole_arr[4] / monopole_arr[4], 'm-o',\
#  label ='Gamma = {}'.format(gamma_arr[4]))
# plt.plot(mono_rad_arr[5], quadpole_arr[5] / monopole_arr[5], 'y-o',\
#  label ='Gamma = {}'.format(gamma_arr[5]))
# plt.plot(mono_rad_arr[6], quadpole_arr[6] / monopole_arr[6], 'k-o',\
#  label ='Gamma = {}'.format(gamma_arr[6]))

# # Make the x axis of the plot logarithmic
# ax1.set_xscale('log')

# # Make the y axis of the plot logarithmic
# #ax1.set_yscale('log')

# # Add a label to the x-axis
# plt.xlabel('Radial Separation R', fontsize = 20)

# # Add a label to the y-axis
# plt.ylabel('Quad / Monopole', fontsize = 20)

# # Add a title to the plot
# plt.title('Quad / Monopole z {}'.format(spec_loc), fontsize = 20)

# # Force the legend to appear on the plot
# plt.legend(loc = 4)

# # Save the figure using the given filename and format
# plt.savefig(data_loc + 'Quad_Mono_Comp_z.png', format = 'png')

# # Print a message to the screen to show that the plot of the quadrupole divided
# # by the monopole for all synchrotron maps has been saved
# print 'Plot of the quadrupole divided by the monopole'\
# + ' for synchrotron intensity saved'

# # Close the figure, now that it has been saved.
# plt.close()

# # Create a figure to display a plot comparing the octopole divided by the
# # monopole for all of the synchrotron maps, i.e. for all gamma
# fig2 = plt.figure()

# # Create an axis for this figure
# ax2 = fig2.add_subplot(111)

# # Plot the octopole divided by the monopole for each gamma
# plt.plot(mono_rad_arr[0], octopole_arr[0] / monopole_arr[0], 'b-o',\
#  label ='Gamma = {}'.format(gamma_arr[0]))
# plt.plot(mono_rad_arr[1], octopole_arr[1] / monopole_arr[1], 'r-o',\
#  label ='Gamma = {}'.format(gamma_arr[1]))
# plt.plot(mono_rad_arr[2], octopole_arr[2] / monopole_arr[2], 'g-o',\
#  label ='Gamma = {}'.format(gamma_arr[2]))
# plt.plot(mono_rad_arr[3], octopole_arr[3] / monopole_arr[3], 'c-o',\
#  label ='Gamma = {}'.format(gamma_arr[3]))
# plt.plot(mono_rad_arr[4], octopole_arr[4] / monopole_arr[4], 'm-o',\
#  label ='Gamma = {}'.format(gamma_arr[4]))
# plt.plot(mono_rad_arr[5], octopole_arr[5] / monopole_arr[5], 'y-o',\
#  label ='Gamma = {}'.format(gamma_arr[5]))
# plt.plot(mono_rad_arr[6], octopole_arr[6] / monopole_arr[6], 'k-o',\
#  label ='Gamma = {}'.format(gamma_arr[6]))

# # Make the x axis of the plot logarithmic
# ax2.set_xscale('log')

# # Make the y axis of the plot logarithmic
# #ax2.set_yscale('log')

# # Set the axis limits on the y axis
# plt.ylim(-0.2, 0.2)

# # Add a label to the x-axis
# plt.xlabel('Radial Separation R', fontsize = 20)

# # Add a label to the y-axis
# plt.ylabel('Octo / Monopole', fontsize = 20)

# # Add a title to the plot
# plt.title('Octo / Monopole z {}'.format(spec_loc), fontsize = 20)

# # Force the legend to appear on the plot
# plt.legend(loc = 4)

# # Save the figure using the given filename and format
# plt.savefig(data_loc + 'Octo_Mono_Comp_z.png', format = 'png')

# # Print a message to the screen to show that the plot of the octopole divided
# # by the monopole for all synchrotron maps has been saved
# print 'Plot of the octopole divided by the monopole'\
# + ' for synchrotron intensity saved'

# # Close the figure, now that it has been saved.
# plt.close()