#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the monopole and quadrupole of the normalised    #
# structure functions of the synchrotron intensity for various angles between  #
# the line of sight and the mean magnetic field. For a each value of gamma, a  #
# plot is produced of the quadrupole / monopole ratio for each line of sight.  #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 6/11/2014                                                        #
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
spec_loc = 'fractal_data/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an array that specifies the rotation angles relative to the z axis of
# the MHD cubes, of the synchrotron maps to be used
rot_ang_arr = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,\
	80.0, 90.0]) 

# We want to produce one plot for each value of gamma, so loop over the values
# of gamma 
for gam_index in range(len(gamma_arr)):
	# Print a message to show what value of gamma is being used in the 
	# current calculations
	print 'Starting calculation for gamma = {}'.format(gamma_arr[gam_index])

	# Create an array of zeroes, which will hold the values of the multipole
	# ratios calculated for the synchrotron data. This array is 2 dimensional, 
	# with the same number of rows as simulations, the number of columns is equal
	# to the number of bins being used to calculate the multipole ratios.
	quad_mat = np.zeros((len(rot_ang_arr), num_bins))

	# Create an array of zeroes, which will hold the radius values used to calculate
	# each multipole ratio. This array has the same shape as the array holding
	# the values of the multipole ratios.
	rad_arr = np.zeros((len(rot_ang_arr), num_bins))

	# We want to produce one line for each rotation angle, so loop over the values
	# of the rotation angle
	for rot_index in range(len(rot_ang_arr)):
		# Open the FITS file that contains the simulated synchrotron intensity maps
		sync_fits = fits.open(data_loc + 'synint_p1-4_{}_frac.fits'.format(rot_ang_arr[rot_index]))

		# Extract the data for the simulated synchrotron intensities
		# This is a 3D data cube, where the slices along the third axis are the
		# synchrotron intensities observed for different values of gamma, the power law 
		# index of the cosmic ray electrons emitting the synchrotron emission.
		sync_data = sync_fits[0].data

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised structure
		# function, which only takes values between 0 and 2.
		strfn = sf_fft(sync_data[gam_index], no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image.
		strfn = np.fft.fftshift(strfn)

		# Calculate the monopole for the normalised structure function
		monopole_arr, mono_rad_arr = calc_multipole_2D(strfn, order = 0,\
		 num_bins = num_bins)

		# Calculate the quadrupole for the normalised structure function
		quadpole_arr, quad_rad_arr = calc_multipole_2D(strfn, order = 2,\
		 num_bins = num_bins)

		# Insert the calculated multipole ratios into the matrix that stores all
		# of the calculated multipole
		quad_mat[rot_index] = quadpole_arr / monopole_arr

		# Insert the radius values used to calculate this multipole ratio
		# into the matrix that stores the radius values
		rad_arr[rot_index] = mono_rad_arr

		# Print a message to show that the multipole ratio has been calculated
		print 'Multipoles calculated for {}'.format(rot_ang_arr[rot_index])

	# Now that the multipole ratios have been calculated, start plotting them
	# all on the same plot 

	# Create a figure to display a plot comparing the multipole ratios for all
	# of the synchrotron maps
	fig1 = plt.figure()

	# Create an axis for this figure
	ax1 = fig1.add_subplot(111)

	# Plot all of the multipole ratios that have been calculated 
	plt.plot(rad_arr[0], quad_mat[0], 'b-o', label ='Angle = {}'.format(rot_ang_arr[0]))
	plt.plot(rad_arr[1], quad_mat[1], 'b--o', label ='Angle = {}'.format(rot_ang_arr[1]))
	plt.plot(rad_arr[2], quad_mat[2], 'r-o', label ='Angle = {}'.format(rot_ang_arr[2]))
	plt.plot(rad_arr[3], quad_mat[3], 'r--o', label ='Angle = {}'.format(rot_ang_arr[3]))
	plt.plot(rad_arr[4], quad_mat[4], 'g-o', label ='Angle = {}'.format(rot_ang_arr[4]))
	plt.plot(rad_arr[5], quad_mat[5], 'g--o', label ='Angle = {}'.format(rot_ang_arr[5]))
	plt.plot(rad_arr[6], quad_mat[6], 'c-o', label ='Angle = {}'.format(rot_ang_arr[6]))
	plt.plot(rad_arr[7], quad_mat[7], 'c--o', label ='Angle = {}'.format(rot_ang_arr[7]))
	plt.plot(rad_arr[8], quad_mat[8], 'm-o', label ='Angle = {}'.format(rot_ang_arr[8]))
	plt.plot(rad_arr[9], quad_mat[9], 'm--o', label ='Angle = {}'.format(rot_ang_arr[9]))

	# Make the x axis of the plot logarithmic
	ax1.set_xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation R', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Quad / Monopole', fontsize = 20)

	# Add a title to the plot
	plt.title('Quad / Monopole Frac Gamma {}'.format(gamma_arr[gam_index]), fontsize = 20)

	# Force the legend to appear on the plot
	plt.legend(loc = 4)

	# Save the figure using the given filename and format
	plt.savefig(data_loc + 'Quad_Mono_AngleComp_Gam{}_frac.png'.\
		format(gamma_arr[gam_index]), format = 'png')

	# Print a message to the screen to show that the plot of all of the 
	# multipole ratios has been saved
	print 'Plot of the multipole ratios'\
	+ ' for synchrotron intensity saved, for gamma = {}'.format(gamma_arr[gam_index])

	# Close the figures so that they don't stay in memory
	plt.close(fig1)