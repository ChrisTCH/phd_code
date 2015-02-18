#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the normalised correlation function of       #
# synchrotron emissivity for different simulations, and different values of    #
# gamma. Four subplots are present, split into low and high magnetic field,    #
# and low and high pressure.                                                   #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 9/2/2015                                                         #
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

# Create a string for the specific simulated data sets to use in calculations.
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
spec_locs = ['b.1p.01_Oct_Burk/', 'b.1p2_Aug_Burk/', 'b1p.01_Oct_Burk/',\
'b1p2_Aug_Burk/']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,2.0,3.0,4.0])

# Create a three dimensional array that will hold all of the information
# for the normalised correlation functions. The first index gives the simulation
# the second gives the gamma value used, and the third axis goes along radius.
norm_corr_arr = np.zeros((len(spec_locs), len(gamma_arr), num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the normalised correlation functions. The first axis represents
# the simulation used, the second represents the particular value of gamma, and 
# the third axis goes over radius.
norm_rad_arr = np.zeros((len(spec_locs), len(gamma_arr), num_bins))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]
	 
	# Open the FITS files that contain the x-component of the simulated magnetic
	# field
	mag_x_fits = fits.open(data_loc + 'magx.fits')

	# Extract the data for the simulated x-component of the magnetic field
	mag_x_data = mag_x_fits[0].data

	# Open the FITS file that contains the y-component of the simulated magnetic 
	# field
	mag_y_fits = fits.open(data_loc + 'magy.fits')

	# Extract the data for the simulated y-component of the magnetic field
	mag_y_data = mag_y_fits[0].data

	# Print a message to the screen to show that the data has been loaded
	print 'Magnetic field components loaded successfully'

	# Calculate the magnitude of the magnetic field perpendicular to the line of 
	# sight, which is just the square root of the sum of the x and y component
	# magnitudes squared.
	mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

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
	for j in range(len(gamma_arr)):
		# Calculate the result of raising the magnetic field strength perpendicular 
		# to the line of sight to the power of gamma
		mag_perp_gamma = np.power(mag_perp, gamma_arr[j])

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
		+ ' field calculated for gamma = {}'.format(gamma_arr[j])

		# Calculate the radially averaged correlation function for the perpendicular
		# component of the magnetic field, raised to the power of gamma
		perp_gamma_rad_corr = sfr(perp_gamma_corr, num_bins)

		# Calculate the normalised correlation function for the magnetic field
		# perpendicular to the line of sight, for gamma. This is the right hand
		# side of equation 15.
		mag_gamma_norm_corr = (perp_gamma_rad_corr[1] - mag_sq_mean_gamma)\
		 / (mag_mean_sq_gamma - mag_sq_mean_gamma)

		# Insert the calculated normalised, radially averaged correlation function
		# into the matrix that stores all of the calculated correlation functions
		cf_mat[j] = mag_gamma_norm_corr

		# Insert the radius values used to calculate this correlation function
		# into the matrix that stores the radius values
		rad_arr[j] = perp_gamma_rad_corr[0]

		# Print a message to show that the correlation function has been calculated
		print 'Normalised, radially averaged correlation function calculated for'\
		+ ' gamma = {}'.format(gamma_arr[j])

	# Add the matrix of normalised correlation function values to the final array
	norm_corr_arr[i] = cf_mat

	# Add the matrix of radius values to the final array being used
	norm_rad_arr[i] = rad_arr

	# Print a message to show that the calculation has finished successfully
	# for this simulation
	print 'All normalised correlation functions calculated for {}'.format(spec_locs[i])

# When the code reaches this point, the normalised correlation functions have
# been saved for every simulation, and every value of gamma, so start making 
# the final plot.
	

# ----------------- Plots of normalised correlation functions ------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will be the low 
# magnetic field simulation, and the bottom row will be high magnetic field 
# simulations. The left column will be low pressure simulations, and the right
# column will be high pressure simulations.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the low 
# magnetic field, low pressure simulation
ax1 = fig.add_subplot(221)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the normalised correlation function for this simulation, for this gamma
	plt.plot(norm_rad_arr[0,i], norm_corr_arr[0,i], '-o')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(norm_rad_arr[0,0])), \
	np.zeros(np.shape(norm_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the low
# magnetic field, high pressure simulation. Make the y axis limits the same as
# for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the normalised correlation function for this simulation, for this gamma
	plt.plot(norm_rad_arr[1,i], norm_corr_arr[1,i], '-o')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(norm_rad_arr[1,0])), \
	np.zeros(np.shape(norm_rad_arr[1,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the high
# magnetic field, low pressure simulation. Make the x axis limits the same as
# for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the normalised correlation function for this simulation, for this gamma
	plt.plot(norm_rad_arr[2,i], norm_corr_arr[2,i], '-o')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(norm_rad_arr[2,0])), \
	np.zeros(np.shape(norm_rad_arr[2,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Create an axis for the fourth subplot to be produced, which is for the high
# magnetic field, high pressure simulation. Make the x axis limits the same as
# for the second plot, and the y axis limits the same as for the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the normalised correlation function for this simulation, for this gamma
	plt.plot(norm_rad_arr[3,i], norm_corr_arr[3,i], '-o', label = 'Gamma={}'.format(gamma_arr[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(norm_rad_arr[3,0])), \
	np.zeros(np.shape(norm_rad_arr[3,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Normalized Correlation Function', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Force the legend to appear on the plot
plt.legend(fontsize = 10)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) Sim 3: b.1p.01', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Sim 8: b.1p2', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Sim 11: b1p.01', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Sim 16: b1p2', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig3.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()