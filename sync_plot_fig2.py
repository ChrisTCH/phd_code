#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates equation 19 of Lazarian and Pogosyan 2012,   #
# to check if this equation is correct. Two plots are produced, comparing the  #
# left and right hand sides of equation 19 for two simulations; one with a low #
# magnetic field, and the other with a high magnetic field. These plots are    #
# designed to be publication quality.                                          #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 23/1/2015                                                        #
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
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

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

# Create a three dimensional array that will hold all of the information
# for the calculated left hand sides of Eq 19. The first axis goes over the 
# different simulations, and then along each row we have the values of the
# left hand side of Eq 19 for different radial separations.
LHS_19_arr = np.zeros((len(spec_locs), num_bins))

# Create a three dimensional array that will hold all of the information
# for the calculated right hand sides of Eq 19. The first axis goes over the 
# different simulations, and then along each row we have the values of the
# right hand side of Eq 19 for different radial separations.
RHS_19_arr = np.zeros((len(spec_locs), num_bins))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]
	
	# Print a message saying that calculations are starting for the current
	# simulation
	print 'Starting calculations for the {} simulation'.format(spec_locs[i])

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

	# Calculate the average of the x-component of the magnetic field squared
	mag_x_mean_sq = np.mean( np.power(mag_x_data, 2.0), dtype = np.float64 )

	# Calculate the average of the y-component of the magnetic field squared
	mag_y_mean_sq = np.mean( np.power(mag_y_data, 2.0), dtype = np.float64 )

	# Print a message to show that the perpendicular component of the magnetic
	# field has been calculated
	print 'Perpendicular component of the magnetic field calculated'

	# -------------- Normalised correlation x-comp B field ---------------------

	# Calculate the correlation function for the x-component of the magnetic
	# field (this has already been normalised)
	x_corr = cf_fft(mag_x_data, no_fluct = True, normalise = True)

	# Print a message to show that the correlation function of the x-component of
	# the magnetic field has been calculated
	print 'Correlation function of the x-component of the magnetic field calculated'

	# Calculate the radially averaged correlation function for the x-component
	# of the magnetic field
	x_rad_av_corr = sfr(x_corr, num_bins, verbose = False)

	# Extract the radius values used to calculate the radially averaged 
	# correlation function
	radius_array = x_rad_av_corr[0]

	# Calculate the normalised radially averaged correlation function for the 
	# x-component of the magnetic field. This is equation 13 of Lazarian and 
	# Pogosyan 2012.
	c_1 = x_rad_av_corr[1]

	# Print a message to show that c_1 has been calculated
	print 'Normalised correlation function for the x-component of the magnetic'\
	+ ' has been calculated'

	# ---------------- Normalised correlation y-comp B field -------------------

	# Calculate the correlation function for the y-component of the magnetic
	# field (this has already been normalised)
	y_corr = cf_fft(mag_y_data, no_fluct = True, normalise = True)

	# Print a message to show that the correlation function of the y-component of
	# the magnetic field has been calculated
	print 'Correlation function of the y-component of the magnetic field calculated'

	# Calculate the radially averaged correlation function for the y-component
	# of the magnetic field
	y_rad_av_corr = sfr(y_corr, num_bins, verbose = False)

	# Calculate the normalised radially averaged correlation function for the 
	# y-component of the magnetic field. This is equation 14 of Lazarian and
	# Pogosyan 2012.
	c_2 = y_rad_av_corr[1]

	# Print a message to show that c_2 has been calculated
	print 'Normalised correlation function for the y-component of the magnetic'\
	+ ' has been calculated'

	# Calculate the right hand side of equation 19 of Lazarian and Pogosyan 2012
	# and enter it into the corresponding array
	RHS_19_arr[i] = 0.5 * ( np.power(c_1, 2.0) + np.power(c_2, 2.0) )

	# -------------- Normalised correlation B_perp gamma = 2 -------------------

	# Calculate the result of raising the magnetic field strength perpendicular 
	# to the line of sight to the power of gamma = 2
	mag_perp_gamma_2 = np.power(mag_perp, 2.0)

	# Calculate the square of the mean of the perpendicular component of the 
	# magnetic field raised to the power of gamma = 2
	mag_sq_mean_gamma_2 = np.power(np.mean(mag_perp_gamma_2, dtype = np.float64), 2.0)

	# Calculate the mean of the squared perpendicular component of the magnetic
	# field raised to the power of gamma = 2
	mag_mean_sq_gamma_2 = np.mean( np.power(mag_perp_gamma_2, 2.0), dtype = np.float64 )

	# Calculate the correlation function for the perpendicular component of the
	# magnetic field, when raised to the power of gamma = 2
	perp_gamma_2_corr = cf_fft(mag_perp_gamma_2, no_fluct = True)

	# Print a message to show that the correlation function of the perpendicular 
	# component of the magnetic field has been calculated for gamma = 2
	print 'Correlation function of the perpendicular component of the magnetic'\
	+ ' field calculated for gamma = 2'

	# Calculate the radially averaged correlation function for the perpendicular
	# component of the magnetic field, raised to the power of gamma = 2
	perp_gamma_2_rad_corr = (sfr(perp_gamma_2_corr, num_bins, verbose = False))[1]

	# Calculate the normalised correlation function for the magnetic field
	# perpendicular to the line of sight, for gamma = 2. This is the left hand
	# side of equation 19, and the right hand side of equation 15.
	LHS_19_arr[i] = (perp_gamma_2_rad_corr - mag_sq_mean_gamma_2)\
	/ (mag_mean_sq_gamma_2 - mag_sq_mean_gamma_2)

	# Print a message to show that the normalised correlation function for 
	# gamma = 2 has been calculated
	print 'The normalised correlation function for gamma = 2 has been calculated'

# -------------------- Plots of LHS and RHS Equation 19 ------------------------

# Here we want to produce one plot with four subplots. These subplots should
# compare the left and right hand sides of Eq 19 for the four simulations,
# covering the different regimes of turbulence

# Create a figure to hold both of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the low 
# magnetic field simulation
ax1 = fig.add_subplot(221)

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array, RHS_19_arr[0], 'b-o') 
plt.plot(radius_array, LHS_19_arr[0], 'r-^')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000, len(radius_array)),\
 np.zeros(np.shape(radius_array)), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the low
# magnetic field, high pressure simulation. Make the y axis limits the same as
# for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array, RHS_19_arr[1], 'b-o') 
plt.plot(radius_array, LHS_19_arr[1], 'r-^')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000, len(radius_array)),\
 np.zeros(np.shape(radius_array)), 'k--', alpha = 0.5)

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

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array, RHS_19_arr[2], 'b-o') 
plt.plot(radius_array, LHS_19_arr[2], 'r-^')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000, len(radius_array)),\
 np.zeros(np.shape(radius_array)), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Create an axis for the fourth subplot to be produced, which is for the high
# magnetic field, high pressure simulation. Make the x axis limits the same as
# for the second plot, and the y axis limits the same as for the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array, RHS_19_arr[3], 'b-o', label = 'RHS Eq. 6') 
plt.plot(radius_array, LHS_19_arr[3], 'r-^', label = 'LHS Eq. 6')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000, len(radius_array)),\
 np.zeros(np.shape(radius_array)), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Force the legend to appear on the plot
plt.legend(fontsize=10, numpoints=1)

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Normalized Correlation Function', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) Ms7.02Ma1.76', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Ms0.45Ma1.72', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Ms6.78Ma0.52', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Ms0.48Ma0.65', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig2_RHS_normalise.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()