#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the normalised correlation functions, structure functions,    #
# and quadrupole ratios of the synchrotron intensity maps, for different       #
# values of gamma. Plots are then produced of the normalised correlation       #
# functions, structure functions, quadrupole ratios.                           #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 10/2/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the functions that calculate the structure and correlation functions
# using FFT, as well as the function that calculates the radially averaged 
# structure or correlation functions. Also import the function that calculates
# multipoles of the 2D structure functions, and the function that calculates the
# magnitude and argument of the quadrupole ratio
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio

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
corr_rad_arr = np.zeros((len(spec_locs), len(gamma_arr), num_bins))

# Create a three dimensional array that will hold all of the information
# for the structure functions. The first index gives the simulation
# the second gives the gamma value used, and the third axis goes along radius.
sf_arr = np.zeros((len(spec_locs), len(gamma_arr), num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the structure functions. The first axis represents the 
# simulation used, the second represents the particular value of gamma, and 
# the third axis goes over radius.
sf_rad_arr = np.zeros((len(spec_locs), len(gamma_arr), num_bins))

# Create a three dimensional array that will hold all of the information
# for the magnitude of the quadrupole ratios. The first index gives the 
# simulation the second gives the gamma value used, and the third axis goes 
# along radius.
quad_arr = np.zeros((len(spec_locs), len(gamma_arr), num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the quadrupole ratios. The first axis represents the 
# simulation used, the second represents the particular value of gamma, and 
# the third axis goes over radius.
quad_rad_arr = np.zeros((len(spec_locs), len(gamma_arr), num_bins))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]
	 
	# Open the FITS file that contains the synchrotron intensity maps for this
	# simulation
	sync_fits = fits.open(data_loc + 'synint_p1-4.fits')

	# Extract the data for the simulated synchrotron intensities
	sync_data = sync_fits[0].data

	# Print a message to the screen to show that the data has been loaded
	print 'Synchrotron intensity loaded successfully'

	# Loop over the gamma values, to calculate the correlation function, 
	# structure function and quadrupole ratio for each gamma value
	for j in range(len(gamma_arr)):
		# Calculate the 2D correlation function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the correlation function
		corr = cf_fft(sync_data[2*j], no_fluct = True)

		# Radially average the calculated 2D correlation function, using the 
		# specified number of bins
		rad_corr = sfr(corr, num_bins, verbose = False)

		# Calculate the square of the mean of the synchrotron intensity values
		sync_sq_mean = np.power( np.mean(sync_data[2*j], dtype = np.float64), 2.0 )

		# Calculate the mean of the synchrotron intensity values squared
		sync_mean_sq = np.mean( np.power(sync_data[2*j], 2.0), dtype = np.float64 )

		# Calculate the normalised, radially averaged correlation function for
		# this value of gamma
		norm_rad_corr = (rad_corr[1] - sync_sq_mean) / (sync_mean_sq - sync_sq_mean)

		# Print a message to show that the correlation function of the
		# synchrotron intensity has been calculated for this gamma
		print 'Correlation function of synchrotron intensity'\
		+ ' calculated for gamma = {}'.format(gamma_arr[j])

		# Insert the calculated normalised, radially averaged correlation function
		# into the matrix that stores all of the calculated correlation functions
		norm_corr_arr[i,j] = norm_rad_corr

		# Insert the radius values used to calculate this correlation function
		# into the matrix that stores the radius values
		corr_rad_arr[i,j] = rad_corr[0]

		# Print a message to show that the correlation function has been calculated
		print 'Normalised, radially averaged correlation function calculated for'\
		+ ' gamma = {}'.format(gamma_arr[j])

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not 
		# subtracting the mean from the synchrotron maps before calculating the
		# structure function.
		strfn = sf_fft(sync_data[2*j], no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins.
		rad_sf = sfr(strfn, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function
		sf_arr[i,j] = rad_sf[1]

		# Extract the radius values used to calculate this structure function
		sf_rad_arr[i,j] = rad_sf[0]

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn = sf_fft(sync_data[2*j], no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image.
		norm_strfn = np.fft.fftshift(norm_strfn)

		# Calculate the magnitude and argument of the quadrupole ratio
		quad_mod, quad_arg, quad_rad = calc_quad_ratio(norm_strfn, num_bins)

		# Add the calculated modulus of the quadrupole ratio to the final array
		quad_arr[i,j] = quad_mod

		# Add the radius values used to calculate the quadrupole ratio to the
		# corresponding array
		quad_rad_arr[i,j] = quad_rad

	# Print a message to show that the calculation has finished successfully
	# for this simulation
	print 'All normalised correlation functions calculated for {}'.format(spec_locs[i])

# When the code reaches this point, the normalised correlation functions,
# structure functions, and quadrupole ratios have been saved for every 
# simulation, and every value of gamma, so start making the final plots.

# ----------------- Plots of normalised correlation functions ------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will be the low 
# magnetic field simulation, and the bottom row will be high magnetic field 
# simulations. The left column will be low pressure simulations, and the right
# column will be high pressure simulations.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an array of marker symbols, so that the plot for each gamma value has
# a different plot symbol
symbol_arr = ['o','^','s','*']

# Create an axis for the first subplot to be produced, which is for the low 
# magnetic field, low pressure simulation
ax1 = fig.add_subplot(221)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the normalised correlation function for this simulation, for this gamma
	plt.plot(corr_rad_arr[0,i], norm_corr_arr[0,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,0])), \
	np.zeros(np.shape(corr_rad_arr[0,0])), 'k--', alpha = 0.5)

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
	plt.plot(corr_rad_arr[1,i], norm_corr_arr[1,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[1,0])), \
	np.zeros(np.shape(corr_rad_arr[1,0])), 'k--', alpha = 0.5)

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
	plt.plot(corr_rad_arr[2,i], norm_corr_arr[2,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[2,0])), \
	np.zeros(np.shape(corr_rad_arr[2,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Create an axis for the fourth subplot to be produced, which is for the high
# magnetic field, high pressure simulation. Make the x axis limits the same as
# for the second plot, and the y axis limits the same as for the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the normalised correlation function for this simulation, for this gamma
	plt.plot(corr_rad_arr[3,i], norm_corr_arr[3,i], '-' + symbol_arr[i],\
	 label = 'Gamma={}'.format(gamma_arr[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[3,0])), \
	np.zeros(np.shape(corr_rad_arr[3,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'NCF Sync Intensity', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Force the legend to appear on the plot
plt.legend(fontsize = 10, numpoints=1)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) Ms7.02Ma1.76', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Ms0.45Ma1.72', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Ms6.78Ma0.52', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Ms0.48Ma0.65', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig4.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#--------------------------- Structure Functions -------------------------------

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
	# Plot the structure function for this simulation, for this gamma
	plt.plot(sf_rad_arr[0,i], sf_arr[0,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,0])), \
	np.zeros(np.shape(sf_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
ax1.set_yscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the low
# magnetic field, high pressure simulation. Make the y axis limits the same as
# for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the structure function for this simulation, for this gamma
	plt.plot(sf_rad_arr[1,i], sf_arr[1,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[1,0])), \
	np.zeros(np.shape(sf_rad_arr[1,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
ax2.set_yscale('log')

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
	# Plot the structure function for this simulation, for this gamma
	plt.plot(sf_rad_arr[2,i], sf_arr[2,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[2,0])), \
	np.zeros(np.shape(sf_rad_arr[2,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis of the plot logarithmic
ax3.set_yscale('log')

# Create an axis for the fourth subplot to be produced, which is for the high
# magnetic field, high pressure simulation. Make the x axis limits the same as
# for the second plot, and the y axis limits the same as for the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the structure function for this simulation, for this gamma
	plt.plot(sf_rad_arr[3,i], sf_arr[3,i], '-' + symbol_arr[i],\
	 label = 'Gamma={}'.format(gamma_arr[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[3,0])), \
	np.zeros(np.shape(sf_rad_arr[3,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis of the plot logarithmic
ax4.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Structure Function Amplitude', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Force the legend to appear on the plot
plt.legend(loc = 4, fontsize = 10, numpoints=1)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) Ms7.02Ma1.76', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Ms0.45Ma1.72', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Ms6.78Ma0.52', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Ms0.48Ma0.65', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig5.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#----------------------------- Quadrupole Ratios -------------------------------

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
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[0,i], quad_arr[0,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

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
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[1,i], quad_arr[1,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[1,0])), \
	np.zeros(np.shape(quad_rad_arr[1,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the high
# magnetic field, low pressure simulation. Make the x axis limits the same as
# for the first plot
ax3 = fig.add_subplot(223, sharex = ax1, sharey = ax1)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[2,i], quad_arr[2,i], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[2,0])), \
	np.zeros(np.shape(quad_rad_arr[2,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Create an axis for the fourth subplot to be produced, which is for the high
# magnetic field, high pressure simulation. Make the x axis limits the same as
# for the second plot, and the y axis limits the same as for the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax1)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[3,i], quad_arr[3,i], '-' + symbol_arr[i],\
	 label = 'Gamma={}'.format(gamma_arr[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[3,0])), \
	np.zeros(np.shape(quad_rad_arr[3,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Quadrupole Ratio', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Force the legend to appear on the plot
plt.legend(fontsize = 10, numpoints=1)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) Ms7.02Ma1.76', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Ms0.45Ma1.72', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Ms6.78Ma0.52', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Ms0.48Ma0.65', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig6.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()