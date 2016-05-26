#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# for the Alfven, slow and fast modes of a simulation, and calculates the      #
# quadrupole ratio for the contribution of each mode, as well as for the       #
# overall turbulence. A plot comparing these quadrupole ratios is then         #
# produced.                                                                    #
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
spec_loc = 'b1p2_Aug_Burk/'

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,2.0,3.0,4.0])

# Create a three dimensional array that will hold all of the information
# for the magnitude of the quadrupole ratios. The first index specifies whether 
# the quadrupole ratio is for Alfven, slow, fast modes, or the full turbulence,
# the second gives the gamma value used, and the third axis goes along radius.
quad_arr = np.zeros((4, len(gamma_arr), num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the quadrupole ratios. The first index specifies whether 
# the quadrupole ratio is for Alfven, slow, fast modes, or the full turbulence,
# the second gives the gamma value used, and the third axis goes along radius.
quad_rad_arr = np.zeros((4, len(gamma_arr), num_bins))

# Create a string for the full directory path to use in this calculation
data_loc = simul_loc + spec_loc
 
# Open the FITS files that contains the synchrotron intensity maps for the
# different MHD modes, and the full turbulence.
sync_fits_full = fits.open(data_loc + 'synint_p1-4.fits')
sync_fits_alf = fits.open(data_loc + 'synint_p1-4z_alf.fits')
sync_fits_slow = fits.open(data_loc + 'synint_p1-4z_slow.fits')
sync_fits_fast = fits.open(data_loc + 'synint_p1-4z_fast.fits')

# Extract the data for the simulated synchrotron intensities, for the different
# MHD modes, and the full turbulence
sync_data_full = sync_fits_full[0].data
sync_data_alf = sync_fits_alf[0].data
sync_data_slow = sync_fits_slow[0].data
sync_data_fast = sync_fits_fast[0].data

# Print a message to the screen to show that the data has been loaded
print 'Synchrotron intensities loaded successfully'

# Loop over the gamma values, to calculate the quadrupole ratio for each gamma
# value, for the different MHD modes and the full turbulence.
for j in range(len(gamma_arr)):
	# Calculate the 2D structure function for this slice of the synchrotron
	# intensity data cube. Note that no_fluct = True is set, because we are
	# not subtracting the mean from the synchrotron maps before calculating
	# the structure function. We are also calculating the normalised 
	# structure function, which only takes values between 0 and 2.
	# This is done for all MHD modes, and the full turbulence
	norm_strfn_full = sf_fft(sync_data_full[2*j], no_fluct = True, normalise = True)
	norm_strfn_alf = sf_fft(sync_data_alf[2*j], no_fluct = True, normalise = True)
	norm_strfn_slow = sf_fft(sync_data_slow[2*j], no_fluct = True, normalise = True)
	norm_strfn_fast = sf_fft(sync_data_fast[2*j], no_fluct = True, normalise = True)

	# Shift the 2D structure function so that the zero radial separation
	# entry is in the centre of the image. Do this for all MHD modes, and the
	# full turbulence
	norm_strfn_full = np.fft.fftshift(norm_strfn_full)
	norm_strfn_alf = np.fft.fftshift(norm_strfn_alf)
	norm_strfn_slow = np.fft.fftshift(norm_strfn_slow)
	norm_strfn_fast = np.fft.fftshift(norm_strfn_fast)

	# Calculate the magnitude and argument of the quadrupole ratio, for all
	# MHD modes, and for the full turbulence
	quad_mod_full, quad_arg_full, quad_rad_full = calc_quad_ratio(norm_strfn_full, num_bins)
	quad_mod_alf, quad_arg_alf, quad_rad_alf = calc_quad_ratio(norm_strfn_alf, num_bins)
	quad_mod_slow, quad_arg_slow, quad_rad_slow = calc_quad_ratio(norm_strfn_slow, num_bins)
	quad_mod_fast, quad_arg_fast, quad_rad_fast = calc_quad_ratio(norm_strfn_fast, num_bins)

	# Add the calculated modulus of the quadrupole ratio to the final array, 
	# for all MHD modes, and for the full turbulence
	quad_arr[0,j] = quad_mod_alf
	quad_arr[1,j] = quad_mod_slow
	quad_arr[2,j] = quad_mod_fast
	quad_arr[3,j] = quad_mod_full

	# Add the radius values used to calculate the quadrupole ratio to the
	# corresponding array, for all MHD modes, and the full turbulence
	quad_rad_arr[0,j] = quad_rad_alf
	quad_rad_arr[1,j] = quad_rad_slow
	quad_rad_arr[2,j] = quad_rad_fast
	quad_rad_arr[3,j] = quad_rad_full

# Print a message to show that the calculation has finished successfully
print 'All quadrupole ratios calculated'

# When the code reaches this point, the quadrupole ratios have been saved for 
# every MHD mode and the full turbulence, and every value of gamma, so start
# making the final plots.

# ------------------------ Plots of quadrupole ratios --------------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will be the alfven
# and slow mode results, and the bottom row will be the fast mode and total 
# turbulence results

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for Alfven mode
ax1 = fig.add_subplot(221)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[0,i], quad_arr[0,i], '-o')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the slow
# mode. Make the y axis limits the same as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[1,i], quad_arr[1,i], '-o')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[1,0])), \
	np.zeros(np.shape(quad_rad_arr[1,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the fast
# mode. Make the x axis limits the same as for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[2,i], quad_arr[2,i], '-o')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[2,0])), \
	np.zeros(np.shape(quad_rad_arr[2,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Create an axis for the fourth subplot to be produced, which is for the full
# turbulence. Make the x axis limits the same as for the second plot, and the 
# y axis limits the same as for the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the values of gamma to produce plots for each gamma
for i in range(len(gamma_arr)):
	# Plot the quadrupole ratio for this simulation, for this gamma
	plt.plot(quad_rad_arr[3,i], quad_arr[3,i], '-o', label = 'Gamma={}'.format(gamma_arr[i]))

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
plt.figtext(0.19, 0.95, 'a) Alfven Mode', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Slow Mode', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Fast Mode', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Full Turbulence', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig7.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()