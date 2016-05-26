#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the normalised correlation functions, structure functions,    #
# and quadrupole ratios of the synchrotron intensity maps, for different       #
# lines of sight. Plots are then produced of the normalised correlation        #
# functions, structure functions, quadrupole ratios. This code is intended to  #
# be used with simulations produced by Christoph Federrath.                    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 14/1/2016                                                        #
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
simul_loc = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create a string for the specific simulated data sets to use in calculations.
# The directories end in:
# 512sM5Bs5886_20 (Solenoidal turbulence, timestep 20)
# 512sM5Bs5886_25 (Solenoidal turbulence, timestep 25)
# 512sM5Bs5886_30 (Solenoidal turbulence, timestep 30)
# 512sM5Bs5886_35 (Solenoidal turbulence, timestep 35)
# 512sM5Bs5886_40 (Solenoidal turbulence, timestep 40)
# 512cM5Bs5886_20 (Compressive turbulence, timestep 20)
# 512cM5Bs5886_25 (Compressive turbulence, timestep 25)
# 512cM5Bs5886_30 (Compressive turbulence, timestep 30)
# 512cM5Bs5886_35 (Compressive turbulence, timestep 35)
# 512cM5Bs5886_40 (Compressive turbulence, timestep 40)
spec_locs = ['512sM5Bs5886_20/', '512cM5Bs5886_20/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation
sim_labels = ['Sol', 'Comp']

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Create a three dimensional array that will hold all of the information
# for the normalised correlation functions. The first index gives the simulation
# the second gives the line of sight, and the third axis goes along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
norm_corr_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the normalised correlation functions. The first axis represents
# the simulation used, the second represents the line of sight, and 
# the third axis goes over radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
corr_rad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the structure functions. The first index gives the simulation
# the second gives the line of sight, and the third axis goes along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
sf_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the structure functions. The first axis represents the 
# simulation used, the second represents the line of sight, and 
# the third axis goes over radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
sf_rad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the magnitude of the quadrupole ratios. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes 
# along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the quadrupole ratios. The first axis represents the 
# simulation used, the second represents the line of sight, and 
# the third axis goes over radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_rad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the real part of the quadrupole ratios. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes 
# along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_real_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the imaginary part of the quadrupole ratios. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes 
# along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_imag_arr = np.zeros((len(spec_locs), 3, num_bins))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]

	# Loop over the lines of sight, to calculate the correlation function, 
	# structure function and quadrupole ratio for each line of sight
	for j in range(3):
		# Open the FITS file that contains the synchrotron intensity maps for this
		# simulation
		sync_fits = fits.open(data_loc + 'synint_{}_gam{}.fits'.format(line_o_sight[j],gamma))

		# Extract the data for the simulated synchrotron intensities
		sync_data = sync_fits[0].data

		# Print a message to the screen to show that the data has been loaded
		print 'Synchrotron intensity loaded successfully'

		# Calculate the 2D correlation function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the correlation function
		corr = cf_fft(sync_data, no_fluct = True)

		# Radially average the calculated 2D correlation function, using the 
		# specified number of bins
		rad_corr = sfr(corr, num_bins, verbose = False)

		# Calculate the square of the mean of the synchrotron intensity values
		sync_sq_mean = np.power( np.mean(sync_data, dtype = np.float64), 2.0 )

		# Calculate the mean of the synchrotron intensity values squared
		sync_mean_sq = np.mean( np.power(sync_data, 2.0), dtype = np.float64 )

		# Calculate the normalised, radially averaged correlation function for
		# this value of gamma
		norm_rad_corr = (rad_corr[1] - sync_sq_mean) / (sync_mean_sq - sync_sq_mean)

		# Print a message to show that the correlation function of the
		# synchrotron intensity has been calculated for this line of sight
		print 'Correlation function of synchrotron intensity'\
		+ ' calculated for {} LOS'.format(line_o_sight[j])

		# Insert the calculated normalised, radially averaged correlation function
		# into the matrix that stores all of the calculated correlation functions
		norm_corr_arr[i,j] = norm_rad_corr

		# Insert the radius values used to calculate this correlation function
		# into the matrix that stores the radius values
		corr_rad_arr[i,j] = rad_corr[0]

		# Print a message to show that the correlation function has been calculated
		print 'Normalised, radially averaged correlation function calculated for'\
		+ ' {} LOS'.format(line_o_sight[j])

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not 
		# subtracting the mean from the synchrotron maps before calculating the
		# structure function.
		strfn = sf_fft(sync_data, no_fluct = True)

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
		norm_strfn = sf_fft(sync_data, no_fluct = True, normalise = True)

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

		# Calculate the real part of the quadrupole ratio
		quad_real_arr[i,j] = quad_mod * np.cos(quad_arg)

		# Calculate the imaginary part of the quadrupole ratio
		quad_imag_arr[i,j] = quad_mod * np.sin(quad_arg)

	# Print a message to show that the calculation has finished successfully
	# for this simulation
	print 'All normalised correlation functions calculated for {}'.format(spec_locs[i])

# When the code reaches this point, the normalised correlation functions,
# structure functions, and quadrupole ratios have been saved for every 
# simulation, and every line of sight, so start making the final plots.

# ----------------- Plots of normalised correlation functions ------------------

# Here we want to produce one plot with three subplots. There should be one row
# of subplots, with three subplots in the row. The left subplot will be the 
# normalised correlation functions for a line of sight along the x axis, the
# centre plot will be for the y axis, and the right subplot will be the 
# normalised correlation functions for the z axis. In each plot the solenoidal
# and compressive simulations will be compared

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an array of marker symbols, so that the plot for each line of sight has
# a different plot symbol
symbol_arr = ['o','^']

# Create an axis for the first subplot to be produced, which is for the
# x line of sight
ax1 = fig.add_subplot(131)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,0], norm_corr_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,0])), \
	np.zeros(np.shape(corr_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Create an axis for the second subplot to be produced, which is for the
# y line of sight. Make the y axis limits the same as for the x axis plot
ax2 = fig.add_subplot(132, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,1], norm_corr_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,1])), \
	np.zeros(np.shape(corr_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight. Make the y axis limits the same as for the x axis plot
ax3 = fig.add_subplot(133, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,2], norm_corr_arr[i,2], '-' + symbol_arr[i],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,2])), \
	np.zeros(np.shape(corr_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 10, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'NCF Sync Intensity', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.18, 0.93, 'a) x-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.445, 0.93, 'b) y-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure c
plt.figtext(0.73, 0.93, 'c) z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'ncfs_all_sims_gam{}.eps'.format(gamma), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#--------------------------- Structure Functions -------------------------------

# Here we want to produce one plot with three subplots. There should be one row
# of subplots, with three subplots in the row. The left subplot will be the 
# structure functions for a line of sight along the x axis, the centre plot will
# be for the y axis, and the right subplot will be the structure functions for 
# the z axis. In each plot the solenoidal and compressive simulations will be 
# compared

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an array of marker symbols, so that the plot for each line of sight has
# a different plot symbol
symbol_arr = ['o','^']

# Create an axis for the first subplot to be produced, which is for the
# x line of sight
ax1 = fig.add_subplot(131)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,0], sf_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,0])), \
	np.zeros(np.shape(sf_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
ax1.set_yscale('log')

# Create an axis for the second subplot to be produced, which is for the
# y line of sight. Make the y axis limits the same as for the x axis plot
ax2 = fig.add_subplot(132, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,1], sf_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,1])), \
	np.zeros(np.shape(sf_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
ax2.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight. Make the y axis limits the same as for the x axis plot
ax3 = fig.add_subplot(133, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,2], sf_arr[i,2], '-' + symbol_arr[i],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,2])), \
	np.zeros(np.shape(sf_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis of the plot logarithmic
ax3.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 10, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Structure Function Amplitude', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.18, 0.93, 'a) x-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.445, 0.93, 'b) y-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure c
plt.figtext(0.73, 0.93, 'c) z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'sfs_all_sims_gam{}.eps'.format(gamma), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#----------------------------- Quadrupole Ratios -------------------------------

# Here we want to produce one plot with three subplots. There should be one row
# of subplots, with three subplots in the row. The left subplot will be the 
# quadrupole ratio modulus for a line of sight along the x axis, the centre plot
# will be for the y axis, and the right subplot will be the quadrupole ratio
# modulus for the z axis. In each plot the solenoidal and compressive 
# simulations will be compared

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an array of marker symbols, so that the plot for each line of sight has
# a different plot symbol
symbol_arr = ['o','^']

# Create an axis for the first subplot to be produced, which is for the
# x line of sight
ax1 = fig.add_subplot(131)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,0], quad_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Create an axis for the second subplot to be produced, which is for the
# y line of sight. Make the y axis limits the same as for the x axis plot
ax2 = fig.add_subplot(132, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,1], quad_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,1])), \
	np.zeros(np.shape(quad_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight. Make the y axis limits the same as for the x axis plot
ax3 = fig.add_subplot(133, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,2], quad_arr[i,2], '-' + symbol_arr[i],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,2])), \
	np.zeros(np.shape(quad_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 10, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Quadrupole Ratio', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.18, 0.93, 'a) x-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.445, 0.93, 'b) y-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure c
plt.figtext(0.73, 0.93, 'c) z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'quad_ratio_all_sims_gam{}.eps'.format(gamma), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#----------------------- Real and Imaginary Parts of Quad Ratio ----------------

# Here we want to produce one plot with six subplots. There should be two rows
# of subplots, with three subplots in each row. The top row will be the real 
# part of the quadrupole ratio, and the bottom row will be the imaginary part.
# The left column will be for a line of sight along the x axis, the centre
# column for a line of sight along the y axis, and the right column will be for
# a line of sight along the z axis.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the real
# part of the quadrupole ratio for a line of sight along the x axis
ax1 = fig.add_subplot(231)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio for this simulation, for this line of sight
	plt.plot(quad_rad_arr[i,0], quad_real_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the real
# part of the quadrupole ratio for a line of sight along the y axis. Make the y
# axis limits the same as for the x axis plot
ax2 = fig.add_subplot(232, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio for this simulation, for this line of sight
	plt.plot(quad_rad_arr[i,1], quad_real_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,1])), \
	np.zeros(np.shape(quad_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the real
# part of the quadrupole ratio for a line of sight along the z axis. Make the y
# axis limits the same as for the x axis plot
ax3 = fig.add_subplot(233, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio for this simulation, for this line of sight
	plt.plot(quad_rad_arr[i,2], quad_real_arr[i,2], '-' + symbol_arr[i],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,2])), \
	np.zeros(np.shape(quad_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax3.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 10, numpoints=1)

# Create an axis for the fourth subplot to be produced, which is for the 
# imaginary part of the quadrupole ratio for a line of sight along the x axis.
# Make the x axis limits the same as for the first plot
ax4 = fig.add_subplot(234, sharex = ax1, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio for this simulation, for this line of sight
	plt.plot(quad_rad_arr[i,0], quad_imag_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Create an axis for the fifth subplot to be produced, which is for the 
# imaginary part of the quadrupole ratio for a line of sigth along the y axis.
# Make the x axis limits the same as for the second plot, and the y axis limits
# the same as for the fourth plot
ax5 = fig.add_subplot(235, sharex = ax2, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio for this simulation, for this line of sight
	plt.plot(quad_rad_arr[i,1], quad_imag_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,1])), \
	np.zeros(np.shape(quad_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax5.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax5.get_yticklabels(), visible=False)

# Create an axis for the sixth subplot to be produced, which is for the 
# imaginary part of the quadrupole ratio for a line of sigth along the z axis.
# Make the x axis limits the same as for the third plot, and the y axis limits
# the same as for the fourth plot
ax6 = fig.add_subplot(236, sharex = ax3, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(len(spec_locs)):
	# Plot the quadrupole ratio for this simulation, for this line of sight
	plt.plot(quad_rad_arr[i,2], quad_imag_arr[i,2], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,2])), \
	np.zeros(np.shape(quad_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax6.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax6.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Quadrupole Ratio', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.15, 0.94, 'a) x-LOS Real', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure b
plt.figtext(0.42, 0.94, 'b) y-LOS Real', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.7, 0.94, 'c) z-LOS Real', fontsize = 18)

# Add some text to the figure, to label the left plot as figure d
plt.figtext(0.15, 0.475, 'd) x-LOS Imag', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure e
plt.figtext(0.42, 0.475, 'e) y-LOS Imag', fontsize = 18)

# Add some text to the figure, to label the right plot as figure f
plt.figtext(0.7, 0.475, 'f) z-LOS Imag', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'real_imag_quad_all_sims_gam{}.eps'.format(gamma), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()