#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the skewness, kurtosis, structure function slope, and         #
# integrated quadrupole ratio as a function of sonic and Alfvenic Mach number  #
# for simulations with low and high magnetic field. Plots of these statistics  #
# as a function of sonic and Alfvenic Mach number are then saved.              #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 11/2/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

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
# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/', 'c512b3p.01/', 'c512b5p.01/']

# Create a list, where each entry is a string describing the initial magnetic
# field and pressure used to run each simulation.
short_simul = ['b.1p.0049', 'b.1p.0077', 'b.1p.01', 'b.1p.025', 'b.1p.05',\
'b.1p.1', 'b.1p.7', 'b.1p2', 'b1p.0049', 'b1p.0077', 'b1p.01', 'b1p.025',\
'b1p.05', 'b1p.1', 'b1p.7', 'b1p2', 'b3p.01', 'b5p.01']

# Create an array, where each entry specifies the pressure of the corresponding
# simulation in the list of simulation directories
press_arr = np.array([0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0, 0.01, 0.01])

# Create an array, where each entry specifies the initial mean magnetic field of
# the corresponding simulation to study 
mag_arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,\
	1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 5.0])

# Create an array, where each entry specifies the calculated sonic Mach number 
# for each simulation
sonic_mach_arr = np.array([8.85306946, 5.42555035, 5.81776713, 3.71658244,\
 2.75242104, 2.13759125, 0.81017387, 0.44687901, 7.5584105, 6.13642211,\
 5.47297919, 3.63814214, 2.69179409, 2.22693767, 0.83800535, 0.47029213,\
 6.57849578, 7.17334893])

# Create an array, where each entry specifies the calculated Alfvenic Mach 
# number for each simulation
alf_mach_arr = np.array([1.41278383, 1.77294593, 1.75575508, 1.50830194,\
 1.69455875, 1.85993991, 1.74231524, 1.71939152, 0.49665052, 0.50288954,\
 0.51665006, 0.54928564, 0.57584022, 0.67145057, 0.70015313, 0.65195539,\
 0.21894299, 0.14357068])

# Create an array of index values that sorts the sonic Mach number values from
# smallest to largest
sonic_sort = np.argsort(sonic_mach_arr)

# Create an array of index values that sorts the Alfvenic Mach number values 
# from smallest to largest
alf_sort = np.argsort(alf_mach_arr)

# Create an array of the sonic Mach number values, from smallest to largest
sonic_mach_sort = sonic_mach_arr[sonic_sort]

# Create an array of the Alfvenic Mach number values, from smallest to largest
alf_mach_sort = alf_mach_arr[alf_sort]

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma.
# NOTE: We will calculate the biased skewness
skew_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. 
# NOTE: We will calculate the biased Fisher kurtosis
kurt_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of gamma.
m_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity 
# image, for the corresponding simulation, for a particular value of gamma. 
int_quad_arr = np.zeros(len(simul_arr))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS files that contain the simulated synchrotron intensity maps
	sync_fits = fits.open(data_loc + 'synint_p1-4.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensities observed for different values of gamma, the power
	# law index of the cosmic ray electrons emitting the synchrotron emission.
	sync_data = sync_fits[0].data

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Extract the synchrotron intensity map for the value of gamma
	sync_map = sync_data[gam_index]

	# Flatten the synchrotron intensity maps for the value of gamma
	flat_sync = sync_map.flatten()

	# Calculate the biased skewness of the synchrotron intensity maps, and store
	# the results in the corresponding array.
	skew_arr[j] = stats.skew(flat_sync)

	# Calculate the biased Fisher kurtosis of the synchrotron intensity 
	# maps, and store the results in the corresponding array.
	kurt_arr[j] = stats.kurtosis(flat_sync)

	# Calculate the structure function (two-dimensional) of the synchrotron
	# intensity maps. Note that no_fluct = True is set, because we are not
	# subtracting the mean from the synchrotron maps before calculating the 
	# structure function.
	strfn = sf_fft(sync_map, no_fluct = True)

	# Radially average the calculated 2D structure function, using the 
	# specified number of bins.
	rad_sf = sfr(strfn, num_bins, verbose = False)

	# Extract the calculated radially averaged structure function 
	sf = rad_sf[1]

	# Extract the radius values used to calculate this structure function
	sf_rad_arr = rad_sf[0]

	# Calculate the spectral index of the structure function calculated for
	# this value of gamma. Note that only the first third of the structure
	# function is used in the calculation, as this is the part that is 
	# close to a straight line. 
	spec_ind_data = np.polyfit(np.log10(\
		sf_rad_arr[0:np.ceil(num_bins/3.0)]),\
		np.log10(sf[0:np.ceil(num_bins/3.0)]), 1, full = True)

	# Extract the returned coefficients from the polynomial fit
	coeff = spec_ind_data[0]

	# Enter the value of m, the slope of the structure function minus 1,
	# into the corresponding array
	m_arr[j] = coeff[0]-1.0

	# Calculate the 2D structure function for this slice of the synchrotron
	# intensity data cube. Note that no_fluct = True is set, because we are
	# not subtracting the mean from the synchrotron maps before calculating
	# the structure function. We are also calculating the normalised 
	# structure function, which only takes values between 0 and 2.
	norm_strfn = sf_fft(sync_map, no_fluct = True, normalise = True)

	# Shift the 2D structure function so that the zero radial separation
	# entry is in the centre of the image.
	norm_strfn = np.fft.fftshift(norm_strfn)

	# Calculate the magnitude and argument of the quadrupole ratio
	quad_mod, quad_arg, quad_rad = calc_quad_ratio(norm_strfn, num_bins)

	# Integrate the magnitude of the quadrupole / monopole ratio from one sixth 
	# of the way along the radial separation bins, until three quarters of the 
	# way along the radial separation bins. This integration is performed with
	# respect to log separation (i.e. I am ignoring the fact that the 
	# points are equally separated in log space, to calculate the area under
	# the quadrupole / monopole ratio plot when the x axis is scaled 
	# logarithmically). I normalise the value that is returned by dividing
	# by the number of increments in log radial separation used in the
	# calculation. 
	int_quad_arr[j] = np.trapz(quad_mod[np.floor(num_bins/6.0):\
		3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
		 - np.floor(num_bins/6.0))

	# Close the fits files, to save memory
	sync_fits.close()

	# Print a message to show that the calculation has finished successfully
	# for this simulation
	print 'All statistics calculated for simulation {}'.format(simul_arr[j])

# When the code reaches this point, the statistics have been saved for every 
# simulation, so start making the final plots.

# ---------------------- Plots of skewness and kurtosis ------------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will be skewness, and
# the bottom row will be kurtosis. The left column will be plots against sonic
# Mach number, and the right column will be plots against Alfvenic Mach number.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for skewness 
# against sonic Mach number
ax1 = fig.add_subplot(221)

# Plot the skewness as a function of sonic Mach number 
plt.scatter(sonic_mach_arr[0:8], skew_arr[0:8], s = 35, c = 'b')
plt.scatter(sonic_mach_arr[8:], skew_arr[8:], s = 35, c = 'r')

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for skewness
# as a function of Alfvenic Mach number. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Plot the skewness as a function of Alfvenic Mach number
plt.scatter(alf_mach_arr[0:8], skew_arr[0:8], s = 35, c = 'b')
plt.scatter(alf_mach_arr[8:], skew_arr[8:], s = 35, c = 'r')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for kurtosis 
# as a function of sonic Mach number. Make the x axis limits the same as
# for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Plot the kurtosis as a function of sonic Mach number
plt.scatter(sonic_mach_arr[0:8], kurt_arr[0:8], s = 35, c = 'b')
plt.scatter(sonic_mach_arr[8:], kurt_arr[8:], s = 35, c = 'r')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for
# kurtosis as a function of Alfvenic Mach number. Make the x axis limits the
# same as for the second plot, and the y axis limits the same as the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Plot the kurtosis as a function of Alfvenic Mach number
plt.scatter(alf_mach_arr[0:8], kurt_arr[0:8], s = 35, c = 'b')
plt.scatter(alf_mach_arr[8:], kurt_arr[8:], s = 35, c = 'r')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.14, 0.94, 'a)', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.57, 0.94, 'b)', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.14, 0.485, 'c)', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.57, 0.485, 'd)', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig9.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#-------------------------- Structure Function Slope ---------------------------

# Here we want to produce one plot with two subplots. There should be one row
# of subplots, with two subplots in the row. The plot on the left should be
# the structure function slope as a function of sonic Mach number, and the plot
# on the right should be the structure function slope as a function of Alfvenic
# Mach number.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,5), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the 
# structure function slope vs sonic Mach number
ax1 = fig.add_subplot(121)

# Plot m as a function of sonic Mach number
plt.scatter(sonic_mach_arr[0:8], m_arr[0:8], s = 35, c = 'b')
plt.scatter(sonic_mach_arr[8:], m_arr[8:], s = 35, c = 'r')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a y-axis label to the plot
plt.ylabel('m', fontsize = 20)

# Create an axis for the second subplot to be produced, which is for the 
# structure function slope as a function of Alfvenic Mach number.
ax2 = fig.add_subplot(122, sharey = ax1)

# Plot m as a function of Alfvenic Mach number 
plt.scatter(alf_mach_arr[0:8], m_arr[0:8], s = 35, c = 'b')
plt.scatter(alf_mach_arr[8:], m_arr[8:], s = 35, c = 'r')

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.14, 0.94, 'a)', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.57, 0.94, 'b)', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig10.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#----------------------------- Quadrupole Ratios -------------------------------

# Here we want to produce one plot with two subplots. There should be one row
# of subplots, with two subplots in the row. The plot on the left should be
# the integrated quadrupole ratio as a function of sonic Mach number, and the
# plot on the right should be the integrated quadrupole ratio as a function of
# Alfvenic Mach number.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,5), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the 
# integrated quadrupole ratio as a function of sonic Mach number
ax1 = fig.add_subplot(121)

# Plot the integrated magnitude of the quad / mono ratio as a function of sonic 
# Mach number 
plt.scatter(sonic_mach_arr[0:8], int_quad_arr[0:8], s = 35, c = 'b')
plt.scatter(sonic_mach_arr[8:], int_quad_arr[8:], s = 35, c = 'r')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a y-axis label to the plot
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Create an axis for the second subplot to be produced, which is for the 
# integrated quadrupole ratio as a function of Alfvenic Mach number. Make the y 
# axis limits the same as for the sonic Mach number plot
ax2 = fig.add_subplot(122, sharey = ax1)

# Plot the integrated magnitude of the quad / mono ratio as a function of 
# Alfvenic Mach number
plt.scatter(alf_mach_arr[0:8], int_quad_arr[0:8], s = 35, c = 'b')
plt.scatter(alf_mach_arr[8:], int_quad_arr[8:], s = 35, c = 'r')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.14, 0.94, 'a)', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.57, 0.94, 'b)', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig11.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()