#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# calculates the quadrupole ratios of the synchrotron intensity maps, as well  #
# their first derivatives, for different lines of sight. Plots are then        #
# produced of the first order derivative of the quadrupole ratios. This code   #
# is intended to be used with simulations produced by Christoph Federrath.     #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 25/1/2016                                                        #
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

# Define a function that calculates the errors in statistics by breaking up
# synchrotron images into quarters, calculating statistics for each quarter, and
# then calculates the standard deviation of the statistics.
def calc_err_bootstrap(sync_map, first_index, end_index):
	'''
	Description
        This function divides the given images into quarters, and then 
        calculates statistics for each quarter. The standard deviation of the 
        calculated statistics is then returned, representing the error on 
        each statistic.
        
    Required Input
        sync_map - The synchrotron intensity map. Should be a 2D Numpy array.
        first_index - A variable to hold the first index to be used to calculate
        			  the standard deviation of the first derivative of the 
        			  quadrupole ratio
        end_index - A variable to hold the final index to be used to calculate
        			the standard deviation of the first derivative of the 
        			quadrupole ratio
                   
    Output
		quad_deriv_std_err - The error calculated for the standard deviation of
							 the first derivative of the quadrupole ratio
					   		 modulus of the synchrotron intensity.
	'''

	# Create an array that will hold the quarters of the synchrotron images
	quarter_arr = np.zeros((4,np.shape(sync_map)[0]/2,np.shape(sync_map)[1]/2))

	# Add the quarters of the images into the array
	quarter_arr[0], quarter_arr[1] = np.split(np.split(sync_map,2,axis=0)[0],2,axis=1) 
	quarter_arr[2], quarter_arr[3] = np.split(np.split(sync_map,2,axis=0)[1],2,axis=1) 

	# Create arrays that will hold the calculated statistics for each quarter
	quad_deriv_std_val = np.zeros(np.shape(quarter_arr)[0])

	# Loop over the quarters, to calculate statistics for each one
	for i in range(np.shape(quarter_arr)[0]):
		# Extract the current image quarter from the array
		image = quarter_arr[i]

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn = sf_fft(image, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image.
		norm_strfn = np.fft.fftshift(norm_strfn)

		# Calculate the magnitude and argument of the quadrupole ratio
		quad_mod, quad_arg, quad_rad = calc_quad_ratio(norm_strfn, num_bins)

		# Calculate the log of the radial spacing between evaluations of the
		# quadrupole ratio
		quad_space = np.log10(quad_rad[1]) - np.log10(quad_rad[0]) 

		# Calculate the first derivative of the quadrupole ratio modulus
		# Note that this assumes data that is equally spaced logarithmically,
		# so that we calculate the derivative as it appears on a semi-log plot 
		quad_mod_deriv = np.gradient(quad_mod, quad_space)

		# Select the array values that are between the dissipation and 
		# injection scales, as these will be used to calculate the standard
		# deviation of the first derivative.
		quad_mod_deriv = quad_mod_deriv[first_index:end_index] /\
		 np.max(quad_mod[first_index:end_index])

		# Calculate the standard deviation of the first derivative of the 
		# quadrupole ratio modulus
		quad_deriv_std_val[i] = np.std(quad_mod_deriv, dtype = np.float64)

	# At this point, the statistics have been calculated for each quarter
	# The next step is to calculate the standard error of the mean of each
	# statistic
	quad_deriv_std_err = np.std(quad_deriv_std_val) / np.sqrt(len(quad_deriv_std_val))

	# Now that all of the calculations have been performed, return the 
	# calculated errors
	return quad_deriv_std_err

# Set a variable to hold the number of bins to use in calculating the 
# correlation functions
num_bins = 25

# Set a variable to hold the final index to be used to calculate the standard
# deviation of the first derivative of the quadrupole ratio
end_index = 17

# Set a variable to hold the first index to be used to calculate the standard
# deviation of the first derivative of the quadrupole ratio
first_index = 8

# Set a variable for how many data points should be used to calculate the
# standard deviation of the first derivative of the quadrupole ratio
num_eval_points = end_index - first_index

# Create a string to hold the location to which plots should be saved
save_loc = '/Users/chrisherron/Documents/PhD/CFed_2016/'

#------------------------------------------------------------------------------

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use, for Christoph's simulations
simul_loc_cfed = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create a string for the specific simulated data sets to use in calculations.
# This is just for Christoph's simulations
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
spec_locs_cfed = ['512sM5Bs5886_20/', '512sM5Bs5886_25/', '512cM5Bs5886_20/',\
'512cM5Bs5886_25/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation, for Christoph's simulations
sim_labels_cfed = ['Sol 20', 'Sol 25', 'Comp 20', 'Comp 25']

# Create an array, where each entry specifies the calculated Alfvenic Mach 
# number for each simulation, for Christoph's simulations
alf_mach_arr_cfed = np.array([2.0,2.1,2.3,2.0])

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube. This can include 'x', 'y', or 'z'. 
# Only use lines of sight perpendicular to the mean magnetic field.
# For Christoph's simulations
line_o_sight_cfed = ['x', 'y']

#------------------------------------------------------------------------------

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. For Blakesley's simulations.
simul_loc_bbur = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# For Blakesley's simulations
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

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study, for Blakesley's simulations
spec_locs_bbur = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/', 'c512b3p.01/', 'c512b5p.01/']

# Create a list, where each entry is a string describing the initial magnetic
# field and pressure used to run each simulation, for Blakesley's simulations
short_labels_bbur = ['b.1p.0049', 'b.1p.0077', 'b.1p.01', 'b.1p.025', 'b.1p.05',\
'b.1p.1', 'b.1p.7', 'b.1p2', 'b1p.0049', 'b1p.0077', 'b1p.01', 'b1p.025',\
'b1p.05', 'b1p.1', 'b1p.7', 'b1p2', 'b3p.01', 'b5p.01']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# low magnetic field simulations used to produce plots
low_B_short_M = ['Ms7.02Ma1.76', 'Ms2.38Ma1.86', 'Ms0.83Ma1.74', 'Ms0.45Ma1.72']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# high magnetic field simulations used to produce plots
high_B_short_M = ['Ms6.78Ma0.52', 'Ms2.41Ma0.67', 'Ms0.87Ma0.7', 'Ms0.48Ma0.65']

# Create an array, where each entry specifies the calculated Alfvenic Mach 
# number for each simulation, for Blakesley's simulations
alf_mach_arr_bbur = np.array([1.41278383, 1.77294593, 1.75575508, 1.50830194,\
 1.69455875, 1.85993991, 1.74231524, 1.71939152, 0.49665052, 0.50288954,\
 0.51665006, 0.54928564, 0.57584022, 0.67145057, 0.70015313, 0.65195539,\
 0.21894299, 0.14357068])

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube. This can include 'x', 'y', or 'z'. 
# Only use lines of sight perpendicular to the mean magnetic field.
# For Blakesley's simulations
line_o_sight_bbur = ['z', 'y']

#------------------------------------------------------------------------------

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Set the index of the gamma array, that is needed to obtain the synchrotron 
# intensity map produced for this value of gamma in Blakesley's simulations
gam_index = 2

# Create a three dimensional array that will hold all of the information
# for the derivative of the quadrupole ratio moduli. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes 
# along radius.
# Index 1 is Christoph, x line of sight
# Index 2 is Christoph, y line of sight
quad_deriv_arr_cfed = np.zeros((len(spec_locs_cfed), 2, num_eval_points))
# Index 1 is Blakesley, z line of sight
# Index 2 is Blakesley, y line of sight
quad_deriv_arr_bbur = np.zeros((len(spec_locs_bbur), 2, num_eval_points))

# Create a three dimensional array that will just hold the radius values used
# to make all of the quadrupole ratios. The first axis represents the 
# simulation used, the second represents the line of sight, and 
# the third axis goes over radius.
# Index 1 is Christoph, x line of sight
# Index 2 is Christoph, y line of sight
quad_rad_arr_cfed = np.zeros((len(spec_locs_cfed), 2, num_eval_points))
# Index 1 is Blakesley, z line of sight
# Index 2 is Blakesley, y line of sight
quad_rad_arr_bbur = np.zeros((len(spec_locs_bbur), 2, num_eval_points))

# Create an empty array, where each entry specifies the calculated standard
# deviation of the first derivative of the quadrupole ratio modulus for the 
# corresponding simulation, for a particular value of gamma. The first index 
# gives the simulation, and the second index gives the line of sight.
quad_deriv_std_arr_cfed = np.zeros((len(spec_locs_cfed),2))
quad_deriv_std_arr_bbur = np.zeros((len(spec_locs_bbur),2))

# Create error arrays for each of the statistics. These errors are calculated
# by the standard deviation of the statistics calculated for sub-images of the
# synchrotron maps. The first index gives the simulation, and the 
# second index gives the line of sight.
quad_deriv_std_err_arr_cfed = np.zeros((len(spec_locs_cfed),2))
quad_deriv_std_err_arr_bbur = np.zeros((len(spec_locs_bbur),2))

# Loop over Christoph's simulations that we are using to make the plot
for i in range(len(spec_locs_cfed)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc_cfed + spec_locs_cfed[i]

	# Loop over the lines of sight, to calculate the quadrupole ratio for each 
	# line of sight
	for j in range(2):
		# Open the FITS file that contains the synchrotron intensity maps for this
		# simulation
		sync_fits = fits.open(data_loc + 'synint_{}_gam{}.fits'.format(\
			line_o_sight_cfed[j],gamma))

		# Extract the data for the simulated synchrotron intensities
		sync_data = sync_fits[0].data

		# Print a message to the screen to show that the data has been loaded
		print 'Synchrotron intensity loaded successfully'

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

		# Add the radius values used to calculate the quadrupole ratio to the
		# corresponding array
		quad_rad_arr_cfed[i,j] = quad_rad[first_index:end_index]

		# Calculate the log of the radial spacing between evaluations of the
		# quadrupole ratio
		quad_space = np.log10(quad_rad[1]) - np.log10(quad_rad[0]) 

		# Calculate the first derivative of the quadrupole ratio modulus
		# Note that this assumes data that is equally spaced logarithmically,
		# so that we calculate the derivative as it appears on a semi-log plot 
		quad_mod_deriv = np.gradient(quad_mod, quad_space)

		# Select the array values that are between the dissipation and 
		# injection scales, as these will be used to calculate the standard
		# deviation of the first derivative.
		quad_deriv_arr_cfed[i,j] = quad_mod_deriv[first_index:end_index] /\
		 np.max(quad_mod[first_index:end_index])

		# Calculate the standard deviation of the first derivative of the 
		# quadrupole ratio modulus
		quad_deriv_std_arr_cfed[i,j] = np.std(quad_deriv_arr_cfed[i,j], dtype = np.float64)

		# Create errors for each of the statistics. These errors are only for the
		# statistics calculated from this line of sight (perpendicular to the mean 
		# magnetic field), and are calculated by the standard deviation of the 
		# statistics calculated for sub-images of the synchrotron maps.
		quad_deriv_std_err_arr_cfed[i,j]= calc_err_bootstrap(sync_data,\
		 first_index, end_index)

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			spec_locs_cfed[i], line_o_sight_cfed[j])

# Loop over Blakesley's simulations that we are using to make the plot
for i in range(len(spec_locs_bbur)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc_bbur + spec_locs_bbur[i]

	# Loop over the lines of sight, to calculate the quadrupole ratio for each 
	# line of sight
	for j in range(2):
		# Open the FITS file that contains the synchrotron intensity maps for this
		# simulation
		if line_o_sight_bbur[j] == 'z':
			# Open the FITS file for a line of sight along the z axis
			sync_fits = fits.open(data_loc + 'synint_p1-4.fits'.format(\
				line_o_sight_bbur[j]))
		elif line_o_sight_bbur[j] == 'y':
			# Open the FITS file for a line of sight along the y axis
			sync_fits = fits.open(data_loc + 'synint_p1-4y.fits'.format(\
				line_o_sight_bbur[j]))

		# Extract the data for the simulated synchrotron intensities
		sync_data = sync_fits[0].data

		# Extract the slice for the value of gamma
		sync_data = sync_data[gam_index]

		# Print a message to the screen to show that the data has been loaded
		print 'Synchrotron intensity loaded successfully'

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

		# Add the radius values used to calculate the quadrupole ratio to the
		# corresponding array
		quad_rad_arr_bbur[i,j] = quad_rad[first_index:end_index]

		# Calculate the log of the radial spacing between evaluations of the
		# quadrupole ratio
		quad_space = np.log10(quad_rad[1]) - np.log10(quad_rad[0]) 

		# Calculate the first derivative of the quadrupole ratio modulus
		# Note that this assumes data that is equally spaced logarithmically,
		# so that we calculate the derivative as it appears on a semi-log plot 
		quad_mod_deriv = np.gradient(quad_mod, quad_space)

		# Select the array values that are between the dissipation and 
		# injection scales, as these will be used to calculate the standard
		# deviation of the first derivative.
		quad_deriv_arr_bbur[i,j] = quad_mod_deriv[first_index:end_index] /\
		 np.max(quad_mod[first_index:end_index])

		# Calculate the standard deviation of the first derivative of the 
		# quadrupole ratio modulus
		quad_deriv_std_arr_bbur[i,j] = np.std(quad_deriv_arr_bbur[i,j], dtype = np.float64)

		# Create errors for each of the statistics. These errors are only for the
		# statistics calculated from this line of sight (perpendicular to the mean 
		# magnetic field), and are calculated by the standard deviation of the 
		# statistics calculated for sub-images of the synchrotron maps.
		quad_deriv_std_err_arr_bbur[i,j]= calc_err_bootstrap(sync_data,\
		 first_index, end_index)

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			spec_locs_bbur[i], line_o_sight_bbur[j])

# When the code reaches this point, the quadrupole ratio has been saved for every 
# simulation, and every line of sight, so start making the final plots.

#------------------------- Quadrupole Ratios Christoph -------------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The left subplot will be the 
# quadrupole ratio modulus for a line of sight along the x axis, the right plot
# will be for the y axis. The top row is for the solenoidal simulation, and the
# bottom row is for the compressive simulation. In each plot the different
# timesteps will be compared.

# Create a figure to hold all of the subplots
fig1 = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an array of marker symbols, so that the plot for each line of sight has
# a different plot symbol
symbol_arr = ['o','^']

# Create an axis for the first subplot to be produced, which is for the
# x line of sight, solenoidal simulation
ax1 = fig1.add_subplot(221)

# Plot the quadrupole ratio modulus for this simulation, for this 
# line of sight
plt.plot(quad_rad_arr_cfed[0,0], quad_deriv_arr_cfed[0,0], '-' + symbol_arr[0])
plt.plot(quad_rad_arr_cfed[1,0], quad_deriv_arr_cfed[1,0], '-' + symbol_arr[1])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_cfed[0,0])), \
	np.zeros(np.shape(quad_rad_arr_cfed[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Set the limits for the x axis
ax1.set_xlim(xmin = 5, xmax = 200)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the
# y line of sight. Make the y axis limits the same as for the x axis plot
ax2 = fig1.add_subplot(222, sharey = ax1)

# Plot the quadrupole ratio modulus for this simulation, for this 
# line of sight
plt.plot(quad_rad_arr_cfed[0,1], quad_deriv_arr_cfed[0,1], '-' + symbol_arr[0],\
	label = '{}'.format(sim_labels_cfed[0]))
plt.plot(quad_rad_arr_cfed[1,1], quad_deriv_arr_cfed[1,1], '-' + symbol_arr[1],\
	label = '{}'.format(sim_labels_cfed[1]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_cfed[0,1])), \
	np.zeros(np.shape(quad_rad_arr_cfed[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc = 3, fontsize = 8, numpoints=1)

# Create an axis for the third subplot to be produced, which is for the 
# compressive simulation, and a line of sight along the x axis. Make the x axis
# limits the same as for the first plot
ax3 = fig1.add_subplot(223, sharex = ax1)

# Plot the quadrupole ratio modulus for this simulation, for this 
# line of sight
plt.plot(quad_rad_arr_cfed[2,0], quad_deriv_arr_cfed[2,0], '-' + symbol_arr[0])
plt.plot(quad_rad_arr_cfed[3,0], quad_deriv_arr_cfed[3,0], '-' + symbol_arr[1])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_cfed[2,0])), \
	np.zeros(np.shape(quad_rad_arr_cfed[2,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Create an axis for the fourth subplot to be produced, which is for the 
# compressive simulation, and a line of sight along the y axis. Make the x axis
# limits the same as for the second plot, and y axis the same as the third 
ax4 = fig1.add_subplot(224, sharex = ax2, sharey = ax3)

# Plot the quadrupole ratio modulus for this simulation, for this 
# line of sight
plt.plot(quad_rad_arr_cfed[2,1], quad_deriv_arr_cfed[2,1], '-' + symbol_arr[0],\
	label = '{}'.format(sim_labels_cfed[2]))
plt.plot(quad_rad_arr_cfed[3,1], quad_deriv_arr_cfed[3,1], '-' + symbol_arr[1],\
	label = '{}'.format(sim_labels_cfed[3]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_cfed[2,1])), \
	np.zeros(np.shape(quad_rad_arr_cfed[2,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc = 3, fontsize = 8, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Norm Quadrupole Ratio Deriv', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.94, 'a) Sol x-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure b
plt.figtext(0.61, 0.94, 'b) Sol y-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure c
plt.figtext(0.19, 0.475, 'c) Comp x-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Comp y-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_deriv_cfed_time_gam{}_{}_{}.eps'.format(gamma,\
	first_index, end_index), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#------------------------- Quadrupole Ratios Blakesley -------------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The left subplot will be the 
# quadrupole ratio modulus for a line of sight along the z axis, the right plot
# will be for the y axis. The top row is for the low magnetic field simulations,
# and the bottom row is for the high magnetic field simulations.

# Create a figure to hold all of the subplots
fig2 = plt.figure(2, figsize=(9,6), dpi = 300)

# Create an array of marker symbols, so that the plot for each line of sight has
# a different plot symbol
symbol_arr = ['o','^', 's', '*']

# Create an axis for the first subplot to be produced, which is for the
# z line of sight, low magnetic field simulations
ax5 = fig2.add_subplot(221)

# Plot the quadrupole ratio modulus for these simulations, for this 
# line of sight
plt.plot(quad_rad_arr_bbur[2,0], quad_deriv_arr_bbur[2,0], '-' + symbol_arr[0])
plt.plot(quad_rad_arr_bbur[5,0], quad_deriv_arr_bbur[5,0], '-' + symbol_arr[1])
plt.plot(quad_rad_arr_bbur[6,0], quad_deriv_arr_bbur[6,0], '-' + symbol_arr[2])
plt.plot(quad_rad_arr_bbur[7,0], quad_deriv_arr_bbur[7,0], '-' + symbol_arr[3])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_bbur[2,0])), \
	np.zeros(np.shape(quad_rad_arr_bbur[2,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax5.set_xscale('log')

# Set the limits for the x axis
ax5.set_xlim([5,200])

# Make the x axis tick labels invisible
plt.setp( ax5.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the
# y line of sight. Make the y axis limits the same as for the z axis plot
ax6 = fig2.add_subplot(222, sharey = ax5)

# Plot the quadrupole ratio modulus for these simulations, for this 
# line of sight
plt.plot(quad_rad_arr_bbur[2,1], quad_deriv_arr_bbur[2,1], '-' + symbol_arr[0],\
	label = '{}'.format(low_B_short_M[0]))
plt.plot(quad_rad_arr_bbur[5,1], quad_deriv_arr_bbur[5,1], '-' + symbol_arr[1],\
	label = '{}'.format(low_B_short_M[1]))
plt.plot(quad_rad_arr_bbur[6,1], quad_deriv_arr_bbur[6,1], '-' + symbol_arr[2],\
	label = '{}'.format(low_B_short_M[2]))
plt.plot(quad_rad_arr_bbur[7,1], quad_deriv_arr_bbur[7,1], '-' + symbol_arr[3],\
	label = '{}'.format(low_B_short_M[3]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_bbur[2,1])), \
	np.zeros(np.shape(quad_rad_arr_bbur[2,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax6.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax6.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax6.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc = 3, fontsize = 8, numpoints=1)

# Create an axis for the third subplot to be produced, which is for the 
# high magnetic field simulations, and a line of sight along the z axis. Make 
# the x axis limits the same as for the first plot
ax7 = fig2.add_subplot(223, sharex = ax5)

# Plot the quadrupole ratio modulus for these simulations, for this 
# line of sight
plt.plot(quad_rad_arr_bbur[10,0], quad_deriv_arr_bbur[10,0], '-' + symbol_arr[0])
plt.plot(quad_rad_arr_bbur[13,0], quad_deriv_arr_bbur[13,0], '-' + symbol_arr[1])
plt.plot(quad_rad_arr_bbur[14,0], quad_deriv_arr_bbur[14,0], '-' + symbol_arr[2])
plt.plot(quad_rad_arr_bbur[15,0], quad_deriv_arr_bbur[15,0], '-' + symbol_arr[3])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_bbur[10,0])), \
	np.zeros(np.shape(quad_rad_arr_bbur[10,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax7.set_xscale('log')

# Create an axis for the fourth subplot to be produced, which is for the 
# compressive simulation, and a line of sight along the y axis. Make the x axis
# limits the same as for the second plot, and y axis the same as the third 
ax8 = fig2.add_subplot(224, sharex = ax6, sharey = ax7)

# Plot the quadrupole ratio modulus for this simulation, for this 
# line of sight
plt.plot(quad_rad_arr_bbur[10,1], quad_deriv_arr_bbur[10,1], '-' + symbol_arr[0],\
	label = '{}'.format(high_B_short_M[0]))
plt.plot(quad_rad_arr_bbur[13,1], quad_deriv_arr_bbur[13,1], '-' + symbol_arr[1],\
	label = '{}'.format(high_B_short_M[1]))
plt.plot(quad_rad_arr_bbur[14,1], quad_deriv_arr_bbur[14,1], '-' + symbol_arr[2],\
	label = '{}'.format(high_B_short_M[2]))
plt.plot(quad_rad_arr_bbur[15,1], quad_deriv_arr_bbur[15,1], '-' + symbol_arr[3],\
	label = '{}'.format(high_B_short_M[3]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr_bbur[10,1])), \
	np.zeros(np.shape(quad_rad_arr_bbur[10,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax8.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax8.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc = 3, fontsize = 8, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Norm Quadrupole Ratio Deriv', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.94, 'a) Low B z-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure b
plt.figtext(0.61, 0.94, 'b) Low B y-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure c
plt.figtext(0.19, 0.475, 'c) High B z-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) High B y-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_deriv_bbur_time_gam{}_{}_{}.eps'.format(gamma,\
	first_index, end_index), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#-------------------------------------------------------------------------------

# Create a figure to plot the standard deviation of the first derivative of the
# quadrupole ratio modulus as a function of the Alfvenic Mach number
fig3 = plt.figure(3, figsize=(7,5), dpi = 300)

# Create an axis for the plot
ax9 = fig3.add_subplot(111)

# Plot the values of the standard deviation of the first derivative for 
# Christoph and Blakesley's simulations
plt.plot(alf_mach_arr_bbur, quad_deriv_std_arr_bbur[:,0], 'o', label = 'BB zLOS')
plt.plot(alf_mach_arr_bbur, quad_deriv_std_arr_bbur[:,1], 'o', label = 'BB yLOS')
plt.plot(alf_mach_arr_cfed, quad_deriv_std_arr_cfed[:,0], '^', label = 'CF xLOS')
plt.plot(alf_mach_arr_cfed, quad_deriv_std_arr_cfed[:,1], '^', label = 'CF yLOS')

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 8, numpoints=1)

# Add labels to the axes
plt.xlabel('Alfvenic Mach Number', fontsize = 18)
plt.ylabel('StDev First Deriv Quad Ratio', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'std_quad_deriv_alf_gam{}_{}_{}.eps'.format(gamma,\
	first_index, end_index), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#-------------------------------------------------------------------------------

# Now that all of the statistics have been calculated, print them out to the 
# screen. Loop over all of the lines of sight, and the different simulations,
# and print out results for the simulations
for j in range(2):
	# For this line of sight, loop over Christoph's simulations
	for i in range(len(spec_locs_cfed)):
		# Print out the value of the mean for this line of sight
		print "{} {} LOS Quad Deriv StDev: {}   Error: {}".format(sim_labels_cfed[i],\
		 line_o_sight_cfed[j], quad_deriv_std_arr_cfed[i,j], quad_deriv_std_err_arr_cfed[i,j])

# Loop over Blakesley's simulations
for j in range(2):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs_bbur)):
		# Print out the value of the mean for this line of sight
		print "{} {} LOS Quad Deriv StDev: {}   Error: {}".format(short_labels_bbur[i],\
		 line_o_sight_bbur[j], quad_deriv_std_arr_bbur[i,j], quad_deriv_std_err_arr_bbur[i,j])