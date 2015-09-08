#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the skewness, kurtosis, structure function slope, and         #
# integrated quadrupole ratio as a function of gamma for simulations with low  #
# and high magnetic field. Plots of these statistics as a function of gamma    #
# are then saved.                                                              #
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

# Create strings giving the directories for the simulations produced with a 
# low magnetic field
# low_B_sims = ['b.1p.01_Oct_Burk/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
# 'b.1p2_Aug_Burk/'] 
low_B_sims = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/',\
 'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/'] 

# Create strings giving the directories for the simulations produced with a 
# high magnetic field
# high_B_sims = ['b1p.01_Oct_Burk/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
# 'b1p2_Aug_Burk/']
high_B_sims = ['c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
 'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/'] 

# Create strings giving the simulation codes, for the low magnetic field 
# simulations used to produce plots
# low_B_short = ['b.1p.01', 'b.1p.1', 'b.1p.7', 'b.1p2']
low_B_short = ['b.1p.0049', 'b.1p.0077', 'b.1p.01', 'b.1p.025', 'b.1p.05',\
 'b.1p.1', 'b.1p.7', 'b.1p2']

# Create strings giving the simulation codes, for the high magnetic field
# simulations used to produce plots
# high_B_short = ['b1p.01', 'b1p.1', 'b1p.7', 'b1p2']
high_B_short = ['b1p.0049', 'b1p.0077', 'b1p.01', 'b1p.025', 'b1p.05',\
 'b1p.1', 'b1p.7', 'b1p2']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# low magnetic field simulations used to produce plots
low_B_short_M = ['Ms8.85Ma1.41', 'Ms5.43Ma1.77', 'Ms5.82Ma1.76','Ms3.72Ma1.51',\
'Ms2.75Ma1.69', 'Ms2.14Ma1.86', 'Ms0.81Ma1.74', 'Ms0.47Ma1.72']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# high magnetic field simulations used to produce plots
high_B_short_M = ['Ms7.56Ma0.5','Ms6.14Ma0.5', 'Ms5.47Ma0.52', 'Ms3.64Ma0.55',\
'Ms2.69Ma0.58', 'Ms2.23Ma0.67', 'Ms0.84Ma0.7', 'Ms0.47Ma0.65']

# Create a list that holds the colours to use for each simulation
color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', (1,0.41,0)]

# Create a list that holds the marker type to use for each simulation
symbol_list = ['o', '^', 's', 'p', '*', '+', 'x', 'D']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a two dimensional array that will hold all of the skew values for the
# different low magnetic field simulations. The first index gives the simulation
# the second gives the gamma value used.
skew_low_arr = np.zeros((len(low_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the skew values for the
# different high magnetic field simulations. The first index gives the simulation
# the second gives the gamma value used.
skew_high_arr = np.zeros((len(high_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the kurtosis values for 
# the different low magnetic field simulations. The first index gives the 
# simulation the second gives the gamma value used.
kurt_low_arr = np.zeros((len(low_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the kurtosis values for
# the different high magnetic field simulations. The first index gives the
# simulation the second gives the gamma value used.
kurt_high_arr = np.zeros((len(high_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the structure function 
# slope values for the different low magnetic field simulations. The first index
# gives the simulation the second gives the gamma value used.
sf_low_arr = np.zeros((len(low_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the structure function
# slope values for the different high magnetic field simulations. The first 
# index gives the simulation the second gives the gamma value used.
sf_high_arr = np.zeros((len(high_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the values for the 
# residuals of the linear fit to the structure function, for low magnetic field
# simulations. The first index gives the simulation the second gives the gamma
# value used.
resid_low_arr = np.zeros((len(low_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the values for the 
# residuals of the linear fit to the structure function, for high magnetic field
# simulations. The first index gives the simulation the second gives the gamma
# value used.
resid_high_arr = np.zeros((len(high_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the integrated quadrupole
# ratio values for the different low magnetic field simulations. The first index
# gives the simulation the second gives the gamma value used.
quad_low_arr = np.zeros((len(low_B_sims), len(gamma_arr)))

# Create a two dimensional array that will hold all of the integrated quadrupole
# ratio values for the different high magnetic field simulations. The first 
# index gives the simulation the second gives the gamma value used.
quad_high_arr = np.zeros((len(high_B_sims), len(gamma_arr)))

# Loop over the different simulations that we are using to make the plots
for i in range(len(low_B_sims)):
	# Create a string for the full directory path to use in the calculation for
	# low and high magnetic field simulations
	data_loc_low = simul_loc + low_B_sims[i]
	data_loc_high = simul_loc + high_B_sims[i]
	 
	# Open the FITS file that contains the synchrotron intensity maps for the
	# current low and high magnetic field simulations
	sync_fits_low = fits.open(data_loc_low + 'synint_p1-4.fits')
	sync_fits_high = fits.open(data_loc_high + 'synint_p1-4.fits')

	# Extract the data for the simulated synchrotron intensities for the current
	# low and high magnetic field simulations
	sync_data_low = sync_fits_low[0].data
	sync_data_high = sync_fits_high[0].data

	# Print a message to the screen to show that the data has been loaded
	print 'Synchrotron intensity loaded successfully'

	# Loop over the gamma values, to calculate the skewness, kurtosis, structure
	# function slope and integrated quadrupole ratio for the low and high
	# magnetic field simulations
	for j in range(len(gamma_arr)):
		# Extract the synchrotron intensity map for this value of gamma, for
		# low and high magnetic field simulations
		sync_map_low = sync_data_low[j]
		sync_map_high = sync_data_high[j]

		# Flatten the synchrotron intensity maps for this value of gamma, for
		# low and high magnetic field simulations
		flat_sync_low = sync_map_low.flatten()
		flat_sync_high = sync_map_high.flatten()

		# Calculate the biased skewness of the synchrotron intensity maps, for
		# low and high magnetic field simulations, and store the results in the
		# corresponding array.
		skew_low_arr[i,j] = stats.skew(flat_sync_low)
		skew_high_arr[i,j] = stats.skew(flat_sync_high)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# maps, for low and high magnetic field simulations, and store the 
		# results in the corresponding array.
		kurt_low_arr[i,j] = stats.kurtosis(flat_sync_low)
		kurt_high_arr[i,j] = stats.kurtosis(flat_sync_high)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity maps, for the low and high magnetic field simulations. Note 
		# that no_fluct = True is set, because we are not subtracting the mean
		# from the synchrotron maps before calculating the structure function.
		strfn_low = sf_fft(sync_map_low, no_fluct = True)
		strfn_high = sf_fft(sync_map_high, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins, for low and high magnetic field simulations.
		rad_sf_low = sfr(strfn_low, num_bins, verbose = False)
		rad_sf_high = sfr(strfn_high, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function for low 
		# and high magnetic field simulations
		sf_low = rad_sf_low[1]
		sf_high = rad_sf_high[1]

		# Extract the radius values used to calculate this structure function,
		# for low and high magnetic field simulations.
		sf_rad_arr_low = rad_sf_low[0]
		sf_rad_arr_high = rad_sf_high[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. Perform a linear fit for the low magnetic 
		# field simulation
		spec_ind_data_low = np.polyfit(np.log10(\
			sf_rad_arr_low[0:np.ceil(num_bins/3.0)]),\
			np.log10(sf_low[0:np.ceil(num_bins/3.0)]), 1, full = True)
		# Perform a linear fit for the high magnetic field simulation
		spec_ind_data_high = np.polyfit(np.log10(\
			sf_rad_arr_high[0:np.ceil(num_bins/3.0)]),\
			np.log10(sf_high[0:np.ceil(num_bins/3.0)]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit, for low and
		# high magnetic field simulations
		coeff_low = spec_ind_data_low[0]
		coeff_high = spec_ind_data_high[0]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array, for low and high magnetic field 
		# simulations
		sf_low_arr[i,j] = coeff_low[0]-1.0
		sf_high_arr[i,j] = coeff_high[0]-1.0

		# Enter the value of the residuals of the linear fit to the structure
		# function to the corresponding array, for low and high magnetic field
		# simulations
		resid_low_arr[i,j] = spec_ind_data_low[1]
		resid_high_arr[i,j] = spec_ind_data_high[1]

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn_low = sf_fft(sync_map_low, no_fluct = True, normalise = True)
		norm_strfn_high = sf_fft(sync_map_high, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image. This is done for low and high 
		# magnetic field simulations
		norm_strfn_low = np.fft.fftshift(norm_strfn_low)
		norm_strfn_high = np.fft.fftshift(norm_strfn_high)

		# Calculate the magnitude and argument of the quadrupole ratio, for 
		# low and high magnetic field simulations
		quad_mod_low, quad_arg_low, quad_rad_low = calc_quad_ratio(norm_strfn_low, num_bins)
		quad_mod_high, quad_arg_high, quad_rad_high = calc_quad_ratio(norm_strfn_high, num_bins)

		# Integrate the magnitude of the quadrupole / monopole ratio from 
		# one sixth of the way along the radial separation bins, until three 
		# quarters of the way along the radial separation bins. This integration
		# is performed with respect to log separation (i.e. I am ignoring the 
		# fact that the points are equally separated in log space, to calculate 
		# the area under the quadrupole / monopole ratio plot when the x axis 
		# is scaled logarithmically). I normalise the value that is returned by 
		# dividing by the number of increments in log radial separation used in 
		# the calculation. This is done for low and high magnetic field 
		# simulations
		quad_low_arr[i,j] = np.trapz(quad_mod_low[np.floor(num_bins/6.0):\
			3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
			 - np.floor(num_bins/6.0))
		quad_high_arr[i,j] = np.trapz(quad_mod_high[np.floor(num_bins/6.0):\
			3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
			 - np.floor(num_bins/6.0))

	# Print a message to show that the calculation has finished successfully
	# for these simulation
	print 'All statistics calculated for simulation group {}'.format(i)

# When the code reaches this point, the statistics have been saved for every 
# simulation, and every value of gamma, so start making the final plots.

# ---------------------- Plots of skewness and kurtosis ------------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will be skewness, and
# the bottom row will be kurtosis. The left column will be low magnetic field
# simulations, and the right column will be high magnetic field simulations.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the skewness 
# of low magnetic field simulations
ax1 = fig.add_subplot(221)

# Loop over the low magnetic field simulations to produce plots for each simulation
for i in [2,5,6,7]:
	# Plot the skewness for this simulation, against gamma
	plt.plot(gamma_arr, skew_low_arr[i], '-' + symbol_list[i],\
		label = '{}'.format(low_B_short_M[i]))

# Force the legends to appear on the plot
plt.legend(loc = 2, fontsize = 10)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the 
# skewness of high magnetic field simulations. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the high magnetic field simulations to produce plots for each simulation
for i in [2,5,6,7]:
	# Plot the skewness for this simulation, against gamma
	plt.plot(gamma_arr, skew_high_arr[i], '-' + symbol_list[i],\
		label = '{}'.format(high_B_short_M[i]))

# Force the legends to appear on the plot
plt.legend(loc = 2, fontsize = 10)

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the kurtosis 
# of low magnetic field simulations. Make the x axis limits the same as
# for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Loop over the low magnetic field simulations to produce plots for each simulation
for i in [2,5,6,7]:
	# Plot the kurtosis for this simulation, against gamma
	plt.plot(gamma_arr, kurt_low_arr[i], '-' + symbol_list[i])

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for the 
# kurtosis of high magnetic field simulations. Make the x axis limits the same 
# as for the second plot
ax4 = fig.add_subplot(224, sharex = ax2)

# Loop over the high magnetic field simulation to produce plots for each simulation
for i in [2,5,6,7]:
	# Plot the kurtosis for this simulation, against gamma
	plt.plot(gamma_arr, kurt_high_arr[i], '-' + symbol_list[i])

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Gamma', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) Skew, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Skew, high B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Kurt, low B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Kurt, high B', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig8a.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

# #-------------------------- Structure Function Slope ---------------------------

# Here we want to produce one plot with two subplots. There should be one row
# of subplots, with two subplots in the row. The plot on the left should be
# the structure function slope as a function of gamma for low magnetic field 
# simulations, and the plot on the right should be the structure function slope
# as a function of gamma for high magnetic field simulations.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,5), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the 
# structure function slope of low magnetic field simulations
ax1 = fig.add_subplot(221)

# Loop over the simulations to plot structure function slope vs gamma
for i in [2,5,6,7]:
	# Plot the structure function slope as a function of gamma for this simulation
	plt.plot(gamma_arr, sf_low_arr[i], '-' + symbol_list[i],\
		label = '{}'.format(low_B_short_M[i]))

# Force the legend to appear on the plot
plt.legend(loc = 3, fontsize = 9)

# Add a y-axis label to the plot
plt.ylabel('m', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the 
# structure function slope of high magnetic field simulations.
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the simulations to plot the structure function slope vs gamma
for i in [2,5,6,7]:
	# Plot the structure function slope as a function of gamma for this simulation
	plt.plot(gamma_arr, sf_high_arr[i], '-' + symbol_list[i],\
		label = '{}'.format(high_B_short_M[i]))

# Force the legend to appear on the plot
plt.legend(loc = 3, fontsize = 9)

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the residual 
# of the linear fit to the structure functions, for low magnetic field
# simulations. Make the x axis limits the same as for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Loop over the low magnetic field simulations to produce plots for each simulation
for i in [2,5,6,7]:
	# Plot the residuals for this simulation, against gamma
	plt.plot(gamma_arr, resid_low_arr[i], '-' + symbol_list[i])

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for the 
# residuals of the linear fit to the structure functions, for high magnetic
# field simulations. Make the x axis limits the same as for the second plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the high magnetic field simulation to produce plots for each simulation
for i in [2,5,6,7]:
	# Plot the residuals for this simulation, against gamma
	plt.plot(gamma_arr, resid_high_arr[i], '-' + symbol_list[i])

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Gamma', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) m, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) m, high B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Residuals, low B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Residuals, high B', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig8b.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#----------------------------- Quadrupole Ratios -------------------------------

# Here we want to produce one plot with two subplots. There should be one row
# of subplots, with two subplots in the row. The plot on the left should be
# the integrated quadrupole ratio as a function of gamma for low magnetic field 
# simulations, and the plot on the right should be the integrated quadrupole
# ratio as a function of gamma for high magnetic field simulations.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,5), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the 
# integrated quadrupole ratio of low magnetic field simulations
ax1 = fig.add_subplot(121)

# Loop over the simulations to plot integrated quadrupole ratio vs gamma
for i in range(len(low_B_sims)):
	# Plot the integrated quadrupole ratio as a function of gamma for this simulation
	plt.plot(gamma_arr, quad_low_arr[i], '-' + symbol_list[i],\
		color = color_list[i], label = '{}'.format(low_B_short_M[i]))

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 10)

# Add a y-axis label to the plot
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Create an axis for the second subplot to be produced, which is for the 
# integrated quadrupole ratio of high magnetic field simulations. Make the y axis
# limits the same as for the low magnetic field plot
ax2 = fig.add_subplot(122, sharey = ax1)

# Loop over the simulations to plot the integrated quadrupole ratio vs gamma
for i in range(len(high_B_sims)):
	# Plot the integrated quadrupole ratio as a function of gamma for this simulation
	plt.plot(gamma_arr, quad_high_arr[i], '-' + symbol_list[i],\
		color = color_list[i], label = '{}'.format(high_B_short_M[i]))

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Gamma', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3, fontsize = 10)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.15, 0.95, 'a) Quad Ratio, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.58, 0.95, 'b) Quad Ratio, high B', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig8d.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()