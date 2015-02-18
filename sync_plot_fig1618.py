#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the structure function slope and integrated quadrupole ratio  #
# for different simulations as a function of an observational effect. Two      #
# plots are produced, looking at noise and angular resolution.                 #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 13/2/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, astropy.convolution for convolution functions, 
# scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
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
low_B_sims = ['b.1p.01_Oct_Burk/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/'] 

# Create strings giving the directories for the simulations produced with a 
# high magnetic field
high_B_sims = ['b1p.01_Oct_Burk/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/']

# Create strings giving the simulation codes, for the low magnetic field 
# simulations used to produce plots
low_B_short = ['b.1p.01', 'b.1p.1', 'b.1p.7', 'b.1p2']

# Create strings giving the simulation codes, for the high magnetic field
# simulations used to produce plots
high_B_short = ['b1p.01', 'b1p.1', 'b1p.7', 'b1p2']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a string that determines what observational effect will be studied
# String can be one of the following:
# noise - Study how statistics change as noise level is varied
# res - Study how statistics change as the spatial resolution is varied
obs_effect = 'res'

# Create a variable that controls how many data points are being used for the
# free parameter
free_num = 20

# Depending on what observational effect is being studied, create an array of 
# values over which we will iterate. This array represents the values of the 
# free parameter related to the observational effect 
if obs_effect == 'noise':
	# Create an array of values that will be used to determine the standard
	# deviation of the Gaussian distribution from which noise values are 
	# generated. The standard deviation will be calculated by multiplying the
	# median synchrotron intensity by the values in this array.
	iter_array = np.linspace(0.02, 0.5, free_num)

	# Create a label for the x-axis of plots that are made against noise
	# standard deviation
	xlabel = 'Noise StandDev [perc median inten]'

	# Create a string to be used in the titles of any plots that are made 
	# against noise standard deviation
	title_string = 'Noise StandDev'

	# Create a string to be used in legends involving spectral channel width
	leg_string = 'Noise = ' 
elif obs_effect == 'res':
	# Create an array of values that represent the standard deviation of the 
	# Gaussian used to smooth the synchrotron maps. All values are in pixels.
	iter_array = np.linspace(1.0, 50.0, free_num)

	# Create an array of values representing the final angular resolution of
	# the image after smoothing. The final resolution is calculated by 
	# quadrature from the initial resolution (1 pixel) and the standard 
	# deviation of the convolving Gaussian.
	final_res = np.sqrt(1.0 + np.power(iter_array,2.0))

	# Create a label for the x-axis of plots that are made against angular 
	# resolution
	xlabel = 'Angular Resolution [pixels]'

	# Create a string to be used in the titles of any plots that are made 
	# against angular resolution
	title_string = 'Angular Resolution'

	# Create a string to be used in legends involving angular resolution
	leg_string = 'AngRes = ' 

# Create a two dimensional array that will hold all of the structure function 
# slope values for the different low magnetic field simulations. The first index
# gives the simulation the second gives the strength of the observational effect
sf_low_arr = np.zeros((len(low_B_sims), len(iter_array)))

# Create a two dimensional array that will hold all of the structure function
# slope values for the different high magnetic field simulations. The first 
# index gives the simulation the second gives the strength of the observational 
# effect
sf_high_arr = np.zeros((len(high_B_sims), len(iter_array)))

# Create a two dimensional array that will hold all of the integrated quadrupole
# ratio values for the different low magnetic field simulations. The first index
# gives the simulation the second gives the strength of the observational effect
quad_low_arr = np.zeros((len(low_B_sims), len(iter_array)))

# Create a two dimensional array that will hold all of the integrated quadrupole
# ratio values for the different high magnetic field simulations. The first 
# index gives the simulation the second gives the strength of the observational 
# effect
quad_high_arr = np.zeros((len(high_B_sims), len(iter_array)))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for i in range(len(low_B_sims)):
	# Create a string for the full directory path to use in the calculation for
	# low and high magnetic field simulations
	data_loc_low = simul_loc + low_B_sims[i]
	data_loc_high = simul_loc + high_B_sims[i]

	# Open the FITS file that contains the simulated synchrotron intensity
	# map for this line of sight, for low and high magnetic fields
	sync_fits_low = fits.open(data_loc_low + 'synint_p1-4.fits')
	sync_fits_high = fits.open(data_loc_high + 'synint_p1-4.fits')

	# Extract the data for the simulated synchrotron intensities for the current
	# low and high magnetic field simulations
	sync_data_low = sync_fits_low[0].data
	sync_data_high = sync_fits_high[0].data

	# Extract the synchrotron intensity map for the value of gamma, for
	# low and high magnetic field simulations
	sync_map_low = sync_data_low[gam_index]
	sync_map_high = sync_data_high[gam_index]

	# Print a message to the screen to show what simulation group is being used
	print 'Starting calculation for simulation group {}'.format(i)

	# Loop over the values for the parameter related to the observational
	# effect, to calculate the structure function slope and integrated 
	# quadrupole ratio for the low and high magnetic field simulations
	for j in range(len(iter_array)):
		# Check to see which observational effect is being studied
		if obs_effect == 'noise':
			# In this case, we are taking into account the effect of noise in
			# the telescope. We start with an array of values that, when 
			# multiplied by the median intensity of the synchrotron map, give
			# the standard deviation of the Gaussian noise. 

			# Take into account an observing frequency of 1.4 GHz, by multiplying
			# the extracted synchrotron maps by a gamma dependent frequency factor
			sync_map_low_f = sync_map_low * np.power(1.4, -(gamma - 1))
			sync_map_high_f = sync_map_high * np.power(1.4, -(gamma - 1))

			# Calculate the standard deviation of the Gaussian noise that will 
			# affect the synchrotron maps. This needs to be done individually 
			# for low and high magnetic field simulations
			noise_stdev_low = iter_array[j] * np.median(sync_map_low_f)
			noise_stdev_high = iter_array[j] * np.median(sync_map_high_f)

			# Create an array of values that are randomly drawn from a Gaussian
			# distribution with the specified standard deviation. This 
			# represents the noise at each pixel of the image. 
			noise_matrix_low = np.random.normal(scale = noise_stdev_low,\
			 size = np.shape(sync_map_low))
			noise_matrix_high = np.random.normal(scale = noise_stdev_high,\
			 size = np.shape(sync_map_high))

			# Add the noise maps onto the synchrotron intensity maps, to produce
			# the mock 'observed' maps
			sync_map_free_param_low = sync_map_low_f + noise_matrix_low
			sync_map_free_param_high = sync_map_high_f + noise_matrix_high

		elif obs_effect == 'res':
			# In this case, we are taking into account the effect of spatial 
			# resolution. We start with an array of values that specifies the
			# standard deviation of the Gaussian to be used to smooth the data.

			# Take into account an observing frequency of 1.4 GHz, by multiplying
			# the extracted synchrotron maps by a gamma dependent frequency factor
			sync_map_low_f = sync_map_low * np.power(1.4, -(gamma - 1))
			sync_map_high_f = sync_map_high * np.power(1.4, -(gamma - 1))

			# Create a Gaussian kernel to use to smooth the synchrotron map,
			# using the given standard deviation
			gauss_kernel = Gaussian2DKernel(iter_array[j])

			# Smooth the synchrotron maps to the required resolution by 
			# convolution with the above Gaussian kernel.
			sync_map_free_param_low = convolve_fft(sync_map_low_f, gauss_kernel, boundary = 'wrap')
			sync_map_free_param_high = convolve_fft(sync_map_high_f, gauss_kernel, boundary = 'wrap')

			# Replace the array of standard deviations with the array of final
			# resolutions, so that the final resolutions are used in all plots
			iter_array[j] = final_res[j]

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity maps, for the low and high magnetic field simulations. Note 
		# that no_fluct = True is set, because we are not subtracting the mean
		# from the synchrotron maps before calculating the structure function.
		strfn_low = sf_fft(sync_map_free_param_low, no_fluct = True)
		strfn_high = sf_fft(sync_map_free_param_high, no_fluct = True)

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

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn_low = sf_fft(sync_map_free_param_low, no_fluct = True, normalise = True)
		norm_strfn_high = sf_fft(sync_map_free_param_high, no_fluct = True, normalise = True)

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

	# Close the FITS files, now that we are finished using them, to save
	# memory
	sync_fits_low.close()
	sync_fits_high.close()

	# Print a message to show that the calculation has finished successfully
	# for this simulation group
	print 'All statistics calculated for simulation group {}'.format(i)

# When the code reaches this point, the statistics have been saved for every 
# simulation, so start making the final plots.

# ------------------- Plots of SF slope and quadrupole ratio -------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will be SF slope, and
# the bottom row will be quadrupole ratio. The left column will be low magnetic 
# field simulations, and the right column will be high magnetic field 
# simulations.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the SF slope 
# of low magnetic field simulations
ax1 = fig.add_subplot(221)

# Loop over the low magnetic field simulations to produce plots for each simulation
for i in range(len(low_B_sims)):
	# Plot the SF slope for this simulation, against the observational effect
	plt.plot(iter_array, sf_low_arr[i], '-o', label = '{}'.format(low_B_short[i]))

# Force the legends to appear on the plot
plt.legend(loc = 4, fontsize = 10)

# Add a label to the y-axis
plt.ylabel('m', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the 
# SF slope of high magnetic field simulations. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the high magnetic field simulations to produce plots for each simulation
for i in range(len(high_B_sims)):
	# Plot the SF slope for this simulation, against the observational effect
	plt.plot(iter_array, sf_high_arr[i], '-o', label = '{}'.format(high_B_short[i]))

# Force the legends to appear on the plot
plt.legend(loc = 4, fontsize = 10)

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the 
# integrated quadrupole ratio of low magnetic field simulations. Make the x axis
# limits the same as for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Loop over the low magnetic field simulations to produce plots for each simulation
for i in range(len(low_B_sims)):
	# Plot the integrated quadrupole ratio for this simulation, against the
	# observational effect
	plt.plot(iter_array, quad_low_arr[i], '-o')

# Add a label to the y-axis
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for the 
# integrated quadrupole ratio of high magnetic field simulations. Make the x 
# axis limits the same as for the second plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the high magnetic field simulation to produce plots for each simulation
for i in range(len(high_B_sims)):
	# Plot the integrated quadrupole ratio for this simulation, against the 
	# observational effect
	plt.plot(iter_array, quad_high_arr[i], '-o')

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, xlabel, ha = 'center', va = 'bottom', fontsize = 20)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) m, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) m, high B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Quad, low B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Quad, high B', fontsize = 18)

# Depending on the observational effect being studied, change the filename used
# to save the figure
if obs_effect == 'noise':
	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Publication_Plots/fig18.eps', format = 'eps')
elif obs_effect == 'res':
	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Publication_Plots/fig16.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()