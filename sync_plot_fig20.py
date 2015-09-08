#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the structure function slope and integrated quadrupole ratio  #
# for different simulations as a function of the final noise level of the      #
# image, for a fixed angular resolution.                                       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 8/8/2015                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, astropy.convolution for convolution functions, 
# scipy.stats for calculating statistical quantities,
# scipy.ndimage for smoothing and convolution
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy import stats
from scipy import ndimage

# Import the functions that calculate the structure and correlation functions
# using FFT, the function that calculates the radially averaged structure or 
# correlation functions, the function that calculates multipoles of 2D images,
# and the function that calculates the magnitude and argument of the quadrupole
# ratio. The function that converts Numpy arrays into FITS files is also 
# imported.
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio

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

# Create strings giving the simulation codes in terms of Mach numbers, for the
# low magnetic field simulations used to produce plots
low_B_short_M = ['Ms5.82Ma1.76', 'Ms2.14Ma1.86', 'Ms0.81Ma1.74', 'Ms0.47Ma1.72']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# high magnetic field simulations used to produce plots
high_B_short_M = ['Ms5.47Ma0.52', 'Ms2.23Ma0.67', 'Ms0.84Ma0.7', 'Ms0.47Ma0.65']

# Create strings giving the simulation codes in terms of sonic Mach numbers, for
# the low magnetic field simulations used to produce plots
low_B_short_M_latex = ['$M_s = 5.82$', '$M_s = 2.14$', '$M_s = 0.81$', '$M_s = 0.47$']

# Create strings giving the simulation codes in terms of sonic Mach numbers, for
# the high magnetic field simulations used to produce plots
high_B_short_M_latex = ['$M_s = 5.47$', '$M_s = 2.23$', '$M_s = 0.84$', '$M_s = 0.47$']

# Create an array of marker symbols, so that the plot for each gamma value has
# a different plot symbol
symbol_arr = ['o','^','s','*']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a variable that controls how many data points are being used for the
# free parameter
free_num = 25

# Create an array of values that will be used to determine the standard
# deviation of the Gaussian distribution from which noise values are 
# generated. The standard deviation will be calculated by multiplying the
# median synchrotron intensity by the values in this array.
iter_array = np.linspace(0.02, 0.7, free_num)

# Create an array that will hold the values for the noise level of the final
# synchrotron maps produced, in the same units as the generated noise. 
# Each row corresponds to a simulation, and each column corresponds to a 
# different noise value. There is one array for low magnetic field
# simulations, and another for high magnetic field simulations.
final_noise_low = np.zeros((len(low_B_short), len(iter_array)))
final_noise_high = np.zeros((len(high_B_short), len(iter_array)))

# Create a variable that represents the standard deviation of the 
# Gaussian used to smooth the synchrotron maps. Value is in pixels.
smooth_stdev = 1.3

# Create a variable representing the final angular resolution of
# the image after smoothing. The final resolution is calculated by 
# quadrature from the initial resolution (1 pixel) and the standard 
# deviation of the convolving Gaussian.
final_res = np.sqrt(1.0 + np.power(smooth_stdev,2.0))

# Print the final resolution to the screen
print 'The final resolution is {} pixels'.format(final_res)

# Create a label for the x-axis of the produced plot
#xlabel = 'Noise StandDev [frac median inten]'
xlabel = 'Noise-to-Signal Ratio'

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

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/'

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

	# Take into account an observing frequency of 1.4 GHz, by multiplying
	# the extracted synchrotron maps by a gamma dependent frequency factor
	sync_map_low_f = sync_map_low * np.power(1.4, -(gamma - 1))
	sync_map_high_f = sync_map_high * np.power(1.4, -(gamma - 1))

	# Print a message to the screen to show what simulation group is being used
	print 'Starting calculation for simulation group {}'.format(i)

	# Loop over the noise values to calculate the structure function slope and
	# integrated quadrupole ratio for the low and high magnetic field simulations
	for j in range(len(iter_array)):
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

		# Create a Gaussian kernel to use to smooth the synchrotron map,
		# using the given standard deviation
		gauss_kernel = Gaussian2DKernel(smooth_stdev)

		# Smooth the synchrotron maps to the required resolution by 
		# convolution with the above Gaussian kernel.
		sync_map_free_param_low = convolve_fft(sync_map_free_param_low,\
		 gauss_kernel, boundary = 'wrap')
		sync_map_free_param_high = convolve_fft(sync_map_free_param_high,\
		 gauss_kernel, boundary = 'wrap')

		# To plot against the final noise level, we need to perform some 
		# additional calculations
		
		# Start by smoothing the initial synchrotron intensity map to
		# the required resolution. (No noise added)
		sync_map_low_no_noise = convolve_fft(sync_map_low_f,\
		 gauss_kernel, boundary = 'wrap')
		sync_map_high_no_noise = convolve_fft(sync_map_high_f,\
		 gauss_kernel, boundary = 'wrap')

		# Subtract this smoothed synchrotron map (with no noise) from the
		# full map (noise added, then smoothed)
		noise_map_low = sync_map_free_param_low - sync_map_low_no_noise
		noise_map_high = sync_map_free_param_high - sync_map_high_no_noise

		# Calculate the standard deviation of the noise (in same units as
		# the intensity)
		stdev_final_noise_low = np.std(noise_map_low)
		stdev_final_noise_high = np.std(noise_map_high)

		# Express the calculated standard deviation as a fraction of the 
		# median synchrotron intensity of the map, and store the value in
		# the corresponding matrix
		final_noise_low[i,j] = stdev_final_noise_low / np.median(sync_map_low_f)
		final_noise_high[i,j] = stdev_final_noise_high / np.median(sync_map_high_f)

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
	plt.plot(final_noise_low[i], sf_low_arr[i], '-' + symbol_arr[i],\
	 label = r'{}'.format(low_B_short_M[i]))

# Force the legends to appear on the plot
plt.legend(loc = 1, fontsize = 10)

# Add a label to the y-axis
plt.ylabel('m', fontsize = 20)

# Set the x axis limits for the plot
ax1.set_xlim([np.min(final_noise_low), np.max(final_noise_low)])

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the 
# SF slope of high magnetic field simulations. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the high magnetic field simulations to produce plots for each simulation
for i in range(len(high_B_sims)):
	# Plot the SF slope for this simulation, against the observational effect
	plt.plot(final_noise_high[i], sf_high_arr[i], '-' + symbol_arr[i],\
	 label = r'{}'.format(high_B_short_M[i]))

# Force the legends to appear on the plot
plt.legend(loc = 1, fontsize = 10)

# Set the x axis limits for the plot
ax2.set_xlim([np.min(final_noise_high), np.max(final_noise_high)])

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
	plt.plot(final_noise_low[i], quad_low_arr[i], '-' + symbol_arr[i])

# Add a label to the y-axis
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Set the x axis limits for the plot
ax3.set_xlim([np.min(final_noise_low), np.max(final_noise_low)])

# Create an axis for the fourth subplot to be produced, which is for the 
# integrated quadrupole ratio of high magnetic field simulations. Make the x 
# axis limits the same as for the second plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the high magnetic field simulation to produce plots for each simulation
for i in range(len(high_B_sims)):
	# Plot the integrated quadrupole ratio for this simulation, against the 
	# observational effect
	plt.plot(final_noise_high[i], quad_high_arr[i], '-' + symbol_arr[i])

# Set the x axis limits for the plot
ax4.set_xlim([np.min(final_noise_high), np.max(final_noise_high)])

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

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig20_2.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()