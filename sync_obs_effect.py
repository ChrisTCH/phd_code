#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the skewness, kurtosis, radially averaged        #
# structure function, and quadrupole/monopole ratio of synchrotron intensity   #
# maps that are influenced by observational effects. Each of these quantities  #
# is plotted against the sonic and Alfvenic Mach numbers, to see which         #
# quantities are sensitive tracers of the sonic and Alfvenic Mach numbers.     #
# Quantities are also plotted against the free parameters of the observational #
# effects, to directly show how observations affect measured statistics.       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 12/12/2014                                                       #
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
from mat2FITS_Image import mat2FITS_Image

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
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a string that determines what observational effect will be studied
# String can be one of the following:
# spec - Study how statistics change as spectral resolution is varied
# noise - Study how statistics change as noise level is varied
# res - Study how statistics change as the spatial resolution is varied
obs_effect = 'res'

# Create a variable that controls how many data points are being used for the
# free parameter
free_num = 20

# Create a variable that controls how many sub-channels are used if we are 
# studying the effect of spectral resolution.
sub_channels = 20

# Depending on what observational effect is being studied, create an array of 
# values over which we will iterate. This array represents the values of the 
# free parameter related to the observational effect 
if obs_effect == 'spec':
	# Create an array of spectral channel width values (in MHz), over which
	# to iterate. Each value represents the width of a spectral channel that 
	# is centred on 1.4 GHz.
	iter_array = np.linspace(0.1, 10, free_num)

	# Create a label for the x-axis of plots that are made against spectral
	# channel width
	xlabel = 'Spectral Resolution [MHz]'

	# Create a string to be used in the titles of any plots that are made 
	# against spectral channel width
	title_string = 'Spectral Resolution'

	# Create a string to be used in legends involving spectral channel width
	leg_string = 'SpecRes = ' 
elif obs_effect == 'noise':
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

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of the free parameter related to the observational effect 
# being studied. Each row corresponds to a value of the free parameter, and each 
# column corresponds to a simulation. There is one array for a line of sight
# along the z axis, and another for a line of sight along the x axis.
# NOTE: We will calculate the biased skewness
skew_z_arr = np.zeros((len(iter_array),len(simul_arr)))
skew_x_arr = np.zeros((len(iter_array),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of the free parameter related to the observational effect 
# being studied. Each row corresponds to a value of the free parameter, and each 
# column corresponds to a simulation. There is one array for a line of sight
# along the z axis, and another for a line of sight along the x axis.
# NOTE: We will calculate the biased Fisher kurtosis
kurt_z_arr = np.zeros((len(iter_array),len(simul_arr)))
kurt_x_arr = np.zeros((len(iter_array),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of the free parameter related
# to the observational effect being studied. Each row corresponds to a value of 
# the free parameter, and each column corresponds to a simulation. There is one 
# array for a line of sight along the z axis, and another for a line of sight
# along the x axis.
m_z_arr = np.zeros((len(iter_array),len(simul_arr)))
m_x_arr = np.zeros((len(iter_array),len(simul_arr)))

# Create an empty array, where each entry specifies the residuals of the linear
# fit to the structure function of the synchrotron intensity image, of the 
# corresponding simulation, for a particular value of the free parameter related
# to the observational effect being studied. Each row corresponds to a value of 
# the free parameter, and each column corresponds to a simulation. There is one 
# array for a line of sight along the z axis, and another for a line of sight 
# along the x axis.
residual_z_arr = np.zeros((len(iter_array),len(simul_arr)))
residual_x_arr = np.zeros((len(iter_array),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity 
# image, for the corresponding simulation, for a particular value of the free 
# parameter related to the observational effect being studied. Each row 
# corresponds to a value of the free parameter, and each column corresponds to a
# simulation. There is one array for a line of sight along the z axis, and 
# another for a line of sight along the x axis.
int_quad_z_arr = np.zeros((len(iter_array),len(simul_arr)))
int_quad_x_arr = np.zeros((len(iter_array),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated magnitude of 
# the quadrupole/monopole ratio of the synchrotron intensity image at a 
# particular radial separation, for the corresponding simulation, for a 
# particular value of the free parameter related to the observational effect 
# being studied. Each row corresponds to a value of the free parameter, and each
# column corresponds to a simulation. There is one array for a line of sight 
# along the z axis, and another for a line of sight along the x axis.
quad_point_z_arr = np.zeros((len(iter_array),len(simul_arr)))
quad_point_x_arr = np.zeros((len(iter_array),len(simul_arr)))

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/'

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS files that contain the simulated synchrotron intensity maps
	# for lines of sight along the z axis and x axis
	sync_fits_z = fits.open(data_loc + 'synint_p1-4.fits')
	sync_fits_x = fits.open(data_loc + 'synint_p1-4x.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensities observed for different values of gamma, the power
	# law index of the cosmic ray electrons emitting the synchrotron emission.
	sync_data_z = sync_fits_z[0].data
	sync_data_x = sync_fits_x[0].data

	# Extract the synchrotron intensity map for the value of gamma, for
	# lines of sight along the x and z axes
	sync_map_z = sync_data_z[gam_index]
	sync_map_x = sync_data_x[gam_index]

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Create empty arrays, that will store the synchrotron intensity maps
	# produced. Each slice of these arrays corresponds to a different value
	# of the free parameter being studied. There is one array for a line of 
	# sight along the z axis, and another for a line of sight along the x axis.
	sync_param_z = np.zeros((free_num, np.shape(sync_map_z)[0], \
		np.shape(sync_map_z)[1]))
	sync_param_x = np.zeros((free_num, np.shape(sync_map_x)[0], \
		np.shape(sync_map_x)[1]))

	# Loop over the various values of the free parameter related to the 
	# observational effect being studied, to calculate the various statistics
	# for the synchrotron map observed for each value of the free parameter
	for i in range(len(iter_array)):
		# Check to see which observational effect is being studied
		if obs_effect == 'spec':
			# In this case, we are studying the effect of spectral resolution,
			# so we start with an array of channel widths in MHz, and need to 
			# convert this into the observing frequencies of sub-channels in
			# GHz. 

			# Construct an array of the sub-channel locations in MHz, specified
			# relative to the central observing frequency.
			sub_channel_loc = np.linspace(-iter_array[i]/2.0, iter_array[i]/2.0\
				, sub_channels)

			# Construct an array of the sub-channel locations in GHz, then raise
			# this to a power related to gamma. This is done so that the 
			# produced synchrotron intensity maps scale correctly with frequency
			sub_channel_GHz = np.power(1.4 + 1e-3 * sub_channel_loc,-(gamma- 1))

			# Construct a 3D array, where each slice gives the synchrotron map
			# observed in that sub-channel. There is one array for a line of 
			# sight along the z axis, and one for a line of sight along the x
			# axis.
			sync_maps_sub_chan_z = np.tile(sync_map_z, (sub_channels,1,1))\
			 * np.transpose(np.tile(sub_channel_GHz, (np.shape(sync_map_z)[0]\
			 	, np.shape(sync_map_z)[1], 1 )))
			sync_maps_sub_chan_x = np.tile(sync_map_x, (sub_channels,1,1))\
			 * np.transpose(np.tile(sub_channel_GHz, (np.shape(sync_map_x)[0]\
			 	, np.shape(sync_map_x)[1], 1 )))

			# Integrate the synchrotron maps measured in the sub-channels over
			# frequency (the third axis, first index), and normalise by the 
			# number of sub-channels used in the calculation. This is performed
			# for lines of sight along the z and x axes
			sync_map_free_param_z = np.mean(sync_maps_sub_chan_z, dtype = np.float64, axis = 0)
			# np.trapz(sync_maps_sub_chan_z, dx = \
			# 	sub_channel_GHz[1] - sub_channel_GHz[0],axis = 0) / sub_channels
			sync_map_free_param_x = np.mean(sync_maps_sub_chan_x, dtype = np.float64, axis = 0)
			# np.trapz(sync_maps_sub_chan_x, dx = \
			# 	sub_channel_GHz[1] - sub_channel_GHz[0],axis = 0) / sub_channels

		elif obs_effect == 'noise':
			# In this case, we are taking into account the effect of noise in
			# the telescope. We start with an array of values that, when 
			# multiplied by the median intensity of the synchrotron map, give
			# the standard deviation of the Gaussian noise. 

			# Take into account an observing frequency of 1.4 GHz, by multiplying
			# the extracted synchrotron maps by a gamma dependent frequency factor
			sync_map_z_f = sync_map_z * np.power(1.4, -(gamma - 1))
			sync_map_x_f = sync_map_x * np.power(1.4, -(gamma - 1))

			# Calculate the standard deviation of the Gaussian noise that will 
			# affect the synchrotron maps. This needs to be done individually 
			# for lines of sight along the z and x axes, because of the lines of
			# sight have different intensity maps.
			noise_stdev_z = iter_array[i] * np.median(sync_map_z_f)
			noise_stdev_x = iter_array[i] * np.median(sync_map_x_f)

			# Create an array of values that are randomly drawn from a Gaussian
			# distribution with the specified standard deviation. This 
			# represents the noise at each pixel of the image. 
			noise_matrix_z = np.random.normal(scale = noise_stdev_z,\
			 size = np.shape(sync_map_z))
			noise_matrix_x = np.random.normal(scale = noise_stdev_x,\
			 size = np.shape(sync_map_x))

			# Add the noise maps onto the synchrotron intensity maps, to produce
			# the mock 'observed' maps
			sync_map_free_param_z = sync_map_z_f + noise_matrix_z
			sync_map_free_param_x = sync_map_x_f + noise_matrix_x

		elif obs_effect == 'res':
			# In this case, we are taking into account the effect of spatial 
			# resolution. We start with an array of values that specifies the
			# standard deviation of the Gaussian to be used to smooth the data.

			# Take into account an observing frequency of 1.4 GHz, by multiplying
			# the extracted synchrotron maps by a gamma dependent frequency factor
			sync_map_z_f = sync_map_z * np.power(1.4, -(gamma - 1))
			sync_map_x_f = sync_map_x * np.power(1.4, -(gamma - 1))

			# Create a Gaussian kernel to use to smooth the synchrotron map,
			# using the given standard deviation
			gauss_kernel = Gaussian2DKernel(iter_array[i])

			# Smooth the synchrotron maps to the required resolution by 
			# convolution with the above Gaussian kernel.
			sync_map_free_param_z = convolve_fft(sync_map_z_f, gauss_kernel, boundary = 'wrap')
			sync_map_free_param_x = convolve_fft(sync_map_x_f, gauss_kernel, boundary = 'wrap')

			# Replace the array of standard deviations with the array of final
			# resolutions, so that the final resolutions are used in all plots
			iter_array[i] = final_res[i]

			# # Create empty arrays that will contain the final smoothed image.
			# # This step is required to ensure that the smooth is as accurate as
			# # possible
			# sync_map_free_param_z = np.zeros(np.shape(sync_map_z), dtype =\
			#  np.float64)
			# sync_map_free_param_x = np.zeros(np.shape(sync_map_x), dtype =\
			#  np.float64)

			# # Smooth the synchrotron maps to the required resolution by 
			# # convolution. The results for lines of sight along the z and x axes
			# # are automatically stored in the arrays that were created.
			# ndimage.filters.gaussian_filter(sync_map_z,\
			# 	sigma = iter_array[i], output = sync_map_free_param_z)
			# ndimage.filters.gaussian_filter(sync_map_x,\
			# 	sigma = iter_array[i], output = sync_map_free_param_x)

		# Now that the synchrotron map has been produced for this value of the
		# free parameter, store it in the array that will hold all of the
		# produced synchrotron maps
		sync_param_z[i] = sync_map_free_param_z
		sync_param_x[i] = sync_map_free_param_x

		# Flatten the synchrotron intensity maps for this value of gamma, for
		# lines of sight along the x and z axes
		flat_sync_z = sync_map_free_param_z.flatten()
		flat_sync_x = sync_map_free_param_x.flatten()

		# Calculate the biased skewness of the synchrotron intensity maps, for
		# lines of sight along the x and z axes, and store the results in the
		# corresponding array.
		skew_z_arr[i,j] = stats.skew(flat_sync_z)
		skew_x_arr[i,j] = stats.skew(flat_sync_x)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# maps, for lines of sight along the x and z axes, and store the results
		# in the corresponding array.
		kurt_z_arr[i,j] = stats.kurtosis(flat_sync_z)
		kurt_x_arr[i,j] = stats.kurtosis(flat_sync_x)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity maps, for the lines of sight along the x and z axes. Note 
		# that no_fluct = True is set, because we are not subtracting the mean
		# from the synchrotron maps before calculating the structure function.
		strfn_z = sf_fft(sync_map_free_param_z, no_fluct = True)
		strfn_x = sf_fft(sync_map_free_param_x, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins, for lines of sight along the x and z axes.
		rad_sf_z = sfr(strfn_z, num_bins, verbose = False)
		rad_sf_x = sfr(strfn_x, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function for lines
		# of sight along the x and z axes.
		sf_z = rad_sf_z[1]
		sf_x = rad_sf_x[1]

		# Extract the radius values used to calculate this structure function,
		# for line os sight along the x and z axes.
		sf_rad_arr_z = rad_sf_z[0]
		sf_rad_arr_x = rad_sf_x[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. Perform a linear fit for a line
		# of sight along the z axis.
		spec_ind_data_z = np.polyfit(np.log10(\
			sf_rad_arr_z[0:np.ceil(num_bins/3.0)]),\
			np.log10(sf_z[0:np.ceil(num_bins/3.0)]), 1, full = True)
		# Perform a linear fit for a line of sight along the x axis
		spec_ind_data_x = np.polyfit(np.log10(\
			sf_rad_arr_x[0:np.ceil(num_bins/3.0)]),\
			np.log10(sf_x[0:np.ceil(num_bins/3.0)]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit, for lines
		# of sight along the x and z axes
		coeff_z = spec_ind_data_z[0]
		coeff_x = spec_ind_data_x[0]

		# Extract the sum of the residuals from the polynomial fit, for lines
		# of sight along the x and z axes
		residual_z_arr[i,j] = spec_ind_data_z[1]
		residual_x_arr[i,j] = spec_ind_data_x[1]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array, for lines of sight along the x and z
		# axes
		m_z_arr[i,j] = coeff_z[0]-1.0
		m_x_arr[i,j] = coeff_x[0]-1.0

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn_z = sf_fft(sync_map_free_param_z, no_fluct = True, normalise = True)
		norm_strfn_x = sf_fft(sync_map_free_param_x, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image. This is done for lines of sight
		# along the x and z axes
		norm_strfn_z = np.fft.fftshift(norm_strfn_z)
		norm_strfn_x = np.fft.fftshift(norm_strfn_x)

		# Calculate the magnitude and argument of the quadrupole ratio, for 
		# lines of sight along the x and z axes.
		quad_mod_z, quad_arg_z, quad_rad_z = calc_quad_ratio(norm_strfn_z, num_bins)
		quad_mod_x, quad_arg_x, quad_rad_x = calc_quad_ratio(norm_strfn_x, num_bins)

		# Find the value of the magnitude of the quadrupole/monopole ratio for a
		# radial separation that is one third of the way along the radial 
		# separation range that is probed, and store it in the corresponding 
		# array. This is done for lines of sight along the x and z axes.
		quad_point_z_arr[i,j] = quad_mod_z[np.floor(num_bins/3.0)]
		quad_point_x_arr[i,j] = quad_mod_x[np.floor(num_bins/3.0)]

		# Integrate the magnitude of the quadrupole/monopole ratio from one 
		# sixth of the way along the radial separation bins, until three 
		# quarters of the way along the radial separation bins. This integration
		# is performed with respect to log separation (i.e. I am ignoring the 
		# fact that the points are equally separated in log space, to calculate 
		# the area under the quadrupole / monopole ratio plot when the x axis is
		# scaled logarithmically). I normalise the value that is returned by 
		# dividing by the number of increments in log radial separation used in 
		# the calculation. This is done for lines of sight along the x and z axes.
		int_quad_z_arr[i,j] = np.trapz(quad_mod_z[np.floor(num_bins/6.0):\
			3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
			 - np.floor(num_bins/6.0))
		int_quad_x_arr[i,j] = np.trapz(quad_mod_x[np.floor(num_bins/6.0):\
			3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
			 - np.floor(num_bins/6.0))

		# At this point, all of the statistics that need to be calculated for
		# every value of gamma have been calculated.

	# # Convert the arrays that show the synchrotron map for each value of the 
	# # free parameter into a FITS file, and save it.
	# mat2FITS_Image(sync_param_z, filename = save_loc + short_simul[j] + '_' + obs_effect + '_z.fits')
	# mat2FITS_Image(sync_param_x, filename = save_loc + short_simul[j] + '_' + obs_effect + '_x.fits')

	# Close the fits files, to save memory
	sync_fits_z.close()
	sync_fits_x.close()

# When the code reaches this point, the statistics have been calculated for
# every simulation and every value of gamma, so it is time to start plotting

#------------------------------ Skewness zLOS ----------------------------------

# Skewness vs sonic Mach number

# Create a figure to display a plot of the skewness as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the skewness as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (skew_z_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (skew_z_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (skew_z_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Sonic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_sonic_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the skewness as a function of sonic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number 

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (skew_z_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (skew_z_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (skew_z_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Alfvenic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_alf_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the skewness as a function of Alfvenic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the skewness as a function of the
# observational effect, for simulations with b = 0.1 
fig3 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the skewness as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, skew_z_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, skew_z_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, skew_z_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, skew_z_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, skew_z_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, skew_z_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, skew_z_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, skew_z_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs ' + title_string + ' b.1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box3 = ax3.get_position()
ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])

# Force the legend to appear on the plot
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_{}_b.1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the observational effect has been saved
print 'Plot of the skewness as a function of observational effect saved b=0.1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the skewness as a function of the
# observational effect, for simulations with b = 1 
fig4 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the skewness as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, skew_z_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, skew_z_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, skew_z_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, skew_z_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, skew_z_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, skew_z_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, skew_z_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, skew_z_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs ' + title_string + ' b1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box4 = ax4.get_position()
ax4.set_position([box4.x0, box4.y0, box4.width * 0.8, box4.height])

# Force the legend to appear on the plot
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_{}_b1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the observational effect has been saved
print 'Plot of the skewness as a function of observational effect saved b=1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ Skewness xLOS ----------------------------------

# Skewness vs sonic Mach number

# Create a figure to display a plot of the skewness as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig5 = plt.figure()

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot the skewness as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (skew_x_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (skew_x_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (skew_x_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Sonic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_sonic_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the skewness as a function of sonic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number 

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig6 = plt.figure()

# Create an axis for this figure
ax6 = fig6.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (skew_x_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (skew_x_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (skew_x_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Alfvenic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_alf_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the skewness as a function of Alfvenic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the skewness as a function of the
# observational effect, for simulations with b = 0.1 
fig7 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax7 = fig7.add_subplot(111)

# Plot the skewness as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, skew_x_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, skew_x_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, skew_x_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, skew_x_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, skew_x_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, skew_x_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, skew_x_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, skew_x_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs ' + title_string + ' b.1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box7 = ax7.get_position()
ax7.set_position([box7.x0, box7.y0, box7.width * 0.8, box7.height])

# Force the legend to appear on the plot
ax7.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_{}_b.1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the observational effect has been saved
print 'Plot of the skewness as a function of observational effect saved b=0.1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the skewness as a function of the
# observational effect, for simulations with b = 1 
fig8 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax8 = fig8.add_subplot(111)

# Plot the skewness as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, skew_x_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, skew_x_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, skew_x_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, skew_x_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, skew_x_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, skew_x_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, skew_x_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, skew_x_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs ' + title_string + ' b1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box8 = ax8.get_position()
ax8.set_position([box8.x0, box8.y0, box8.width * 0.8, box8.height])

# Force the legend to appear on the plot
ax8.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_{}_b1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the observational effect has been saved
print 'Plot of the skewness as a function of observational effect saved b=1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ Kurtosis zLOS ----------------------------------

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig9 = plt.figure()

# Create an axis for this figure
ax9 = fig9.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (kurt_z_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (kurt_z_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (kurt_z_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs Sonic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_sonic_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the kurtosis as a function of sonic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number 

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig10 = plt.figure()

# Create an axis for this figure
ax10 = fig10.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (kurt_z_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (kurt_z_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (kurt_z_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs Alfvenic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_alf_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the kurtosis as a function of the
# observational effect, for simulations with b = 0.1 
fig11 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax11 = fig11.add_subplot(111)

# Plot the kurtosis as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, kurt_z_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, kurt_z_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, kurt_z_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, kurt_z_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, kurt_z_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, kurt_z_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, kurt_z_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, kurt_z_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs ' + title_string + ' b.1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box11 = ax11.get_position()
ax11.set_position([box11.x0, box11.y0, box11.width * 0.8, box11.height])

# Force the legend to appear on the plot
ax11.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_{}_b.1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the observational effect has been saved
print 'Plot of the kurtosis as a function of observational effect saved b=0.1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the kurtosis as a function of the
# observational effect, for simulations with b = 1 
fig12 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax12 = fig12.add_subplot(111)

# Plot the kurtosis as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, kurt_z_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, kurt_z_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, kurt_z_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, kurt_z_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, kurt_z_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, kurt_z_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, kurt_z_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, kurt_z_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs ' + title_string + ' b1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box12 = ax12.get_position()
ax12.set_position([box12.x0, box12.y0, box12.width * 0.8, box12.height])

# Force the legend to appear on the plot
ax12.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_{}_b1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the observational effect has been saved
print 'Plot of the kurtosis as a function of observational effect saved b=1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ Kurtosis xLOS ----------------------------------

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig13 = plt.figure()

# Create an axis for this figure
ax13 = fig13.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (kurt_x_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (kurt_x_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (kurt_x_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs Sonic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_sonic_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the kurtosis as a function of sonic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number 

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig14 = plt.figure()

# Create an axis for this figure
ax14 = fig14.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (kurt_x_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (kurt_x_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (kurt_x_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs Alfvenic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_alf_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the kurtosis as a function of the
# observational effect, for simulations with b = 0.1 
fig15 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax15 = fig15.add_subplot(111)

# Plot the kurtosis as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, kurt_x_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, kurt_x_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, kurt_x_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, kurt_x_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, kurt_x_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, kurt_x_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, kurt_x_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, kurt_x_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs ' + title_string + ' b.1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box15 = ax15.get_position()
ax15.set_position([box15.x0, box15.y0, box15.width * 0.8, box15.height])

# Force the legend to appear on the plot
ax15.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_{}_b.1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the observational effect has been saved
print 'Plot of the kurtosis as a function of observational effect saved b=0.1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the kurtosis as a function of the
# observational effect, for simulations with b = 1 
fig16 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax16 = fig16.add_subplot(111)

# Plot the kurtosis as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, kurt_x_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, kurt_x_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, kurt_x_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, kurt_x_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, kurt_x_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, kurt_x_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, kurt_x_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, kurt_x_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs ' + title_string + ' b1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box16 = ax16.get_position()
ax16.set_position([box16.x0, box16.y0, box16.width * 0.8, box16.height])

# Force the legend to appear on the plot
ax16.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_{}_b1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the observational effect has been saved
print 'Plot of the kurtosis as a function of observational effect saved b=1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ SF Slope -1 zLOS -------------------------------

# SF Slope - 1 vs sonic Mach number

# Create a figure to display a plot of the SF slope - 1 as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig17 = plt.figure()

# Create an axis for this figure
ax17 = fig17.add_subplot(111)

# Plot the SF slope - 1 as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (m_z_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (m_z_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (m_z_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Sonic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_sonic_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the SF slope - 1 as a function of sonic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# SF slope - 1 vs Alfvenic Mach number 

# Create a figure to display a plot of the SF slope - 1 as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig18 = plt.figure()

# Create an axis for this figure
ax18 = fig18.add_subplot(111)

# Plot the SF slope - 1 as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (m_z_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (m_z_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (m_z_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Alfvenic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_alf_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the SF slope - 1 as a function of Alfvenic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# SF slope - 1 vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the SF slope - 1 as a function of the
# observational effect, for simulations with b = 0.1 
fig19 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax19 = fig19.add_subplot(111)

# Plot the SF slope - 1 as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, m_z_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, m_z_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, m_z_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, m_z_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, m_z_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, m_z_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, m_z_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, m_z_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs ' + title_string + ' b.1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box19 = ax19.get_position()
ax19.set_position([box19.x0, box19.y0, box19.width * 0.8, box19.height])

# Force the legend to appear on the plot
ax19.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_{}_b.1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the observational effect has been saved
print 'Plot of the SF slope - 1 as a function of observational effect saved b=0.1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# SF slope - 1 vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the SF slope - 1 as a function of the
# observational effect, for simulations with b = 1 
fig20 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax20 = fig20.add_subplot(111)

# Plot the SF slope - 1 as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, m_z_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, m_z_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, m_z_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, m_z_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, m_z_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, m_z_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, m_z_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, m_z_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs ' + title_string + ' b1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box20 = ax20.get_position()
ax20.set_position([box20.x0, box20.y0, box20.width * 0.8, box20.height])

# Force the legend to appear on the plot
ax20.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_{}_b1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the observational effect has been saved
print 'Plot of the SF slope - 1 as a function of observational effect saved b=1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ SF slope - 1 xLOS ------------------------------

# SF slope - 1 vs sonic Mach number

# Create a figure to display a plot of the SF slope - 1 as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig21 = plt.figure()

# Create an axis for this figure
ax21 = fig21.add_subplot(111)

# Plot the SF slope - 1 as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (m_x_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (m_x_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (m_x_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Sonic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_sonic_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the SF slope - 1 as a function of sonic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# SF slope - 1 vs Alfvenic Mach number 

# Create a figure to display a plot of the SF slope - 1 as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig22 = plt.figure()

# Create an axis for this figure
ax22 = fig22.add_subplot(111)

# Plot the SF slope - 1 as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (m_x_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (m_x_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (m_x_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Alfvenic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_alf_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the SF slope - 1 as a function of Alfvenic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# SF slope - 1 vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the SF slope - 1 as a function of the
# observational effect, for simulations with b = 0.1 
fig23 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax23 = fig23.add_subplot(111)

# Plot the SF slope - 1 as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, m_x_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, m_x_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, m_x_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, m_x_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, m_x_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, m_x_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, m_x_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, m_x_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs ' + title_string + ' b.1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box23 = ax23.get_position()
ax23.set_position([box23.x0, box23.y0, box23.width * 0.8, box23.height])

# Force the legend to appear on the plot
ax23.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_{}_b.1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the observational effect has been saved
print 'Plot of the SF slope - 1 as a function of observational effect saved b=0.1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# SF slope - 1 vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the SF slope - 1 as a function of the
# observational effect, for simulations with b = 1 
fig24 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax24 = fig24.add_subplot(111)

# Plot the SF slope - 1 as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, m_x_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, m_x_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, m_x_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, m_x_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, m_x_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, m_x_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, m_x_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, m_x_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs ' + title_string + ' b1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box24 = ax24.get_position()
ax24.set_position([box24.x0, box24.y0, box24.width * 0.8, box24.height])

# Force the legend to appear on the plot
ax24.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_{}_b1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the observational effect has been saved
print 'Plot of the SF slope - 1 as a function of observational effect saved b=1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ Residuals zLOS ---------------------------------

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig25 = plt.figure()

# Create an axis for this figure
ax25 = fig25.add_subplot(111)

# Plot the residuals as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (residual_z_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (residual_z_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (residual_z_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Sonic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_sonic_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the residuals as a function of sonic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number 

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig26 = plt.figure()

# Create an axis for this figure
ax26 = fig26.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (residual_z_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (residual_z_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (residual_z_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Alfvenic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_alf_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the residuals as a function of Alfvenic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the residuals as a function of the
# observational effect, for simulations with b = 0.1 
fig27 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax27 = fig27.add_subplot(111)

# Plot the residuals as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, residual_z_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, residual_z_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, residual_z_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, residual_z_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, residual_z_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, residual_z_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, residual_z_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, residual_z_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs ' + title_string + ' b.1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box27 = ax27.get_position()
ax27.set_position([box27.x0, box27.y0, box27.width * 0.8, box27.height])

# Force the legend to appear on the plot
ax27.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_{}_b.1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the observational effect has been saved
print 'Plot of the residuals as a function of observational effect saved b=0.1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the residuals as a function of the
# observational effect, for simulations with b = 1 
fig28 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax28 = fig28.add_subplot(111)

# Plot the residuals as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, residual_z_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, residual_z_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, residual_z_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, residual_z_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, residual_z_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, residual_z_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, residual_z_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, residual_z_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs ' + title_string + ' b1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box28 = ax28.get_position()
ax28.set_position([box28.x0, box28.y0, box28.width * 0.8, box28.height])

# Force the legend to appear on the plot
ax28.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_{}_b1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the observational effect has been saved
print 'Plot of the residuals as a function of observational effect saved b=1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ Residuals xLOS ---------------------------------

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig29 = plt.figure()

# Create an axis for this figure
ax29 = fig29.add_subplot(111)

# Plot the residuals as a function of sonic Mach number for various values of the
# free parameter related to the observational effect being studied.
plt.plot(sonic_mach_sort, (residual_x_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (residual_x_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (residual_x_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Sonic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_sonic_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the residuals as a function of sonic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number 

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all values of the free 
# parameter related to the observational effect being studied.
fig30 = plt.figure()

# Create an axis for this figure
ax30 = fig30.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number for various values of
# the free parameter related to the observational effect being studied.
plt.plot(alf_mach_sort, (residual_x_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (residual_x_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (residual_x_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Alfvenic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_alf_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the residuals as a function of Alfvenic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the residuals as a function of the
# observational effect, for simulations with b = 0.1 
fig31 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax31 = fig31.add_subplot(111)

# Plot the residuals as a function of the observational effect for simulations
# with b = 0.1
plt.plot(iter_array, residual_x_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, residual_x_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, residual_x_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, residual_x_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, residual_x_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, residual_x_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, residual_x_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, residual_x_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs ' + title_string + ' b.1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box31 = ax31.get_position()
ax31.set_position([box31.x0, box31.y0, box31.width * 0.8, box31.height])

# Force the legend to appear on the plot
ax31.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_{}_b.1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the observational effect has been saved
print 'Plot of the residuals as a function of observational effect saved b=0.1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the residuals as a function of the
# observational effect, for simulations with b = 1 
fig32 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax32 = fig32.add_subplot(111)

# Plot the residuals as a function of the observational effect for simulations
# with b = 1
plt.plot(iter_array, residual_x_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, residual_x_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, residual_x_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, residual_x_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, residual_x_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, residual_x_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, residual_x_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, residual_x_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs ' + title_string + ' b1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box32 = ax32.get_position()
ax32.set_position([box32.x0, box32.y0, box32.width * 0.8, box32.height])

# Force the legend to appear on the plot
ax32.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'residuals_{}_b1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the observational effect has been saved
print 'Plot of the residuals as a function of observational effect saved b=1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

#---------------------- Integrated magnitude quad ratio zLOS -------------------

# Integrated magnitude quad ratio vs sonic Mach number

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of sonic Mach number for all of the synchrotron maps, i.e. for all 
# values of the free parameter related to the observational effect being studied.
fig33 = plt.figure()

# Create an axis for this figure
ax33 = fig33.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of sonic Mach number 
# for various values of the free parameter related to the observational effect being studied
plt.plot(sonic_mach_sort, (int_quad_z_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (int_quad_z_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (int_quad_z_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Sonic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_sonic_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the integrated quad ratio as a function of sonic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude quad ratio vs Alfvenic Mach number 

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of Alfvenic Mach number for all of the synchrotron maps, i.e. for all
# values of the free parameter related to the observational effect being studied.
fig34 = plt.figure()

# Create an axis for this figure
ax34 = fig34.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of Alfvenic Mach number
# for various values of the free parameter related to the observational effect 
# being studied.
plt.plot(alf_mach_sort, (int_quad_z_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (int_quad_z_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (int_quad_z_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Alfvenic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_alf_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the integrated quad ratio as a function of Alfvenic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude quad ratio vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of the observational effect, for simulations with b = 0.1 
fig35 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax35 = fig35.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of the observational 
# effect for simulations with b = 0.1
plt.plot(iter_array, int_quad_z_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, int_quad_z_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, int_quad_z_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, int_quad_z_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, int_quad_z_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, int_quad_z_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, int_quad_z_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, int_quad_z_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs ' + title_string + ' b.1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box35 = ax35.get_position()
ax35.set_position([box35.x0, box35.y0, box35.width * 0.8, box35.height])

# Force the legend to appear on the plot
ax35.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_{}_b.1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the observational effect has been saved
print 'Plot of the integrated quad ratio as a function of observational effect saved b=0.1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude quad ratio vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of the observational effect, for simulations with b = 1 
fig36 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax36 = fig36.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of the observational 
# effect for simulations with b = 1
plt.plot(iter_array, int_quad_z_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, int_quad_z_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, int_quad_z_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, int_quad_z_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, int_quad_z_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, int_quad_z_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, int_quad_z_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, int_quad_z_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs ' + title_string + ' b1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box36 = ax36.get_position()
ax36.set_position([box36.x0, box36.y0, box36.width * 0.8, box36.height])

# Force the legend to appear on the plot
ax36.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_{}_b1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the observational effect has been saved
print 'Plot of the integrated quad ratio as a function of observational effect saved b=1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

#---------------------- Integrated magnitude quad ratio xLOS -------------------

# Integrated magnitude quad ratio vs sonic Mach number

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of sonic Mach number for all of the synchrotron maps, i.e. for all 
# values of the free parameter related to the observational effect being studied.
fig37 = plt.figure()

# Create an axis for this figure
ax37 = fig37.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of sonic Mach number 
# for various values of the free parameter related to the observational effect 
# being studied
plt.plot(sonic_mach_sort, (int_quad_x_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (int_quad_x_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (int_quad_x_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Sonic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_sonic_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the integrated quad ratio as a function of sonic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude quad ratio vs Alfvenic Mach number 

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of Alfvenic Mach number for all of the synchrotron maps, i.e. for all
# values of the free parameter related to the observational effect being studied.
fig38 = plt.figure()

# Create an axis for this figure
ax38 = fig38.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of Alfvenic Mach number
# for various values of the free parameter related to the observational effect 
# being studied.
plt.plot(alf_mach_sort, (int_quad_x_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (int_quad_x_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (int_quad_x_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Alfvenic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_alf_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the integrated quad ratio as a function of Alfvenic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude quad ratio vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of the observational effect, for simulations with b = 0.1 
fig39 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax39 = fig39.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of the observational 
# effect for simulations with b = 0.1
plt.plot(iter_array, int_quad_x_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, int_quad_x_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, int_quad_x_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, int_quad_x_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, int_quad_x_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, int_quad_x_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, int_quad_x_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, int_quad_x_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs ' + title_string + ' b.1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box39 = ax39.get_position()
ax39.set_position([box39.x0, box39.y0, box39.width * 0.8, box39.height])

# Force the legend to appear on the plot
ax39.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_{}_b.1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the observational effect has been saved
print 'Plot of the integrated quad ratio as a function of observational effect saved b=0.1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude quad ratio vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the integrated magnitude quad ratio as a 
# function of the observational effect, for simulations with b = 1 
fig40 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax40 = fig40.add_subplot(111)

# Plot the integrated magnitude quad ratio as a function of the observational 
# effect for simulations with b = 1
plt.plot(iter_array, int_quad_x_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, int_quad_x_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, int_quad_x_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, int_quad_x_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, int_quad_x_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, int_quad_x_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, int_quad_x_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, int_quad_x_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs ' + title_string + ' b1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box40 = ax40.get_position()
ax40.set_position([box40.x0, box40.y0, box40.width * 0.8, box40.height])

# Force the legend to appear on the plot
ax40.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_{}_b1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated quad ratio as a 
# function of the observational effect has been saved
print 'Plot of the integrated quad ratio as a function of observational effect saved b=1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

#---------------------- Magnitude Quad ratio at a point zLOS -------------------

# Magnitude Quad ratio at a point vs sonic Mach number

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of sonic Mach number for all of the synchrotron maps, i.e.
# for all values of the free parameter related to the observational effect being
# studied.
fig41 = plt.figure()

# Create an axis for this figure
ax41 = fig41.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of sonic Mach 
# number for various values of the free parameter related to the observational 
# effect being studied
plt.plot(sonic_mach_sort, (quad_point_z_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (quad_point_z_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (quad_point_z_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Sonic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_sonic_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the quad ratio at a point as a function of sonic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude Quad ratio at a point vs Alfvenic Mach number 

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of Alfvenic Mach number for all of the synchrotron maps, 
# i.e. for all values of the free parameter related to the observational effect 
# being studied.
fig42 = plt.figure()

# Create an axis for this figure
ax42 = fig42.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of Alfvenic Mach
# number for various values of the free parameter related to the observational 
# effect being studied.
plt.plot(alf_mach_sort, (quad_point_z_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (quad_point_z_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (quad_point_z_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Alfvenic Mach Number Gam{} z'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_alf_{}_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the quad ratio at a point as a function of Alfvenic Mach number saved, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude Quad ratio at a point vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of the observational effect, for simulations with b = 0.1 
fig43 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax43 = fig43.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of the 
# observational effect for simulations with b = 0.1
plt.plot(iter_array, quad_point_z_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, quad_point_z_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, quad_point_z_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, quad_point_z_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, quad_point_z_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, quad_point_z_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, quad_point_z_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, quad_point_z_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs ' + title_string + ' b.1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box43 = ax43.get_position()
ax43.set_position([box43.x0, box43.y0, box43.width * 0.8, box43.height])

# Force the legend to appear on the plot
ax43.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_{}_b.1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the observational effect has been saved
print 'Plot of the quad ratio at a point as a function of observational effect saved b=0.1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude Quad ratio at a point vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of the observational effect, for simulations with b = 1 
fig44 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax44 = fig44.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of the 
# observational effect for simulations with b = 1
plt.plot(iter_array, quad_point_z_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, quad_point_z_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, quad_point_z_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, quad_point_z_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, quad_point_z_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, quad_point_z_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, quad_point_z_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, quad_point_z_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs ' + title_string + ' b1 Gam{} z'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box44 = ax44.get_position()
ax44.set_position([box44.x0, box44.y0, box44.width * 0.8, box44.height])

# Force the legend to appear on the plot
ax44.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_{}_b1_gam{}_z.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the observational effect has been saved
print 'Plot of the quad ratio at a point as a function of observational effect saved b=1, zLOS'

# Close the figure, now that it has been saved.
plt.close()

#---------------------- Magnitude Quad ratio at a point xLOS -------------------

# Magnitude of the quad ratio at a point vs sonic Mach number

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of sonic Mach number for all of the synchrotron maps, i.e.
# for all values of the free parameter related to the observational effect being
# studied.
fig45 = plt.figure()

# Create an axis for this figure
ax45 = fig45.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of sonic Mach 
# number for various values of the free parameter related to the observational 
# effect being studied
plt.plot(sonic_mach_sort, (quad_point_x_arr[0])[sonic_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(sonic_mach_sort, (quad_point_x_arr[free_num/2 - 1])[sonic_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(sonic_mach_sort, (quad_point_x_arr[free_num - 1])[sonic_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Sonic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_sonic_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the sonic Mach number has been saved for this line of sight
print 'Plot of the quad ratio at a point as a function of sonic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of the Quad ratio at a point vs Alfvenic Mach number 

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of Alfvenic Mach number for all of the synchrotron maps, 
# i.e. for all values of the free parameter related to the observational effect
# being studied.
fig46 = plt.figure()

# Create an axis for this figure
ax46 = fig46.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of Alfvenic Mach
# number for various values of the free parameter related to the observational 
# effect being studied.
plt.plot(alf_mach_sort, (quad_point_x_arr[0])[alf_sort],'b-o',label = leg_string\
	+'{}'.format(iter_array[0]))
plt.plot(alf_mach_sort, (quad_point_x_arr[free_num/2 - 1])[alf_sort],'r-o',\
	label= leg_string +'{0:.2f}'.format(iter_array[free_num/2 - 1]))
plt.plot(alf_mach_sort, (quad_point_x_arr[free_num - 1])[alf_sort],'c-o',\
	label = leg_string + '{}'.format(iter_array[free_num - 1]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Alfvenic Mach Number Gam{} x'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_alf_{}_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the Alfvenic Mach number has been saved for this line of sight
print 'Plot of the quad ratio at a point as a function of Alfvenic Mach number saved, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of Quad ratio at a point vs Observational Effect - Low Magnetic Field

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of the observational effect, for simulations with b = 0.1 
fig47 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax47 = fig47.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of the 
# observational effect for simulations with b = 0.1
plt.plot(iter_array, quad_point_x_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, quad_point_x_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, quad_point_x_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, quad_point_x_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, quad_point_x_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, quad_point_x_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, quad_point_x_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, quad_point_x_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs ' + title_string + ' b.1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box47 = ax47.get_position()
ax47.set_position([box47.x0, box47.y0, box47.width * 0.8, box47.height])

# Force the legend to appear on the plot
ax47.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_{}_b.1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the observational effect has been saved
print 'Plot of the quad ratio at a point as a function of observational effect saved b=0.1, xLOS'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of Quad ratio at a point vs Observational Effect - High Magnetic Field

# Create a figure to display a plot of the magnitude of the quad ratio at a 
# point as a function of the observational effect, for simulations with b = 1 
fig48 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax48 = fig48.add_subplot(111)

# Plot the magnitude of the quad ratio at a point as a function of the 
# observational effect for simulations with b = 1
plt.plot(iter_array, quad_point_x_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, quad_point_x_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, quad_point_x_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, quad_point_x_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, quad_point_x_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, quad_point_x_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, quad_point_x_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, quad_point_x_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs ' + title_string + ' b1 Gam{} x'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box48 = ax48.get_position()
ax48.set_position([box48.x0, box48.y0, box48.width * 0.8, box48.height])

# Force the legend to appear on the plot
ax48.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_{}_b1_gam{}_x.png'.format(obs_effect,gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad ratio at a point as a 
# function of the observational effect has been saved
print 'Plot of the quad ratio at a point as a function of observational effect saved b=1, xLOS'

# Close the figure, now that it has been saved.
plt.close()