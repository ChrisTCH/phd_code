#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the skewness, kurtosis, radially averaged        #
# structure function, and quadrupole/monopole ratio of the synchrotron         #
# intensity for various values of gamma. Each of these quantities is plotted   #
# against the sonic and Alfvenic Mach numbers, to see which quantities are     #
# sensitive tracers of the sonic and Alfvenic Mach numbers.                    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 12/11/2014                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

# Import the functions that calculate the structure and correlation functions
# using FFT, the function that calculates the radially averaged structure or 
# correlation functions, the function that calculates multipoles of 2D 
# images, and the function that calculates the magnitude and argument of the
# complex quadrupole ratio
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio

# Define a function that calculates the errors in statistics by breaking up
# synchrotron images into quarters, calculating statistics for each quarter, and
# then calculates the standard deviation of the statistics.
def calc_err_bootstrap(sync_map_y, sync_map_z):
	'''
	Description
        This function divides the given images into quarters, and then 
        calculates statistics for each quarter. The standard deviation of the 
        calculated statistics is then returned, representing the error on 
        each statistic.
        
    Required Input
        sync_map_y - The synchrotron intensity map observed for a line of sight
        			 along the y axis.
        sync_map_z - The synchrotron intensity map observed for a line of sight 
        			 along the z axis. Must have the same size as the map 
        			 for a line of sight along the y axis.
                   
    Output
        skew_err - The error calculated for the skewness of synchrotron 
        		   intensity
        kurt_err - The error calculated for the kurtosis of synchrotron 
        		   intensity
        m_err - The error calculated for the structure function slope of the 
        		synchrotron intensity
		residual_err - The error calculated for the residual of the linear fit 
					   to the structure function of synchrotron intensity
		int_quad_err - The error calculated for the integrated quadrupole ratio
					   modulus of the synchrotron intensity
		quad_point_err - The error calculated for the value of the quadrupole 
						 ratio modulus at a point of synchrotron intensity
	'''

	# Create an array that will hold the quarters of the synchrotron images
	quarter_arr = np.zeros((8,np.shape(sync_map_y)[0]/2,np.shape(sync_map_y)[1]/2))

	# Add the quarters of the images into the array
	quarter_arr[0], quarter_arr[1] = np.split(np.split(sync_map_y,2,axis=0)[0],2,axis=1) 
	quarter_arr[2], quarter_arr[3] = np.split(np.split(sync_map_y,2,axis=0)[1],2,axis=1) 
	quarter_arr[4], quarter_arr[5] = np.split(np.split(sync_map_z,2,axis=0)[0],2,axis=1)
	quarter_arr[6], quarter_arr[7] = np.split(np.split(sync_map_z,2,axis=0)[1],2,axis=1)

	# Create arrays that will hold the calculated statistics for each quarter
	skew_val = np.zeros(np.shape(quarter_arr)[0])
	kurt_val = np.zeros(np.shape(quarter_arr)[0])
	m_val = np.zeros(np.shape(quarter_arr)[0])
	resid_val = np.zeros(np.shape(quarter_arr)[0])
	int_quad_val = np.zeros(np.shape(quarter_arr)[0])
	quad_point_val = np.zeros(np.shape(quarter_arr)[0])

	# Loop over the quarters, to calculate statistics for each one
	for i in range(np.shape(quarter_arr)[0]):
		# Extract the current image quarter from the array
		image = quarter_arr[i]

		# Flatten the image, so that we can calculate the skewness and kurtosis
		flat_image = image.flatten()

		# Calculate the biased skewness of the synchrotron intensity map
		skew_val[i] = stats.skew(flat_image)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# maps
		kurt_val[i] = stats.kurtosis(flat_image)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not 
		# subtracting the mean from the synchrotron maps before calculating the 
		# structure function.
		strfn = sf_fft(image, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins.
		rad_sf = sfr(strfn, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function
		sf = rad_sf[1]

		# Extract the radius values used to calculate this structure function.
		sf_rad_arr = rad_sf[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. 
		spec_ind_data = np.polyfit(np.log10(\
			sf_rad_arr[11:16]),\
			np.log10(sf[11:16]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit
		coeff = spec_ind_data[0]

		# Extract the sum of the residuals from the polynomial fit
		resid_val[i] = spec_ind_data[1]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array
		m_val[i] = coeff[0]-1.0

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

		# Find the value of the magnitude of the quadrupole / monopole ratio 
		# for a radial separation that is one third of the way along the radial 
		# separation range that is probed, and store it in the corresponding 
		# array.
		quad_point_val[i] = quad_mod[np.floor(num_bins/3.0)]

		# Integrate the magnitude of the quadrupole / monopole ratio from 
		# one sixth of the way along the radial separation bins, until three 
		# quarters of the way along the radial separation bins. This integration
		# is performed with respect to log separation (i.e. I am ignoring the 
		# fact that the points are equally separated in log space, to calculate 
		# the area under the quadrupole / monopole ratio plot when the x axis 
		# is scaled logarithmically). I normalise the value that is returned by 
		# dividing by the number of increments in log radial separation used in 
		# the calculation.
		int_quad_val[i] = np.trapz(quad_mod[11:20], dx = 1.0)\
		 / (19 - 11)

	# At this point, the statistics have been calculated for each quarter
	# The next step is to calculate the standard deviation of each statistic
	skew_err = np.std(skew_val)
	kurt_err = np.std(kurt_val)
	m_err = np.std(m_val)
	residual_err = np.std(resid_val)
	int_quad_err = np.std(int_quad_val)
	quad_point_err = np.std(quad_point_val)

	# Now that all of the calculations have been performed, return the 
	# calculated errors
	return skew_err, kurt_err, m_err, residual_err, int_quad_err, quad_point_err

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 25

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

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
sonic_mach_arr = np.array([10.95839209, 9.16414046, 7.02482223, 4.32383325,\
 3.11247421, 2.37827562, 0.82952013, 0.44891885, 9.92156478, 7.89086655,\
 6.78273351, 4.45713648, 3.15831577, 2.40931069, 0.87244536, 0.47752262,\
 8.42233298, 8.39066797])

# Create an array, where each entry specifies the calculated Alfvenic Mach 
# number for each simulation
alf_mach_arr = np.array([1.41278383, 1.77294593, 1.75575508, 1.50830194,\
 1.69455875, 1.85993991, 1.74231524, 1.71939152, 0.49665052, 0.50288954,\
 0.51665006, 0.54928564, 0.57584022, 0.67145057, 0.70015313, 0.65195539,\
 0.21894299, 0.14357068])

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. Each row corresponds to a value of gamma, and each 
# column corresponds to a simulation. There is one array for a line of sight
# along each of the x, y and z axes (x is parallel to B).
# NOTE: We will calculate the biased skewness
skew_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
skew_y_arr = np.zeros((len(gamma_arr),len(simul_arr)))
skew_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. Each row corresponds to a value of gamma, and each 
# column corresponds to a simulation. There is one array for a line of sight
# along each of the x, y and z axes (x is parallel to B).
# NOTE: We will calculate the biased Fisher kurtosis
kurt_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
kurt_y_arr = np.zeros((len(gamma_arr),len(simul_arr)))
kurt_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of gamma. Each row 
# corresponds to a value of gamma, and each column corresponds to a simulation.
# There is one array for a line of sight along each of the x, y and z axes 
# (x is parallel to B).
m_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
m_y_arr = np.zeros((len(gamma_arr),len(simul_arr)))
m_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the residuals of the linear
# fit to the structure function of the synchrotron intensity image, of the 
# corresponding simulation, for a particular value of gamma. Each row 
# corresponds to a value of gamma, and each column corresponds to a simulation.
# There is one array for a line of sight along each of the x, y and z axes
# (x is parallel to B).
residual_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
residual_y_arr = np.zeros((len(gamma_arr),len(simul_arr)))
residual_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity
# image, for the corresponding simulation, for a particular value of gamma. Each
# row corresponds to a value of gamma, and each column corresponds to a 
# simulation. There is one array for a line of sight along each of the x, y and
# z axes (x is parallel to B).
int_quad_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
int_quad_y_arr = np.zeros((len(gamma_arr),len(simul_arr)))
int_quad_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated magnitude of 
# the quadrupole/monopole ratio of the synchrotron intensity image at a 
# particular radial separation, for the corresponding simulation, for a 
# particular value of gamma. Each row corresponds to a value of gamma, and each 
# column corresponds to a simulation. There is one array for a line of sight 
# along each of the x, y and z axes (x is parallel to B).
quad_point_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
quad_point_y_arr = np.zeros((len(gamma_arr),len(simul_arr)))
quad_point_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create error arrays for each of the statistics. These errors are only for the
# statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field), and are calculated by the standard deviation of the 
# statistics calculated for sub-images of the synchrotron maps.
skew_err_arr = np.zeros((len(gamma_arr),len(simul_arr)))
kurt_err_arr = np.zeros((len(gamma_arr),len(simul_arr)))
m_err_arr = np.zeros((len(gamma_arr),len(simul_arr)))
residual_err_arr = np.zeros((len(gamma_arr),len(simul_arr)))
int_quad_err_arr = np.zeros((len(gamma_arr),len(simul_arr)))
quad_point_err_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS files that contain the simulated synchrotron intensity maps
	# for lines of sight along each of the axes
	sync_fits_z = fits.open(data_loc + 'synint_p1-4.fits')
	sync_fits_y = fits.open(data_loc + 'synint_p1-4y.fits')
	sync_fits_x = fits.open(data_loc + 'synint_p1-4x.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensities observed for different values of gamma, the power
	# law index of the cosmic ray electrons emitting the synchrotron emission.
	sync_data_z = sync_fits_z[0].data
	sync_data_y = sync_fits_y[0].data
	sync_data_x = sync_fits_x[0].data

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Loop over the various values of gamma, to calculate the various statistics
	# for the synchrotron map observed for each value of gamma
	for i in range(len(gamma_arr)):
		# Extract the synchrotron intensity map for this value of gamma, for
		# lines of sight along each of the axes
		sync_map_z = sync_data_z[i]
		sync_map_y = sync_data_y[i]
		sync_map_x = sync_data_x[i]

		# Flatten the synchrotron intensity maps for this value of gamma, for
		# lines of sight along each of the axes
		flat_sync_z = sync_map_z.flatten()
		flat_sync_y = sync_map_y.flatten()
		flat_sync_x = sync_map_x.flatten()

		# Calculate the biased skewness of the synchrotron intensity maps, for
		# lines of sight along each of the axes, and store the results in the
		# corresponding array.
		skew_z_arr[i,j] = stats.skew(flat_sync_z)
		skew_y_arr[i,j] = stats.skew(flat_sync_y)
		skew_x_arr[i,j] = stats.skew(flat_sync_x)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# maps, for lines of sight along each of the axes, and store the results
		# in the corresponding array.
		kurt_z_arr[i,j] = stats.kurtosis(flat_sync_z)
		kurt_y_arr[i,j] = stats.kurtosis(flat_sync_y)
		kurt_x_arr[i,j] = stats.kurtosis(flat_sync_x)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity maps, for lines of sight along each of the axes. Note 
		# that no_fluct = True is set, because we are not subtracting the mean
		# from the synchrotron maps before calculating the structure function.
		strfn_z = sf_fft(sync_map_z, no_fluct = True)
		strfn_y = sf_fft(sync_map_y, no_fluct = True)
		strfn_x = sf_fft(sync_map_x, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins, for lines of sight along each of the axes.
		rad_sf_z = sfr(strfn_z, num_bins, verbose = False)
		rad_sf_y = sfr(strfn_y, num_bins, verbose = False)
		rad_sf_x = sfr(strfn_x, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function for lines
		# of sight along each of the axes.
		sf_z = rad_sf_z[1]
		sf_y = rad_sf_y[1]
		sf_x = rad_sf_x[1]

		# Extract the radius values used to calculate this structure function,
		# for lines of sight along each of the axes.
		sf_rad_arr_z = rad_sf_z[0]
		sf_rad_arr_y = rad_sf_y[0]
		sf_rad_arr_x = rad_sf_x[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. Perform a linear fit for a line
		# of sight along the z axis.
		spec_ind_data_z = np.polyfit(np.log10(\
			sf_rad_arr_z[11:16]),\
			np.log10(sf_z[11:16]), 1, full = True)
		# Perform a linear fit for a line of sight along the y axis
		spec_ind_data_y = np.polyfit(np.log10(\
			sf_rad_arr_y[11:16]),\
			np.log10(sf_y[11:16]), 1, full = True)
		# Perform a linear fit for a line of sight along the x axis
		spec_ind_data_x = np.polyfit(np.log10(\
			sf_rad_arr_x[11:16]),\
			np.log10(sf_x[11:16]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit, for lines
		# of sight along each of the axes.
		coeff_z = spec_ind_data_z[0]
		coeff_y = spec_ind_data_y[0]
		coeff_x = spec_ind_data_x[0]

		# Extract the sum of the residuals from the polynomial fit, for lines
		# of sight along each of the axes.
		residual_z_arr[i,j] = spec_ind_data_z[1]
		residual_y_arr[i,j] = spec_ind_data_y[1]
		residual_x_arr[i,j] = spec_ind_data_x[1]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array, for lines of sight along each of the
		# axes
		m_z_arr[i,j] = coeff_z[0]-1.0
		m_y_arr[i,j] = coeff_y[0]-1.0
		m_x_arr[i,j] = coeff_x[0]-1.0

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn_z = sf_fft(sync_map_z, no_fluct = True, normalise = True)
		norm_strfn_y = sf_fft(sync_map_y, no_fluct = True, normalise = True)
		norm_strfn_x = sf_fft(sync_map_x, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image. This is done for lines of sight
		# along each of the axes
		norm_strfn_z = np.fft.fftshift(norm_strfn_z)
		norm_strfn_y = np.fft.fftshift(norm_strfn_y)
		norm_strfn_x = np.fft.fftshift(norm_strfn_x)

		# Calculate the magnitude and argument of the quadrupole ratio, for 
		# lines of sight along each of the axes.
		quad_mod_z, quad_arg_z, quad_rad_z = calc_quad_ratio(norm_strfn_z, num_bins)
		quad_mod_y, quad_arg_y, quad_rad_y = calc_quad_ratio(norm_strfn_y, num_bins)
		quad_mod_x, quad_arg_x, quad_rad_x = calc_quad_ratio(norm_strfn_x, num_bins)

		# Find the value of the magnitude of the quadrupole / monopole ratio 
		# for a radial separation that is one third of the way along the radial 
		# separation range that is probed, and store it in the corresponding 
		# array. This is done for lines of sight along each of the axes.
		quad_point_z_arr[i,j] = quad_mod_z[np.floor(num_bins/3.0)]
		quad_point_y_arr[i,j] = quad_mod_y[np.floor(num_bins/3.0)]
		quad_point_x_arr[i,j] = quad_mod_x[np.floor(num_bins/3.0)]

		# Integrate the magnitude of the quadrupole / monopole ratio from 
		# one sixth of the way along the radial separation bins, until three 
		# quarters of the way along the radial separation bins. This integration
		# is performed with respect to log separation (i.e. I am ignoring the 
		# fact that the points are equally separated in log space, to calculate 
		# the area under the quadrupole / monopole ratio plot when the x axis 
		# is scaled logarithmically). I normalise the value that is returned by 
		# dividing by the number of increments in log radial separation used in 
		# the calculation. This is done for lines of sight along each of the 
		# axes.
		int_quad_z_arr[i,j] = np.trapz(quad_mod_z[11:20], dx = 1.0)\
		 / (19 - 11)
		int_quad_y_arr[i,j] = np.trapz(quad_mod_y[11:20], dx = 1.0)\
		 / (19 - 11)
		int_quad_x_arr[i,j] = np.trapz(quad_mod_x[11:20], dx = 1.0)\
		 / (19 - 11)

		# Create errors for each of the statistics. These errors are only for the
		# statistics calculated from the y and z axes (perpendicular to the mean 
		# magnetic field), and are calculated by the standard deviation of the 
		# statistics calculated for sub-images of the synchrotron maps.
		skew_err_arr[i,j], kurt_err_arr[i,j], m_err_arr[i,j],\
		residual_err_arr[i,j], int_quad_err_arr[i,j], quad_point_err_arr[i,j]\
		= calc_err_bootstrap(sync_map_y, sync_map_z)

		# At this point, all of the statistics that need to be calculated for
		# every value of gamma have been calculated.

	# Close the fits files, to save memory
	sync_fits_z.close()
	sync_fits_y.close()
	sync_fits_x.close()

# Create mean value arrays for each of the statistics. These values are only for
# the statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field).
skew_mean_arr = (skew_y_arr + skew_z_arr) / 2.0
kurt_mean_arr = (kurt_y_arr + kurt_z_arr) / 2.0
m_mean_arr = (m_y_arr + m_z_arr) / 2.0
residual_mean_arr = (residual_y_arr + residual_z_arr) / 2.0
int_quad_mean_arr = (int_quad_y_arr + int_quad_z_arr) / 2.0
quad_point_mean_arr = (quad_point_y_arr + quad_point_z_arr) / 2.0

# When the code reaches this point, the statistics have been calculated for
# every simulation and every value of gamma, so it is time to start plotting

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/'

#------------------------- Line of sight along z -------------------------------

# Skewness vs sonic Mach number

# Create a figure to display a plot of the skewness as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the skewness as a function of sonic Mach number for each gamma
plt.errorbar(sonic_mach_arr, skew_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = skew_err_arr[0])
plt.errorbar(sonic_mach_arr, skew_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = skew_err_arr[1])
plt.errorbar(sonic_mach_arr, skew_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = skew_err_arr[2])
plt.errorbar(sonic_mach_arr, skew_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = skew_err_arr[3])
plt.errorbar(sonic_mach_arr, skew_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = skew_err_arr[4])
plt.errorbar(sonic_mach_arr, skew_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = skew_err_arr[5])
plt.errorbar(sonic_mach_arr, skew_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = skew_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mean Skew vs Sonic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_sonic_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved
print 'Plot of the skewness as a function of sonic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number for each gamma
plt.errorbar(alf_mach_arr, skew_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = skew_err_arr[0])
plt.errorbar(alf_mach_arr, skew_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = skew_err_arr[1])
plt.errorbar(alf_mach_arr, skew_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = skew_err_arr[2])
plt.errorbar(alf_mach_arr, skew_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = skew_err_arr[3])
plt.errorbar(alf_mach_arr, skew_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = skew_err_arr[4])
plt.errorbar(alf_mach_arr, skew_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = skew_err_arr[5])
plt.errorbar(alf_mach_arr, skew_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = skew_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mean Skew vs Alfvenic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_alf_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the skewness as a function of Alfvenic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig3 = plt.figure()

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number for each gamma
plt.errorbar(sonic_mach_arr, kurt_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = kurt_err_arr[0])
plt.errorbar(sonic_mach_arr, kurt_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = kurt_err_arr[1])
plt.errorbar(sonic_mach_arr, kurt_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = kurt_err_arr[2])
plt.errorbar(sonic_mach_arr, kurt_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = kurt_err_arr[3])
plt.errorbar(sonic_mach_arr, kurt_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = kurt_err_arr[4])
plt.errorbar(sonic_mach_arr, kurt_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = kurt_err_arr[5])
plt.errorbar(sonic_mach_arr, kurt_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = kurt_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Mean Kurtosis vs Sonic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_sonic_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved
print 'Plot of the kurtosis as a function of sonic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig4 = plt.figure()

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number for each gamma
plt.errorbar(alf_mach_arr, kurt_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = kurt_err_arr[0])
plt.errorbar(alf_mach_arr, kurt_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = kurt_err_arr[1])
plt.errorbar(alf_mach_arr, kurt_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = kurt_err_arr[2])
plt.errorbar(alf_mach_arr, kurt_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = kurt_err_arr[3])
plt.errorbar(alf_mach_arr, kurt_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = kurt_err_arr[4])
plt.errorbar(alf_mach_arr, kurt_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = kurt_err_arr[5])
plt.errorbar(alf_mach_arr, kurt_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = kurt_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Mean Kurtosis vs Alfvenic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_alf_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Structure Function slope vs sonic Mach number

# Create a figure to display a plot of the SF slope-1 (m) as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig5 = plt.figure()

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot m as a function of sonic Mach number for each gamma
plt.errorbar(sonic_mach_arr, m_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = m_err_arr[0])
plt.errorbar(sonic_mach_arr, m_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = m_err_arr[1])
plt.errorbar(sonic_mach_arr, m_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = m_err_arr[2])
plt.errorbar(sonic_mach_arr, m_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = m_err_arr[3])
plt.errorbar(sonic_mach_arr, m_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = m_err_arr[4])
plt.errorbar(sonic_mach_arr, m_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = m_err_arr[5])
plt.errorbar(sonic_mach_arr, m_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = m_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mean SF Slope - 1 vs Sonic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_sonic_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the sonic Mach number has been saved
print 'Plot of m as a function of sonic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# SF slope vs Alfvenic Mach number

# Create a figure to display a plot of the SF slope-1 as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig6 = plt.figure()

# Create an axis for this figure
ax6 = fig6.add_subplot(111)

# Plot m as a function of Alfvenic Mach number for each gamma
plt.errorbar(alf_mach_arr, m_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = m_err_arr[0])
plt.errorbar(alf_mach_arr, m_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = m_err_arr[1])
plt.errorbar(alf_mach_arr, m_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = m_err_arr[2])
plt.errorbar(alf_mach_arr, m_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = m_err_arr[3])
plt.errorbar(alf_mach_arr, m_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = m_err_arr[4])
plt.errorbar(alf_mach_arr, m_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = m_err_arr[5])
plt.errorbar(alf_mach_arr, m_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = m_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mean SF Slope - 1 vs Alfvenic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_alf_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of m as a function of Alfvenic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig7 = plt.figure()

# Create an axis for this figure
ax7 = fig7.add_subplot(111)

# Plot the residuals as a function of sonic Mach number for each gamma
plt.errorbar(sonic_mach_arr, residual_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = residual_err_arr[0])
plt.errorbar(sonic_mach_arr, residual_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = residual_err_arr[1])
plt.errorbar(sonic_mach_arr, residual_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = residual_err_arr[2])
plt.errorbar(sonic_mach_arr, residual_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = residual_err_arr[3])
plt.errorbar(sonic_mach_arr, residual_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = residual_err_arr[4])
plt.errorbar(sonic_mach_arr, residual_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = residual_err_arr[5])
plt.errorbar(sonic_mach_arr, residual_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = residual_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Mean Residuals SF Fit vs Sonic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_sonic_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved
print 'Plot of the residuals as a function of sonic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig8 = plt.figure()

# Create an axis for this figure
ax8 = fig8.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number for each gamma
plt.errorbar(alf_mach_arr, residual_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = residual_err_arr[0])
plt.errorbar(alf_mach_arr, residual_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = residual_err_arr[1])
plt.errorbar(alf_mach_arr, residual_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = residual_err_arr[2])
plt.errorbar(alf_mach_arr, residual_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = residual_err_arr[3])
plt.errorbar(alf_mach_arr, residual_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = residual_err_arr[4])
plt.errorbar(alf_mach_arr, residual_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = residual_err_arr[5])
plt.errorbar(alf_mach_arr, residual_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = residual_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Mean Residuals SF Fit vs Alfvenic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_alf_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the residuals as a function of Alfvenic Mach number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude of quad / mono ratio vs sonic Mach number

# Create a figure to display a plot of the integrated magnitude of the quad / 
# mono ratio as a function of sonic Mach number for all of the synchrotron maps,
# i.e. for all gamma
fig9 = plt.figure()

# Create an axis for this figure
ax9 = fig9.add_subplot(111)

# Plot the integrated magnitude of the quad / mono ratio as a function of sonic
# Mach number for each gamma
plt.errorbar(sonic_mach_arr, int_quad_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = int_quad_err_arr[0])
plt.errorbar(sonic_mach_arr, int_quad_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = int_quad_err_arr[1])
plt.errorbar(sonic_mach_arr, int_quad_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = int_quad_err_arr[2])
plt.errorbar(sonic_mach_arr, int_quad_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = int_quad_err_arr[3])
plt.errorbar(sonic_mach_arr, int_quad_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = int_quad_err_arr[4])
plt.errorbar(sonic_mach_arr, int_quad_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = int_quad_err_arr[5])
plt.errorbar(sonic_mach_arr, int_quad_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = int_quad_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Mean Integrated Mag quad vs Sonic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_sonic_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the sonic Mach number has been saved
print 'Plot of the integrated quad/mono ratio as a function of sonic Mach'\
+ ' number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude of quad / mono ratio vs Alfvenic Mach number

# Create a figure to display a plot of the integrated magnitude of the quad/mono
# ratio as a function of Alfvenic Mach number for all of the synchrotron maps, 
# i.e. for all gamma
fig10 = plt.figure()

# Create an axis for this figure
ax10 = fig10.add_subplot(111)

# Plot the integrated magnitude of the quad / mono ratio as a function of 
# Alfvenic Mach number for each gamma
plt.errorbar(alf_mach_arr, int_quad_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = int_quad_err_arr[0])
plt.errorbar(alf_mach_arr, int_quad_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = int_quad_err_arr[1])
plt.errorbar(alf_mach_arr, int_quad_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = int_quad_err_arr[2])
plt.errorbar(alf_mach_arr, int_quad_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = int_quad_err_arr[3])
plt.errorbar(alf_mach_arr, int_quad_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = int_quad_err_arr[4])
plt.errorbar(alf_mach_arr, int_quad_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = int_quad_err_arr[5])
plt.errorbar(alf_mach_arr, int_quad_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = int_quad_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Mean Integrated Mag quad vs Alfvenic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_alf_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the Alfvenic Mach number has been saved
print 'Plot of the integrated quad/mono as a function of Alfvenic Mach'\
+ ' number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of the quad / mono ratio at a point vs sonic Mach number

# Create a figure to display a plot of the magnitude of the quad/mono ratio at a
# point as a function of sonic Mach number for all of the synchrotron maps, 
# i.e. for all gamma
fig11 = plt.figure()

# Create an axis for this figure
ax11 = fig11.add_subplot(111)

# Plot the quad/mono ratio as a point as a function of sonic Mach number for
# each gamma
plt.errorbar(sonic_mach_arr, quad_point_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = quad_point_err_arr[0])
plt.errorbar(sonic_mach_arr, quad_point_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = quad_point_err_arr[1])
plt.errorbar(sonic_mach_arr, quad_point_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = quad_point_err_arr[2])
plt.errorbar(sonic_mach_arr, quad_point_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = quad_point_err_arr[3])
plt.errorbar(sonic_mach_arr, quad_point_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = quad_point_err_arr[4])
plt.errorbar(sonic_mach_arr, quad_point_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = quad_point_err_arr[5])
plt.errorbar(sonic_mach_arr, quad_point_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = quad_point_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mean Mag Quad/mono at point vs Sonic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_sonic_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio at
# a single radius value as a function of the sonic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of sonic Mach'\
+ ' number saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of the quad / mono ratio at a point vs Alfvenic Mach number

# Create a figure to display a plot of the magnitude of the quad/mono ratio at a
# point as a function of Alfvenic Mach number for all of the synchrotron maps, 
# i.e. for all gamma
fig12 = plt.figure()

# Create an axis for this figure
ax12 = fig12.add_subplot(111)

# Plot the quad/mono ratio at a point as a function of Alfvenic Mach number
# for each gamma
plt.errorbar(alf_mach_arr, quad_point_mean_arr[0], fmt ='bo',\
	label ='Gamma = {}'.format(gamma_arr[0]), yerr = quad_point_err_arr[0])
plt.errorbar(alf_mach_arr, quad_point_mean_arr[1], fmt ='ro',\
	label ='Gamma = {}'.format(gamma_arr[1]), yerr = quad_point_err_arr[1])
plt.errorbar(alf_mach_arr, quad_point_mean_arr[2], fmt ='go',\
	label ='Gamma = {}'.format(gamma_arr[2]), yerr = quad_point_err_arr[2])
plt.errorbar(alf_mach_arr, quad_point_mean_arr[3], fmt ='co',\
	label ='Gamma = {}'.format(gamma_arr[3]), yerr = quad_point_err_arr[3])
plt.errorbar(alf_mach_arr, quad_point_mean_arr[4], fmt ='mo',\
	label ='Gamma = {}'.format(gamma_arr[4]), yerr = quad_point_err_arr[4])
plt.errorbar(alf_mach_arr, quad_point_mean_arr[5], fmt ='yo',\
	label ='Gamma = {}'.format(gamma_arr[5]), yerr = quad_point_err_arr[5])
plt.errorbar(alf_mach_arr, quad_point_mean_arr[6], fmt ='ko',\
	label ='Gamma = {}'.format(gamma_arr[6]), yerr = quad_point_err_arr[6])

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mean Mag Quad/mono at point vs Alfvenic Mach Number', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_alf_mach_mean.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio
# at a point as a function of the Alfvenic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of Alfvenic Mach'\
+ ' number saved mean'

# Close the figure, now that it has been saved.
plt.close()

#------------------------- Line of sight along x -------------------------------

# Skewness vs sonic Mach number

# Create a figure to display a plot of the skewness as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig13 = plt.figure()

# Create an axis for this figure
ax13 = fig13.add_subplot(111)

# Plot the skewness as a function of sonic Mach number for each gamma
plt.plot(sonic_mach_arr, skew_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, skew_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, skew_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, skew_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, skew_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, skew_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, skew_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Sonic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved
print 'Plot of the skewness as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig14 = plt.figure()

# Create an axis for this figure
ax14 = fig14.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, skew_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, skew_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, skew_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, skew_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, skew_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, skew_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, skew_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Alfvenic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the skewness as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig15 = plt.figure()

# Create an axis for this figure
ax15 = fig15.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number for each gamma
plt.plot(sonic_mach_arr, kurt_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, kurt_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, kurt_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, kurt_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, kurt_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, kurt_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, kurt_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Sonic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved
print 'Plot of the kurtosis as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig16 = plt.figure()

# Create an axis for this figure
ax16 = fig16.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, kurt_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, kurt_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, kurt_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, kurt_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, kurt_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, kurt_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, kurt_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Alfvenic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Structure Function slope vs sonic Mach number

# Create a figure to display a plot of the SF slope-1 (m) as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig17 = plt.figure()

# Create an axis for this figure
ax17 = fig17.add_subplot(111)

# Plot m as a function of sonic Mach number for each gamma
plt.plot(sonic_mach_arr, m_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, m_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, m_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, m_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, m_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, m_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, m_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Sonic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the sonic Mach number has been saved
print 'Plot of m as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# SF slope vs Alfvenic Mach number

# Create a figure to display a plot of the SF slope-1 as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig18 = plt.figure()

# Create an axis for this figure
ax18 = fig18.add_subplot(111)

# Plot m as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, m_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, m_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, m_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, m_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, m_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, m_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, m_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Alfvenic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of m as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig19 = plt.figure()

# Create an axis for this figure
ax19 = fig19.add_subplot(111)

# Plot the residuals as a function of sonic Mach number for each gamma
plt.plot(sonic_mach_arr, residual_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, residual_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, residual_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, residual_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, residual_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, residual_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, residual_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Residuals SF Fit vs Sonic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved
print 'Plot of the residuals as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig20 = plt.figure()

# Create an axis for this figure
ax20 = fig20.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, residual_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, residual_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, residual_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, residual_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, residual_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, residual_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, residual_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Residuals SF Fit vs Alfvenic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the residuals as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude of quad / mono ratio vs sonic Mach number

# Create a figure to display a plot of the integrated magnitude of the quad / 
# mono ratio as a function of sonic Mach number for all of the synchrotron maps,
# i.e. for all gamma
fig21 = plt.figure()

# Create an axis for this figure
ax21 = fig21.add_subplot(111)

# Plot the integrated magnitude of the quad / mono ratio as a function of sonic 
# Mach number for each gamma
plt.plot(sonic_mach_arr, int_quad_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, int_quad_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, int_quad_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, int_quad_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, int_quad_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, int_quad_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, int_quad_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Integrated Mag quad/mono vs Sonic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the sonic Mach number has been saved
print 'Plot of the integrated quad/mono ratio as a function of sonic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude of the quad / mono ratio vs Alfvenic Mach number

# Create a figure to display a plot of the integrated magnitude of the quad/mono
# ratio as a function of Alfvenic Mach number for all of the synchrotron maps, 
# i.e. for all gamma
fig22 = plt.figure()

# Create an axis for this figure
ax22 = fig22.add_subplot(111)

# Plot the integrated magnitude of the quad / mono ratio as a function of 
# Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, int_quad_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, int_quad_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, int_quad_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, int_quad_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, int_quad_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, int_quad_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, int_quad_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Integrated Mag quad/mono vs Alfvenic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the Alfvenic Mach number has been saved
print 'Plot of the integrated quad/mono as a function of Alfvenic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of the quad / mono ratio at a point vs sonic Mach number

# Create a figure to display a plot of the magnitude of the quad/mono ratio at a
# point as a function of sonic Mach number for all of the synchrotron maps, i.e.
# for all gamma
fig23 = plt.figure()

# Create an axis for this figure
ax23 = fig23.add_subplot(111)

# Plot the magnitude of the quad/mono ratio as a point as a function of sonic 
# Mach number for each gamma
plt.plot(sonic_mach_arr, quad_point_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, quad_point_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, quad_point_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, quad_point_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, quad_point_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, quad_point_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, quad_point_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Sonic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio at
# a single radius value as a function of the sonic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of sonic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of the quad / mono ratio at a point vs Alfvenic Mach number

# Create a figure to display a plot of the magnitude of the quad/mono ratio at a
# point as a function of Alfvenic Mach number for all of the synchrotron maps, 
# i.e. for all gamma
fig24 = plt.figure()

# Create an axis for this figure
ax24 = fig24.add_subplot(111)

# Plot the magnitude of the quad/mono ratio at a point as a function of Alfvenic
# Mach number for each gamma
plt.plot(alf_mach_arr, quad_point_x_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, quad_point_x_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, quad_point_x_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, quad_point_x_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, quad_point_x_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, quad_point_x_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, quad_point_x_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Alfvenic Mach Number x', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio
# at a point as a function of the Alfvenic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of Alfvenic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()

#--------------------------- Plots against gamma -------------------------------

# Skewness vs gamma z-LOS low magnetic field

# Create a figure to display a plot of the skewness as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig25 = plt.figure()

# Create an axis for this figure
ax25 = fig25.add_subplot(111)

# Plot the skewness as a function of gamma for each simulation
plt.errorbar(gamma_arr, skew_mean_arr[:,1], fmt ='bo',\
	label ='{}'.format(short_simul[1]), yerr = skew_err_arr[:,1])
plt.errorbar(gamma_arr, skew_mean_arr[:,3], fmt ='ro',\
	label ='{}'.format(short_simul[3]), yerr = skew_err_arr[:,3])
plt.errorbar(gamma_arr, skew_mean_arr[:,5], fmt ='go',\
	label ='{}'.format(short_simul[5]), yerr = skew_err_arr[:,5])
plt.errorbar(gamma_arr, skew_mean_arr[:,7], fmt ='co',\
	label ='{}'.format(short_simul[7]), yerr = skew_err_arr[:,7])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mean Skew vs Gamma B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_gamma_mean_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of gamma has been saved
print 'Plot of the skewness as a function of gamma saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs gamma z-LOS high magnetic field

# Create a figure to display a plot of the skewness as a function of gamma
# for some of the synchrotron maps with B = 1
fig26 = plt.figure()

# Create an axis for this figure
ax26 = fig26.add_subplot(111)

# Plot the skewness as a function of gamma for each simulation
plt.errorbar(gamma_arr, skew_mean_arr[:,9], fmt ='bo',\
	label ='{}'.format(short_simul[9]), yerr = skew_err_arr[:,9])
plt.errorbar(gamma_arr, skew_mean_arr[:,11], fmt ='ro',\
	label ='{}'.format(short_simul[11]), yerr = skew_err_arr[:,11])
plt.errorbar(gamma_arr, skew_mean_arr[:,13], fmt ='go',\
	label ='{}'.format(short_simul[13]), yerr = skew_err_arr[:,13])
plt.errorbar(gamma_arr, skew_mean_arr[:,15], fmt ='co',\
	label ='{}'.format(short_simul[15]), yerr = skew_err_arr[:,15])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mean Skew vs Gamma B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_gamma_mean_b1.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of gamma has been saved
print 'Plot of the skewness as a function of gamma saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs gamma x-LOS low magnetic field

# Create a figure to display a plot of the skewness as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig27 = plt.figure()

# Create an axis for this figure
ax27 = fig27.add_subplot(111)

# Plot the skewness as a function of gamma for each simulation
plt.plot(gamma_arr, skew_x_arr[:,1], 'bo',\
	label ='{}'.format(short_simul[1]))
plt.plot(gamma_arr, skew_x_arr[:,3], 'ro',\
	label ='{}'.format(short_simul[3]))
plt.plot(gamma_arr, skew_x_arr[:,5], 'go',\
	label ='{}'.format(short_simul[5]))
plt.plot(gamma_arr, skew_x_arr[:,7], 'co',\
	label ='{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Gamma x LOS B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_gamma_x_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of gamma has been saved
print 'Plot of the skewness as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs gamma x-LOS high magnetic field

# Create a figure to display a plot of the skewness as a function of gamma
# for some of the synchrotron maps with B = 1
fig28 = plt.figure()

# Create an axis for this figure
ax28 = fig28.add_subplot(111)

# Plot the skewness as a function of gamma for each simulation
plt.plot(gamma_arr, skew_x_arr[:,9], 'bo',\
	label ='{}'.format(short_simul[9]))
plt.plot(gamma_arr, skew_x_arr[:,11], 'ro',\
	label ='{}'.format(short_simul[11]))
plt.plot(gamma_arr, skew_x_arr[:,13], 'go',\
	label ='{}'.format(short_simul[13]))
plt.plot(gamma_arr, skew_x_arr[:,15], 'co',\
	label ='{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Gamma x LOS B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_gamma_x_b1.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of gamma has been saved
print 'Plot of the skewness as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs gamma z-LOS low magnetic field

# Create a figure to display a plot of the kurtosis as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig29 = plt.figure()

# Create an axis for this figure
ax29 = fig29.add_subplot(111)

# Plot the kurtosis as a function of gamma for each simulation
plt.errorbar(gamma_arr, kurt_mean_arr[:,1], fmt ='bo',\
	label ='{}'.format(short_simul[1]), yerr = kurt_err_arr[:,1])
plt.errorbar(gamma_arr, kurt_mean_arr[:,3], fmt ='ro',\
	label ='{}'.format(short_simul[3]), yerr = kurt_err_arr[:,3])
plt.errorbar(gamma_arr, kurt_mean_arr[:,5], fmt ='go',\
	label ='{}'.format(short_simul[5]), yerr = kurt_err_arr[:,5])
plt.errorbar(gamma_arr, kurt_mean_arr[:,7], fmt ='co',\
	label ='{}'.format(short_simul[7]), yerr = kurt_err_arr[:,7])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Change the y limit axes to make the plot easier to read
plt.ylim((-5,25))

# Add a title to the plot
plt.title('Mean Kurtosis vs Gamma B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_gamma_mean_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of gamma has been saved
print 'Plot of the kurtosis as a function of gamma saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs gamma z-LOS high magnetic field

# Create a figure to display a plot of the kurtosis as a function of gamma
# for some of the synchrotron maps with B = 1
fig30 = plt.figure()

# Create an axis for this figure
ax30 = fig30.add_subplot(111)

# Plot the kurtosis as a function of gamma for each simulation
plt.errorbar(gamma_arr, kurt_mean_arr[:,9], fmt ='bo',\
	label ='{}'.format(short_simul[9]), yerr = kurt_err_arr[:,9])
plt.errorbar(gamma_arr, kurt_mean_arr[:,11], fmt ='ro',\
	label ='{}'.format(short_simul[11]), yerr = kurt_err_arr[:,11])
plt.errorbar(gamma_arr, kurt_mean_arr[:,13], fmt ='go',\
	label ='{}'.format(short_simul[13]), yerr = kurt_err_arr[:,13])
plt.errorbar(gamma_arr, kurt_mean_arr[:,15], fmt ='co',\
	label ='{}'.format(short_simul[15]), yerr = kurt_err_arr[:,15])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Mean Kurtosis vs Gamma B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_gamma_mean_b1.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of gamma has been saved
print 'Plot of the kurtosis as a function of gamma saved z'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs gamma x-LOS low magnetic field

# Create a figure to display a plot of the kurtosis as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig31 = plt.figure()

# Create an axis for this figure
ax31 = fig31.add_subplot(111)

# Plot the kurtosis as a function of gamma for each simulation
plt.plot(gamma_arr, kurt_x_arr[:,1], 'bo',\
	label ='{}'.format(short_simul[1]))
plt.plot(gamma_arr, kurt_x_arr[:,3], 'ro',\
	label ='{}'.format(short_simul[3]))
plt.plot(gamma_arr, kurt_x_arr[:,5], 'go',\
	label ='{}'.format(short_simul[5]))
plt.plot(gamma_arr, kurt_x_arr[:,7], 'co',\
	label ='{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Gamma x LOS B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_gamma_x_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of gamma has been saved
print 'Plot of the kurtosis as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs gamma x-LOS high magnetic field

# Create a figure to display a plot of the kurtosis as a function of gamma
# for some of the synchrotron maps with B = 1
fig32 = plt.figure()

# Create an axis for this figure
ax32 = fig32.add_subplot(111)

# Plot the kurtosis as a function of gamma for each simulation
plt.plot(gamma_arr, kurt_x_arr[:,9], 'bo',\
	label ='{}'.format(short_simul[9]))
plt.plot(gamma_arr, kurt_x_arr[:,11], 'ro',\
	label ='{}'.format(short_simul[11]))
plt.plot(gamma_arr, kurt_x_arr[:,13], 'go',\
	label ='{}'.format(short_simul[13]))
plt.plot(gamma_arr, kurt_x_arr[:,15], 'co',\
	label ='{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Gamma x LOS B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_gamma_x_b1.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of gamma has been saved
print 'Plot of the kurtosis as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()

# m vs gamma z-LOS low magnetic field

# Create a figure to display a plot of m as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig33 = plt.figure()

# Create an axis for this figure
ax33 = fig33.add_subplot(111)

# Plot m as a function of gamma for each simulation
plt.errorbar(gamma_arr, m_mean_arr[:,1], fmt ='bo',\
	label ='{}'.format(short_simul[1]), yerr = m_err_arr[:,1])
plt.errorbar(gamma_arr, m_mean_arr[:,3], fmt ='ro',\
	label ='{}'.format(short_simul[3]), yerr = m_err_arr[:,3])
plt.errorbar(gamma_arr, m_mean_arr[:,5], fmt ='go',\
	label ='{}'.format(short_simul[5]), yerr = m_err_arr[:,5])
plt.errorbar(gamma_arr, m_mean_arr[:,7], fmt ='co',\
	label ='{}'.format(short_simul[7]), yerr = m_err_arr[:,7])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mean SF Slope - 1 vs Gamma B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_gamma_mean_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of m as a 
# function of gamma has been saved
print 'Plot of m as a function of gamma saved mean'

# Close the figure, now that it has been saved.
plt.close()

# m vs gamma z-LOS high magnetic field

# Create a figure to display a plot of m as a function of gamma
# for some of the synchrotron maps with B = 1
fig34 = plt.figure()

# Create an axis for this figure
ax34 = fig34.add_subplot(111)

# Plot m as a function of gamma for each simulation
plt.errorbar(gamma_arr, m_mean_arr[:,9], fmt ='bo',\
	label ='{}'.format(short_simul[9]), yerr = m_err_arr[:,9])
plt.errorbar(gamma_arr, m_mean_arr[:,11], fmt ='ro',\
	label ='{}'.format(short_simul[11]), yerr = m_err_arr[:,11])
plt.errorbar(gamma_arr, m_mean_arr[:,13], fmt ='go',\
	label ='{}'.format(short_simul[13]), yerr = m_err_arr[:,13])
plt.errorbar(gamma_arr, m_mean_arr[:,15], fmt ='co',\
	label ='{}'.format(short_simul[15]), yerr = m_err_arr[:,15])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mean SF Slope - 1 vs Gamma B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_gamma_mean_b1.png', format = 'png')

# Print a message to the screen to show that the plot of m as a 
# function of gamma has been saved
print 'Plot of m as a function of gamma saved mean'

# Close the figure, now that it has been saved.
plt.close()

# m vs gamma x-LOS low magnetic field

# Create a figure to display a plot of m as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig35 = plt.figure()

# Create an axis for this figure
ax35 = fig35.add_subplot(111)

# Plot m as a function of gamma for each simulation
plt.plot(gamma_arr, m_x_arr[:,1], 'bo',\
	label ='{}'.format(short_simul[1]))
plt.plot(gamma_arr, m_x_arr[:,3], 'ro',\
	label ='{}'.format(short_simul[3]))
plt.plot(gamma_arr, m_x_arr[:,5], 'go',\
	label ='{}'.format(short_simul[5]))
plt.plot(gamma_arr, m_x_arr[:,7], 'co',\
	label ='{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Gamma x LOS B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_gamma_x_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of m as a 
# function of gamma has been saved
print 'Plot of m as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()

# m vs gamma x-LOS high magnetic field

# Create a figure to display a plot of m as a function of gamma
# for some of the synchrotron maps with B = 1
fig36 = plt.figure()

# Create an axis for this figure
ax36 = fig36.add_subplot(111)

# Plot m as a function of gamma for each simulation
plt.plot(gamma_arr, m_x_arr[:,9], 'bo',\
	label ='{}'.format(short_simul[9]))
plt.plot(gamma_arr, m_x_arr[:,11], 'ro',\
	label ='{}'.format(short_simul[11]))
plt.plot(gamma_arr, m_x_arr[:,13], 'go',\
	label ='{}'.format(short_simul[13]))
plt.plot(gamma_arr, m_x_arr[:,15], 'co',\
	label ='{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Gamma x LOS B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_gamma_x_b1.png', format = 'png')

# Print a message to the screen to show that the plot of m as a 
# function of gamma has been saved
print 'Plot of m as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()

# Int quad ratio vs gamma z-LOS low magnetic field

# Create a figure to display a plot of int quad ratio as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig33 = plt.figure()

# Create an axis for this figure
ax33 = fig33.add_subplot(111)

# Plot int quad ratio as a function of gamma for each simulation
plt.errorbar(gamma_arr, int_quad_mean_arr[:,1], fmt ='bo',\
	label ='{}'.format(short_simul[1]), yerr = int_quad_err_arr[:,1])
plt.errorbar(gamma_arr, int_quad_mean_arr[:,3], fmt ='ro',\
	label ='{}'.format(short_simul[3]), yerr = int_quad_err_arr[:,3])
plt.errorbar(gamma_arr, int_quad_mean_arr[:,5], fmt ='go',\
	label ='{}'.format(short_simul[5]), yerr = int_quad_err_arr[:,5])
plt.errorbar(gamma_arr, int_quad_mean_arr[:,7], fmt ='co',\
	label ='{}'.format(short_simul[7]), yerr = int_quad_err_arr[:,7])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Add a title to the plot
plt.title('Mean Int Quad Ratio vs Gamma B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_gamma_mean_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of int quad ratio as a 
# function of gamma has been saved
print 'Plot of int quad ratio as a function of gamma saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Int quad ratio vs gamma z-LOS high magnetic field

# Create a figure to display a plot of int quad ratio as a function of gamma
# for some of the synchrotron maps with B = 1
fig34 = plt.figure()

# Create an axis for this figure
ax34 = fig34.add_subplot(111)

# Plot int quad ratio as a function of gamma for each simulation
plt.errorbar(gamma_arr, int_quad_mean_arr[:,9], fmt ='bo',\
	label ='{}'.format(short_simul[9]), yerr = int_quad_err_arr[:,9])
plt.errorbar(gamma_arr, int_quad_mean_arr[:,11], fmt ='ro',\
	label ='{}'.format(short_simul[11]), yerr = int_quad_err_arr[:,11])
plt.errorbar(gamma_arr, int_quad_mean_arr[:,13], fmt ='go',\
	label ='{}'.format(short_simul[13]), yerr = int_quad_err_arr[:,13])
plt.errorbar(gamma_arr, int_quad_mean_arr[:,15], fmt ='co',\
	label ='{}'.format(short_simul[15]), yerr = int_quad_err_arr[:,15])

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Add a title to the plot
plt.title('Mean Int Quad Ratio vs Gamma B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_gamma_mean_b1.png', format = 'png')

# Print a message to the screen to show that the plot of int quad ratio as a 
# function of gamma has been saved
print 'Plot of int quad ratio as a function of gamma saved mean'

# Close the figure, now that it has been saved.
plt.close()

# Int quad ratio vs gamma x-LOS low magnetic field

# Create a figure to display a plot of int quad ratio as a function of gamma
# for some of the synchrotron maps with B = 0.1
fig35 = plt.figure()

# Create an axis for this figure
ax35 = fig35.add_subplot(111)

# Plot int quad ratio as a function of gamma for each simulation
plt.plot(gamma_arr, int_quad_x_arr[:,1], 'bo',\
	label ='{}'.format(short_simul[1]))
plt.plot(gamma_arr, int_quad_x_arr[:,3], 'ro',\
	label ='{}'.format(short_simul[3]))
plt.plot(gamma_arr, int_quad_x_arr[:,5], 'go',\
	label ='{}'.format(short_simul[5]))
plt.plot(gamma_arr, int_quad_x_arr[:,7], 'co',\
	label ='{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Add a title to the plot
plt.title('Int Quad Ratio vs Gamma x LOS B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_gamma_x_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of int quad ratio as a 
# function of gamma has been saved
print 'Plot of int quad ratio as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()

# Int quad ratio vs gamma x-LOS high magnetic field

# Create a figure to display a plot of int quad ratio as a function of gamma
# for some of the synchrotron maps with B = 1
fig36 = plt.figure()

# Create an axis for this figure
ax36 = fig36.add_subplot(111)

# Plot int quad ratio as a function of gamma for each simulation
plt.plot(gamma_arr, int_quad_x_arr[:,9], 'bo',\
	label ='{}'.format(short_simul[9]))
plt.plot(gamma_arr, int_quad_x_arr[:,11], 'ro',\
	label ='{}'.format(short_simul[11]))
plt.plot(gamma_arr, int_quad_x_arr[:,13], 'go',\
	label ='{}'.format(short_simul[13]))
plt.plot(gamma_arr, int_quad_x_arr[:,15], 'co',\
	label ='{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Add a title to the plot
plt.title('Int Quad Ratio vs Gamma x LOS B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_gamma_x_b1.png', format = 'png')

# Print a message to the screen to show that the plot of int quad ratio as a 
# function of gamma has been saved
print 'Plot of int quad ratio as a function of gamma saved x'

# Close the figure, now that it has been saved.
plt.close()