#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the skewness, kurtosis, structure function slope, and         #
# integrated quadrupole ratio as a function of the angle between the line of   #
# sight and the mean magnetic field, for simulations with a small and large    #
# magnetic field.                                                              #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 13/2/2015                                                        #
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
	# The next step is to calculate the standard error of the mean of each
	# statistic
	skew_err = np.std(skew_val) / np.sqrt(len(skew_val))
	kurt_err = np.std(kurt_val) / np.sqrt(len(kurt_val))
	m_err = np.std(m_val) / np.sqrt(len(m_val))
	residual_err = np.std(resid_val) / np.sqrt(len(resid_val))
	int_quad_err = np.std(int_quad_val) / np.sqrt(len(int_quad_val))

	# Now that all of the calculations have been performed, return the 
	# calculated errors
	return skew_err, kurt_err, m_err, residual_err, int_quad_err

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
low_B_short_M = ['Ms7.02Ma1.76', 'Ms2.38Ma1.86', 'Ms0.83Ma1.74', 'Ms0.45Ma1.72']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# high magnetic field simulations used to produce plots
high_B_short_M = ['Ms6.78Ma0.52', 'Ms2.41Ma0.67', 'Ms0.87Ma0.7', 'Ms0.48Ma0.65']

# Create an array specifying the sonic Mach numbers for the low and
# high magnetic field simulations
low_B_mach = [7.02, 2.38, 0.83, 0.45]
high_B_mach = [6.78, 2.41, 0.87, 0.48]

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

# Create an array that specifies the rotation angles relative to the z axis of
# the MHD cubes, of the synchrotron maps to be used
rot_ang_arr = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,\
	80.0, 90.0]) 

# Create an array that just specifies the relative angle between the line of
# sight and the mean magnetic field direction
rel_ang_arr = 90.0 - rot_ang_arr

# Create a two dimensional array that will hold all of the skew values for the
# different low magnetic field simulations. The first index gives the simulation
# the second gives the angle between the line of sight and the mean magnetic 
# field. Do this for lines of sight along the y and z axes.
skew_low_arr_y = np.zeros((len(low_B_sims), len(rot_ang_arr)))
skew_low_arr_z = np.zeros((len(low_B_sims), len(rot_ang_arr)))

# Create a two dimensional array that will hold all of the skew values for the
# different high magnetic field simulations. The first index gives the simulation
# the second gives the angle between the line of sight and the mean magnetic 
# field. Do this for lines of sight along the y and z axes.
skew_high_arr_y = np.zeros((len(high_B_sims), len(rot_ang_arr)))
skew_high_arr_z = np.zeros((len(high_B_sims), len(rot_ang_arr)))

# Create a two dimensional array that will hold all of the kurtosis values for 
# the different low magnetic field simulations. The first index gives the 
# simulation the second gives the angle between the line of sight and the mean 
# magnetic field. Do this for lines of sight along the y and z axes.
kurt_low_arr_y = np.zeros((len(low_B_sims), len(rot_ang_arr)))
kurt_low_arr_z = np.zeros((len(low_B_sims), len(rot_ang_arr)))

# Create a two dimensional array that will hold all of the kurtosis values for
# the different high magnetic field simulations. The first index gives the
# simulation the second gives the angle between the line of sight and the mean 
# magnetic field. Do this for lines of sight along the y and z axes.
kurt_high_arr_y = np.zeros((len(high_B_sims), len(rot_ang_arr)))
kurt_high_arr_z = np.zeros((len(high_B_sims), len(rot_ang_arr)))

# Create a two dimensional array that will hold all of the structure function 
# slope values for the different low magnetic field simulations. The first index
# gives the simulation the second gives the angle between the line of sight and 
# the mean magnetic field. Do this for lines of sight along the y and z axes.
sf_low_arr_y = np.zeros((len(low_B_sims), len(rot_ang_arr)))
sf_low_arr_z = np.zeros((len(low_B_sims), len(rot_ang_arr)))

# Create a two dimensional array that will hold all of the structure function
# slope values for the different high magnetic field simulations. The first 
# index gives the simulation the second gives the angle between the line of 
# sight and the mean magnetic field. Do this for lines of sight along the y and
# z axes.
sf_high_arr_y = np.zeros((len(high_B_sims), len(rot_ang_arr)))
sf_high_arr_z = np.zeros((len(high_B_sims), len(rot_ang_arr)))

# Create a two dimensional array that will hold all of the integrated quadrupole
# ratio values for the different low magnetic field simulations. The first index
# gives the simulation the second gives the angle between the line of sight and 
# the mean magnetic field. Do this for lines of sight along the y and z axes.
quad_low_arr_y = np.zeros((len(low_B_sims), len(rot_ang_arr)))
quad_low_arr_z = np.zeros((len(low_B_sims), len(rot_ang_arr)))

# Create a two dimensional array that will hold all of the integrated quadrupole
# ratio values for the different high magnetic field simulations. The first 
# index gives the simulation the second gives the angle between the line of 
# sight and the mean magnetic field. Do this for lines of sight along the y and 
# z axes.
quad_high_arr_y = np.zeros((len(high_B_sims), len(rot_ang_arr)))
quad_high_arr_z = np.zeros((len(high_B_sims), len(rot_ang_arr)))

# Create error arrays for each of the statistics. These errors are only for the
# statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field), and are calculated by the standard deviation of the 
# statistics calculated for sub-images of the synchrotron maps.
skew_err_low_arr = np.zeros((len(low_B_sims), len(rot_ang_arr)))
kurt_err_low_arr = np.zeros((len(low_B_sims), len(rot_ang_arr)))
m_err_low_arr = np.zeros((len(low_B_sims), len(rot_ang_arr)))
residual_err_low_arr = np.zeros((len(low_B_sims), len(rot_ang_arr)))
int_quad_err_low_arr = np.zeros((len(low_B_sims), len(rot_ang_arr)))
# For high magnetic field simulations
skew_err_high_arr = np.zeros((len(high_B_sims), len(rot_ang_arr)))
kurt_err_high_arr = np.zeros((len(high_B_sims), len(rot_ang_arr)))
m_err_high_arr = np.zeros((len(high_B_sims), len(rot_ang_arr)))
residual_err_high_arr = np.zeros((len(high_B_sims), len(rot_ang_arr)))
int_quad_err_high_arr = np.zeros((len(high_B_sims), len(rot_ang_arr)))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for i in range(len(low_B_sims)):
	# Create a string for the full directory path to use in the calculation for
	# low and high magnetic field simulations
	data_loc_low = simul_loc + low_B_sims[i]
	data_loc_high = simul_loc + high_B_sims[i]

	# Print a message to the screen to show what simulation group is being used
	print 'Starting calculation for simulation group {}'.format(i)

	# Loop over the values for the angle between the line of sight and the mean
	# magnetic field, to calculate the skewness, kurtosis, structure
	# function slope and integrated quadrupole ratio for the low and high
	# magnetic field simulations
	for j in range(len(rot_ang_arr)):
		# Open the FITS file that contains the simulated synchrotron intensity
		# map for this line of sight, for low and high magnetic fields
		sync_fits_low_y = fits.open(data_loc_low + 'synint_p1-4y_rot_{}.fits'.\
			format(rot_ang_arr[j]))
		sync_fits_high_y = fits.open(data_loc_high + 'synint_p1-4y_rot_{}.fits'.\
			format(rot_ang_arr[j]))
		# For z LOS
		sync_fits_low_z = fits.open(data_loc_low + 'synint_p1-4_rot_{}.fits'.\
			format(rot_ang_arr[j]))
		sync_fits_high_z = fits.open(data_loc_high + 'synint_p1-4_rot_{}.fits'.\
			format(rot_ang_arr[j]))

		# Extract the data for the simulated synchrotron intensities for the current
		# low and high magnetic field simulations
		sync_data_low_y = sync_fits_low_y[0].data
		sync_data_high_y = sync_fits_high_y[0].data
		# For z LOS
		sync_data_low_z = sync_fits_low_z[0].data
		sync_data_high_z = sync_fits_high_z[0].data

		# Extract the synchrotron intensity map for the value of gamma, for
		# low and high magnetic field simulations
		sync_map_low_y = sync_data_low_y[gam_index]
		sync_map_high_y = sync_data_high_y[gam_index]
		# For z LOS
		sync_map_low_z = sync_data_low_z[gam_index]
		sync_map_high_z = sync_data_high_z[gam_index]

		# Flatten the synchrotron intensity maps for the value of gamma, for
		# low and high magnetic field simulations
		flat_sync_low_y = sync_map_low_y.flatten()
		flat_sync_high_y = sync_map_high_y.flatten()
		# For z LOS
		flat_sync_low_z = sync_map_low_z.flatten()
		flat_sync_high_z = sync_map_high_z.flatten()

		# Calculate the biased skewness of the synchrotron intensity maps, for
		# low and high magnetic field simulations, and store the results in the
		# corresponding array.
		skew_low_arr_y[i,j] = stats.skew(flat_sync_low_y)
		skew_high_arr_y[i,j] = stats.skew(flat_sync_high_y)
		# For z LOS
		skew_low_arr_z[i,j] = stats.skew(flat_sync_low_z)
		skew_high_arr_z[i,j] = stats.skew(flat_sync_high_z)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# maps, for low and high magnetic field simulations, and store the 
		# results in the corresponding array.
		kurt_low_arr_y[i,j] = stats.kurtosis(flat_sync_low_y)
		kurt_high_arr_y[i,j] = stats.kurtosis(flat_sync_high_y)
		# For z LOS
		kurt_low_arr_z[i,j] = stats.kurtosis(flat_sync_low_z)
		kurt_high_arr_z[i,j] = stats.kurtosis(flat_sync_high_z)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity maps, for the low and high magnetic field simulations. Note 
		# that no_fluct = True is set, because we are not subtracting the mean
		# from the synchrotron maps before calculating the structure function.
		strfn_low_y = sf_fft(sync_map_low_y, no_fluct = True)
		strfn_high_y = sf_fft(sync_map_high_y, no_fluct = True)
		# For z LOS
		strfn_low_z = sf_fft(sync_map_low_z, no_fluct = True)
		strfn_high_z = sf_fft(sync_map_high_z, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins, for low and high magnetic field simulations.
		rad_sf_low_y = sfr(strfn_low_y, num_bins, verbose = False)
		rad_sf_high_y = sfr(strfn_high_y, num_bins, verbose = False)
		# For z LOS
		rad_sf_low_z = sfr(strfn_low_z, num_bins, verbose = False)
		rad_sf_high_z = sfr(strfn_high_z, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function for low 
		# and high magnetic field simulations
		sf_low_y = rad_sf_low_y[1]
		sf_high_y = rad_sf_high_y[1]
		# For z LOS
		sf_low_z = rad_sf_low_z[1]
		sf_high_z = rad_sf_high_z[1]

		# Extract the radius values used to calculate this structure function,
		# for low and high magnetic field simulations.
		sf_rad_arr_low_y = rad_sf_low_y[0]
		sf_rad_arr_high_y = rad_sf_high_y[0]
		# For z LOS
		sf_rad_arr_low_z = rad_sf_low_z[0]
		sf_rad_arr_high_z = rad_sf_high_z[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. Perform a linear fit for the low magnetic 
		# field simulation
		spec_ind_data_low_y = np.polyfit(np.log10(\
			sf_rad_arr_low_y[11:16]),\
			np.log10(sf_low_y[11:16]), 1, full = True)
		# Perform a linear fit for the high magnetic field simulation
		spec_ind_data_high_y = np.polyfit(np.log10(\
			sf_rad_arr_high_y[11:16]),\
			np.log10(sf_high_y[11:16]), 1, full = True)
		# For z LOS
		spec_ind_data_low_z = np.polyfit(np.log10(\
			sf_rad_arr_low_z[11:16]),\
			np.log10(sf_low_z[11:16]), 1, full = True)
		# Perform a linear fit for the high magnetic field simulation
		spec_ind_data_high_z = np.polyfit(np.log10(\
			sf_rad_arr_high_z[11:16]),\
			np.log10(sf_high_z[11:16]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit, for low and
		# high magnetic field simulations
		coeff_low_y = spec_ind_data_low_y[0]
		coeff_high_y = spec_ind_data_high_y[0]
		# For z LOS
		coeff_low_z = spec_ind_data_low_z[0]
		coeff_high_z = spec_ind_data_high_z[0]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array, for low and high magnetic field 
		# simulations
		sf_low_arr_y[i,j] = coeff_low_y[0]-1.0
		sf_high_arr_y[i,j] = coeff_high_y[0]-1.0
		# For z LOS
		sf_low_arr_z[i,j] = coeff_low_z[0]-1.0
		sf_high_arr_z[i,j] = coeff_high_z[0]-1.0

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn_low_y = sf_fft(sync_map_low_y, no_fluct = True, normalise = True)
		norm_strfn_high_y = sf_fft(sync_map_high_y, no_fluct = True, normalise = True)
		# For z LOS
		norm_strfn_low_z = sf_fft(sync_map_low_z, no_fluct = True, normalise = True)
		norm_strfn_high_z = sf_fft(sync_map_high_z, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image. This is done for low and high 
		# magnetic field simulations
		norm_strfn_low_y = np.fft.fftshift(norm_strfn_low_y)
		norm_strfn_high_y = np.fft.fftshift(norm_strfn_high_y)
		# For z LOS
		norm_strfn_low_z = np.fft.fftshift(norm_strfn_low_z)
		norm_strfn_high_z = np.fft.fftshift(norm_strfn_high_z)

		# Calculate the magnitude and argument of the quadrupole ratio, for 
		# low and high magnetic field simulations
		quad_mod_low_y, quad_arg_low_y, quad_rad_low_y = calc_quad_ratio(norm_strfn_low_y, num_bins)
		quad_mod_high_y, quad_arg_high_y, quad_rad_high_y = calc_quad_ratio(norm_strfn_high_y, num_bins)
		# For z LOS
		quad_mod_low_z, quad_arg_low_z, quad_rad_low_z = calc_quad_ratio(norm_strfn_low_z, num_bins)
		quad_mod_high_z, quad_arg_high_z, quad_rad_high_z = calc_quad_ratio(norm_strfn_high_z, num_bins)

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
		quad_low_arr_y[i,j] = np.trapz(quad_mod_low_y[11:20], dx = 1.0) / (19 - 11)
		quad_high_arr_y[i,j] = np.trapz(quad_mod_high_y[11:20], dx = 1.0) / (19 - 11)
		# For z LOS
		quad_low_arr_z[i,j] = np.trapz(quad_mod_low_z[11:20], dx = 1.0) / (19 - 11)
		quad_high_arr_z[i,j] = np.trapz(quad_mod_high_z[11:20], dx = 1.0) / (19 - 11)

		# Create errors for each of the statistics. These errors are only for the
		# statistics calculated from the y and z axes (perpendicular to the mean 
		# magnetic field), and are calculated by the standard deviation of the 
		# statistics calculated for sub-images of the synchrotron maps.
		skew_err_low_arr[i,j], kurt_err_low_arr[i,j], m_err_low_arr[i,j],\
		residual_err_low_arr[i,j], int_quad_err_low_arr[i,j]\
		= calc_err_bootstrap(sync_map_low_y, sync_map_low_z)
		skew_err_high_arr[i,j], kurt_err_high_arr[i,j], m_err_high_arr[i,j],\
		residual_err_high_arr[i,j], int_quad_err_high_arr[i,j]\
		= calc_err_bootstrap(sync_map_high_y, sync_map_high_z)

		# Close the FITS files, now that we are finished using them, to save
		# memory
		sync_fits_low_y.close()
		sync_fits_high_y.close()
		# For z LOS
		sync_fits_low_z.close()
		sync_fits_high_z.close()

	# Print a message to show that the calculation has finished successfully
	# for this simulation group
	print 'All statistics calculated for simulation group {}'.format(i)

# Create mean value arrays for each of the statistics. These values are only for
# the statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field), for y and z lines of sight
skew_mean_low_arr = (skew_low_arr_y + skew_low_arr_z) / 2.0
kurt_mean_low_arr = (kurt_low_arr_y + kurt_low_arr_z) / 2.0
m_mean_low_arr = (sf_low_arr_y + sf_low_arr_z) / 2.0
int_quad_mean_low_arr = (quad_low_arr_y + quad_low_arr_z) / 2.0
# For high magnetic field simulations
skew_mean_high_arr = (skew_high_arr_y + skew_high_arr_z) / 2.0
kurt_mean_high_arr = (kurt_high_arr_y + kurt_high_arr_z) / 2.0
m_mean_high_arr = (sf_high_arr_y + sf_high_arr_z) / 2.0
int_quad_mean_high_arr = (quad_high_arr_y + quad_high_arr_z) / 2.0

# When the code reaches this point, the statistics have been saved for every 
# simulation, so start making the final plots.

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
for i in range(len(low_B_sims)):
	# Plot the skewness for this simulation, against the relative angle between
	# the line of sight and the mean magnetic field
	plt.errorbar(rel_ang_arr, skew_mean_low_arr[i], fmt='-' + symbol_arr[i],\
	 label = '{}'.format(low_B_short_M[i]),yerr=skew_err_low_arr[i])

# Force the legends to appear on the plot
plt.legend(loc = 4, fontsize = 8, numpoints=1)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the 
# skewness of high magnetic field simulations. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Loop over the high magnetic field simulations to produce plots for each simulation
for i in range(len(high_B_sims)):
	# Plot the skewness for this simulation, against the relative angle between
	# the line of sight and the mean magnetic field
	plt.errorbar(rel_ang_arr, skew_mean_high_arr[i], fmt='-' + symbol_arr[i],\
	 label = '{}'.format(high_B_short_M[i]),yerr=skew_err_high_arr[i])

# Force the legends to appear on the plot
plt.legend(loc = 1, fontsize = 8, numpoints=1)

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the kurtosis 
# of low magnetic field simulations. Make the x axis limits the same as
# for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Loop over the low magnetic field simulations to produce plots for each simulation
for i in range(len(low_B_sims)):
	# Plot the kurtosis for this simulation, against the relative angle between
	# the line of sight and the mean magnetic field
	plt.errorbar(rel_ang_arr, kurt_mean_low_arr[i], fmt='-' + symbol_arr[i],\
		yerr=kurt_err_low_arr[i])

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for the 
# kurtosis of high magnetic field simulations. Make the x axis limits the same 
# as for the second plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Loop over the high magnetic field simulation to produce plots for each simulation
for i in range(len(high_B_sims)):
	# Plot the kurtosis for this simulation, against the relative angle between
	# the line of sight and the mean magnetic field
	plt.errorbar(rel_ang_arr, kurt_mean_high_arr[i], fmt='-' + symbol_arr[i],\
		yerr=kurt_err_high_arr[i])

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Angle Between LOS and Mean B [degrees]', ha = 'center', \
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
plt.savefig(simul_loc + 'Publication_Plots/fig12.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#-------------------------- Structure Function Slope ---------------------------

# Here we want to produce one plot with two subplots. There should be one row
# of subplots, with two subplots in the row. The plot on the left should be
# the structure function slope as a function of gamma for low magnetic field 
# simulations, and the plot on the right should be the structure function slope
# as a function of gamma for high magnetic field simulations.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,5), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the 
# structure function slope of low magnetic field simulations
ax1 = fig.add_subplot(121)

# Loop over the simulations to plot structure function slope vs the relative 
# angle between the line of sight and the mean magnetic field
for i in range(len(low_B_sims)):
	# Plot the structure function slope as a function of the relative angle 
	# between the line of sight and the mean magnetic field for this simulation
	plt.errorbar(rel_ang_arr, m_mean_low_arr[i], fmt='-' + symbol_arr[i],\
	 label = '{}'.format(low_B_short_M[i]),yerr=m_mean_low_arr[i])

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 10, numpoints=1)

# Add a y-axis label to the plot
plt.ylabel('m', fontsize = 20)

# Create an axis for the second subplot to be produced, which is for the 
# structure function slope of high magnetic field simulations.
ax2 = fig.add_subplot(122, sharey = ax1)

# Loop over the simulations to plot the structure function slope vs the relative
# angle between the line of sight and the mean magnetic field
for i in range(len(high_B_sims)):
	# Plot the structure function slope as a function of the relative angle 
	# between the line of sight and the mean magnetic field for this simulation
	plt.errorbar(rel_ang_arr, m_mean_high_arr[i], fmt='-' + symbol_arr[i],\
	 label = '{}'.format(high_B_short_M[i]),yerr=m_mean_high_arr[i])

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Angle Between LOS and Mean B [degrees]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 10, numpoints=1)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) m, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) m, high B', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig13.eps', format = 'eps')

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

# Loop over the simulations to plot integrated quadrupole ratio vs the relative 
# angle between the line of sight and the mean magnetic field
for i in range(len(low_B_sims)):
	# Plot the integrated quadrupole ratio as a function of the relative angle 
	# between the line of sight and the mean magnetic field for this simulation
	plt.errorbar(rel_ang_arr, int_quad_mean_low_arr[i], fmt='-' + symbol_arr[i],\
	 label = '{}'.format(low_B_short_M[i]),yerr=int_quad_err_low_arr[i])

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 10, numpoints=1)

# Add a y-axis label to the plot
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Create an axis for the second subplot to be produced, which is for the 
# integrated quadrupole ratio of high magnetic field simulations. Make the y axis
# limits the same as for the low magnetic field plot
ax2 = fig.add_subplot(122, sharey = ax1)

# Loop over the simulations to plot the integrated quadrupole ratio vs the 
# relative angle between the line of sight and the mean magnetic field
for i in range(len(high_B_sims)):
	# Plot the integrated quadrupole ratio as a function of the relative angle 
	# between the line of sight and the mean magnetic field for this simulation
	plt.errorbar(rel_ang_arr, int_quad_mean_high_arr[i], fmt='-' + symbol_arr[i],\
	 label = '{}'.format(high_B_short_M[i]),yerr=int_quad_err_high_arr[i])

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Angle Between LOS and Mean B [degrees]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 10, numpoints=1)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.15, 0.95, 'a) Quad Ratio, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.58, 0.95, 'b) Quad Ratio, high B', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig14.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()