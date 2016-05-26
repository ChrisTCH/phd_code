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
# particular value of gamma. For y and z lines of sight.
# NOTE: We will calculate the biased skewness
skew_arr_y = np.zeros(len(simul_arr))
skew_arr_z = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. For y and z lines of sight
# NOTE: We will calculate the biased Fisher kurtosis
kurt_arr_y = np.zeros(len(simul_arr))
kurt_arr_z = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of gamma. For y and z LOS
m_arr_y = np.zeros(len(simul_arr))
m_arr_z = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity 
# image, for the corresponding simulation, for a particular value of gamma. 
# For y and z lines of sight
int_quad_arr_y = np.zeros(len(simul_arr))
int_quad_arr_z = np.zeros(len(simul_arr))

# Create error arrays for each of the statistics. These errors are only for the
# statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field), and are calculated by the standard deviation of the 
# statistics calculated for sub-images of the synchrotron maps.
skew_err_arr = np.zeros(len(simul_arr))
kurt_err_arr = np.zeros(len(simul_arr))
m_err_arr = np.zeros(len(simul_arr))
residual_err_arr = np.zeros(len(simul_arr))
int_quad_err_arr = np.zeros(len(simul_arr))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS files that contain the simulated synchrotron intensity maps,
	# for lines of sight along the y and z axes
	sync_fits_y = fits.open(data_loc + 'synint_p1-4y.fits')
	sync_fits_z = fits.open(data_loc + 'synint_p1-4.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensities observed for different values of gamma, the power
	# law index of the cosmic ray electrons emitting the synchrotron emission.
	sync_data_y = sync_fits_y[0].data
	sync_data_z = sync_fits_z[0].data

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Extract the synchrotron intensity map for the value of gamma, for y and
	# z lines of sight
	sync_map_y = sync_data_y[gam_index]
	sync_map_z = sync_data_z[gam_index]

	# Flatten the synchrotron intensity maps for the value of gamma, for y and
	# z lines of sight
	flat_sync_y = sync_map_y.flatten()
	flat_sync_z = sync_map_z.flatten()

	# Calculate the biased skewness of the synchrotron intensity maps, and store
	# the results in the corresponding array, for y and z lines of sight
	skew_arr_y[j] = stats.skew(flat_sync_y)
	skew_arr_z[j] = stats.skew(flat_sync_z)

	# Calculate the biased Fisher kurtosis of the synchrotron intensity 
	# maps, and store the results in the corresponding array, for y and z LOS
	kurt_arr_y[j] = stats.kurtosis(flat_sync_y)
	kurt_arr_z[j] = stats.kurtosis(flat_sync_z)

	# Calculate the structure function (two-dimensional) of the synchrotron
	# intensity maps. Note that no_fluct = True is set, because we are not
	# subtracting the mean from the synchrotron maps before calculating the 
	# structure function, for y and z lines of sight
	strfn_y = sf_fft(sync_map_y, no_fluct = True)
	strfn_z = sf_fft(sync_map_z, no_fluct = True)

	# Radially average the calculated 2D structure function, using the 
	# specified number of bins, for y and z lines of sight
	rad_sf_y = sfr(strfn_y, num_bins, verbose = False)
	rad_sf_z = sfr(strfn_z, num_bins, verbose = False)

	# Extract the calculated radially averaged structure function, for y and
	# z lines of sight
	sf_y = rad_sf_y[1]
	sf_z = rad_sf_z[1]

	# Extract the radius values used to calculate this structure function, for
	# y and z lines of sight
	sf_rad_arr_y = rad_sf_y[0]
	sf_rad_arr_z = rad_sf_z[0]

	# Calculate the spectral index of the structure function calculated for
	# this value of gamma. Note that only the first third of the structure
	# function is used in the calculation, as this is the part that is 
	# close to a straight line. Do this for y and z lines of sight.
	spec_ind_data_y = np.polyfit(np.log10(\
		sf_rad_arr_y[11:16]),\
		np.log10(sf_y[11:16]), 1, full = True)
	spec_ind_data_z = np.polyfit(np.log10(\
		sf_rad_arr_z[11:16]),\
		np.log10(sf_z[11:16]), 1, full = True)

	# Extract the returned coefficients from the polynomial fit, for y and z
	# lines of sight
	coeff_y = spec_ind_data_y[0]
	coeff_z = spec_ind_data_z[0]

	# Enter the value of m, the slope of the structure function minus 1,
	# into the corresponding array, for y and z lines of sight
	m_arr_y[j] = coeff_y[0]-1.0
	m_arr_z[j] = coeff_z[0]-1.0

	# Calculate the 2D structure function for this slice of the synchrotron
	# intensity data cube. Note that no_fluct = True is set, because we are
	# not subtracting the mean from the synchrotron maps before calculating
	# the structure function. We are also calculating the normalised 
	# structure function, which only takes values between 0 and 2.
	norm_strfn_y = sf_fft(sync_map_y, no_fluct = True, normalise = True)
	norm_strfn_z = sf_fft(sync_map_z, no_fluct = True, normalise = True)

	# Shift the 2D structure function so that the zero radial separation
	# entry is in the centre of the image.
	norm_strfn_y = np.fft.fftshift(norm_strfn_y)
	norm_strfn_z = np.fft.fftshift(norm_strfn_z)

	# Calculate the magnitude and argument of the quadrupole ratio
	quad_mod_y, quad_arg_y, quad_rad_y = calc_quad_ratio(norm_strfn_y, num_bins)
	quad_mod_z, quad_arg_z, quad_rad_z = calc_quad_ratio(norm_strfn_z, num_bins)

	# Integrate the magnitude of the quadrupole / monopole ratio from one sixth 
	# of the way along the radial separation bins, until three quarters of the 
	# way along the radial separation bins. This integration is performed with
	# respect to log separation (i.e. I am ignoring the fact that the 
	# points are equally separated in log space, to calculate the area under
	# the quadrupole / monopole ratio plot when the x axis is scaled 
	# logarithmically). I normalise the value that is returned by dividing
	# by the number of increments in log radial separation used in the
	# calculation. 
	int_quad_arr_y[j] = np.trapz(quad_mod_y[11:20], dx = 1.0) / (19 - 11)
	int_quad_arr_z[j] = np.trapz(quad_mod_z[11:20], dx = 1.0) / (19 - 11)

	# Create errors for each of the statistics. These errors are only for the
	# statistics calculated from the y and z axes (perpendicular to the mean 
	# magnetic field), and are calculated by the standard deviation of the 
	# statistics calculated for sub-images of the synchrotron maps.
	skew_err_arr[j], kurt_err_arr[j], m_err_arr[j],\
	residual_err_arr[j], int_quad_err_arr[j]\
	= calc_err_bootstrap(sync_map_y, sync_map_z)

	# Close the fits files, to save memory
	sync_fits_y.close()
	sync_fits_z.close()

	# Print a message to show that the calculation has finished successfully
	# for this simulation
	print 'All statistics calculated for simulation {}'.format(simul_arr[j])

# Create mean value arrays for each of the statistics. These values are only for
# the statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field).
skew_mean_arr = (skew_arr_y + skew_arr_z) / 2.0
kurt_mean_arr = (kurt_arr_y + kurt_arr_z) / 2.0
m_mean_arr = (m_arr_y + m_arr_z) / 2.0
int_quad_mean_arr = (int_quad_arr_y + int_quad_arr_z) / 2.0

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
plt.errorbar(sonic_mach_arr[0:8], skew_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=skew_err_arr[0:8])
plt.errorbar(sonic_mach_arr[8:16], skew_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=skew_err_arr[8:16])
plt.errorbar(sonic_mach_arr[16:], skew_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=skew_err_arr[16:])

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for skewness
# as a function of Alfvenic Mach number. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Plot the skewness as a function of Alfvenic Mach number
plt.errorbar(alf_mach_arr[0:8], skew_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=skew_err_arr[0:8])
plt.errorbar(alf_mach_arr[8:16], skew_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=skew_err_arr[8:16])
plt.errorbar(alf_mach_arr[16:], skew_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=skew_err_arr[16:])

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for kurtosis 
# as a function of sonic Mach number. Make the x axis limits the same as
# for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Plot the kurtosis as a function of sonic Mach number
plt.errorbar(sonic_mach_arr[0:8], kurt_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=kurt_err_arr[0:8])
plt.errorbar(sonic_mach_arr[8:16], kurt_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=kurt_err_arr[8:16])
plt.errorbar(sonic_mach_arr[16:], kurt_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=kurt_err_arr[16:])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for
# kurtosis as a function of Alfvenic Mach number. Make the x axis limits the
# same as for the second plot, and the y axis limits the same as the third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Plot the kurtosis as a function of Alfvenic Mach number
plt.errorbar(alf_mach_arr[0:8], kurt_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=kurt_err_arr[0:8])
plt.errorbar(alf_mach_arr[8:16], kurt_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=kurt_err_arr[8:16])
plt.errorbar(alf_mach_arr[16:], kurt_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=kurt_err_arr[16:])

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
plt.errorbar(sonic_mach_arr[0:8], m_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=m_err_arr[0:8])
plt.errorbar(sonic_mach_arr[8:16], m_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=m_err_arr[8:16])
plt.errorbar(sonic_mach_arr[16:], m_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=m_err_arr[16:])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a y-axis label to the plot
plt.ylabel('m', fontsize = 20)

# Create an axis for the second subplot to be produced, which is for the 
# structure function slope as a function of Alfvenic Mach number.
ax2 = fig.add_subplot(122, sharey = ax1)

# Plot m as a function of Alfvenic Mach number 
plt.errorbar(alf_mach_arr[0:8], m_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=m_err_arr[0:8])
plt.errorbar(alf_mach_arr[8:16], m_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=m_err_arr[8:16])
plt.errorbar(alf_mach_arr[16:], m_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=m_err_arr[16:])

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
plt.errorbar(sonic_mach_arr[0:8], int_quad_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=int_quad_err_arr[0:8])
plt.errorbar(sonic_mach_arr[8:16], int_quad_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=int_quad_err_arr[8:16])
plt.errorbar(sonic_mach_arr[16:], int_quad_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=int_quad_err_arr[16:])

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
plt.errorbar(alf_mach_arr[0:8], int_quad_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=int_quad_err_arr[0:8])
plt.errorbar(alf_mach_arr[8:16], int_quad_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=int_quad_err_arr[8:16])
plt.errorbar(alf_mach_arr[16:], int_quad_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=int_quad_err_arr[16:])

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