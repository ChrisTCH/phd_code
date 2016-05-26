#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the skewness, kurtosis, structure function slope, and         #
# integrated quadrupole ratio as a function of sonic and Alfvenic Mach number  #
# for simulations with low and high magnetic field. This code is intended to   #
# be used with simulations produced by Christoph Federrath.                    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 18/1/2016                                                        #
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
def calc_err_bootstrap(sync_map):
	'''
	Description
        This function divides the given images into quarters, and then 
        calculates statistics for each quarter. The standard deviation of the 
        calculated statistics is then returned, representing the error on 
        each statistic.
        
    Required Input
        sync_map_y - The synchrotron intensity map. Should be a 2D Numpy array.
                   
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
	quarter_arr = np.zeros((4,np.shape(sync_map)[0]/2,np.shape(sync_map)[1]/2))

	# Add the quarters of the images into the array
	quarter_arr[0], quarter_arr[1] = np.split(np.split(sync_map,2,axis=0)[0],2,axis=1) 
	quarter_arr[2], quarter_arr[3] = np.split(np.split(sync_map,2,axis=0)[1],2,axis=1) 

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
			sf_rad_arr[5:14]),\
			np.log10(sf[5:14]), 1, full = True)

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
		int_quad_val[i] = np.trapz(quad_mod[8:23], dx = 1.0)\
		 / (22 - 8)

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
# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['512sM5Bs5886_20/', '512cM5Bs5886_20/']

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 1.0

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
# NOTE: We will calculate the biased skewness
skew_arr = np.zeros((len(simul_arr),3))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
# NOTE: We will calculate the biased Fisher kurtosis
kurt_arr = np.zeros((len(simul_arr),3))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of gamma. The first index 
# gives the simulation, and the second index gives the line of sight as (x,y,z).
m_arr = np.zeros((len(simul_arr),3))

# Create an empty array, where each entry specifies the residuals of the
# linear fit to the structure function, of the corresponding simulation, for a
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
residual_arr = np.zeros((len(simul_arr),3))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity 
# image, for the corresponding simulation, for a particular value of gamma. 
# The first index gives the simulation, and the second index gives the line of
# sight as (x,y,z).
int_quad_arr = np.zeros((len(simul_arr),3))

# Create error arrays for each of the statistics. These errors are calculated
# by the standard deviation of the statistics calculated for sub-images of the
# synchrotron maps. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
skew_err_arr =     np.zeros((len(simul_arr),3))
kurt_err_arr =     np.zeros((len(simul_arr),3))
m_err_arr =        np.zeros((len(simul_arr),3))
residual_err_arr = np.zeros((len(simul_arr),3))
int_quad_err_arr = np.zeros((len(simul_arr),3))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for i in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[i]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[i])

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

		# Flatten the synchrotron intensity map
		flat_sync = sync_data.flatten()

		# Calculate the biased skewness of the synchrotron intensity map, and store
		# the result in the corresponding array
		skew_arr[i,j] = stats.skew(flat_sync)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# map, and store the result in the corresponding array
		kurt_arr[i,j] = stats.kurtosis(flat_sync)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not
		# subtracting the mean from the synchrotron maps before calculating the 
		# structure function
		strfn = sf_fft(sync_data, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins
		rad_sf = sfr(strfn, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function
		sf = rad_sf[1]

		# Extract the radius values used to calculate this structure function
		sf_rad_arr = rad_sf[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma.
		spec_ind_data = np.polyfit(np.log10(\
			sf_rad_arr[5:14]),\
			np.log10(sf[5:14]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit
		coeff = spec_ind_data[0]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array
		m_arr[i,j] = coeff[0]-1.0

		# Enter the value of the residuals into the corresponding array
		residual_arr[i,j] = spec_ind_data[1]

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

		# Integrate the magnitude of the quadrupole / monopole ratio from one sixth 
		# of the way along the radial separation bins, until three quarters of the 
		# way along the radial separation bins. This integration is performed with
		# respect to log separation (i.e. I am ignoring the fact that the 
		# points are equally separated in log space, to calculate the area under
		# the quadrupole / monopole ratio plot when the x axis is scaled 
		# logarithmically). I normalise the value that is returned by dividing
		# by the number of increments in log radial separation used in the
		# calculation. 
		int_quad_arr[i,j] = np.trapz(quad_mod[8:23], dx = 1.0) / (22 - 8)

		# Create errors for each of the statistics. These errors are only for the
		# statistics calculated from the y and z axes (perpendicular to the mean 
		# magnetic field), and are calculated by the standard deviation of the 
		# statistics calculated for sub-images of the synchrotron maps.
		skew_err_arr[i,j], kurt_err_arr[i,j], m_err_arr[i,j],\
		residual_err_arr[i,j], int_quad_err_arr[i,j]\
		= calc_err_bootstrap(sync_data)

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			simul_arr[i], line_o_sight[j])

# Now that all of the statistics have been calculated, print them out to the 
# screen. Loop over all of the lines of sight, and print out results for the 
# solenoidal simulation
for j in range(3):
	# Print out the value of skewness for this line of sight
	print "Solenoidal {} LOS Skewness: {}   Error: {}".format(line_o_sight[j],\
		skew_arr[0,j], skew_err_arr[0,j])
	# Print out the value of kurtosis for this line of sight
	print "Solenoidal {} LOS Kurtosis: {}   Error: {}".format(line_o_sight[j],\
		kurt_arr[0,j], kurt_err_arr[0,j])
	# Print out the structure function slope for this line of sight
	print "Solenoidal {} LOS SF Slope: {}   Error: {}".format(line_o_sight[j],\
		m_arr[0,j], m_err_arr[0,j])
	# Print out the residuals for this line of sight
	print "Solenoidal {} LOS Residuals: {}   Error: {}".format(line_o_sight[j],\
		residual_arr[0,j], residual_err_arr[0,j])
	# Print out the value of the quadrupole ratio for this line of sight
	print "Solenoidal {} LOS Quad Ratio: {}   Error: {}".format(line_o_sight[j],\
		int_quad_arr[0,j], int_quad_err_arr[0,j])

# Print out results for the compressive simulation
for j in range(3):
	# Print out the value of skewness for this line of sight
	print "Compressive {} LOS Skewness: {}   Error: {}".format(line_o_sight[j],\
		skew_arr[1,j], skew_err_arr[1,j])
	# Print out the value of kurtosis for this line of sight
	print "Compressive {} LOS Kurtosis: {}   Error: {}".format(line_o_sight[j],\
		kurt_arr[1,j], kurt_err_arr[1,j])
	# Print out the structure function slope for this line of sight
	print "Compressive {} LOS SF Slope: {}   Error: {}".format(line_o_sight[j],\
		m_arr[1,j], m_err_arr[1,j])
	# Print out the residuals for this line of sight
	print "Compressive {} LOS Residuals: {}   Error: {}".format(line_o_sight[j],\
		residual_arr[1,j], residual_err_arr[1,j])
	# Print out the value of the quadrupole ratio for this line of sight
	print "Compressive {} LOS Quad Ratio: {}   Error: {}".format(line_o_sight[j],\
		int_quad_arr[1,j], int_quad_err_arr[1,j])