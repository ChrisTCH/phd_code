#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# calculates the normalised correlation functions, structure functions,        #
# and quadrupole ratios of the synchrotron intensity maps, for different       #
# lines of sight. Plots are then produced of the normalised correlation        #
# functions, structure functions, quadrupole ratios. This code is intended to  #
# be used with simulations produced by Christoph Federrath.                    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 20/1/2016                                                        #
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
def calc_err_bootstrap(sync_map, log = False):
	'''
	Description
        This function divides the given images into quarters, and then 
        calculates statistics for each quarter. The standard deviation of the 
        calculated statistics is then returned, representing the error on 
        each statistic.
        
    Required Input
        sync_map - The synchrotron intensity map. Should be a 2D Numpy array.
        log - A boolean value. If True, then the moments are calculated for the 
        	  the logarithm of the PDF, and not the PDF itself
                   
    Output
        mean_err - The error calculated for the mean of synchrotron intensity
        stdev_err - The error calculated for the standard deviation of the 
        			synchrotron intensity
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
	mean_val = np.zeros(np.shape(quarter_arr)[0])
	stdev_val = np.zeros(np.shape(quarter_arr)[0])
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

		# If we are calculating moments of the log PDFs, then calculate the
		# logarithm of the flat image
		if log == True:
			# In this case we are calculating the moments of the log PDFs, so 
			# calculate the log PDFs
			flat_image = np.log10(flat_image)

		# Calculate the mean of the synchrotron intensity map
		mean_val[i] = np.mean(flat_image, dtype=np.float64)

		# Calculate the standard deviation of the synchrotron intensity map
		stdev_val[i] = np.std(flat_image, dtype=np.float64)

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
	mean_err = np.std(mean_val) / np.sqrt(len(mean_val))
	stdev_err = np.std(stdev_val) / np.sqrt(len(stdev_val))
	skew_err = np.std(skew_val) / np.sqrt(len(skew_val))
	kurt_err = np.std(kurt_val) / np.sqrt(len(kurt_val))
	m_err = np.std(m_val) / np.sqrt(len(m_val))
	residual_err = np.std(resid_val) / np.sqrt(len(resid_val))
	int_quad_err = np.std(int_quad_val) / np.sqrt(len(int_quad_val))

	# Now that all of the calculations have been performed, return the 
	# calculated errors
	return mean_err,stdev_err,skew_err,kurt_err,m_err,residual_err, int_quad_err

# Set a variable to hold the number of bins to use in calculating the 
# correlation functions
num_bins = 25

# Create a variable that controls whether the moments of the log PDFs are 
# calculated
log = True

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
spec_locs = ['512sM5Bs5886_20/', '512sM5Bs5886_25/', '512sM5Bs5886_30/',\
'512sM5Bs5886_35/', '512sM5Bs5886_40/', '512cM5Bs5886_20/', '512cM5Bs5886_25/',\
'512cM5Bs5886_30/', '512cM5Bs5886_35/', '512cM5Bs5886_40/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation
sim_labels = ['Sol 20', 'Sol 25', 'Sol 30', 'Sol 35', 'Sol 40',\
 'Comp 20', 'Comp 25', 'Comp 30', 'Comp 35', 'Comp 40']

# Create a variable that holds the number of timesteps being used
num_timestep = 5

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Create a three dimensional array that will hold all of the information
# for the normalised correlation functions. The first index gives the simulation
# the second gives the line of sight, and the third axis goes along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
norm_corr_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the normalised correlation functions. The first axis represents
# the simulation used, the second represents the line of sight, and 
# the third axis goes over radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
corr_rad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the structure functions. The first index gives the simulation
# the second gives the line of sight, and the third axis goes along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
sf_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the structure functions. The first axis represents the 
# simulation used, the second represents the line of sight, and 
# the third axis goes over radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
sf_rad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the magnitude of the quadrupole ratios. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes 
# along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the quadrupole ratios. The first axis represents the 
# simulation used, the second represents the line of sight, and 
# the third axis goes over radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_rad_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the real part of the quadrupole ratios. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes 
# along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_real_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the imaginary part of the quadrupole ratios. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes 
# along radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
quad_imag_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create an empty array, where each entry specifies the calculated mean of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
mean_arr = np.zeros((len(spec_locs),3))

# Create an empty array, where each entry specifies the calculated standard
# deviation of the synchrotron intensity image of the corresponding simulation
# for a particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
stdev_arr = np.zeros((len(spec_locs),3))

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
# NOTE: We will calculate the biased skewness
skew_arr = np.zeros((len(spec_locs),3))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
# NOTE: We will calculate the biased Fisher kurtosis
kurt_arr = np.zeros((len(spec_locs),3))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of gamma. The first index 
# gives the simulation, and the second index gives the line of sight as (x,y,z).
m_arr = np.zeros((len(spec_locs),3))

# Create an empty array, where each entry specifies the residuals of the
# linear fit to the structure function, of the corresponding simulation, for a
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
residual_arr = np.zeros((len(spec_locs),3))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity 
# image, for the corresponding simulation, for a particular value of gamma. 
# The first index gives the simulation, and the second index gives the line of
# sight as (x,y,z).
int_quad_arr = np.zeros((len(spec_locs),3))

# Create error arrays for each of the statistics. These errors are calculated
# by the standard deviation of the statistics calculated for sub-images of the
# synchrotron maps. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z).
mean_err_arr =     np.zeros((len(spec_locs),3))
stdev_err_arr =    np.zeros((len(spec_locs),3))
skew_err_arr =     np.zeros((len(spec_locs),3))
kurt_err_arr =     np.zeros((len(spec_locs),3))
m_err_arr =        np.zeros((len(spec_locs),3))
residual_err_arr = np.zeros((len(spec_locs),3))
int_quad_err_arr = np.zeros((len(spec_locs),3))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]

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

		# If we are calculating the moments of the log PDFs, then calculate the
		# logarithm of the synchrotron intensity values
		if log == True:
			# In this case we are calculating the moments of the log PDFs, so 
			# calculate the log of the synchrotron intensities
			flat_sync = np.log10(flat_sync/ np.mean(flat_sync, dtype = np.float64))

		# Calculate the mean of the synchrotron intensity map, and store the
		# result in the corresponding array
		mean_arr[i,j] = np.mean(flat_sync, dtype=np.float64)

		# Calculate the standard deviation of the synchrotron intensity map, and
		# store the result in the corresponding array
		stdev_arr[i,j] = np.std(flat_sync, dtype=np.float64)

		# Calculate the biased skewness of the synchrotron intensity map, and store
		# the result in the corresponding array
		skew_arr[i,j] = stats.skew(flat_sync)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# map, and store the result in the corresponding array
		kurt_arr[i,j] = stats.kurtosis(flat_sync)

		# Calculate the 2D correlation function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the correlation function
		corr = cf_fft(sync_data, no_fluct = True)

		# Radially average the calculated 2D correlation function, using the 
		# specified number of bins
		rad_corr = sfr(corr, num_bins, verbose = False)

		# Calculate the square of the mean of the synchrotron intensity values
		sync_sq_mean = np.power( np.mean(sync_data, dtype = np.float64), 2.0 )

		# Calculate the mean of the synchrotron intensity values squared
		sync_mean_sq = np.mean( np.power(sync_data, 2.0), dtype = np.float64 )

		# Calculate the normalised, radially averaged correlation function for
		# this value of gamma
		norm_rad_corr = (rad_corr[1] - sync_sq_mean) / (sync_mean_sq - sync_sq_mean)

		# Print a message to show that the correlation function of the
		# synchrotron intensity has been calculated for this line of sight
		print 'Correlation function of synchrotron intensity'\
		+ ' calculated for {} LOS'.format(line_o_sight[j])

		# Insert the calculated normalised, radially averaged correlation function
		# into the matrix that stores all of the calculated correlation functions
		norm_corr_arr[i,j] = norm_rad_corr

		# Insert the radius values used to calculate this correlation function
		# into the matrix that stores the radius values
		corr_rad_arr[i,j] = rad_corr[0]

		# Print a message to show that the correlation function has been calculated
		print 'Normalised, radially averaged correlation function calculated for'\
		+ ' {} LOS'.format(line_o_sight[j])

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not 
		# subtracting the mean from the synchrotron maps before calculating the
		# structure function.
		strfn = sf_fft(sync_data, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins.
		rad_sf = sfr(strfn, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function
		sf_arr[i,j] = rad_sf[1]

		# Extract the radius values used to calculate this structure function
		sf_rad_arr[i,j] = rad_sf[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma.
		spec_ind_data = np.polyfit(np.log10(\
			sf_rad_arr[i,j,5:14]),\
			np.log10(sf_arr[i,j,5:14]), 1, full = True)

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

		# Add the calculated modulus of the quadrupole ratio to the final array
		quad_arr[i,j] = quad_mod

		# Add the radius values used to calculate the quadrupole ratio to the
		# corresponding array
		quad_rad_arr[i,j] = quad_rad

		# Calculate the real part of the quadrupole ratio
		quad_real_arr[i,j] = quad_mod * np.cos(quad_arg)

		# Calculate the imaginary part of the quadrupole ratio
		quad_imag_arr[i,j] = quad_mod * np.sin(quad_arg)

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
		mean_err_arr[i,j], stdev_err_arr[i,j], skew_err_arr[i,j],\
		kurt_err_arr[i,j], m_err_arr[i,j],\
		residual_err_arr[i,j], int_quad_err_arr[i,j]\
		= calc_err_bootstrap(sync_data, log = log)

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			spec_locs[i], line_o_sight[j])

# When the code reaches this point, the normalised correlation functions,
# structure functions, and quadrupole ratios have been saved for every 
# simulation, and every line of sight, so start making the final plots.

# Create an array of marker symbols, so that the plot for each line of sight has
# a different plot symbol
symbol_arr = ['o','^','s','*','D']

# ----------------- Plots of normalised correlation functions ------------------

# Here we want to produce one plot with six subplots. There should be two rows
# of subplots, with three subplots in each row. The left subplot will be the 
# normalised correlation functions for a line of sight along the x axis, the
# centre plot will be for the y axis, and the right subplot will be the 
# normalised correlation functions for the z axis. In each plot the timesteps
# of the solenoidal and compressive simulations will be compared, with 
# solenoidal simulations on the top row, and compressive on the bottom.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the
# x line of sight, solenoidal simulations
ax1 = fig.add_subplot(231)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,0], norm_corr_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,0])), \
	np.zeros(np.shape(corr_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the
# y line of sight, solenoidal simulation. Make the y axis limits the same as
# for the x axis plot
ax2 = fig.add_subplot(232, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,1], norm_corr_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,1])), \
	np.zeros(np.shape(corr_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight, solenoidal simulation. Make the y axis limits the same as for
# the x axis plot
ax3 = fig.add_subplot(233, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,2], norm_corr_arr[i,2], '-' + symbol_arr[i],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,2])), \
	np.zeros(np.shape(corr_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax3.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 8, numpoints=1)

# Create an axis for the fourth subplot to be produced, which is for the
# x line of sight, compressive simulations
ax4 = fig.add_subplot(234, sharex = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,0], norm_corr_arr[i,0], '-' + symbol_arr[i - num_timestep])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,0])), \
	np.zeros(np.shape(corr_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Create an axis for the fifth subplot to be produced, which is for the
# y line of sight, compressive simulation. Make the y axis limits the same as
# for the x axis plot
ax5 = fig.add_subplot(235, sharex = ax2, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,1], norm_corr_arr[i,1], '-' + symbol_arr[i - num_timestep])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,1])), \
	np.zeros(np.shape(corr_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax5.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax5.get_yticklabels(), visible=False)

# Create an axis for the sixth subplot to be produced, which is for the
# z line of sight, compressive simulation. Make the y axis limits the same as for
# the x axis plot
ax6 = fig.add_subplot(236, sharex = ax3, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the normalised correlation function for this simulation, for this 
	# line of sight
	plt.plot(corr_rad_arr[i,2], norm_corr_arr[i,2], '-' + symbol_arr[i - num_timestep],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(corr_rad_arr[0,2])), \
	np.zeros(np.shape(corr_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax6.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax6.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 8, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'NCF Sync Intensity', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.15, 0.94, 'a) Sol x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure b
plt.figtext(0.42, 0.94, 'b) Sol y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.7, 0.94, 'c) Sol z-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure d
plt.figtext(0.15, 0.475, 'd) Comp x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure e
plt.figtext(0.42, 0.475, 'e) Comp y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure f
plt.figtext(0.7, 0.475, 'f) Comp z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'ncfs_all_sims_time_gam{}.eps'.format(gamma), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#--------------------------- Structure Functions -------------------------------

# Here we want to produce one plot with six subplots. There should be two rows
# of subplots, with three subplots in each row. The left subplot will be the 
# structure functions for a line of sight along the x axis, the centre plot will
# be for the y axis, and the right subplot will be the structure functions for 
# the z axis. In each plot the solenoidal and compressive simulations will be 
# compared for different timesteps. The top row is for the solenoidal simulation
# and the bottom row is for the compressive simulation.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the
# x line of sight, solenoidal simulation
ax1 = fig.add_subplot(231)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,0], sf_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,0])), \
	np.zeros(np.shape(sf_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
ax1.set_yscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the
# y line of sight, solenoidal simulation. Make the y axis limits the same as for
# the x axis plot
ax2 = fig.add_subplot(232, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,1], sf_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,1])), \
	np.zeros(np.shape(sf_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
ax2.set_yscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight, solenoidal simulation. Make the y axis limits the same as for
# the x axis plot
ax3 = fig.add_subplot(233, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,2], sf_arr[i,2], '-' + symbol_arr[i],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,2])), \
	np.zeros(np.shape(sf_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis of the plot logarithmic
ax3.set_yscale('log')

# Make the x axis tick labels invisible
plt.setp( ax3.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc = 4, fontsize = 8, numpoints=1)

# Create an axis for the fourth subplot to be produced, which is for the
# x line of sight, compressive simulation
ax4 = fig.add_subplot(234, sharex = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,0], sf_arr[i,0], '-' + symbol_arr[i - num_timestep])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,0])), \
	np.zeros(np.shape(sf_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis of the plot logarithmic
ax4.set_yscale('log')

# Create an axis for the fifth subplot to be produced, which is for the
# y line of sight, compressive simulation. Make the y axis limits the same as for
# the x axis plot
ax5 = fig.add_subplot(235, sharex = ax2, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,1], sf_arr[i,1], '-' + symbol_arr[i - num_timestep])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,1])), \
	np.zeros(np.shape(sf_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax5.set_xscale('log')

# Make the y axis of the plot logarithmic
ax5.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax5.get_yticklabels(), visible=False)

# Create an axis for the sixth subplot to be produced, which is for the
# z line of sight, compressive simulation. Make the y axis limits the same as for
# the x axis plot
ax6 = fig.add_subplot(236, sharex = ax3, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.plot(sf_rad_arr[i,2], sf_arr[i,2], '-' + symbol_arr[i - num_timestep],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_arr[0,2])), \
	np.zeros(np.shape(sf_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax6.set_xscale('log')

# Make the y axis of the plot logarithmic
ax6.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax6.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc = 4, fontsize = 8, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Structure Function Amplitude', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.15, 0.94, 'a) Sol x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure b
plt.figtext(0.42, 0.94, 'b) Sol y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.7, 0.94, 'c) Sol z-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure d
plt.figtext(0.15, 0.475, 'd) Comp x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure e
plt.figtext(0.42, 0.475, 'e) Comp y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure f
plt.figtext(0.7, 0.475, 'f) Comp z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'sfs_all_sims_time_gam{}.eps'.format(gamma), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#----------------------------- Quadrupole Ratios -------------------------------

# Here we want to produce one plot with six subplots. There should be two rows
# of subplots, with three subplots in each row. The left subplot will be the 
# quadrupole ratio modulus for a line of sight along the x axis, the centre plot
# will be for the y axis, and the right subplot will be the quadrupole ratio
# modulus for the z axis. In each plot the solenoidal and compressive 
# simulations will be compared for different timesteps. The top row is for the
# solenoidal simulation, and the bottom row for the compressive simulation.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the
# x line of sight, solenoidal simulation
ax1 = fig.add_subplot(231)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,0], quad_arr[i,0], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the
# y line of sight, solenoidal simulation. Make the y axis limits the same as for
# the x axis plot
ax2 = fig.add_subplot(232, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,1], quad_arr[i,1], '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,1])), \
	np.zeros(np.shape(quad_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight, solenoidal simulation. Make the y axis limits the same as for
# the x axis plot
ax3 = fig.add_subplot(233, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,2], quad_arr[i,2], '-' + symbol_arr[i],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,2])), \
	np.zeros(np.shape(quad_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the x axis tick labels invisible
plt.setp( ax3.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 8, numpoints=1)

# Create an axis for the fourth subplot to be produced, which is for the
# x line of sight, compressive simulation
ax4 = fig.add_subplot(234, sharex = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,0], quad_arr[i,0], '-' + symbol_arr[i - num_timestep])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Create an axis for the fifth subplot to be produced, which is for the
# y line of sight, compressive simulation. Make the y axis limits the same as for
# the x axis plot
ax5 = fig.add_subplot(235, sharex = ax2, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,1], quad_arr[i,1], '-' + symbol_arr[i - num_timestep])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,1])), \
	np.zeros(np.shape(quad_rad_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax5.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax5.get_yticklabels(), visible=False)

# Create an axis for the sixth subplot to be produced, which is for the
# z line of sight, compressive simulation. Make the y axis limits the same as for
# the x axis plot
ax6 = fig.add_subplot(236, sharex = ax3, sharey = ax4)

# Loop over the simulations to produce plots for each simulation
for i in range(num_timestep, 2*num_timestep):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.plot(quad_rad_arr[i,2], quad_arr[i,2], '-' + symbol_arr[i - num_timestep],\
		label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,2])), \
	np.zeros(np.shape(quad_rad_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax6.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax6.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 8, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Quadrupole Ratio', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.15, 0.94, 'a) Sol x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure b
plt.figtext(0.42, 0.94, 'b) Sol y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.7, 0.94, 'c) Sol z-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure d
plt.figtext(0.15, 0.475, 'd) Comp x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure e
plt.figtext(0.42, 0.475, 'e) Comp y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure f
plt.figtext(0.7, 0.475, 'f) Comp z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'quad_ratio_all_sims_time_gam{}.eps'.format(gamma), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

# #----------------------- Real and Imaginary Parts of Quad Ratio ----------------

# # Here we want to produce one plot with six subplots. There should be two rows
# # of subplots, with three subplots in each row. The top row will be the real 
# # part of the quadrupole ratio, and the bottom row will be the imaginary part.
# # The left column will be for a line of sight along the x axis, the centre
# # column for a line of sight along the y axis, and the right column will be for
# # a line of sight along the z axis.

# # Create a figure to hold all of the subplots
# fig = plt.figure(1, figsize=(10,6), dpi = 300)

# # Create an axis for the first subplot to be produced, which is for the real
# # part of the quadrupole ratio for a line of sight along the x axis
# ax1 = fig.add_subplot(231)

# # Loop over the simulations to produce plots for each simulation
# for i in range(len(spec_locs)):
# 	# Plot the quadrupole ratio for this simulation, for this line of sight
# 	plt.plot(quad_rad_arr[i,0], quad_real_arr[i,0], '-' + symbol_arr[i])

# # Plot a faded dashed line to represent the line y = 0
# plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
# 	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# # Make the x axis of the plot logarithmic
# ax1.set_xscale('log')

# # Make the x axis tick labels invisible
# plt.setp( ax1.get_xticklabels(), visible=False)

# # Create an axis for the second subplot to be produced, which is for the real
# # part of the quadrupole ratio for a line of sight along the y axis. Make the y
# # axis limits the same as for the x axis plot
# ax2 = fig.add_subplot(232, sharey = ax1)

# # Loop over the simulations to produce plots for each simulation
# for i in range(len(spec_locs)):
# 	# Plot the quadrupole ratio for this simulation, for this line of sight
# 	plt.plot(quad_rad_arr[i,1], quad_real_arr[i,1], '-' + symbol_arr[i])

# # Plot a faded dashed line to represent the line y = 0
# plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,1])), \
# 	np.zeros(np.shape(quad_rad_arr[0,1])), 'k--', alpha = 0.5)

# # Make the x axis of the plot logarithmic
# ax2.set_xscale('log')

# # Make the x axis tick labels invisible
# plt.setp( ax2.get_xticklabels(), visible=False)

# # Make the y axis tick labels invisible
# plt.setp( ax2.get_yticklabels(), visible=False)

# # Create an axis for the third subplot to be produced, which is for the real
# # part of the quadrupole ratio for a line of sight along the z axis. Make the y
# # axis limits the same as for the x axis plot
# ax3 = fig.add_subplot(233, sharey = ax1)

# # Loop over the simulations to produce plots for each simulation
# for i in range(len(spec_locs)):
# 	# Plot the quadrupole ratio for this simulation, for this line of sight
# 	plt.plot(quad_rad_arr[i,2], quad_real_arr[i,2], '-' + symbol_arr[i],\
# 		label = '{}'.format(sim_labels[i]))

# # Plot a faded dashed line to represent the line y = 0
# plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,2])), \
# 	np.zeros(np.shape(quad_rad_arr[0,2])), 'k--', alpha = 0.5)

# # Make the x axis of the plot logarithmic
# ax3.set_xscale('log')

# # Make the x axis tick labels invisible
# plt.setp( ax3.get_xticklabels(), visible=False)

# # Make the y axis tick labels invisible
# plt.setp( ax3.get_yticklabels(), visible=False)

# # Force the legend to appear on the plot
# plt.legend(loc=4, fontsize = 9, numpoints=1)

# # Create an axis for the fourth subplot to be produced, which is for the 
# # imaginary part of the quadrupole ratio for a line of sight along the x axis.
# # Make the x axis limits the same as for the first plot
# ax4 = fig.add_subplot(234, sharex = ax1, sharey = ax1)

# # Loop over the simulations to produce plots for each simulation
# for i in range(len(spec_locs)):
# 	# Plot the quadrupole ratio for this simulation, for this line of sight
# 	plt.plot(quad_rad_arr[i,0], quad_imag_arr[i,0], '-' + symbol_arr[i])

# # Plot a faded dashed line to represent the line y = 0
# plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,0])), \
# 	np.zeros(np.shape(quad_rad_arr[0,0])), 'k--', alpha = 0.5)

# # Make the x axis of the plot logarithmic
# ax4.set_xscale('log')

# # Create an axis for the fifth subplot to be produced, which is for the 
# # imaginary part of the quadrupole ratio for a line of sigth along the y axis.
# # Make the x axis limits the same as for the second plot, and the y axis limits
# # the same as for the fourth plot
# ax5 = fig.add_subplot(235, sharex = ax2, sharey = ax4)

# # Loop over the simulations to produce plots for each simulation
# for i in range(len(spec_locs)):
# 	# Plot the quadrupole ratio for this simulation, for this line of sight
# 	plt.plot(quad_rad_arr[i,1], quad_imag_arr[i,1], '-' + symbol_arr[i])

# # Plot a faded dashed line to represent the line y = 0
# plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,1])), \
# 	np.zeros(np.shape(quad_rad_arr[0,1])), 'k--', alpha = 0.5)

# # Make the x axis of the plot logarithmic
# ax5.set_xscale('log')

# # Make the y axis tick labels invisible
# plt.setp( ax5.get_yticklabels(), visible=False)

# # Create an axis for the sixth subplot to be produced, which is for the 
# # imaginary part of the quadrupole ratio for a line of sigth along the z axis.
# # Make the x axis limits the same as for the third plot, and the y axis limits
# # the same as for the fourth plot
# ax6 = fig.add_subplot(236, sharex = ax3, sharey = ax4)

# # Loop over the simulations to produce plots for each simulation
# for i in range(len(spec_locs)):
# 	# Plot the quadrupole ratio for this simulation, for this line of sight
# 	plt.plot(quad_rad_arr[i,2], quad_imag_arr[i,2], '-' + symbol_arr[i])

# # Plot a faded dashed line to represent the line y = 0
# plt.plot(np.linspace(0,1000,len(quad_rad_arr[0,2])), \
# 	np.zeros(np.shape(quad_rad_arr[0,2])), 'k--', alpha = 0.5)

# # Make the x axis of the plot logarithmic
# ax6.set_xscale('log')

# # Make the y axis tick labels invisible
# plt.setp( ax6.get_yticklabels(), visible=False)

# # Add a label to the x-axis
# plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
# 	va = 'bottom', fontsize = 20)

# # Add a label to the y-axis
# plt.figtext(0.03, 0.5, 'Quadrupole Ratio', ha = 'left', \
# 	va = 'center', fontsize = 20, rotation = 'vertical')

# # Add some text to the figure, to label the left plot as figure a
# plt.figtext(0.15, 0.94, 'a) x-LOS Real', fontsize = 18)

# # Add some text to the figure, to label the centre plot as figure b
# plt.figtext(0.42, 0.94, 'b) y-LOS Real', fontsize = 18)

# # Add some text to the figure, to label the right plot as figure c
# plt.figtext(0.7, 0.94, 'c) z-LOS Real', fontsize = 18)

# # Add some text to the figure, to label the left plot as figure d
# plt.figtext(0.15, 0.475, 'd) x-LOS Imag', fontsize = 18)

# # Add some text to the figure, to label the centre plot as figure e
# plt.figtext(0.42, 0.475, 'e) y-LOS Imag', fontsize = 18)

# # Add some text to the figure, to label the right plot as figure f
# plt.figtext(0.7, 0.475, 'f) z-LOS Imag', fontsize = 18)

# # Make sure that all of the labels are clearly visible in the plot
# #plt.tight_layout()

# # Save the figure using the given filename and format
# plt.savefig(simul_loc + 'real_imag_quad_all_sims_time_gam{}.eps'.format(gamma), format = 'eps')

# # Close the figure so that it does not stay in memory
# plt.close()

#-------------------------------------------------------------------------------

# Now that all of the statistics have been calculated, print them out to the 
# screen. Loop over all of the lines of sight, and the different simulations,
# and print out results for the simulations
for j in range(3):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs)):
		# Print out the value of the mean for this line of sight
		print "{} {} LOS Mean: {}   Error: {}".format(sim_labels[i], line_o_sight[j],\
			mean_arr[i,j], mean_err_arr[i,j])

for j in range(3):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs)):
		# Print out the value of the standard deviation for this line of sight
		print "{} {} LOS St Dev: {}   Error: {}".format(sim_labels[i], line_o_sight[j],\
			stdev_arr[i,j], stdev_err_arr[i,j])

for j in range(3):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs)):
		# Print out the value of skewness for this line of sight
		print "{} {} LOS Skewness: {}   Error: {}".format(sim_labels[i], line_o_sight[j],\
			skew_arr[i,j], skew_err_arr[i,j])

for j in range(3):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs)):
		# Print out the value of kurtosis for this line of sight
		print "{} {} LOS Kurtosis: {}   Error: {}".format(sim_labels[i], line_o_sight[j],\
			kurt_arr[i,j], kurt_err_arr[i,j])

for j in range(3):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs)):
		# Print out the structure function slope for this line of sight
		print "{} {} LOS SF Slope: {}   Error: {}".format(sim_labels[i], line_o_sight[j],\
			m_arr[i,j], m_err_arr[i,j])

for j in range(3):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs)):
		# Print out the residuals for this line of sight
		print "{} {} LOS Residuals: {}   Error: {}".format(sim_labels[i], line_o_sight[j],\
			residual_arr[i,j], residual_err_arr[i,j])

for j in range(3):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs)):
		# Print out the value of the quadrupole ratio for this line of sight
		print "{} {} LOS Quad Ratio: {}   Error: {}".format(sim_labels[i], line_o_sight[j],\
			int_quad_arr[i,j], int_quad_err_arr[i,j])