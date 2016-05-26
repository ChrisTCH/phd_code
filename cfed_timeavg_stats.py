#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# calculates the time-averaged structure functions and quadrupole ratios of    #
# the synchrotron intensity maps, for different lines of sight. Plots are then #
# produced of the structure functions and quadrupole ratios. The time-averaged #
# moments of the synchrotron intensity PDFs are also calculated, along with    #
# the time-averaged structure function slope and integrated quadrupole ratio.  #
# This code is intended to be used with simulations produced by Christoph      #
# Federrath.                                                                   #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 8/2/2016                                                         #
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

# Define a function that takes some calculated statistics, and plots them 
# against the zeta parameter, which defines the type of turbulent driving
def plot_vs_zeta(stat_arr, zeta_arr, stat_err_arr, symbol_arr, y_label,\
 lines_o_sight, save_name, legend_loc):
	'''
	Description
        This function takes a list of values for a statistic, and the errors on 
        the calculated statistics, and then plots the statistic against the 
        zeta parameter for each line of sight. The lines of sight are assumed to
        occupy the second index of the array. 
        
    Required Input
        stat_arr - A two-dimensional Numpy array, whose first dimension gives 
        		   the value of the statistic for different simulations, and 
        		   whose second dimension gives the value of the statistic for 
        		   different lines of sight.
        zeta_arr - A one-dimensional Numpy array that gives the values of zeta 
        		   for which statistics have been calculated.
        stat_err_arr - A two-dimensional Numpy array of the same format as the
        			   stat_arr. Each entry gives the uncertainty on the 
        			   measured statistic.
        symbol_arr - A one-dimensional list giving the symbol and linestyle to
        			 use for each line of sight.
        y_label - A string that gives the label to place on the y-axis of the 
        		  plot.
        lines_o_sight - A one-dimensional list giving the line of sight for 
        				each entry in the second dimension of stat_arr.
        save_name - A string that gives the full filename to use when saving the
        			file. Must include the .eps extension.
        legend_loc - A number that specifies where the legend should be placed
                   
    Output
        A plot is saved using the given filename. Zero is returned if the save 
        is successful, and one is returned if an error occurs.
	'''

	# Create a new figure instance, to create the new plot
	fig = plt.figure(dpi = 300)

	# Create an axis for figure
	ax = fig.add_subplot(111)

	# Loop over the lines of sight, to produce a plot for each line of sight
	for i in range(len(lines_o_sight)):
		# Plot the structure function for this simulation, for this 
		# line of sight
		plt.errorbar(zeta_arr, stat_arr[:,i], yerr = stat_err_arr[:,i],\
		 fmt=symbol_arr[i], label = '{}-LOS'.format(lines_o_sight[i]))

	# Force the legend to appear on the plot
	plt.legend(loc = legend_loc, fontsize = 10, numpoints=1)

	# Change the limits of the x axis
	plt.xlim([np.amin(zeta_arr)-0.1, np.amax(zeta_arr)+0.1])

	# Add a label to the x-axis
	plt.xlabel(r'$\zeta$ [unitless]', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel(y_label, fontsize = 20,)

	# Save the figure, using the given filename
	plt.savefig(save_name, format = 'eps')

	# Close the figure, now that it has been saved
	plt.close()

	# Now that the figure has been produced, return zero
	return 0

# Set a variable to hold the number of bins to use in calculating the 
# correlation functions
num_bins = 25

# Set a variable that holds the number of timesteps we have for the simulations
num_timestep = 5

# Create a variable that controls whether the moments of the log normalised PDFs
# are calculated
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
# 512m075M5Bs5887_20 (zeta = 0.75, timestep 20)
# 512m075M5Bs5887_25 (zeta = 0.75, timestep 25)
# 512m075M5Bs5887_30 (zeta = 0.75, timestep 30)
# 512m075M5Bs5887_35 (zeta = 0.75, timestep 35)
# 512m075M5Bs5887_40 (zeta = 0.75, timestep 40)
# 512mM5Bs5887_20 (zeta = 0.5, timestep 20)
# 512mM5Bs5887_25 (zeta = 0.5, timestep 25)
# 512mM5Bs5887_30 (zeta = 0.5, timestep 30)
# 512mM5Bs5887_35 (zeta = 0.5, timestep 35)
# 512mM5Bs5887_40 (zeta = 0.5, timestep 40)
# 512m025M5Bs5887_20 (zeta = 0.25, timestep 20)
# 512m025M5Bs5887_25 (zeta = 0.25, timestep 25)
# 512m025M5Bs5887_30 (zeta = 0.25, timestep 30)
# 512m025M5Bs5887_35 (zeta = 0.25, timestep 35)
# 512m025M5Bs5887_40 (zeta = 0.25, timestep 40)
# 512cM5Bs5886_20 (Compressive turbulence, timestep 20)
# 512cM5Bs5886_25 (Compressive turbulence, timestep 25)
# 512cM5Bs5886_30 (Compressive turbulence, timestep 30)
# 512cM5Bs5886_35 (Compressive turbulence, timestep 35)
# 512cM5Bs5886_40 (Compressive turbulence, timestep 40)
spec_locs = ['512sM5Bs5886_20/', '512sM5Bs5886_25/', '512sM5Bs5886_30/',\
'512sM5Bs5886_35/', '512sM5Bs5886_40/', '512m075M5Bs5887_20/',\
'512m075M5Bs5887_25/', '512m075M5Bs5887_30/', '512m075M5Bs5887_35/',\
'512m075M5Bs5887_40/', '512mM5Bs5887_20/', '512mM5Bs5887_25/',\
'512mM5Bs5887_30/', '512mM5Bs5887_35/', '512mM5Bs5887_40/',\
'512m025M5Bs5887_20/', '512m025M5Bs5887_25/', '512m025M5Bs5887_30/',\
'512m025M5Bs5887_35/', '512m025M5Bs5887_40/', '512cM5Bs5886_20/',\
'512cM5Bs5886_25/', '512cM5Bs5886_30/', '512cM5Bs5886_35/', '512cM5Bs5886_40/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation
sim_labels = [r'$\zeta = 1.0$', r'$\zeta = 0.75$', r'$\zeta = 0.5$',\
 r'$\zeta = 0.25$', r'$\zeta = 0$']

# Create an array that gives the value of zeta for each simulation
zeta = np.array([1.0,0.75,0.5,0.25,0.0])

# Create a variable that holds the number of simulations being used
num_sims = len(sim_labels)

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

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

# Create arrays for the time-averaged versions of the structure functions and
# integrated quadrupole ratios, and the radii used to calculate these.
sf_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))
sf_rad_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))
quad_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))
quad_rad_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))

# Create arrays for the errors on the time-averaged versions of the structure 
# functions and integrated quadrupole ratios.
sf_err_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))
quad_err_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))

# Create arrays for the time-averaged versions of the above statistics. The 
# first index gives the simulation, and the second index gives the line of sight
# as (x,y,z).
mean_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
stdev_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
skew_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
kurt_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
m_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
residual_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
int_quad_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))

# Create error arrays for each of the statistics. These errors are calculated
# by the standard deviation of the statistics calculated for different 
# timesteps. The first index gives the simulation, and the second index gives 
# the line of sight as (x,y,z).
mean_err_arr =     np.zeros((len(spec_locs)/num_timestep,3))
stdev_err_arr =    np.zeros((len(spec_locs)/num_timestep,3))
skew_err_arr =     np.zeros((len(spec_locs)/num_timestep,3))
kurt_err_arr =     np.zeros((len(spec_locs)/num_timestep,3))
m_err_arr =        np.zeros((len(spec_locs)/num_timestep,3))
residual_err_arr = np.zeros((len(spec_locs)/num_timestep,3))
int_quad_err_arr = np.zeros((len(spec_locs)/num_timestep,3))

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
			# In this case we are calculating the moments of the log normalised
			# PDFs, so calculate the log of the synchrotron intensities
			flat_sync = np.log10(flat_sync / np.mean(flat_sync, dtype = np.float64))

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

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not 
		# subtracting the mean from the synchrotron maps before calculating the
		# structure function.
		strfn = sf_fft(np.log10(sync_data/np.mean(sync_data,dtype=np.float64)), no_fluct = True)

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

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			spec_locs[i], line_o_sight[j])

# Calculate the time-averaged structure function and quadrupole ratio, as well as 
# the radii used to calculate these quantities
for i in range(num_sims):
	sf_timeavg_arr[i] = np.mean(sf_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	sf_rad_timeavg_arr[i] = np.mean(sf_rad_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	quad_timeavg_arr[i] = np.mean(quad_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	quad_rad_timeavg_arr[i] = np.mean(quad_rad_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)

	# Calculate the standard deviation on each point of the structure function and
	# quadrupole ratio modulus
	sf_err_arr[i] = np.std(sf_arr[i*num_timestep:(i+1)*num_timestep], axis = 0,\
	 dtype = np.float64)/ np.sqrt(num_timestep)
	quad_err_arr[i] = np.std(quad_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)

	# Calculate the time-averaged versions of the statistics
	mean_timeavg_arr[i] = np.mean(mean_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	stdev_timeavg_arr[i] = np.mean(stdev_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	skew_timeavg_arr[i] = np.mean(skew_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	kurt_timeavg_arr[i] = np.mean(kurt_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	m_timeavg_arr[i] = np.mean(m_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	residual_timeavg_arr[i] = np.mean(residual_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	int_quad_timeavg_arr[i] = np.mean(int_quad_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)

	# Calculate the standard deviation in the time-averaged versions of the statistics
	mean_err_arr[i] = np.std(mean_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)
	stdev_err_arr[i] = np.std(stdev_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)
	skew_err_arr[i] = np.std(skew_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)
	kurt_err_arr[i] = np.std(kurt_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)
	m_err_arr[i] = np.std(m_arr[i*num_timestep:(i+1)*num_timestep], axis = 0,\
	 dtype = np.float64)/ np.sqrt(num_timestep)
	residual_err_arr[i] = np.std(residual_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)
	int_quad_err_arr[i] = np.std(int_quad_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)

# When the code reaches this point, the time-averaged structure functions, and
# quadrupole ratios have been saved for every simulation, and every line of 
# sight, so start making the final plots.

#--------------------------- Structure Functions -------------------------------

# Here we want to produce one plot with three subplots. There should be one row
# of subplots, with three subplots in the row. The left subplot will be the 
# structure functions for a line of sight along the x axis, the centre plot will
# be for the y axis, and the right subplot will be the structure functions for 
# the z axis. In each plot the solenoidal and compressive simulations will be 
# compared

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an array of marker symbols, so that the plot for simulation has
# a different plot symbol
symbol_arr = ['o','^', 's', '*', '+']

# Create an axis for the first subplot to be produced, which is for the
# x line of sight
ax1 = fig.add_subplot(131)

# Loop over the simulations to produce plots for each simulation
for i in range(num_sims):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.errorbar(sf_rad_timeavg_arr[i,0], sf_timeavg_arr[i,0],\
	yerr = sf_err_arr[i,0], fmt='-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_timeavg_arr[0,0])), \
	np.zeros(np.shape(sf_rad_timeavg_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
ax1.set_yscale('log')

# Create an axis for the second subplot to be produced, which is for the
# y line of sight. Make the y axis limits the same as for the x axis plot
ax2 = fig.add_subplot(132, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_sims):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.errorbar(sf_rad_timeavg_arr[i,1], sf_timeavg_arr[i,1],\
	yerr = sf_err_arr[i,1], fmt='-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_timeavg_arr[0,1])), \
	np.zeros(np.shape(sf_rad_timeavg_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
ax2.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight. Make the y axis limits the same as for the x axis plot
ax3 = fig.add_subplot(133, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_sims):
	# Plot the structure function for this simulation, for this 
	# line of sight
	plt.errorbar(sf_rad_timeavg_arr[i,2], sf_timeavg_arr[i,2],yerr= sf_err_arr[i,2],\
	 fmt= '-' + symbol_arr[i], label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(sf_rad_timeavg_arr[0,2])), \
	np.zeros(np.shape(sf_rad_timeavg_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis of the plot logarithmic
ax3.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc=4, fontsize = 10, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Structure Function Amplitude', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.18, 0.93, 'a) x-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.445, 0.93, 'b) y-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure c
plt.figtext(0.73, 0.93, 'c) z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'sfs_all_sims_timeavg_gam{}_log_sims{}.eps'.\
	format(gamma,num_sims), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#----------------------------- Quadrupole Ratios -------------------------------

# Here we want to produce one plot with three subplots. There should be one row
# of subplots, with three subplots in the row. The left subplot will be the 
# quadrupole ratio modulus for a line of sight along the x axis, the centre plot
# will be for the y axis, and the right subplot will be the quadrupole ratio
# modulus for the z axis. In each plot the solenoidal and compressive 
# simulations will be compared

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the
# x line of sight
ax1 = fig.add_subplot(131)

# Loop over the simulations to produce plots for each simulation
for i in range(num_sims):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.errorbar(quad_rad_timeavg_arr[i,0], quad_timeavg_arr[i,0],\
	yerr = quad_err_arr[i,0], fmt= '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_timeavg_arr[0,0])), \
	np.zeros(np.shape(quad_rad_timeavg_arr[0,0])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Create an axis for the second subplot to be produced, which is for the
# y line of sight. Make the y axis limits the same as for the x axis plot
ax2 = fig.add_subplot(132, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_sims):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.errorbar(quad_rad_timeavg_arr[i,1], quad_timeavg_arr[i,1],\
	yerr = quad_err_arr[i,1], fmt= '-' + symbol_arr[i])

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_timeavg_arr[0,1])), \
	np.zeros(np.shape(quad_rad_timeavg_arr[0,1])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# z line of sight. Make the y axis limits the same as for the x axis plot
ax3 = fig.add_subplot(133, sharey = ax1)

# Loop over the simulations to produce plots for each simulation
for i in range(num_sims):
	# Plot the quadrupole ratio modulus for this simulation, for this 
	# line of sight
	plt.errorbar(quad_rad_timeavg_arr[i,2], quad_timeavg_arr[i,2],\
	yerr = quad_err_arr[i,2], fmt='-' + symbol_arr[i],\
	label = '{}'.format(sim_labels[i]))

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000,len(quad_rad_timeavg_arr[0,2])), \
	np.zeros(np.shape(quad_rad_timeavg_arr[0,2])), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(fontsize = 10, numpoints=1)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Quadrupole Ratio', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.18, 0.93, 'a) x-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.445, 0.93, 'b) y-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure c
plt.figtext(0.73, 0.93, 'c) z-LOS', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'quad_ratio_all_sims_timeavg_gam{}_sims{}.eps'.\
	format(gamma,num_sims), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#-------------------------------------------------------------------------------

# Create an array of symbols to use for the following plots
sym_list = ['-o','-^','-s']

# Now produce plots of each statistic against the zeta parameter, using the
# plot_vs_zeta function
plot_vs_zeta(mean_timeavg_arr, zeta, mean_err_arr, sym_list, 'Mean',\
 line_o_sight, simul_loc + 'mean_all_sims_timeavg_gam{}_sims{}.eps'.\
	format(gamma,num_sims), 4)

plot_vs_zeta(stdev_timeavg_arr, zeta, stdev_err_arr, sym_list,\
 'Standard Deviation', line_o_sight, simul_loc +\
  'stdev_all_sims_timeavg_gam{}_sims{}.eps'.format(gamma,num_sims), 1)

plot_vs_zeta(skew_timeavg_arr, zeta, skew_err_arr, sym_list, 'Skewness',\
 line_o_sight, simul_loc + 'skew_all_sims_timeavg_gam{}_sims{}.eps'.\
	format(gamma,num_sims), 1)

plot_vs_zeta(kurt_timeavg_arr, zeta, kurt_err_arr, sym_list, 'Kurtosis',\
 line_o_sight, simul_loc + 'kurt_all_sims_timeavg_gam{}_sims{}.eps'.\
	format(gamma,num_sims), 1)

plot_vs_zeta(m_timeavg_arr, zeta, m_err_arr, sym_list, 'm', line_o_sight,\
 simul_loc + 'm_all_sims_timeavg_gam{}_sims{}.eps'.\
	format(gamma,num_sims), 1)

plot_vs_zeta(residual_timeavg_arr, zeta, residual_err_arr, sym_list,\
 'Residuals', line_o_sight, simul_loc + 'resid_all_sims_timeavg_gam{}_sims{}.eps'.\
	format(gamma,num_sims), 1)

plot_vs_zeta(int_quad_timeavg_arr, zeta, int_quad_err_arr, sym_list,\
 'Int Quad Ratio', line_o_sight, simul_loc + 'int_quad_all_sims_timeavg_gam{}_sims{}.eps'.\
	format(gamma,num_sims), 1)