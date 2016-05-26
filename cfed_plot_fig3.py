#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# calculates the time-averaged moments of the synchrotron intensity PDFs. A    #
# plot is produced of the moments against zeta. This code is intended to be    #
# used with simulations produced by Christoph Federrath.                       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 2/3/2016                                                         #
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

# Set the dpi at which to save the image
save_dpi = 300

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

# Create arrays for the time-averaged versions of the above statistics. The 
# first index gives the simulation, and the second index gives the line of sight
# as (x,y,z).
mean_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
stdev_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
skew_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))
kurt_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3))

# Create error arrays for each of the statistics. These errors are calculated
# by the standard deviation of the statistics calculated for different 
# timesteps. The first index gives the simulation, and the second index gives 
# the line of sight as (x,y,z).
mean_err_arr =     np.zeros((len(spec_locs)/num_timestep,3))
stdev_err_arr =    np.zeros((len(spec_locs)/num_timestep,3))
skew_err_arr =     np.zeros((len(spec_locs)/num_timestep,3))
kurt_err_arr =     np.zeros((len(spec_locs)/num_timestep,3))

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

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			spec_locs[i], line_o_sight[j])

# Calculate the time-averaged structure function and quadrupole ratio, as well as 
# the radii used to calculate these quantities
for i in range(num_sims):
	# Calculate the time-averaged versions of the statistics
	mean_timeavg_arr[i] = np.mean(mean_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	stdev_timeavg_arr[i] = np.mean(stdev_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	skew_timeavg_arr[i] = np.mean(skew_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)
	kurt_timeavg_arr[i] = np.mean(kurt_arr[i*num_timestep:(i+1)*num_timestep],\
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

# Print out all of the calculated statistics, along with their errors
# print "Mean", mean_timeavg_arr
# print "StDev", stdev_timeavg_arr
# print "Skew", skew_timeavg_arr
# print "Kurt", kurt_timeavg_arr

# print "Mean Err", mean_err_arr
# print "StDev Err", stdev_err_arr
# print "Skew Err", skew_err_arr
# print "Kurt Err", kurt_err_arr

# Average the x and y axis columns for the mean and standard deviation
mean_perp = np.mean(mean_timeavg_arr[:,0:2], axis = 1, dtype = np.float64)
stdev_perp = np.mean(stdev_timeavg_arr[:,0:2], axis = 1, dtype = np.float64)

# Extract the values of the mean and standard deviation for a line of sight
# that is parallel to the mean magnetic field
mean_par = mean_timeavg_arr[:,2]
stdev_par = stdev_timeavg_arr[:,2]

# Perform a linear fit to the last three values of the mean, for the two lines
# of sight
mean_fit_perp = np.polyfit(zeta[2:], mean_perp[2:], 1)
mean_fit_par = np.polyfit(zeta[2:], mean_par[2:], 1)

# Perform a linear fit to the last three values of the standard deviation, for
# the two lines of sight
stdev_fit_perp = np.polyfit(zeta[2:], stdev_perp[2:], 1)
stdev_fit_par = np.polyfit(zeta[2:], stdev_par[2:], 1)

# Print out the intercepts and gradients for the lines of best fit
print "Mean perp:", mean_fit_perp
print "Mean par:", mean_fit_par
print "StDev perp:", stdev_fit_perp
print "StDev par:", stdev_fit_par

# When the code reaches this point, the time-averaged structure functions, and
# quadrupole ratios have been saved for every simulation, and every line of 
# sight, so start making the final plots.

#-------------------------------------------------------------------------------

# Create an array of symbols to use for the following plots
sym_list = ['-o','-^','-s']

# ---------------------------- Plots of the moments ----------------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will show the mean
# and standard deviation of the log normalised synchrotron intensity, and the 
# bottom row will show the skewness and kurtosis.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the mean
ax1 = fig.add_subplot(221)

# Loop over the lines of sight, to produce a plot for each line of sight
for i in range(len(line_o_sight)):
	# Plot the statistic for this simulation, for this 
	# line of sight
	plt.errorbar(zeta, mean_timeavg_arr[:,i], yerr = mean_err_arr[:,i],\
	 fmt=sym_list[i])

# Change the limits of the x axis
plt.xlim([np.amin(zeta)-0.1, np.amax(zeta)+0.1])

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the 
# standard deviation
ax2 = fig.add_subplot(222)

# Loop over the lines of sight, to produce a plot for each line of sight
for i in range(len(line_o_sight)):
	# Plot the statistic for this simulation, for this 
	# line of sight
	plt.errorbar(zeta, stdev_timeavg_arr[:,i], yerr = stdev_err_arr[:,i],\
	 fmt=sym_list[i], label = '{}-LOS'.format(line_o_sight[i]))

# Force the legends to appear on the plot
plt.legend(loc = 1, fontsize = 10, numpoints=1)

# Change the limits of the x axis
plt.xlim([np.amin(zeta)-0.1, np.amax(zeta)+0.1])

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the skewness
ax3 = fig.add_subplot(223)

# Loop over the lines of sight, to produce a plot for each line of sight
for i in range(len(line_o_sight)):
	# Plot the statistic for this simulation, for this 
	# line of sight
	plt.errorbar(zeta, skew_timeavg_arr[:,i], yerr = skew_err_arr[:,i],\
	 fmt=sym_list[i])

# Change the limits of the x axis
plt.xlim([np.amin(zeta)-0.1, np.amax(zeta)+0.1])

# Create an axis for the fourth subplot to be produced, which is for the 
# kurtosis
ax4 = fig.add_subplot(224)

# Loop over the lines of sight, to produce a plot for each line of sight
for i in range(len(line_o_sight)):
	# Plot the statistic for this simulation, for this 
	# line of sight
	plt.errorbar(zeta, kurt_timeavg_arr[:,i], yerr = kurt_err_arr[:,i],\
	 fmt=sym_list[i])

# Change the limits of the x axis
plt.xlim([np.amin(zeta)-0.1, np.amax(zeta)+0.1])

# Add a label to the x-axis
plt.figtext(0.5, 0.0, r'Turbulent Driving Parameter $\zeta$', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.23, 0.94, 'a) Mean', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.57, 0.94, 'b) Standard Deviation', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.20, 0.48, 'c) Skewness', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.64, 0.48, 'd) Kurtosis', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig3.eps', dpi = save_dpi, format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()