#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity,  #
# and calculates the mean and standard deviation as a function of sonic and    #
# Alfvenic Mach number for simulations with low and high magnetic field. Plots #
# of these statistics as a function of sonic and Alfvenic Mach number are then #
# saved.                                                                       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 9/5/2016                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

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
        mean_err - The error calculated for the mean of synchrotron 
        		   intensity
        stdev_err - The error calculated for the standard deviation of 
        			synchrotron intensity
	'''

	# Create an array that will hold the quarters of the synchrotron images
	quarter_arr = np.zeros((8,np.shape(sync_map_y)[0]/2,np.shape(sync_map_y)[1]/2))

	# Add the quarters of the images into the array
	quarter_arr[0], quarter_arr[1] = np.split(np.split(sync_map_y,2,axis=0)[0],2,axis=1) 
	quarter_arr[2], quarter_arr[3] = np.split(np.split(sync_map_y,2,axis=0)[1],2,axis=1) 
	quarter_arr[4], quarter_arr[5] = np.split(np.split(sync_map_z,2,axis=0)[0],2,axis=1)
	quarter_arr[6], quarter_arr[7] = np.split(np.split(sync_map_z,2,axis=0)[1],2,axis=1)

	# Create arrays that will hold the calculated statistics for each quarter
	mean_val = np.zeros(np.shape(quarter_arr)[0])
	stdev_val = np.zeros(np.shape(quarter_arr)[0])

	# Loop over the quarters, to calculate statistics for each one
	for i in range(np.shape(quarter_arr)[0]):
		# Extract the current image quarter from the array
		image = quarter_arr[i]

		# Flatten the image, so that we can calculate the mean and standard 
		# deviation
		flat_image = image.flatten()

		# In this case we are calculating the moments of the log normalised
		# PDFs, so calculate the log of the synchrotron intensities
		flat_image = np.log10(flat_image / np.mean(flat_image, dtype = np.float64))

		# Calculate the mean of the synchrotron intensity map
		mean_val[i] = np.mean(flat_image, dtype=np.float64)

		# Calculate the standard deviation of the synchrotron intensity maps
		stdev_val[i] = np.std(flat_image, dtype=np.float64)

	# At this point, the statistics have been calculated for each quarter
	# The next step is to calculate the standard error of the mean of each
	# statistic
	mean_err = np.std(mean_val) / np.sqrt(len(mean_val))
	stdev_err = np.std(stdev_val) / np.sqrt(len(stdev_val))

	# Now that all of the calculations have been performed, return the 
	# calculated errors
	return mean_err, stdev_err

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

# Create an empty array, where each entry specifies the calculated mean of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. For y and z lines of sight.
mean_arr_y = np.zeros(len(simul_arr))
mean_arr_z = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated standard 
# deviation of the synchrotron intensity image of the corresponding simulation 
# for a particular value of gamma. For y and z lines of sight
stdev_arr_y = np.zeros(len(simul_arr))
stdev_arr_z = np.zeros(len(simul_arr))

# Create error arrays for each of the statistics. These errors are only for the
# statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field), and are calculated by the standard deviation of the 
# statistics calculated for sub-images of the synchrotron maps.
mean_err_arr = np.zeros(len(simul_arr))
stdev_err_arr = np.zeros(len(simul_arr))

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

	# In this case we are calculating the moments of the log normalised
	# PDFs, so calculate the log of the synchrotron intensities
	flat_sync_y = np.log10(flat_sync_y / np.mean(flat_sync_y, dtype = np.float64))
	flat_sync_z = np.log10(flat_sync_z / np.mean(flat_sync_z, dtype = np.float64))

	# Calculate the mean of the synchrotron intensity maps, and store
	# the results in the corresponding array, for y and z lines of sight
	mean_arr_y[j] = np.mean(flat_sync_y, dtype=np.float64)
	mean_arr_z[j] = np.mean(flat_sync_z, dtype=np.float64)

	# Calculate the standard deviation of the synchrotron intensity 
	# maps, and store the results in the corresponding array, for y and z LOS
	stdev_arr_y[j] = np.std(flat_sync_y, dtype=np.float64)
	stdev_arr_z[j] = np.std(flat_sync_z, dtype=np.float64)

	# Create errors for each of the statistics. These errors are only for the
	# statistics calculated from the y and z axes (perpendicular to the mean 
	# magnetic field), and are calculated by the standard deviation of the 
	# statistics calculated for sub-images of the synchrotron maps.
	mean_err_arr[j], stdev_err_arr[j] = calc_err_bootstrap(sync_map_y, sync_map_z)

	# Close the fits files, to save memory
	sync_fits_y.close()
	sync_fits_z.close()

	# Print a message to show that the calculation has finished successfully
	# for this simulation
	print 'All statistics calculated for simulation {}'.format(simul_arr[j])

# Create mean value arrays for each of the statistics. These values are only for
# the statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field).
mean_mean_arr = (mean_arr_y + mean_arr_z) / 2.0
stdev_mean_arr = (stdev_arr_y + stdev_arr_z) / 2.0

# When the code reaches this point, the statistics have been saved for every 
# simulation, so start making the final plots.

# ------------------- Plots of mean and standard deviation ---------------------

# Here we want to produce one plot with four subplots. There should be two rows
# of subplots, with two subplots in each row. The top row will be the mean, and
# the bottom row will be standard deviation. The left column will be plots 
# against sonic Mach number, and the right column will be plots against Alfvenic
# Mach number.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for mean
# against sonic Mach number
ax1 = fig.add_subplot(221)

# Plot the mean as a function of sonic Mach number 
plt.errorbar(sonic_mach_arr[0:8], mean_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=mean_err_arr[0:8])
plt.errorbar(sonic_mach_arr[8:16], mean_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=mean_err_arr[8:16])
plt.errorbar(sonic_mach_arr[16:], mean_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=mean_err_arr[16:])

# Add a label to the y-axis
plt.ylabel('Mean', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for mean
# as a function of Alfvenic Mach number. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Plot the mean as a function of Alfvenic Mach number
plt.errorbar(alf_mach_arr[0:8], mean_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=mean_err_arr[0:8])
plt.errorbar(alf_mach_arr[8:16], mean_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=mean_err_arr[8:16])
plt.errorbar(alf_mach_arr[16:], mean_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=mean_err_arr[16:])

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for standard 
# deviation as a function of sonic Mach number. Make the x axis limits the same 
# as for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Plot the standard deviation as a function of sonic Mach number
plt.errorbar(sonic_mach_arr[0:8], stdev_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=stdev_err_arr[0:8])
plt.errorbar(sonic_mach_arr[8:16], stdev_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=stdev_err_arr[8:16])
plt.errorbar(sonic_mach_arr[16:], stdev_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=stdev_err_arr[16:])

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('St Dev', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for
# standard deviation as a function of Alfvenic Mach number. Make the x axis 
# limits the same as for the second plot, and the y axis limits the same as the 
# third plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Plot the standard deviation as a function of Alfvenic Mach number
plt.errorbar(alf_mach_arr[0:8], stdev_mean_arr[0:8], ms = 5, mfc = 'b',fmt='o',yerr=stdev_err_arr[0:8])
plt.errorbar(alf_mach_arr[8:16], stdev_mean_arr[8:16], ms = 7,ecolor='r', mfc = 'r',fmt='*',yerr=stdev_err_arr[8:16])
plt.errorbar(alf_mach_arr[16:], stdev_mean_arr[16:], ms = 7,ecolor='g', mfc = 'g',fmt='^',yerr=stdev_err_arr[16:])

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
plt.savefig(simul_loc + 'Ultimate_Output/mean_stdev_mach.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()