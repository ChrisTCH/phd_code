#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# calculates the time-averaged structure functions and quadrupole ratios of    #
# the synchrotron intensity maps, for different lines of sight. This is done   #
# for the compressive simulation only, as it has the biggest differences       #
# between lines of sight.                                                      #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 26/2/2016                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats
import h5py

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

# Create a variable that controls whether the moments of the log normalised PDFs
# are calculated
log = True

# Set a variable that holds the number of timesteps we have for the simulations
num_timestep = 5

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
spec_locs = ['512cM5Bs5886_20/',\
'512cM5Bs5886_25/', '512cM5Bs5886_30/', '512cM5Bs5886_35/', '512cM5Bs5886_40/']

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create an array that gives the timestep for each simulation snapshot
timestep = ['20', '25', '30', '35', '40']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Create an empty array, where each entry specifies the calculated mean of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z). The third index gives the 
# integration axis, as (0,1,2).
mean_arr = np.zeros((len(spec_locs),3,3))

# Create an empty array, where each entry specifies the calculated standard
# deviation of the synchrotron intensity image of the corresponding simulation
# for a particular value of gamma. The first index gives the simulation, and the 
# second index gives the line of sight as (x,y,z). The third index gives the 
# integration axis, as (0,1,2).
stdev_arr = np.zeros((len(spec_locs),3,3))

# Create arrays for the time-averaged versions of the above statistics. The 
# first index gives the simulation, and the second index gives the line of sight
# as (x,y,z). The third index gives the integration axis, as (0,1,2).
mean_timeavg_arr = np.zeros((3, 3))
stdev_timeavg_arr = np.zeros((3, 3))

# Create error arrays for each of the statistics. These errors are calculated
# by the standard deviation of the statistics calculated for different 
# timesteps. The first index gives the simulation, and the second index gives 
# the line of sight as (x,y,z). The third index gives the integration axis, as 
# (0,1,2).
mean_err_arr =     np.zeros((3, 3))
stdev_err_arr =    np.zeros((3, 3))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]

	# Load magnetic field components here
	mag_x_hdf = h5py.File(data_loc + 'OUT_hdf5_plt_cnt_00{}_magx'.format(timestep[i]),'r')
	mag_y_hdf = h5py.File(data_loc + 'OUT_hdf5_plt_cnt_00{}_magy'.format(timestep[i]),'r')
	mag_z_hdf = h5py.File(data_loc + 'OUT_hdf5_plt_cnt_00{}_magz'.format(timestep[i]),'r')

	# Extract the components of the magnetic field
	mag_x = mag_x_hdf['magx']
	mag_y = mag_y_hdf['magy']
	mag_z = mag_z_hdf['magz']

	# Loop over the lines of sight, to calculate the correlation function, 
	# structure function and quadrupole ratio for each line of sight
	for j in range(3):
		# Calculate the magnetic field perpendicular to this line of sight
		if line_o_sight[j] == 'z':
			# Calculate the magnitude of the magnetic field perpendicular to the line of
			# sight, which is just the square root of the sum of the x and y component
			# magnitudes squared.
			mag_perp = np.sqrt( np.power(mag_x, 2.0) + np.power(mag_y, 2.0) )
		if line_o_sight[j] == 'y':
			# Calculate the magnitude of the magnetic field perpendicular to the line of
			# sight, which is just the square root of the sum of the x and z component
			# magnitudes squared.
			mag_perp = np.sqrt( np.power(mag_x, 2.0) + np.power(mag_z, 2.0) )
		if line_o_sight[j] == 'x':
			# Calculate the magnitude of the magnetic field perpendicular to the line of
			# sight, which is just the square root of the sum of the y and z component
			# magnitudes squared.
			mag_perp = np.sqrt( np.power(mag_y, 2.0) + np.power(mag_z, 2.0) )

		# Calculate the result of raising the perpendicular magnetic field 
		# strength to the power of gamma
		mag_perp_gamma = np.power(mag_perp, gamma)

		# Loop over integration axes
		for k in range(3):
			# Integrate the perpendicular magnetic field strength raised to the 
			# power of gamma along the required axis, to calculate the observed 
			# synchrotron map. This integration is performed by the trapezoidal 
			# rule. To normalise the calculated synchrotron map, divide by the 
			# number of pixels along this integration axis. 
			sync_arr = np.trapz(mag_perp_gamma, dx = 1.0, axis = k) /\
			 np.shape(mag_perp)[0]

			# Flatten the array of synchrotron intensity values
			flat_sync = sync_arr.flatten()

			# If we are calculating the moments of the log PDFs, then calculate the
			# logarithm of the synchrotron intensity values
			if log == True:
				# In this case we are calculating the moments of the log normalised
				# PDFs, so calculate the log of the synchrotron intensities
				flat_sync = np.log10(flat_sync / np.mean(flat_sync, dtype = np.float64))

			# Calculate the mean of the synchrotron intensity map, and store the
			# result in the corresponding array
			mean_arr[i,j,k] = np.mean(flat_sync, dtype=np.float64)

			# Calculate the standard deviation of the synchrotron intensity map, and
			# store the result in the corresponding array
			stdev_arr[i,j,k] = np.std(flat_sync, dtype=np.float64)

	# Close the fits files, to save memory
	mag_x_hdf.close()
	mag_y_hdf.close()
	mag_z_hdf.close()

	# Print a message to show that the calculation has finished successfully
	# for this simulation
	print 'All statistics calculated for simulation {}'.format(spec_locs[i])

# Calculate the time-averaged versions of the statistics
mean_timeavg_arr = np.mean(mean_arr, axis = 0, dtype = np.float64)
stdev_timeavg_arr = np.mean(stdev_arr, axis = 0, dtype = np.float64)

# Calculate the standard deviation in the time-averaged versions of the statistics
mean_err_arr = np.std(mean_arr, axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)
stdev_err_arr = np.std(stdev_arr, axis = 0, dtype = np.float64)/ np.sqrt(num_timestep)

# When the code reaches this point, the time-averaged mean and standard 
# deviation have been saved for every simulation, and every line of 
# sight, so start printing out the results.
print "Mean synchrotron intensity, line of sight on y-axis, integration axis on x axis"
print mean_timeavg_arr
print ""

print "Error in mean synchrotron intensity, line of sight on y-axis, integration axis on x axis"
print mean_err_arr
print ""

print "Stdev synchrotron intensity, line of sight on y-axis, integration axis on x axis"
print stdev_timeavg_arr
print ""

print "Error in stdev synchrotron intensity, line of sight on y-axis, integration axis on x axis"
print stdev_err_arr
print ""