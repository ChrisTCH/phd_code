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

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. Each row corresponds to a value of gamma, and each 
# column corresponds to a simulation. There is one array for a line of sight
# along the z axis, and another for a line of sight along the x axis.
# NOTE: We will calculate the biased skewness
skew_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
skew_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. Each row corresponds to a value of gamma, and each 
# column corresponds to a simulation. There is one array for a line of sight
# along the z axis, and another for a line of sight along the x axis.
# NOTE: We will calculate the biased Fisher kurtosis
kurt_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
kurt_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of gamma. Each row 
# corresponds to a value of gamma, and each column corresponds to a simulation.
# There is one array for a line of sight along the z axis, and another for a
# line of sight along the x axis.
m_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
m_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the residuals of the linear
# fit to the structure function of the synchrotron intensity image, of the 
# corresponding simulation, for a particular value of gamma. Each row 
# corresponds to a value of gamma, and each column corresponds to a simulation.
# There is one array for a line of sight along the z axis, and another for a
# line of sight along the x axis.
residual_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
residual_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity
# image, for the corresponding simulation, for a particular value of gamma. Each
# row corresponds to a value of gamma, and each column corresponds to a 
# simulation. There is one array for a line of sight along the z axis, and 
# another for a line of sight along the x axis.
int_quad_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
int_quad_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated magnitude of 
# the quadrupole/monopole ratio of the synchrotron intensity image at a 
# particular radial separation, for the corresponding simulation, for a 
# particular value of gamma. Each row corresponds to a value of gamma, and each 
# column corresponds to a simulation. There is one array for a line of sight 
# along the z axis, and another for a line of sight along the x axis.
quad_point_z_arr = np.zeros((len(gamma_arr),len(simul_arr)))
quad_point_x_arr = np.zeros((len(gamma_arr),len(simul_arr)))

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

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Loop over the various values of gamma, to calculate the various statistics
	# for the synchrotron map observed for each value of gamma
	for i in range(len(gamma_arr)):
		# Extract the synchrotron intensity map for this value of gamma, for
		# lines of sight along the x and z axes
		sync_map_z = sync_data_z[i]
		sync_map_x = sync_data_x[i]

		# Flatten the synchrotron intensity maps for this value of gamma, for
		# lines of sight along the x and z axes
		flat_sync_z = sync_map_z.flatten()
		flat_sync_x = sync_map_x.flatten()

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
		strfn_z = sf_fft(sync_map_z, no_fluct = True)
		strfn_x = sf_fft(sync_map_x, no_fluct = True)

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
		norm_strfn_z = sf_fft(sync_map_z, no_fluct = True, normalise = True)
		norm_strfn_x = sf_fft(sync_map_x, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image. This is done for lines of sight
		# along the x and z axes
		norm_strfn_z = np.fft.fftshift(norm_strfn_z)
		norm_strfn_x = np.fft.fftshift(norm_strfn_x)

		# Calculate the magnitude and argument of the quadrupole ratio, for 
		# lines of sight along the x and z axes.
		quad_mod_z, quad_arg_z, quad_rad_z = calc_quad_ratio(norm_strfn_z, num_bins)
		quad_mod_x, quad_arg_x, quad_rad_x = calc_quad_ratio(norm_strfn_x, num_bins)

		# Find the value of the magnitude of the quadrupole / monopole ratio 
		# for a radial separation that is one third of the way along the radial 
		# separation range that is probed, and store it in the corresponding 
		# array. This is done for lines of sight along the x and z axes.
		quad_point_z_arr[i,j] = quad_mod_z[np.floor(num_bins/3.0)]
		quad_point_x_arr[i,j] = quad_mod_x[np.floor(num_bins/3.0)]

		# Integrate the magnitude of the quadrupole / monopole ratio from 
		# one sixth of the way along the radial separation bins, until three 
		# quarters of the way along the radial separation bins. This integration
		# is performed with respect to log separation (i.e. I am ignoring the 
		# fact that the points are equally separated in log space, to calculate 
		# the area under the quadrupole / monopole ratio plot when the x axis 
		# is scaled logarithmically). I normalise the value that is returned by 
		# dividing by the number of increments in log radial separation used in 
		# the calculation. This is done for lines of sight along the x and z 
		# axes.
		int_quad_z_arr[i,j] = np.trapz(quad_mod_z[np.floor(num_bins/6.0):\
			3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
			 - np.floor(num_bins/6.0))
		int_quad_x_arr[i,j] = np.trapz(quad_mod_x[np.floor(num_bins/6.0):\
			3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
			 - np.floor(num_bins/6.0))

		# At this point, all of the statistics that need to be calculated for
		# every value of gamma have been calculated.

	# Close the fits files, to save memory
	sync_fits_z.close()
	sync_fits_x.close()

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
plt.plot(sonic_mach_arr, skew_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, skew_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, skew_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, skew_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, skew_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, skew_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, skew_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Sonic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved
print 'Plot of the skewness as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, skew_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, skew_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, skew_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, skew_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, skew_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, skew_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, skew_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Alfvenic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the skewness as a function of Alfvenic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig3 = plt.figure()

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number for each gamma
plt.plot(sonic_mach_arr, kurt_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, kurt_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, kurt_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, kurt_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, kurt_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, kurt_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, kurt_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Sonic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved
print 'Plot of the kurtosis as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig4 = plt.figure()

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, kurt_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, kurt_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, kurt_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, kurt_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, kurt_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, kurt_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, kurt_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Alfvenic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Structure Function slope vs sonic Mach number

# Create a figure to display a plot of the SF slope-1 (m) as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig5 = plt.figure()

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot m as a function of sonic Mach number for each gamma
plt.plot(sonic_mach_arr, m_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, m_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, m_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, m_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, m_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, m_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, m_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Sonic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the sonic Mach number has been saved
print 'Plot of m as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# SF slope vs Alfvenic Mach number

# Create a figure to display a plot of the SF slope-1 as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig6 = plt.figure()

# Create an axis for this figure
ax6 = fig6.add_subplot(111)

# Plot m as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, m_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, m_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, m_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, m_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, m_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, m_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, m_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Alfvenic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of m as a function of Alfvenic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig7 = plt.figure()

# Create an axis for this figure
ax7 = fig7.add_subplot(111)

# Plot the residuals as a function of sonic Mach number for each gamma
plt.plot(sonic_mach_arr, residual_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, residual_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, residual_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, residual_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, residual_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, residual_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, residual_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Residuals SF Fit vs Sonic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved
print 'Plot of the residuals as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all gamma
fig8 = plt.figure()

# Create an axis for this figure
ax8 = fig8.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_arr, residual_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, residual_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, residual_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, residual_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, residual_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, residual_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, residual_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Residuals SF Fit vs Alfvenic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the residuals as a function of Alfvenic Mach number saved z'

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
plt.plot(sonic_mach_arr, int_quad_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, int_quad_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, int_quad_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, int_quad_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, int_quad_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, int_quad_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, int_quad_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Integrated Mag quad vs Sonic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the sonic Mach number has been saved
print 'Plot of the integrated quad/mono ratio as a function of sonic Mach'\
+ ' number saved z'

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
plt.plot(alf_mach_arr, int_quad_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, int_quad_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, int_quad_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, int_quad_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, int_quad_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, int_quad_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, int_quad_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Integrated Mag quad vs Alfvenic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the Alfvenic Mach number has been saved
print 'Plot of the integrated quad/mono as a function of Alfvenic Mach'\
+ ' number saved z'

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
plt.plot(sonic_mach_arr, quad_point_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(sonic_mach_arr, quad_point_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(sonic_mach_arr, quad_point_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(sonic_mach_arr, quad_point_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(sonic_mach_arr, quad_point_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(sonic_mach_arr, quad_point_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(sonic_mach_arr, quad_point_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Sonic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2, bbox_to_anchor=(1.05, 1.0))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio at
# a single radius value as a function of the sonic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of sonic Mach'\
+ ' number saved z'

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
plt.plot(alf_mach_arr, quad_point_z_arr[0], 'bo',\
	label ='Gamma = {}'.format(gamma_arr[0]))
plt.plot(alf_mach_arr, quad_point_z_arr[1], 'ro',\
	label ='Gamma = {}'.format(gamma_arr[1]))
plt.plot(alf_mach_arr, quad_point_z_arr[2], 'go',\
	label ='Gamma = {}'.format(gamma_arr[2]))
plt.plot(alf_mach_arr, quad_point_z_arr[3], 'co',\
	label ='Gamma = {}'.format(gamma_arr[3]))
plt.plot(alf_mach_arr, quad_point_z_arr[4], 'mo',\
	label ='Gamma = {}'.format(gamma_arr[4]))
plt.plot(alf_mach_arr, quad_point_z_arr[5], 'yo',\
	label ='Gamma = {}'.format(gamma_arr[5]))
plt.plot(alf_mach_arr, quad_point_z_arr[6], 'ko',\
	label ='Gamma = {}'.format(gamma_arr[6]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Alfvenic Mach Number z', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio
# at a point as a function of the Alfvenic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of Alfvenic Mach'\
+ ' number saved z'

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
plt.plot(gamma_arr, skew_z_arr[:,1], 'bo',\
	label ='{}'.format(short_simul[1]))
plt.plot(gamma_arr, skew_z_arr[:,3], 'ro',\
	label ='{}'.format(short_simul[3]))
plt.plot(gamma_arr, skew_z_arr[:,5], 'go',\
	label ='{}'.format(short_simul[5]))
plt.plot(gamma_arr, skew_z_arr[:,7], 'co',\
	label ='{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Gamma z LOS B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_gamma_z_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of gamma has been saved
print 'Plot of the skewness as a function of gamma saved z'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs gamma z-LOS high magnetic field

# Create a figure to display a plot of the skewness as a function of gamma
# for some of the synchrotron maps with B = 1
fig26 = plt.figure()

# Create an axis for this figure
ax26 = fig26.add_subplot(111)

# Plot the skewness as a function of gamma for each simulation
plt.plot(gamma_arr, skew_z_arr[:,9], 'bo',\
	label ='{}'.format(short_simul[9]))
plt.plot(gamma_arr, skew_z_arr[:,11], 'ro',\
	label ='{}'.format(short_simul[11]))
plt.plot(gamma_arr, skew_z_arr[:,13], 'go',\
	label ='{}'.format(short_simul[13]))
plt.plot(gamma_arr, skew_z_arr[:,15], 'co',\
	label ='{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Gamma z LOS B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_gamma_z_b1.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of gamma has been saved
print 'Plot of the skewness as a function of gamma saved z'

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
plt.plot(gamma_arr, kurt_z_arr[:,1], 'bo',\
	label ='{}'.format(short_simul[1]))
plt.plot(gamma_arr, kurt_z_arr[:,3], 'ro',\
	label ='{}'.format(short_simul[3]))
plt.plot(gamma_arr, kurt_z_arr[:,5], 'go',\
	label ='{}'.format(short_simul[5]))
plt.plot(gamma_arr, kurt_z_arr[:,7], 'co',\
	label ='{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Change the y limit axes to make the plot easier to read
plt.ylim((-5,25))

# Add a title to the plot
plt.title('Kurtosis vs Gamma z LOS B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_gamma_z_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of gamma has been saved
print 'Plot of the kurtosis as a function of gamma saved z'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs gamma z-LOS high magnetic field

# Create a figure to display a plot of the kurtosis as a function of gamma
# for some of the synchrotron maps with B = 1
fig30 = plt.figure()

# Create an axis for this figure
ax30 = fig30.add_subplot(111)

# Plot the kurtosis as a function of gamma for each simulation
plt.plot(gamma_arr, kurt_z_arr[:,9], 'bo',\
	label ='{}'.format(short_simul[9]))
plt.plot(gamma_arr, kurt_z_arr[:,11], 'ro',\
	label ='{}'.format(short_simul[11]))
plt.plot(gamma_arr, kurt_z_arr[:,13], 'go',\
	label ='{}'.format(short_simul[13]))
plt.plot(gamma_arr, kurt_z_arr[:,15], 'co',\
	label ='{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Gamma z LOS B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_gamma_z_b1.png', format = 'png')

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
plt.plot(gamma_arr, m_z_arr[:,1], 'bo',\
	label ='{}'.format(short_simul[1]))
plt.plot(gamma_arr, m_z_arr[:,3], 'ro',\
	label ='{}'.format(short_simul[3]))
plt.plot(gamma_arr, m_z_arr[:,5], 'go',\
	label ='{}'.format(short_simul[5]))
plt.plot(gamma_arr, m_z_arr[:,7], 'co',\
	label ='{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Gamma z LOS B = 0.1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_gamma_z_b.1.png', format = 'png')

# Print a message to the screen to show that the plot of m as a 
# function of gamma has been saved
print 'Plot of m as a function of gamma saved z'

# Close the figure, now that it has been saved.
plt.close()

# m vs gamma z-LOS high magnetic field

# Create a figure to display a plot of m as a function of gamma
# for some of the synchrotron maps with B = 1
fig34 = plt.figure()

# Create an axis for this figure
ax34 = fig34.add_subplot(111)

# Plot m as a function of gamma for each simulation
plt.plot(gamma_arr, m_z_arr[:,9], 'bo',\
	label ='{}'.format(short_simul[9]))
plt.plot(gamma_arr, m_z_arr[:,11], 'ro',\
	label ='{}'.format(short_simul[11]))
plt.plot(gamma_arr, m_z_arr[:,13], 'go',\
	label ='{}'.format(short_simul[13]))
plt.plot(gamma_arr, m_z_arr[:,15], 'co',\
	label ='{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Gamma', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope - 1 vs Gamma z LOS B = 1', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_gamma_z_b1.png', format = 'png')

# Print a message to the screen to show that the plot of m as a 
# function of gamma has been saved
print 'Plot of m as a function of gamma saved z'

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