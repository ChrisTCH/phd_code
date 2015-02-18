#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of the projected magnetic  #
# field amplitude, and calculates the skewness, kurtosis, radially averaged    #
# structure function, and quadrupole/monopole ratio of the projected magnetic  #
# field amplitude. Each of these quantities is plotted against the sonic and   #
# Alfvenic Mach numbers, to see which quantities are sensitive tracers of the  #
# sonic and Alfvenic Mach numbers.                                             #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 10/12/2014                                                       #
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
# correlation functions, and the function that calculates multipoles of 2D 
# images
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D

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

# Create an empty array, where each entry specifies the calculated skewness of
# the magnetic field amplitude image of the corresponding simulation. Each  
# entry corresponds to a simulation. There is one array for a line of sight
# along the z axis, and another for a line of sight along the x axis.
# NOTE: We will calculate the biased skewness
skew_z_arr = np.zeros(len(simul_arr))
skew_x_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the magnetic field amplitude image of the corresponding simulation. Each  
# entry corresponds to a simulation. There is one array for a line of sight
# along the z axis, and another for a line of sight along the x axis.
# NOTE: We will calculate the biased Fisher kurtosis
kurt_z_arr = np.zeros(len(simul_arr))
kurt_x_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the magnetic field amplitude image minus 1, of the 
# corresponding simulation. Each entry corresponds to a simulation. There is 
# one array for a line of sight along the z axis, and another for a line of
# sight along the x axis.
m_z_arr = np.zeros(len(simul_arr))
m_x_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the residuals of the linear
# fit to the structure function of the magnetic field amplitude image, of the 
# corresponding simulation. Each entry corresponds to a simulation.
# There is one array for a line of sight along the z axis, and another for a
# line of sight along the x axis.
residual_z_arr = np.zeros(len(simul_arr))
residual_x_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated integral of
# the quadrupole / monopole ratio of the magnetic field amplitude image, for the 
# corresponding simulation. Each entry corresponds to a simulation.
# There is one array for a line of sight along the z axis, and another for a
# line of sight along the x axis.
int_quad_z_arr = np.zeros(len(simul_arr))
int_quad_x_arr = np.zeros(len(simul_arr))

# Create an empty array, where each entry specifies the calculated quadrupole /
# monopole ratio of the magnetic field amplitude image at a particular radial 
# separation, for the corresponding simulation. Each entry corresponds to a 
# simulation. There is one array for a line of sight along the z axis, and 
# another for a line of sight along the x axis.
quad_point_z_arr = np.zeros(len(simul_arr))
quad_point_x_arr = np.zeros(len(simul_arr))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS files that contain the simulated magnetic field amplitude
	# maps for lines of sight along the z axis and x axis
	B_amp_fits_z = fits.open(data_loc + 'mag_proj_z.fits')
	B_amp_fits_x = fits.open(data_loc + 'mag_proj_x.fits')

	# Extract the data for the simulated magnetic field amplitudes
	# This is a 2D data cube, where each pixel gives the amplitude of the 
	# magnetic field integrated along the line of sight.
	B_map_z = B_amp_fits_z[0].data
	B_map_x = B_amp_fits_x[0].data

	# Print a message to the screen to show that the magnetic field amplitude
	# data has been loaded successfully
	print 'Simulated magnetic field amplitude data loaded'

	# Flatten the magnetic field amplitude maps for this value of gamma, for
	# lines of sight along the x and z axes
	flat_B_z = B_map_z.flatten()
	flat_B_x = B_map_x.flatten()

	# Calculate the biased skewness of the magnetic field amplitude maps, for
	# lines of sight along the x and z axes, and store the results in the
	# corresponding array.
	skew_z_arr[j] = stats.skew(flat_B_z)
	skew_x_arr[j] = stats.skew(flat_B_x)

	# Calculate the biased Fisher kurtosis of the magnetic field amplitude 
	# maps, for lines of sight along the x and z axes, and store the results
	# in the corresponding array.
	kurt_z_arr[j] = stats.kurtosis(flat_B_z)
	kurt_x_arr[j] = stats.kurtosis(flat_B_x)

	# Calculate the structure function (two-dimensional) of the magnetic field
	# amplitude maps, for the lines of sight along the x and z axes. Note 
	# that no_fluct = True is set, because we are not subtracting the mean
	# from the magnetic field amplitude maps before calculating the structure
	# function.
	strfn_z = sf_fft(B_map_z, no_fluct = True)
	strfn_x = sf_fft(B_map_x, no_fluct = True)

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

	# Calculate the spectral index of the structure function. Note that only the
	# first third of the structure function is used in the calculation, as this
	# is the part that is close to a straight line. Perform a linear fit for a 
	# line of sight along the z axis.
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
	residual_z_arr[j] = spec_ind_data_z[1]
	residual_x_arr[j] = spec_ind_data_x[1]

	# Enter the value of m, the slope of the structure function minus 1,
	# into the corresponding array, for lines of sight along the x and z
	# axes
	m_z_arr[j] = coeff_z[0]-1.0
	m_x_arr[j] = coeff_x[0]-1.0

	# Calculate the 2D structure function for the magnetic field amplitude
	# data cube. Note that no_fluct = True is set, because we are not 
	# subtracting the mean from the magnetic field amplitude maps before 
	# calculating the structure function. We are also calculating the normalised 
	# structure function, which only takes values between 0 and 2.
	norm_strfn_z = sf_fft(B_map_z, no_fluct = True, normalise = True)
	norm_strfn_x = sf_fft(B_map_x, no_fluct = True, normalise = True)

	# Shift the 2D structure function so that the zero radial separation
	# entry is in the centre of the image. This is done for lines of sight
	# along the x and z axes
	norm_strfn_z = np.fft.fftshift(norm_strfn_z)
	norm_strfn_x = np.fft.fftshift(norm_strfn_x)

	# Calculate the monopole for the normalised structure function, for 
	# lines of sight along the x and z axes
	monopole_arr_z, mono_rad_arr_z = calc_multipole_2D(norm_strfn_z,\
	 order = 0, num_bins = num_bins)
	monopole_arr_x, mono_rad_arr_x = calc_multipole_2D(norm_strfn_x,\
	 order = 0, num_bins = num_bins)

	# Calculate the quadrupole for the normalised structure function, for 
	# lines of sight along the x and z axes
	quadpole_arr_z, quad_rad_arr_z = calc_multipole_2D(norm_strfn_z,\
	 order = 2, num_bins = num_bins)
	quadpole_arr_x, quad_rad_arr_x = calc_multipole_2D(norm_strfn_x,\
	 order = 2, num_bins = num_bins)

	# Calculate the quadrupole / monopole ratio, for lines of sight along
	# the x and z axes
	quad_mono_z = quadpole_arr_z / monopole_arr_z
	quad_mono_x = quadpole_arr_x / monopole_arr_x

	# Find the value of the quadrupole / monopole ratio for a radial 
	# separation that is one third of the way along the radial separation
	# range that is probed, and store it in the corresponding array. This
	# is done for lines of sight along the x and z axes.
	quad_point_z_arr[j] = quad_mono_z[np.floor(num_bins/3.0)]
	quad_point_x_arr[j] = quad_mono_x[np.floor(num_bins/3.0)]

	# Integrate the quadrupole / monopole ratio from one sixth of the way
	# along the radial separation bins, until three quarters of the way
	# along the radial separation bins. This integration is performed with
	# respect to log separation (i.e. I am ignoring the fact that the 
	# points are equally separated in log space, to calculate the area under
	# the quadrupole / monopole ratio plot when the x axis is scaled 
	# logarithmically). I normalise the value that is returned by dividing
	# by the number of increments in log radial separation used in the
	# calculation. This is done for lines of sight along the x and z axes.
	int_quad_z_arr[j] = np.trapz(quad_mono_z[np.floor(num_bins/6.0):\
		3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
		 - np.floor(num_bins/6.0))
	int_quad_x_arr[j] = np.trapz(quad_mono_x[np.floor(num_bins/6.0):\
		3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
		 - np.floor(num_bins/6.0))

	# At this point, all of the statistics that need to be calculated
	# have been calculated.

	# Close the fits files, to save memory
	B_amp_fits_z.close()
	B_amp_fits_x.close()

# When the code reaches this point, the statistics have been calculated for
# every simulation, so it is time to start plotting

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/'

#------------------------- Line of sight along z -------------------------------

# Skewness vs sonic Mach number

# Create a figure to display a plot of the skewness as a function of sonic
# Mach number for the magnetic field amplitude maps
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the skewness as a function of sonic Mach number
plt.plot(sonic_mach_sort, skew_z_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mag Skew vs Sonic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, skew_z_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_skew_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved
print 'Plot of the skewness as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, skew_z_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mag Skew vs Alfvenic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, skew_z_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_skew_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the skewness as a function of Alfvenic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for the magnetic field amplitude maps
fig3 = plt.figure()

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number
plt.plot(sonic_mach_sort, kurt_z_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Mag Kurtosis vs Sonic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, kurt_z_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_kurt_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved
print 'Plot of the kurtosis as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig4 = plt.figure()

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, kurt_z_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Mag kurtosis vs Alfvenic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, kurt_z_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_kurt_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Structure Function slope vs sonic Mach number

# Create a figure to display a plot of the SF slope-1 (m) as a function of sonic
# Mach number for the magnetic field amplitude maps
fig5 = plt.figure()

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot m as a function of sonic Mach number
plt.plot(sonic_mach_sort, m_z_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mag SF Slope - 1 vs Sonic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, m_z_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_m_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the sonic Mach number has been saved
print 'Plot of m as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# SF slope vs Alfvenic Mach number

# Create a figure to display a plot of the SF slope-1 as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig6 = plt.figure()

# Create an axis for this figure
ax6 = fig6.add_subplot(111)

# Plot m as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, m_z_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mag SF Slope - 1 vs Alfvenic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, m_z_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_m_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of m as a function of Alfvenic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for the magnetic field amplitude maps
fig7 = plt.figure()

# Create an axis for this figure
ax7 = fig7.add_subplot(111)

# Plot the residuals as a function of sonic Mach number
plt.plot(sonic_mach_sort, residual_z_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Mag Residuals SF Fit vs Sonic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, residual_z_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_residual_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved
print 'Plot of the residuals as a function of sonic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig8 = plt.figure()

# Create an axis for this figure
ax8 = fig8.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, residual_z_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Mag Residuals SF Fit vs Alfvenic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, residual_z_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_residual_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the residuals as a function of Alfvenic Mach number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Integrated quad / mono ratio vs sonic Mach number

# Create a figure to display a plot of the integrated quad / mono ratio as a
# function of sonic Mach number for the magnetic field amplitude maps
fig9 = plt.figure()

# Create an axis for this figure
ax9 = fig9.add_subplot(111)

# Plot the integrated quad / mono ratio as a function of sonic Mach number for
# each gamma
plt.plot(sonic_mach_sort, int_quad_z_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Mag Integrated quad/mono vs Sonic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, int_quad_z_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_int_quad_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the sonic Mach number has been saved
print 'Plot of the integrated quad/mono ratio as a function of sonic Mach'\
+ ' number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Integrated quad / mono ratio vs Alfvenic Mach number

# Create a figure to display a plot of the integrated quad/mono ratio as a
# function of Alfvenic Mach number for the magnetic field amplitude maps
fig10 = plt.figure()

# Create an axis for this figure
ax10 = fig10.add_subplot(111)

# Plot the integrated quad / mono ratio as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, int_quad_z_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Mag Integrated quad/mono vs Alfvenic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, int_quad_z_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_int_quad_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the Alfvenic Mach number has been saved
print 'Plot of the integrated quad/mono as a function of Alfvenic Mach'\
+ ' number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Quad / mono ratio at a point vs sonic Mach number

# Create a figure to display a plot of the quad/mono ratio at a point as a
# function of sonic Mach number for the magnetic field amplitude maps
fig11 = plt.figure()

# Create an axis for this figure
ax11 = fig11.add_subplot(111)

# Plot the quad/mono ratio as a point as a function of sonic Mach number
plt.plot(sonic_mach_sort, quad_point_z_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Quad/mono ratio at R = {0:.2f}'.format(mono_rad_arr_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono ratio at point vs Sonic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, quad_point_z_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_quad_point_sonic_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio at
# a single radius value as a function of the sonic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of sonic Mach'\
+ ' number saved z'

# Close the figure, now that it has been saved.
plt.close()

# Quad / mono ratio at a point vs Alfvenic Mach number

# Create a figure to display a plot of the quad/mono ratio at a point as a
# function of Alfvenic Mach number for the magnetic field amplitude maps
fig12 = plt.figure()

# Create an axis for this figure
ax12 = fig12.add_subplot(111)

# Plot the quad/mono ratio at a point as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, quad_point_z_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Quad/mono ratio at R = {0:.2f}'.format(mono_rad_arr_z\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono ratio at point vs Alfvenic Mach Number z', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, quad_point_z_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_quad_point_alf_mach_z.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio
# at a point as a function of the Alfvenic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of Alfvenic Mach'\
+ ' number saved z'

# Close the figure, now that it has been saved.
plt.close()

#------------------------- Line of sight along x -------------------------------

# Skewness vs sonic Mach number

# Create a figure to display a plot of the skewness as a function of sonic
# Mach number for the magnetic field amplitude maps
fig13 = plt.figure()

# Create an axis for this figure
ax13 = fig13.add_subplot(111)

# Plot the skewness as a function of sonic Mach number
plt.plot(sonic_mach_sort, skew_x_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mag Skew vs Sonic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, skew_x_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_skew_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved
print 'Plot of the skewness as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig14 = plt.figure()

# Create an axis for this figure
ax14 = fig14.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, skew_x_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Mag Skew vs Alfvenic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, skew_x_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_skew_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the skewness as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for the magnetic field amplitude maps
fig15 = plt.figure()

# Create an axis for this figure
ax15 = fig15.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number
plt.plot(sonic_mach_sort, kurt_x_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Mag Kurtosis vs Sonic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, kurt_x_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_kurt_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved
print 'Plot of the kurtosis as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig16 = plt.figure()

# Create an axis for this figure
ax16 = fig16.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number for each gamma
plt.plot(alf_mach_sort, kurt_x_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Mag Kurtosis vs Alfvenic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, kurt_x_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_kurt_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Structure Function slope vs sonic Mach number

# Create a figure to display a plot of the SF slope-1 (m) as a function of sonic
# Mach number for the magnetic field amplitude maps
fig17 = plt.figure()

# Create an axis for this figure
ax17 = fig17.add_subplot(111)

# Plot m as a function of sonic Mach number
plt.plot(sonic_mach_sort, m_x_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mag SF Slope - 1 vs Sonic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, m_x_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_m_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the sonic Mach number has been saved
print 'Plot of m as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# SF slope vs Alfvenic Mach number

# Create a figure to display a plot of the SF slope-1 as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig18 = plt.figure()

# Create an axis for this figure
ax18 = fig18.add_subplot(111)

# Plot m as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, m_x_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Mag SF Slope - 1 vs Alfvenic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, m_x_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_m_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of m as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for the magnetic field amplitude maps
fig19 = plt.figure()

# Create an axis for this figure
ax19 = fig19.add_subplot(111)

# Plot the residuals as a function of sonic Mach number
plt.plot(sonic_mach_sort, residual_x_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Mag Residuals SF Fit vs Sonic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, residual_x_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_residual_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved
print 'Plot of the residuals as a function of sonic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for the magnetic field amplitude maps
fig20 = plt.figure()

# Create an axis for this figure
ax20 = fig20.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, residual_x_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals SF Fit', fontsize = 20)

# Add a title to the plot
plt.title('Mag Residuals SF Fit vs Alfvenic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, residual_x_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_residual_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the residuals as a function of Alfvenic Mach number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Integrated quad / mono ratio vs sonic Mach number

# Create a figure to display a plot of the integrated quad / mono ratio as a
# function of sonic Mach number for the magnetic field amplitude maps
fig21 = plt.figure()

# Create an axis for this figure
ax21 = fig21.add_subplot(111)

# Plot the integrated quad / mono ratio as a function of sonic Mach number
plt.plot(sonic_mach_sort, int_quad_x_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Mag Integrated quad/mono vs Sonic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, int_quad_x_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_int_quad_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the sonic Mach number has been saved
print 'Plot of the integrated quad/mono ratio as a function of sonic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Integrated quad / mono ratio vs Alfvenic Mach number

# Create a figure to display a plot of the integrated quad/mono ratio as a
# function of Alfvenic Mach number for the magnetic field amplitude maps
fig22 = plt.figure()

# Create an axis for this figure
ax22 = fig22.add_subplot(111)

# Plot the integrated quad / mono ratio as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, int_quad_x_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Mag Integrated quad/mono vs Alfvenic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, int_quad_x_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_int_quad_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the integrated quad/
# mono ratio as a function of the Alfvenic Mach number has been saved
print 'Plot of the integrated quad/mono as a function of Alfvenic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Quad / mono ratio at a point vs sonic Mach number

# Create a figure to display a plot of the quad/mono ratio at a point as a
# function of sonic Mach number for the magnetic field amplitude maps
fig23 = plt.figure()

# Create an axis for this figure
ax23 = fig23.add_subplot(111)

# Plot the quad/mono ratio as a point as a function of sonic Mach number
plt.plot(sonic_mach_sort, quad_point_x_arr[sonic_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Quad/mono ratio at R = {0:.2f}'.format(mono_rad_arr_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono ratio at point vs Sonic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the Alfenic Mach
# number to each point
for label, x, y in zip(alf_mach_arr[sonic_sort], sonic_mach_sort, quad_point_x_arr[sonic_sort]):
    # Annotate the current point with the Alfvenic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_quad_point_sonic_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio at
# a single radius value as a function of the sonic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of sonic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()

# Quad / mono ratio at a point vs Alfvenic Mach number

# Create a figure to display a plot of the quad/mono ratio at a point as a
# function of Alfvenic Mach number for the magnetic field amplitude maps
fig24 = plt.figure()

# Create an axis for this figure
ax24 = fig24.add_subplot(111)

# Plot the quad/mono ratio at a point as a function of Alfvenic Mach number
plt.plot(alf_mach_sort, quad_point_x_arr[alf_sort], 'bo')

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Quad/mono ratio at R = {0:.2f}'.format(mono_rad_arr_x\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono ratio at point vs Alfvenic Mach Number x', fontsize = 20)

# Cycle through the data points, to add a text label giving the sonic Mach
# number to each point
for label, x, y in zip(sonic_mach_arr[alf_sort], alf_mach_sort, quad_point_x_arr[alf_sort]):
    # Annotate the current point with the sonic Mach number, to two decimal
    # points
	plt.annotate('{0:.2f}'.format(label), xy = (x, y), xytext = (0,0),\
		textcoords = 'offset points', ha = 'right', va = 'bottom')

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mag_quad_point_alf_mach_x.png', format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio
# at a point as a function of the Alfvenic Mach number has been saved
print 'Plot of quad/mono ratio at point as a function of Alfvenic Mach'\
+ ' number saved x'

# Close the figure, now that it has been saved.
plt.close()