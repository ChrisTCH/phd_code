#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the skewness, kurtosis, radially averaged        #
# structure function, and quadrupole/monopole ratio of the synchrotron         #
# intensity for a single value of gamma. Each of these quantities is plotted   #
# against the sonic and Alfvenic Mach numbers, to see which quantities are     #
# sensitive tracers of the sonic and Alfvenic Mach numbers. Plots are made for #
# different angles between the line of sight and the mean magnetic field, and  #
# statistics are also plotted against this relative angle.                     #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 25/11/2014                                                       #
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
# quadrupole ratio
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
# synchrotron intensity map
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

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. Each row corresponds to a rotation angle, and each 
# column corresponds to a simulation.
# NOTE: We will calculate the biased skewness
skew_arr = np.zeros((len(rot_ang_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. Each row corresponds to a rotation angle, and each 
# column corresponds to a simulation.
# NOTE: We will calculate the biased Fisher kurtosis
kurt_arr = np.zeros((len(rot_ang_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of gamma. Each row 
# corresponds to a rotation angle, and each column corresponds to a simulation.
m_arr = np.zeros((len(rot_ang_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the residuals of the linear
# fit to the structure function of the synchrotron intensity image, of the 
# corresponding simulation, for a particular value of gamma. Each row 
# corresponds to a rotation angle, and each column corresponds to a simulation.
residual_arr = np.zeros((len(rot_ang_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated integral of
# the magnitude of the quadrupole / monopole ratio of the synchrotron intensity 
# image, for the corresponding simulation, for a particular value of gamma. Each
# row corresponds to a rotation angle, and each column corresponds to a 
# simulation.
int_quad_arr = np.zeros((len(rot_ang_arr),len(simul_arr)))

# Create an empty array, where each entry specifies the calculated magnitude of
# the quadrupole/monopole ratio of the synchrotron intensity image at a 
# particular radial separation, for the corresponding simulation, for a 
# particular value of gamma. Each row corresponds to a rotation angle, and each 
# column corresponds to a simulation.
quad_point_arr = np.zeros((len(rot_ang_arr),len(simul_arr)))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Loop over the various rotation angles, to calculate the various statistics
	# for the synchrotron map observed for each rotation angle
	for i in range(len(rot_ang_arr)):
		# Open the FITS file that contains the simulated synchrotron intensity
		# map for this line of sight
		sync_fits = fits.open(data_loc + 'synint_p1-4_rot_{}.fits'.\
			format(rot_ang_arr[i]))

		# Extract the data for the simulated synchrotron intensities
		# This is a 3D data cube, where the slices along the third axis are the
		# synchrotron intensities observed for different values of gamma, the power
		# law index of the cosmic ray electrons emitting the synchrotron emission.
		sync_data = sync_fits[0].data
	
		# Extract the synchrotron intensity map for this value of gamma
		sync_map = sync_data[gam_index]

		# Flatten the synchrotron intensity map for this value of gamma
		flat_sync = sync_map.flatten()

		# Calculate the biased skewness of the synchrotron intensity map, and
		# store the results in the corresponding array.
		skew_arr[i,j] = stats.skew(flat_sync)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# map, and store the results in the corresponding array.
		kurt_arr[i,j] = stats.kurtosis(flat_sync)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not
		# subtracting the mean from the synchrotron maps before calculating the
		# structure function.
		strfn = sf_fft(sync_map, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins.
		rad_sf = sfr(strfn, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function.
		sf = rad_sf[1]

		# Extract the radius values used to calculate this structure function.
		sf_rad_arr = rad_sf[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. Perform a linear fit for a line
		# of sight along the z axis.
		spec_ind_data = np.polyfit(np.log10(\
			sf_rad_arr[0:np.ceil(num_bins/3.0)]),\
			np.log10(sf[0:np.ceil(num_bins/3.0)]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit
		coeff = spec_ind_data[0]

		# Extract the sum of the residuals from the polynomial fit
		residual_arr[i,j] = spec_ind_data[1]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array
		m_arr[i,j] = coeff[0]-1.0

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn = sf_fft(sync_map, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image. 
		norm_strfn = np.fft.fftshift(norm_strfn)

		# Calculate the magnitude and argument of the quadrupole ratio, for this
		# line of sight
		quad_mod, quad_arg, quad_rad = calc_quad_ratio(norm_strfn, num_bins)

		# Find the value of the magnitude of the quadrupole / monopole ratio for
		# a radial separation that is one third of the way along the radial 
		# separation range that is probed, and store it in the corresponding 
		# array.
		quad_point_arr[i,j] = quad_mod[np.floor(num_bins/3.0)]

		# Integrate the magnitude of the quadrupole / monopole ratio from one 
		# sixth of the way along the radial separation bins, until three 
		# quarters of the way along the radial separation bins. This integration
		# is performed with respect to log separation (i.e. I am ignoring the 
		# fact that the points are equally separated in log space, to calculate 
		# the area under the quadrupole / monopole ratio plot when the x axis is
		# scaled logarithmically). I normalise the value that is returned by 
		# dividing by the number of increments in log radial separation used in 
		# the calculation.
		int_quad_arr[i,j] = np.trapz(quad_mod[np.floor(num_bins/6.0):\
			3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
			 - np.floor(num_bins/6.0))

		# At this point, all of the statistics that need to be calculated for
		# every line of sight have been calculated.

		# Close the fits files, to save memory
		sync_fits.close()

# When the code reaches this point, the statistics have been calculated for
# every simulation and every line of sight, so it is time to start plotting

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/'

#-------------------------------- Skewness -------------------------------------

# Skewness vs sonic Mach number

# Create a figure to display a plot of the skewness as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the skewness as a function of sonic Mach number for each line of sight
plt.plot(sonic_mach_sort, (skew_arr[-1])[sonic_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(sonic_mach_sort, (skew_arr[-2])[sonic_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(sonic_mach_sort, (skew_arr[-3])[sonic_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(sonic_mach_sort, (skew_arr[-4])[sonic_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(sonic_mach_sort, (skew_arr[-5])[sonic_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(sonic_mach_sort, (skew_arr[-6])[sonic_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(sonic_mach_sort, (skew_arr[-7])[sonic_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(sonic_mach_sort, (skew_arr[-8])[sonic_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(sonic_mach_sort, (skew_arr[-9])[sonic_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(sonic_mach_sort, (skew_arr[-10])[sonic_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Sonic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_sonic_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the sonic Mach number has been saved
print 'Plot of the skewness as a function of sonic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Alfvenic Mach number 

# Create a figure to display a plot of the skewness as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the skewness as a function of Alfvenic Mach number for each line of sight
plt.plot(alf_mach_sort, (skew_arr[-1])[alf_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(alf_mach_sort, (skew_arr[-2])[alf_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(alf_mach_sort, (skew_arr[-3])[alf_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(alf_mach_sort, (skew_arr[-4])[alf_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(alf_mach_sort, (skew_arr[-5])[alf_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(alf_mach_sort, (skew_arr[-6])[alf_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(alf_mach_sort, (skew_arr[-7])[alf_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(alf_mach_sort, (skew_arr[-8])[alf_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(alf_mach_sort, (skew_arr[-9])[alf_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(alf_mach_sort, (skew_arr[-10])[alf_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Alfvenic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_alf_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the skewness as a function of Alfvenic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Relative Angle - Low Magnetic Field

# Create a figure to display a plot of the skewness as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 0.1 
fig3 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the skewness as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 0.1
plt.plot(rel_ang_arr, skew_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(rel_ang_arr, skew_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(rel_ang_arr, skew_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(rel_ang_arr, skew_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(rel_ang_arr, skew_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(rel_ang_arr, skew_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(rel_ang_arr, skew_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(rel_ang_arr, skew_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Angle LOS - Mean B b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box3 = ax3.get_position()
ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])

# Force the legend to appear on the plot
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_rel_ang_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the skewness as a function of relative angle saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# Skewness vs Relative Angle - High Magnetic Field

# Create a figure to display a plot of the skewness as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 1 
fig4 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the skewness as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 1
plt.plot(rel_ang_arr, skew_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(rel_ang_arr, skew_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(rel_ang_arr, skew_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(rel_ang_arr, skew_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(rel_ang_arr, skew_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(rel_ang_arr, skew_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(rel_ang_arr, skew_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(rel_ang_arr, skew_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Add a title to the plot
plt.title('Skew vs Angle LOS - Mean B b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box4 = ax4.get_position()
ax4.set_position([box4.x0, box4.y0, box4.width * 0.8, box4.height])

# Force the legend to appear on the plot
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Skew_rel_ang_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the skewness as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the skewness as a function of relative angle saved b=1'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------- Kurtosis --------------------------------------

# Kurtosis vs sonic Mach number

# Create a figure to display a plot of the kurtosis as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig5 = plt.figure()

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot the kurtosis as a function of sonic Mach number for each line of sight
plt.plot(sonic_mach_sort, (kurt_arr[-1])[sonic_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(sonic_mach_sort, (kurt_arr[-2])[sonic_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(sonic_mach_sort, (kurt_arr[-3])[sonic_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(sonic_mach_sort, (kurt_arr[-4])[sonic_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(sonic_mach_sort, (kurt_arr[-5])[sonic_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(sonic_mach_sort, (kurt_arr[-6])[sonic_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(sonic_mach_sort, (kurt_arr[-7])[sonic_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(sonic_mach_sort, (kurt_arr[-8])[sonic_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(sonic_mach_sort, (kurt_arr[-9])[sonic_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(sonic_mach_sort, (kurt_arr[-10])[sonic_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Sonic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 2)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_sonic_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the sonic Mach number has been saved
print 'Plot of the kurtosis as a function of sonic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Alfvenic Mach number 

# Create a figure to display a plot of the kurtosis as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig6 = plt.figure()

# Create an axis for this figure
ax6 = fig6.add_subplot(111)

# Plot the kurtosis as a function of Alfvenic Mach number for each line of sight
plt.plot(alf_mach_sort, (kurt_arr[-1])[alf_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(alf_mach_sort, (kurt_arr[-2])[alf_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(alf_mach_sort, (kurt_arr[-3])[alf_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(alf_mach_sort, (kurt_arr[-4])[alf_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(alf_mach_sort, (kurt_arr[-5])[alf_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(alf_mach_sort, (kurt_arr[-6])[alf_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(alf_mach_sort, (kurt_arr[-7])[alf_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(alf_mach_sort, (kurt_arr[-8])[alf_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(alf_mach_sort, (kurt_arr[-9])[alf_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(alf_mach_sort, (kurt_arr[-10])[alf_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Alfvenic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_alf_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the kurtosis as a function of Alfvenic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Relative Angle - Low Magnetic Field

# Create a figure to display a plot of the kurtosis as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 0.1 
fig7 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax7 = fig7.add_subplot(111)

# Plot the kurtosis as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 0.1
plt.plot(rel_ang_arr, kurt_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(rel_ang_arr, kurt_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(rel_ang_arr, kurt_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(rel_ang_arr, kurt_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(rel_ang_arr, kurt_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(rel_ang_arr, kurt_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(rel_ang_arr, kurt_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(rel_ang_arr, kurt_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurt vs Angle LOS - Mean B b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box7 = ax7.get_position()
ax7.set_position([box7.x0, box7.y0, box7.width * 0.8, box7.height])

# Force the legend to appear on the plot
ax7.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_rel_ang_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the kurtosis as a function of relative angle saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# Kurtosis vs Relative Angle - High Magnetic Field

# Create a figure to display a plot of the kurtosis as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 1 
fig8 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax8 = fig8.add_subplot(111)

# Plot the kurtosis as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 1
plt.plot(rel_ang_arr, kurt_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(rel_ang_arr, kurt_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(rel_ang_arr, kurt_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(rel_ang_arr, kurt_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(rel_ang_arr, kurt_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(rel_ang_arr, kurt_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(rel_ang_arr, kurt_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(rel_ang_arr, kurt_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Add a title to the plot
plt.title('Kurtosis vs Angle LOS - Mean B b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box8 = ax8.get_position()
ax8.set_position([box8.x0, box8.y0, box8.width * 0.8, box8.height])

# Force the legend to appear on the plot
ax8.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Kurt_rel_ang_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the kurtosis as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the kurtosis as a function of relative angle saved b=1'

# Close the figure, now that it has been saved.
plt.close()

#-------------------------------- SF Slope - 1 ---------------------------------

# SF Slope - 1 vs sonic Mach number

# Create a figure to display a plot of the SF slope - 1 as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig9 = plt.figure()

# Create an axis for this figure
ax9 = fig9.add_subplot(111)

# Plot the SF slope -1 as a function of sonic Mach number for each line of sight
plt.plot(sonic_mach_sort, (m_arr[-1])[sonic_sort],'bo',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(sonic_mach_sort, (m_arr[-2])[sonic_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(sonic_mach_sort, (m_arr[-3])[sonic_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(sonic_mach_sort, (m_arr[-4])[sonic_sort],'ro',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(sonic_mach_sort, (m_arr[-5])[sonic_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(sonic_mach_sort, (m_arr[-6])[sonic_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(sonic_mach_sort, (m_arr[-7])[sonic_sort],'co',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(sonic_mach_sort, (m_arr[-8])[sonic_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(sonic_mach_sort, (m_arr[-9])[sonic_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(sonic_mach_sort, (m_arr[-10])[sonic_sort],'mo',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Sonic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 3)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_sonic_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the sonic Mach number has been saved
print 'Plot of the SF slope as a function of sonic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# SF Slope - 1 vs Alfvenic Mach number 

# Create a figure to display a plot of the SF Slope -1 as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig10 = plt.figure()

# Create an axis for this figure
ax10 = fig10.add_subplot(111)

# Plot the SF slope as a function of Alfvenic Mach number for each line of sight
plt.plot(alf_mach_sort, (m_arr[-1])[alf_sort],'bo',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(alf_mach_sort, (m_arr[-2])[alf_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(alf_mach_sort, (m_arr[-3])[alf_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(alf_mach_sort, (m_arr[-4])[alf_sort],'ro',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(alf_mach_sort, (m_arr[-5])[alf_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(alf_mach_sort, (m_arr[-6])[alf_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(alf_mach_sort, (m_arr[-7])[alf_sort],'co',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(alf_mach_sort, (m_arr[-8])[alf_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(alf_mach_sort, (m_arr[-9])[alf_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(alf_mach_sort, (m_arr[-10])[alf_sort],'mo',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Alfvenic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 4)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_alf_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the SF slope as a function of Alfvenic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# SF Slope - 1 vs Relative Angle - Low Magnetic Field

# Create a figure to display a plot of the SF slope as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 0.1 
fig11 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax11 = fig11.add_subplot(111)

# Plot the SF slope as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 0.1
plt.plot(rel_ang_arr, m_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(rel_ang_arr, m_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(rel_ang_arr, m_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(rel_ang_arr, m_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(rel_ang_arr, m_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(rel_ang_arr, m_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(rel_ang_arr, m_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(rel_ang_arr, m_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Angle LOS - Mean B b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box11 = ax11.get_position()
ax11.set_position([box11.x0, box11.y0, box11.width * 0.8, box11.height])

# Force the legend to appear on the plot
ax11.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_rel_ang_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the SF slope as a function of relative angle saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# SF Slope - 1 vs Relative Angle - High Magnetic Field

# Create a figure to display a plot of the SF slope as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 1 
fig12 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax12 = fig12.add_subplot(111)

# Plot the SF slope as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 1
plt.plot(rel_ang_arr, m_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(rel_ang_arr, m_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(rel_ang_arr, m_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(rel_ang_arr, m_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(rel_ang_arr, m_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(rel_ang_arr, m_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(rel_ang_arr, m_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(rel_ang_arr, m_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF Slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Angle LOS - Mean B b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box12 = ax12.get_position()
ax12.set_position([box12.x0, box12.y0, box12.width * 0.8, box12.height])

# Force the legend to appear on the plot
ax12.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_rel_ang_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the SF slope as a function of relative angle saved b=1'

# Close the figure, now that it has been saved.
plt.close()

#------------------------------ Residuals SF Fit -------------------------------

# Residuals vs sonic Mach number

# Create a figure to display a plot of the residuals as a function of sonic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig13 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax13 = fig13.add_subplot(111)

# Plot the residuals as a function of sonic Mach number for each line of sight
plt.plot(sonic_mach_sort, (residual_arr[-1])[sonic_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(sonic_mach_sort, (residual_arr[-2])[sonic_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(sonic_mach_sort, (residual_arr[-3])[sonic_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(sonic_mach_sort, (residual_arr[-4])[sonic_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(sonic_mach_sort, (residual_arr[-5])[sonic_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(sonic_mach_sort, (residual_arr[-6])[sonic_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(sonic_mach_sort, (residual_arr[-7])[sonic_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(sonic_mach_sort, (residual_arr[-8])[sonic_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(sonic_mach_sort, (residual_arr[-9])[sonic_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(sonic_mach_sort, (residual_arr[-10])[sonic_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Sonic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box13 = ax13.get_position()
ax13.set_position([box13.x0, box13.y0, box13.width * 0.8, box13.height])

# Force the legend to appear on the plot
ax13.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_sonic_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sonic Mach number has been saved
print 'Plot of the residuals as a function of sonic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Alfvenic Mach number 

# Create a figure to display a plot of the residuals as a function of Alfvenic
# Mach number for all of the synchrotron maps, i.e. for all lines of sight
fig14 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax14 = fig14.add_subplot(111)

# Plot the residuals as a function of Alfvenic Mach number for each line of sight
plt.plot(alf_mach_sort, (residual_arr[-1])[alf_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(alf_mach_sort, (residual_arr[-2])[alf_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(alf_mach_sort, (residual_arr[-3])[alf_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(alf_mach_sort, (residual_arr[-4])[alf_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(alf_mach_sort, (residual_arr[-5])[alf_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(alf_mach_sort, (residual_arr[-6])[alf_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(alf_mach_sort, (residual_arr[-7])[alf_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(alf_mach_sort, (residual_arr[-8])[alf_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(alf_mach_sort, (residual_arr[-9])[alf_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(alf_mach_sort, (residual_arr[-10])[alf_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Alfvenic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box14 = ax14.get_position()
ax14.set_position([box14.x0, box14.y0, box14.width * 0.8, box14.height])

# Force the legend to appear on the plot
ax14.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_alf_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the Alfvenic Mach number has been saved
print 'Plot of the residuals as a function of Alfvenic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Relative Angle - Low Magnetic Field

# Create a figure to display a plot of the residuals as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 0.1 
fig15 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax15 = fig15.add_subplot(111)

# Plot the residuals as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 0.1
plt.plot(rel_ang_arr, residual_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(rel_ang_arr, residual_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(rel_ang_arr, residual_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(rel_ang_arr, residual_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(rel_ang_arr, residual_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(rel_ang_arr, residual_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(rel_ang_arr, residual_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(rel_ang_arr, residual_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Angle LOS - Mean B b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box15 = ax15.get_position()
ax15.set_position([box15.x0, box15.y0, box15.width * 0.8, box15.height])

# Force the legend to appear on the plot
ax15.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residual_rel_ang_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the residuals as a function of relative angle saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Relative Angle - High Magnetic Field

# Create a figure to display a plot of the residuals as a function of the
# relative angle between the line of sight and the mean magnetic field, for 
# simulation with b = 1 
fig16 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax16 = fig16.add_subplot(111)

# Plot the residuals as a function of the relative angle between the line of 
# sight and the mean magnetic field for simulations with b = 1
plt.plot(rel_ang_arr, residual_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(rel_ang_arr, residual_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(rel_ang_arr, residual_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(rel_ang_arr, residual_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(rel_ang_arr, residual_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(rel_ang_arr, residual_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(rel_ang_arr, residual_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(rel_ang_arr, residual_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Angle LOS - Mean B b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box16 = ax16.get_position()
ax16.set_position([box16.x0, box16.y0, box16.width * 0.8, box16.height])

# Force the legend to appear on the plot
ax16.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Residuals_rel_ang_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the relative angle between the line of sight and the mean 
# magnetic field has been saved
print 'Plot of the residuals as a function of relative angle saved b=1'

# Close the figure, now that it has been saved.
plt.close()

#------------------------ Integrated magnitude quad ratio ----------------------

# Integrated magnitude of quadrupole / monopole ratio vs sonic Mach number

# Create a figure to display a plot of the integrated magnitude of the quad 
# ratio as a function of sonic Mach number for all of the synchrotron maps, i.e.
# for all lines of sight
fig17 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax17 = fig17.add_subplot(111)

# Plot the integrated magnitude of the quadrupole ratio as a function of sonic 
# Mach number for each line of sight
plt.plot(sonic_mach_sort, (int_quad_arr[-1])[sonic_sort],'bo',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(sonic_mach_sort, (int_quad_arr[-2])[sonic_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(sonic_mach_sort, (int_quad_arr[-3])[sonic_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(sonic_mach_sort, (int_quad_arr[-4])[sonic_sort],'ro',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(sonic_mach_sort, (int_quad_arr[-5])[sonic_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(sonic_mach_sort, (int_quad_arr[-6])[sonic_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(sonic_mach_sort, (int_quad_arr[-7])[sonic_sort],'co',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(sonic_mach_sort, (int_quad_arr[-8])[sonic_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(sonic_mach_sort, (int_quad_arr[-9])[sonic_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(sonic_mach_sort, (int_quad_arr[-10])[sonic_sort],'mo',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Sonic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box17 = ax17.get_position()
ax17.set_position([box17.x0, box17.y0, box17.width * 0.8, box17.height])

# Force the legend to appear on the plot
ax17.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_sonic_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated 
# quadrupole ratio as a function of the sonic Mach number has been saved
print 'Plot of the integrated quad ratio as a function of sonic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Integrated magnitude of quadrupole / monopole ratio vs Alfvenic Mach number 

# Create a figure to display a plot of the integrated magnitude of the 
# quadrupole / monopole ratio as a function of Alfvenic Mach number for all of 
# the synchrotron maps, i.e. for all lines of sight
fig18 = plt.figure()

# Create an axis for this figure
ax18 = fig18.add_subplot(111)

# Plot the integrated magnitude of the quadrupole / monopole ratio as a function
# of Alfvenic Mach number for each line of sight
plt.plot(alf_mach_sort, (int_quad_arr[-1])[alf_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(alf_mach_sort, (int_quad_arr[-2])[alf_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(alf_mach_sort, (int_quad_arr[-3])[alf_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(alf_mach_sort, (int_quad_arr[-4])[alf_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(alf_mach_sort, (int_quad_arr[-5])[alf_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(alf_mach_sort, (int_quad_arr[-6])[alf_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(alf_mach_sort, (int_quad_arr[-7])[alf_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(alf_mach_sort, (int_quad_arr[-8])[alf_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(alf_mach_sort, (int_quad_arr[-9])[alf_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(alf_mach_sort, (int_quad_arr[-10])[alf_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Alfvenic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_alf_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated 
# quadrupole ratio as a function of the Alfvenic Mach number has been saved
print 'Plot of the integrated quad ratio as a function of Alfvenic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Integrated Mag quadrupole / monopole ratio vs Relative Angle - Low Magnetic Field

# Create a figure to display a plot of the integrated magnitude of the 
# quadrupole / monopole ratio as a function of the relative angle between the 
# line of sight and the mean magnetic field, for simulation with b = 0.1 
fig19 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax19 = fig19.add_subplot(111)

# Plot the integrated magnitude of the quadrupole / monopole ratio as a function
# of the relative angle between the line of sight and the mean magnetic field 
# for simulations with b = 0.1
plt.plot(rel_ang_arr, int_quad_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(rel_ang_arr, int_quad_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(rel_ang_arr, int_quad_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(rel_ang_arr, int_quad_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(rel_ang_arr, int_quad_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(rel_ang_arr, int_quad_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(rel_ang_arr, int_quad_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(rel_ang_arr, int_quad_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Angle LOS - Mean B b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box19 = ax19.get_position()
ax19.set_position([box19.x0, box19.y0, box19.width * 0.8, box19.height])

# Force the legend to appear on the plot
ax19.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_rel_ang_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated 
# quadrupole/monopole ratio as a function of the relative angle between the
# line of sight and the mean magnetic field has been saved
print 'Plot of the integrated quad ratio as a function of relative angle saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# Integrated Mag quadrupole / monopole ratio vs Relative Angle - High Magnetic Field

# Create a figure to display a plot of the integrated magnitude of the 
# quadrupole/monopole ratio as a function of the relative angle between the line
# of sight and the mean magnetic field, for simulation with b = 1 
fig20 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax20 = fig20.add_subplot(111)

# Plot the integrated magnitude of the quadrupole / monopole ratio as a function
# of the relative angle between the line of sight and the mean magnetic field 
# for simulations with b = 1
plt.plot(rel_ang_arr, int_quad_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(rel_ang_arr, int_quad_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(rel_ang_arr, int_quad_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(rel_ang_arr, int_quad_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(rel_ang_arr, int_quad_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(rel_ang_arr, int_quad_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(rel_ang_arr, int_quad_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(rel_ang_arr, int_quad_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Integrated Mag quad/mono', fontsize = 20)

# Add a title to the plot
plt.title('Int Mag quad/mono vs Angle LOS - Mean B b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box20 = ax20.get_position()
ax20.set_position([box20.x0, box20.y0, box20.width * 0.8, box20.height])

# Force the legend to appear on the plot
ax20.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'int_quad_rel_ang_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the integrated 
# quadrupole/monopole ratio as a function of the relative angle between the
# line of sight and the mean magnetic field has been saved
print 'Plot of the integrated quad ratio as a function of relative angle saved b=1'

# Close the figure, now that it has been saved.
plt.close()

#-------------------------- Mag Quad ratio at a point --------------------------

# Magnitude of Quadrupole / Monopole ratio at a point vs sonic Mach number

# Create a figure to display a plot of the magnitude of the quadrupole/monopole
# ratio at a point as a function of sonic Mach number for all of the synchrotron
# maps, i.e. for all lines of sight
fig21 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax21 = fig21.add_subplot(111)

# Plot the magnitude of the quadrupole / monopole ratio at a point as a function
# of sonic Mach number for each line of sight
plt.plot(sonic_mach_sort, (quad_point_arr[-1])[sonic_sort],'bo',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(sonic_mach_sort, (quad_point_arr[-2])[sonic_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(sonic_mach_sort, (quad_point_arr[-3])[sonic_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(sonic_mach_sort, (quad_point_arr[-4])[sonic_sort],'ro',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(sonic_mach_sort, (quad_point_arr[-5])[sonic_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(sonic_mach_sort, (quad_point_arr[-6])[sonic_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(sonic_mach_sort, (quad_point_arr[-7])[sonic_sort],'co',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(sonic_mach_sort, (quad_point_arr[-8])[sonic_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(sonic_mach_sort, (quad_point_arr[-9])[sonic_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(sonic_mach_sort, (quad_point_arr[-10])[sonic_sort],'mo',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Sonic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Sonic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box21 = ax21.get_position()
ax21.set_position([box21.x0, box21.y0, box21.width * 0.8, box21.height])

# Force the legend to appear on the plot
ax21.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_sonic_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio at
# a single radius value as a function of the sonic Mach number has been saved
print 'Plot of quad/mono at a point as a function of sonic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of Quadrupole / Monopole ratio at a point vs Alfvenic Mach number 

# Create a figure to display a plot of the magnitude of the quadrupole/monopole
# ratio at a point as a function of Alfvenic Mach number for all of the 
# synchrotron maps, i.e. for all lines of sight
fig22 = plt.figure()

# Create an axis for this figure
ax22 = fig22.add_subplot(111)

# Plot the magnitude of the quadrupole / monopole ratio at a point as a function
# of Alfvenic Mach number for each line of sight
plt.plot(alf_mach_sort, (quad_point_arr[-1])[alf_sort],'b-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-1]))
# plt.plot(alf_mach_sort, (quad_point_arr[-2])[alf_sort],'b--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-2]))
# plt.plot(alf_mach_sort, (quad_point_arr[-3])[alf_sort],'r-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-3]))
plt.plot(alf_mach_sort, (quad_point_arr[-4])[alf_sort],'r-o',label= 'Angle = {}'\
	.format(rel_ang_arr[-4]))
# plt.plot(alf_mach_sort, (quad_point_arr[-5])[alf_sort],'g-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-5]))
# plt.plot(alf_mach_sort, (quad_point_arr[-6])[alf_sort],'g--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-6]))
plt.plot(alf_mach_sort, (quad_point_arr[-7])[alf_sort],'c-o',label = 'Angle = {}'\
	.format(rel_ang_arr[-7]))
# plt.plot(alf_mach_sort, (quad_point_arr[-8])[alf_sort],'c--o',label= 'Angle = {}'\
# 	.format(rel_ang_arr[-8]))
# plt.plot(alf_mach_sort, (quad_point_arr[-9])[alf_sort],'m-o',label = 'Angle = {}'\
# 	.format(rel_ang_arr[-9]))
plt.plot(alf_mach_sort, (quad_point_arr[-10])[alf_sort],'m-o',label='Angle = {}'\
	.format(rel_ang_arr[-10]))

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Alfvenic Mach Number Gam{}'.format(gamma), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_alf_mach_rot_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the quad/mono ratio at
# a single radius value as a function of the Alfvenic Mach number has been saved
print 'Plot of quad/mono at a point as a function of Alfvenic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of Quadrupole / Monopole ratio at a point vs Relative Angle - Low Magnetic Field

# Create a figure to display a plot of the magnitude of quadrupole / monopole 
# ratio at a point as a function of the relative angle between the line of sight
# and the mean magnetic field, for simulation with b = 0.1 
fig23 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax23 = fig23.add_subplot(111)

# Plot the magnitude of the quadrupole / monopole ratio at a point as a function
# of the relative angle between the line of sight and the mean magnetic field 
# for simulations with b = 0.1
plt.plot(rel_ang_arr, quad_point_arr[:,0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(rel_ang_arr, quad_point_arr[:,1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(rel_ang_arr, quad_point_arr[:,2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(rel_ang_arr, quad_point_arr[:,3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(rel_ang_arr, quad_point_arr[:,4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(rel_ang_arr, quad_point_arr[:,5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(rel_ang_arr, quad_point_arr[:,6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(rel_ang_arr, quad_point_arr[:,7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Angle LOS - Mean B b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box23 = ax23.get_position()
ax23.set_position([box23.x0, box23.y0, box23.width * 0.8, box23.height])

# Force the legend to appear on the plot
ax23.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_rel_ang_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the quadrupole/
# monopole ratio at a point as a function of the relative angle between the 
# line of sight and the mean magnetic field has been saved
print 'Plot of quad/mono at a point as a function of relative angle saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# Magnitude of Quadrupole / Monopole ratio at a point vs Relative Angle - High Magnetic Field

# Create a figure to display a plot of the magnitude of the quadrupole/monopole 
# ratio at a point as a function of the relative angle between the line of sight
# and the mean magnetic field, for simulation with b = 1 
fig24 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax24 = fig24.add_subplot(111)

# Plot the magnitude of the quadrupole / monopole ratio at a point as a function
# of the relative angle between the line of sight and the mean magnetic field 
# for simulations with b = 1
plt.plot(rel_ang_arr, quad_point_arr[:,8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(rel_ang_arr, quad_point_arr[:,9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(rel_ang_arr, quad_point_arr[:,10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(rel_ang_arr, quad_point_arr[:,11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(rel_ang_arr, quad_point_arr[:,12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(rel_ang_arr, quad_point_arr[:,13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(rel_ang_arr, quad_point_arr[:,14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(rel_ang_arr, quad_point_arr[:,15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Relative Orientation LOS - Mean B', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mag Quad/mono at R = {0:.2f}'.format(quad_rad\
	[np.floor(num_bins/3.0)]), fontsize = 20)

# Add a title to the plot
plt.title('Mag Quad/mono at point vs Angle LOS - Mean B b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box24 = ax24.get_position()
ax24.set_position([box24.x0, box24.y0, box24.width * 0.8, box24.height])

# Force the legend to appear on the plot
ax24.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_point_rel_ang_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the quadrupole / 
# monopole ratio at a point as a function of the relative angle between the
# line of sight and the mean magnetic field has been saved
print 'Plot of quad/mono at a point as a function of relative angle saved b=1'

# Close the figure, now that it has been saved.
plt.close()