#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the observed synchrotron emission maps for a #
# cube that is saturated with a uniform, isotropic distribution of cosmic rays #
# with power spectrum index gamma. These maps are generated for sub-cubes of   #
# full simulation cube, and then structure functions are calculated for each   #
# map. The slopes of these structure functions are plotted against the size of #
# the sub-cube, and for each simulation a plot of the structure function for   #
# different sub-cubes is produced. Plots are also produced for how the         #
# skewness, kurtosis, and integrated quadrupole ratio vary with subcube size.  #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 21/4/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.ndimage to handle rotation of data cubes.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage
from scipy import stats

# Import the function that calculates the structure function and the function 
# that calculates the radially averaged structure function.
from sf_fft import sf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 25

# Create a string for the directory that contains the simulated synchrotron
# intensity maps to use. 
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

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

# Create strings giving the simulation codes in terms of Mach numbers, for the
# low magnetic field simulations, then the high magnetic field simulations
short_M = ['Ms10.96Ma1.41', 'Ms9.16Ma1.77', 'Ms7.02Ma1.76','Ms4.32Ma1.51',\
'Ms3.11Ma1.69', 'Ms2.38Ma1.86', 'Ms0.83Ma1.74', 'Ms0.45Ma1.72', 'Ms9.92Ma0.5',\
'Ms7.89Ma0.5', 'Ms6.78Ma0.52', 'Ms4.46Ma0.55', 'Ms3.16Ma0.58', 'Ms2.41Ma0.67',\
'Ms0.87Ma0.7', 'Ms0.48Ma0.65']

# Create a list that holds the colours to use for each simulation
color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', (1,0.41,0)]

# Create a list that holds the marker type to use for each simulation
symbol_list = ['o', '^', 's', 'p', '*', '+', 'x', 'D']

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a variable that controls how many sub-cube sizes are being used
free_num = 41

# Create an array that specifies the different sub-cube sizes to use
# NOTE: The values chosen here are based on the known sizes of the simulation
# cubes
iter_array = np.linspace(312, 512, free_num)

# Create an empty array, where each entry specifies the calculated skewness
# of the synchrotron intensity image, of the corresponding simulation, for a
# particular value of the subcube size. Each row corresponds to a value of 
# the free parameter, and each column corresponds to a simulation. Create an 
# array for lines of sight along the y and z axes.
skew_y_arr = np.zeros((len(simul_arr),len(iter_array)))
skew_z_arr = np.zeros((len(simul_arr),len(iter_array)))

# Create an empty array, where each entry specifies the kurtosis of the 
# synchrotron intensity image, of the corresponding simulation, for a particular
# value of the subcube size. Each row corresponds to a value of the free 
# parameter, and each column corresponds to a simulation. Create an array for 
# lines of sight along the y and z axes.
kurt_y_arr = np.zeros((len(simul_arr),len(iter_array)))
kurt_z_arr = np.zeros((len(simul_arr),len(iter_array)))

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of the subcube size. Each row
# corresponds to a value of the free parameter, and each column corresponds to
# a simulation. Create an array for lines of sight along the y and z axes.
m_y_arr = np.zeros((len(simul_arr),len(iter_array)))
m_z_arr = np.zeros((len(simul_arr),len(iter_array)))

# Create an empty array, where each entry specifies the residuals of the linear
# fit to the structure function of the synchrotron intensity image, of the 
# corresponding simulation, for a particular value of the subcube size. Each row
# corresponds to a value of the free parameter, and each column corresponds to a
# simulation. Create an array for lines of sight along the y and z axes.
residual_y_arr = np.zeros((len(simul_arr),len(iter_array)))
residual_z_arr = np.zeros((len(simul_arr),len(iter_array)))

# Create an empty array, where each entry specifies the integrated quadrupole
# ratio modulus of the synchrotron intensity image, of the corresponding 
# simulation, for a particular value of the subcube size. Each row corresponds 
# to a value of the free parameter, and each column corresponds to a
# simulation. Create an array for lines of sight along the y and z axes.
int_quad_y_arr = np.zeros((len(simul_arr),len(iter_array)))
int_quad_z_arr = np.zeros((len(simul_arr),len(iter_array)))

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS file that contains the x-component of the simulated magnetic
	# field
	mag_x_fits = fits.open(data_loc + 'magx.fits')

	# Extract the data for the simulated x-component of the magnetic field
	mag_x_data = mag_x_fits[0].data

	# Open the FITS file that contains the y-component of the simulated magnetic 
	# field
	mag_y_fits = fits.open(data_loc + 'magy.fits')

	# Extract the data for the simulated y-component of the magnetic field
	mag_y_data = mag_y_fits[0].data

	# Open the FITS file that contains the z-component of the simulated magnetic 
	# field
	mag_z_fits = fits.open(data_loc + 'magz.fits')

	# Extract the data for the simulated z-component of the magnetic field
	mag_z_data = mag_z_fits[0].data

	# Calculate the magnitude of the magnetic field perpendicular to the line of
	# sight, which is just the square root of the sum of the x and y component
	# magnitudes squared.
	mag_perp_z = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the synchrotron maps. Since the line of sight
	# is the z axis, we need to integrate along axis 0. (Numpy convention is 
	# that axes are ordered as (z, y, x))
	int_axis_z = 0

	# Calculate the magnitude of the magnetic field perpendicular to the line of
	# sight, which is just the square root of the sum of the x and z component
	# magnitudes squared.
	mag_perp_y = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_z_data, 2.0) )

	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the synchrotron maps. Since the line of sight
	# is the y axis, we need to integrate along axis 1.
	int_axis_y = 1

	# Calculate the result of raising the perpendicular magnetic field strength
	# to the power of gamma, for these slices
	mag_perp_gamma_y = np.power(mag_perp_y, gamma)
	mag_perp_gamma_z = np.power(mag_perp_z, gamma)

	# Loop over the sub-cube sizes being studied, so that we can calculate the
	# synchrotron map for each sub-cube size
	for i in range(len(iter_array)):
		# Calculate the minimum index to include when extracting the sub-cube
		ind_min = int(256 - iter_array[i] / 2.0)

		# Calculate the maximum index to exclude when extracting the sub-cube
		ind_max = int(256 + iter_array[i] / 2.0)

		# Extract a sub-cube of the required size from the full cube
		sub_mag_perp_gamma_y = mag_perp_gamma_y[ind_min:ind_max,ind_min:ind_max,ind_min:ind_max]
		sub_mag_perp_gamma_z = mag_perp_gamma_z[ind_min:ind_max,ind_min:ind_max,ind_min:ind_max]

		# Integrate the perpendicular magnetic field strength raised to the power
		# of gamma along the required axis, to calculate the observed synchrotron 
		# map for these slices. This integration is performed by the trapezoidal 
		# rule. To normalise the calculated synchrotron map, divide by the number 
		# of pixels along the integration axis. Note the array is ordered by(z,y,x)!
		# NOTE: Set dx to whatever the pixel spacing is
		sync_arr_y = np.trapz(sub_mag_perp_gamma_y, dx = 1.0, axis = int_axis_y) /\
		 np.shape(sub_mag_perp_gamma_y)[int_axis_y]
		sync_arr_z = np.trapz(sub_mag_perp_gamma_z, dx = 1.0, axis = int_axis_z) /\
		 np.shape(sub_mag_perp_gamma_z)[int_axis_z]

		# Flatten the synchrotron intensity maps for this value of gamma, for
		# lines of sight along each of the axes
		flat_sync_y = sync_arr_y.flatten()
		flat_sync_z = sync_arr_z.flatten()

		# Calculate the biased skewness of the synchrotron intensity maps, for
		# lines of sight along each of the axes, and store the results in the
		# corresponding array.
		skew_y_arr[j,i] = stats.skew(flat_sync_y)
		skew_z_arr[j,i] = stats.skew(flat_sync_z)

		# Calculate the biased Fisher kurtosis of the synchrotron intensity 
		# maps, for lines of sight along each of the axes, and store the results
		# in the corresponding array.
		kurt_y_arr[j,i] = stats.kurtosis(flat_sync_y)
		kurt_z_arr[j,i] = stats.kurtosis(flat_sync_z)

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not 
		# subtracting the mean from the synchrotron map before calculating the 
		# structure function.
		strfn_y = sf_fft(sync_arr_y, no_fluct = True)
		strfn_z = sf_fft(sync_arr_z, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins.
		rad_sf_y = sfr(strfn_y, num_bins, verbose = False)
		rad_sf_z = sfr(strfn_z, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function
		sf_y = rad_sf_y[1]
		sf_z = rad_sf_z[1]

		# Extract the radius values used to calculate this structure function
		sf_rad_arr_y = rad_sf_y[0]
		sf_rad_arr_z = rad_sf_z[0]

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. 
		spec_ind_data_y = np.polyfit(np.log10(\
			sf_rad_arr_y[11:16]),\
			np.log10(sf_y[11:16]), 1, full = True)
		spec_ind_data_z = np.polyfit(np.log10(\
			sf_rad_arr_z[11:16]),\
			np.log10(sf_z[11:16]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit
		coeff_y = spec_ind_data_y[0]
		coeff_z = spec_ind_data_z[0]

		# Extract the sum of the residuals from the polynomial fit
		residual_y_arr[j,i] = spec_ind_data_y[1]
		residual_z_arr[j,i] = spec_ind_data_z[1]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array
		m_y_arr[j,i] = coeff_y[0]-1.0
		m_z_arr[j,i] = coeff_z[0]-1.0

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn_y = sf_fft(sync_arr_y, no_fluct = True, normalise = True)
		norm_strfn_z = sf_fft(sync_arr_z, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image. This is done for lines of sight
		# along each of the axes
		norm_strfn_y = np.fft.fftshift(norm_strfn_y)
		norm_strfn_z = np.fft.fftshift(norm_strfn_z)

		# Calculate the magnitude and argument of the quadrupole ratio, for 
		# lines of sight along each of the axes.
		quad_mod_y, quad_arg_y, quad_rad_y = calc_quad_ratio(norm_strfn_y, num_bins)
		quad_mod_z, quad_arg_z, quad_rad_z = calc_quad_ratio(norm_strfn_z, num_bins)

		# Integrate the magnitude of the quadrupole/monopole ratio from one 
		# sixth of the way along the radial separation bins, until three 
		# quarters of the way along the radial separation bins. This integration
		# is performed with respect to log separation (i.e. I am ignoring the 
		# fact that the points are equally separated in log space, to calculate 
		# the area under the quadrupole / monopole ratio plot when the x axis is
		# scaled logarithmically). I normalise the value that is returned by 
		# dividing by the number of increments in log radial separation used in 
		# the calculation. This is done for lines of sight along each of the axes.
		int_quad_y_arr[j,i] = np.trapz(quad_mod_y[11:20], dx = 1.0) / (19 - 11)
		int_quad_z_arr[j,i] = np.trapz(quad_mod_z[11:20], dx = 1.0) / (19 - 11)

	# Close all of the FITS files, to save memory
	mag_x_fits.close()
	mag_y_fits.close()
	mag_z_fits.close()

# When the code reaches this point, structure functions have been calculated
# for all simulations, and for all sub-cube sizes

# Create mean value arrays for each of the statistics. These values are only for
# the statistics calculated from the y and z axes (perpendicular to the mean 
# magnetic field).
skew_mean_arr = (skew_y_arr + skew_z_arr) / 2.0
kurt_mean_arr = (kurt_y_arr + kurt_z_arr) / 2.0
m_mean_arr = (m_y_arr + m_z_arr) / 2.0
residual_mean_arr = (residual_y_arr + residual_z_arr) / 2.0
int_quad_mean_arr = (int_quad_y_arr + int_quad_z_arr) / 2.0

#--------------------------------- Skewness ------------------------------------

# Plot skewness vs sub-cube size for low B

# Create a figure to display a plot of the skewness as a function of the
# sub-cube size, for simulations with b = 0.1 
fig = plt.figure(1, figsize = (9,6), dpi=300)

# Create an axis for this figure
ax1 = fig.add_subplot(221)

# Plot the skewness as a function of the sub-cube size for simulations
# with b = 0.1
plt.plot(iter_array, skew_mean_arr[0],color_list[0]+symbol_list[0],label = '{}'.format(short_M[0]),ms=5)
plt.plot(iter_array, skew_mean_arr[1],color_list[1]+symbol_list[1],label = '{}'.format(short_M[1]),ms=5)
plt.plot(iter_array, skew_mean_arr[2],color_list[2]+symbol_list[2],label = '{}'.format(short_M[2]),ms=5)
plt.plot(iter_array, skew_mean_arr[3],color_list[3]+symbol_list[3],label = '{}'.format(short_M[3]),ms=5)
plt.plot(iter_array, skew_mean_arr[4],color_list[4]+symbol_list[4],label = '{}'.format(short_M[4]),ms=5)
plt.plot(iter_array, skew_mean_arr[5],color_list[5]+symbol_list[5],label = '{}'.format(short_M[5]),ms=5)
plt.plot(iter_array, skew_mean_arr[6],color_list[6]+symbol_list[6],label = '{}'.format(short_M[6]),ms=5)
plt.plot(iter_array, skew_mean_arr[7],color=color_list[7],marker=symbol_list[7],label = '{}'.format(short_M[7]),ms=5,ls='None')

# Set the x axis limits to the minimum and maximum sub-cube sizes
ax1.set_xlim([np.min(iter_array), np.max(iter_array)])

# Add a label to the y-axis
plt.ylabel('Skewness', fontsize = 20)

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the 
# skewness of high magnetic field simulations. Make the y axis limits the same 
# as for the low magnetic field plot
ax2 = fig.add_subplot(222, sharey = ax1)

# Plot the skewness as a function of the sub-cube size for simulations
# with b = 1
plt.plot(iter_array, skew_mean_arr[8] ,color_list[0]+symbol_list[0],label = '{}'.format(short_M[8]) ,ms=5)
plt.plot(iter_array, skew_mean_arr[9] ,color_list[1]+symbol_list[1],label = '{}'.format(short_M[9]) ,ms=5)
plt.plot(iter_array, skew_mean_arr[10],color_list[2]+symbol_list[2],label = '{}'.format(short_M[10]),ms=5)
plt.plot(iter_array, skew_mean_arr[11],color_list[3]+symbol_list[3],label = '{}'.format(short_M[11]),ms=5)
plt.plot(iter_array, skew_mean_arr[12],color_list[4]+symbol_list[4],label = '{}'.format(short_M[12]),ms=5)
plt.plot(iter_array, skew_mean_arr[13],color_list[5]+symbol_list[5],label = '{}'.format(short_M[13]),ms=5)
plt.plot(iter_array, skew_mean_arr[14],color_list[6]+symbol_list[6],label = '{}'.format(short_M[14]),ms=5)
plt.plot(iter_array, skew_mean_arr[15],color=color_list[7],marker=symbol_list[7],label = '{}'.format(short_M[15]),ms=5,ls='None')

# Set the x axis limits to the minimum and maximum sub-cube sizes
ax2.set_xlim([np.min(iter_array), np.max(iter_array)])

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the kurtosis 
# of low magnetic field simulations. Make the x axis limits the same as
# for the first plot
ax3 = fig.add_subplot(223, sharex = ax1)

# Plot the kurtosis as a function of the sub-cube size for simulations
# with b = 0.1
plt.plot(iter_array, kurt_mean_arr[0],color_list[0]+symbol_list[0],label = '{}'.format(short_M[0]),ms=5)
plt.plot(iter_array, kurt_mean_arr[1],color_list[1]+symbol_list[1],label = '{}'.format(short_M[1]),ms=5)
plt.plot(iter_array, kurt_mean_arr[2],color_list[2]+symbol_list[2],label = '{}'.format(short_M[2]),ms=5)
plt.plot(iter_array, kurt_mean_arr[3],color_list[3]+symbol_list[3],label = '{}'.format(short_M[3]),ms=5)
plt.plot(iter_array, kurt_mean_arr[4],color_list[4]+symbol_list[4],label = '{}'.format(short_M[4]),ms=5)
plt.plot(iter_array, kurt_mean_arr[5],color_list[5]+symbol_list[5],label = '{}'.format(short_M[5]),ms=5)
plt.plot(iter_array, kurt_mean_arr[6],color_list[6]+symbol_list[6],label = '{}'.format(short_M[6]),ms=5)
plt.plot(iter_array, kurt_mean_arr[7],color=color_list[7],marker=symbol_list[7],label= '{}'.format(short_M[7]),ms=5,ls='None')

# Force the legends to appear on the plot
plt.legend(loc = 1, fontsize = 7, numpoints=1)

# Add a label to the y-axis
plt.ylabel('Kurtosis', fontsize = 20)

# Create an axis for the fourth subplot to be produced, which is for the 
# kurtosis of high magnetic field simulations. Make the x axis limits the same 
# as for the second plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Plot the kurtosis as a function of the sub-cube size for simulations
# with b = 1
plt.plot(iter_array, kurt_mean_arr[8] ,color_list[0]+symbol_list[0],label = '{}'.format(short_M[8]) ,ms=5)
plt.plot(iter_array, kurt_mean_arr[9] ,color_list[1]+symbol_list[1],label = '{}'.format(short_M[9]) ,ms=5)
plt.plot(iter_array, kurt_mean_arr[10],color_list[2]+symbol_list[2],label = '{}'.format(short_M[10]),ms=5)
plt.plot(iter_array, kurt_mean_arr[11],color_list[3]+symbol_list[3],label = '{}'.format(short_M[11]),ms=5)
plt.plot(iter_array, kurt_mean_arr[12],color_list[4]+symbol_list[4],label = '{}'.format(short_M[12]),ms=5)
plt.plot(iter_array, kurt_mean_arr[13],color_list[5]+symbol_list[5],label = '{}'.format(short_M[13]),ms=5)
plt.plot(iter_array, kurt_mean_arr[14],color_list[6]+symbol_list[6],label = '{}'.format(short_M[14]),ms=5)
plt.plot(iter_array, kurt_mean_arr[15],color=color_list[7],marker=symbol_list[7],label = '{}'.format(short_M[15]),ms=5,ls='None')

# Force the legends to appear on the plot
plt.legend(loc = 1, fontsize = 7, numpoints=1)

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Sub-cube Size [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) Skew, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) Skew, high B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.19, 0.475, 'c) Kurt, low B', fontsize = 18)

# Add some text to the figure, to label the right plot as figure d
plt.figtext(0.61, 0.475, 'd) Kurt, high B', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig_subcube1.eps', format = 'eps')

# Close the figure, now that it has been saved.
plt.close()

#------------------------- SF Slope and quadrupole ratio -----------------------

# Plot structure function slopes vs sub-cube size for low B

# Create a figure to display a plot of the SF slope - 1 as a function of the
# sub-cube size, for simulations with b = 0.1 
fig = plt.figure(1, figsize = (9,6),dpi=300)

# Create an axis for the first subplot to be produced, which is for the 
# structure function slope of low magnetic field simulations
ax1 = fig.add_subplot(221)

# Plot the SF slope - 1 as a function of the sub-cube size for simulations
# with b = 0.1
plt.plot(iter_array, m_mean_arr[0],color_list[0]+symbol_list[0],label = '{}'.format(short_M[0]),ms=5)
plt.plot(iter_array, m_mean_arr[1],color_list[1]+symbol_list[1],label = '{}'.format(short_M[1]),ms=5)
plt.plot(iter_array, m_mean_arr[2],color_list[2]+symbol_list[2],label = '{}'.format(short_M[2]),ms=5)
plt.plot(iter_array, m_mean_arr[3],color_list[3]+symbol_list[3],label = '{}'.format(short_M[3]),ms=5)
plt.plot(iter_array, m_mean_arr[4],color_list[4]+symbol_list[4],label = '{}'.format(short_M[4]),ms=5)
plt.plot(iter_array, m_mean_arr[5],color_list[5]+symbol_list[5],label = '{}'.format(short_M[5]),ms=5)
plt.plot(iter_array, m_mean_arr[6],color_list[6]+symbol_list[6],label = '{}'.format(short_M[6]),ms=5)
plt.plot(iter_array, m_mean_arr[7],color=color_list[7],marker=symbol_list[7],label = '{}'.format(short_M[7]),ms=5,ls='None')

# Set the x axis limits to the minimum and maximum sub-cube sizes
ax1.set_xlim([np.min(iter_array), np.max(iter_array)])

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Add a label to the y-axis
plt.ylabel('m', fontsize = 20)

# Create an axis for the second subplot to be produced, which is for the 
# structure function slope of high magnetic field simulations.
ax2 = fig.add_subplot(222, sharey = ax1)

# Plot the SF slope - 1 as a function of the sub-cube size for simulations
# with b = 1
plt.plot(iter_array, m_mean_arr[8] ,color_list[0]+symbol_list[0],label = '{}'.format(short_M[8]) ,ms=5)
plt.plot(iter_array, m_mean_arr[9] ,color_list[1]+symbol_list[1],label = '{}'.format(short_M[9]) ,ms=5)
plt.plot(iter_array, m_mean_arr[10],color_list[2]+symbol_list[2],label = '{}'.format(short_M[10]),ms=5)
plt.plot(iter_array, m_mean_arr[11],color_list[3]+symbol_list[3],label = '{}'.format(short_M[11]),ms=5)
plt.plot(iter_array, m_mean_arr[12],color_list[4]+symbol_list[4],label = '{}'.format(short_M[12]),ms=5)
plt.plot(iter_array, m_mean_arr[13],color_list[5]+symbol_list[5],label = '{}'.format(short_M[13]),ms=5)
plt.plot(iter_array, m_mean_arr[14],color_list[6]+symbol_list[6],label = '{}'.format(short_M[14]),ms=5)
plt.plot(iter_array, m_mean_arr[15],color=color_list[7],marker=symbol_list[7],label = '{}'.format(short_M[15]),ms=5,ls='None')

# Set the x axis limits to the minimum and maximum sub-cube sizes
ax2.set_xlim([np.min(iter_array), np.max(iter_array)])

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc = 4, fontsize = 7, numpoints=1)

# Create an axis for the third subplot to be produced, which is for the 
# integrated quadrupole ratio of low magnetic field simulations
ax3 = fig.add_subplot(223, sharex = ax1)

# Plot the integrated quadrupole ratios as a function of the sub-cube size for
# simulations with b = 0.1
plt.plot(iter_array, int_quad_mean_arr[0],color_list[0]+symbol_list[0],label = '{}'.format(short_M[0]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[1],color_list[1]+symbol_list[1],label = '{}'.format(short_M[1]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[2],color_list[2]+symbol_list[2],label = '{}'.format(short_M[2]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[3],color_list[3]+symbol_list[3],label = '{}'.format(short_M[3]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[4],color_list[4]+symbol_list[4],label = '{}'.format(short_M[4]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[5],color_list[5]+symbol_list[5],label = '{}'.format(short_M[5]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[6],color_list[6]+symbol_list[6],label = '{}'.format(short_M[6]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[7],color=color_list[7],marker=symbol_list[7],label= '{}'.format(short_M[7]),ms=5,ls='None')

# Add a label to the y-axis
plt.ylabel('Int Quad Ratio', fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 1, fontsize = 7, numpoints=1)

# Set the x axis limits for the plot
ax3.set_xlim([np.min(iter_array), np.max(iter_array)])

# Create an axis for the fourth subplot to be produced, which is for the 
# integrated quadrupole ratio of high magnetic field simulations. Make the y axis
# limits the same as for the low magnetic field plot
ax4 = fig.add_subplot(224, sharex = ax2, sharey = ax3)

# Plot the integrated quadrupole ratio as a function of the sub-cube size for 
# simulations with b = 1
plt.plot(iter_array, int_quad_mean_arr[8] ,color_list[0]+symbol_list[0],label = '{}'.format(short_M[8]) ,ms=5)
plt.plot(iter_array, int_quad_mean_arr[9] ,color_list[1]+symbol_list[1],label = '{}'.format(short_M[9]) ,ms=5)
plt.plot(iter_array, int_quad_mean_arr[10],color_list[2]+symbol_list[2],label = '{}'.format(short_M[10]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[11],color_list[3]+symbol_list[3],label = '{}'.format(short_M[11]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[12],color_list[4]+symbol_list[4],label = '{}'.format(short_M[12]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[13],color_list[5]+symbol_list[5],label = '{}'.format(short_M[13]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[14],color_list[6]+symbol_list[6],label = '{}'.format(short_M[14]),ms=5)
plt.plot(iter_array, int_quad_mean_arr[15],color=color_list[7],marker=symbol_list[7],label = '{}'.format(short_M[15]),ms=5,ls='None')

# Set the x axis limits to the minimum and maximum sub-cube sizes
ax4.set_xlim([np.min(iter_array), np.max(iter_array)])

# Make the y axis tick labels invisible
plt.setp( ax4.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Sub-cube Size [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.95, 'a) m, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.95, 'b) m, high B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.475, 'c) Quad, low B', fontsize = 18)

# Add some text to the figure, to label the left plot as figure b
plt.figtext(0.61, 0.475, 'd) Quad, high B', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig_subcube2.eps', format = 'eps')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the sub-cube size has been saved
print 'Plot of the SF slope - 1 as a function of sub-cube size saved'

# Close the figure, now that it has been saved.
plt.close()

#---------------------------- Residuals ----------------------------------------

# # Plot residuals vs sub-cube size for low B

# # Create a figure to display a plot of the residuals as a function of the
# # sub-cube size, for simulations with b = 0.1 
# fig7 = plt.figure(figsize = (10,6))

# # Create an axis for this figure
# ax7 = fig7.add_subplot(111)

# # Plot the residuals as a function of the sub-cube size for simulations
# # with b = 0.1
# plt.plot(iter_array, residual_mean_arr[0],'b-o',label = '{}'.format(short_simul[0]))
# plt.plot(iter_array, residual_mean_arr[1],'b--o',label= '{}'.format(short_simul[1]))
# plt.plot(iter_array, residual_mean_arr[2],'r-o',label = '{}'.format(short_simul[2]))
# plt.plot(iter_array, residual_mean_arr[3],'r--o',label= '{}'.format(short_simul[3]))
# plt.plot(iter_array, residual_mean_arr[4],'g-o',label = '{}'.format(short_simul[4]))
# plt.plot(iter_array, residual_mean_arr[5],'g--o',label= '{}'.format(short_simul[5]))
# plt.plot(iter_array, residual_mean_arr[6],'c-o',label = '{}'.format(short_simul[6]))
# plt.plot(iter_array, residual_mean_arr[7],'c--o',label= '{}'.format(short_simul[7]))

# # Add a label to the x-axis
# plt.xlabel('Sub-cube Size [pixels]', fontsize = 20)

# # Add a label to the y-axis
# plt.ylabel('Residuals', fontsize = 20)

# # Add a title to the plot
# plt.title('Residuals vs Sub-cube Size b.1 Gam{}'.format(gamma), fontsize = 20)

# # Shrink the width of the plot axes
# box7 = ax7.get_position()
# ax7.set_position([box7.x0, box7.y0, box7.width * 0.8, box7.height])

# # Force the legend to appear on the plot
# ax7.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # Save the figure using the given filename and format
# plt.savefig(save_loc + 'resid_subcube_b.1_gam{}.png'.format(gamma), format = 'png')

# # Print a message to the screen to show that the plot of the residuals as a 
# # function of the sub-cube size has been saved
# print 'Plot of the residuals as a function of sub-cube size saved b=0.1'

# # Close the figure, now that it has been saved.
# plt.close()

# # Residuals vs Sub-Cube Size - High Magnetic Field

# # Create a figure to display a plot of the residuals as a function of the
# # sub-cube size, for simulations with b = 1 
# fig8 = plt.figure(figsize = (10,6))

# # Create an axis for this figure
# ax8 = fig8.add_subplot(111)

# # Plot the residuals as a function of the sub-cube size for simulations
# # with b = 1
# plt.plot(iter_array, residual_mean_arr[8],'b-o',label = '{}'.format(short_simul[8]))
# plt.plot(iter_array, residual_mean_arr[9],'b--o',label= '{}'.format(short_simul[9]))
# plt.plot(iter_array, residual_mean_arr[10],'r-o',label = '{}'.format(short_simul[10]))
# plt.plot(iter_array, residual_mean_arr[11],'r--o',label= '{}'.format(short_simul[11]))
# plt.plot(iter_array, residual_mean_arr[12],'g-o',label = '{}'.format(short_simul[12]))
# plt.plot(iter_array, residual_mean_arr[13],'g--o',label= '{}'.format(short_simul[13]))
# plt.plot(iter_array, residual_mean_arr[14],'c-o',label = '{}'.format(short_simul[14]))
# plt.plot(iter_array, residual_mean_arr[15],'c--o',label= '{}'.format(short_simul[15]))

# # Add a label to the x-axis
# plt.xlabel('Sub-cube Size [pixels]', fontsize = 20)

# # Add a label to the y-axis
# plt.ylabel('Residuals', fontsize = 20)

# # Add a title to the plot
# plt.title('Residuals vs Sub-cube Size b1 Gam{}'.format(gamma), fontsize = 20)

# # Shrink the width of the plot axes
# box8 = ax8.get_position()
# ax8.set_position([box8.x0, box8.y0, box8.width * 0.8, box8.height])

# # Force the legend to appear on the plot
# ax8.legend(loc='upper left', bbox_to_anchor=(1, 1))

# # Save the figure using the given filename and format
# plt.savefig(save_loc + 'resid_subcube_b1_gam{}.png'.format(gamma), format = 'png')

# # Print a message to the screen to show that the plot of the residuals as a 
# # function of the sub-cube size has been saved
# print 'Plot of the residuals as a function of sub-cube saved b=1'

# # Close the figure, now that it has been saved.
# plt.close()

#----------------------------- Quadrupole Ratio --------------------------------

# # Plot integrated quadrupole ratios vs sub-cube size for low B

# # Create a figure to display a plot of the integrated quadrupole ratio as a 
# # function of the sub-cube size, for simulations with b = 0.1 
# fig = plt.figure(1, figsize = (9,5), dpi=300)

# # Create an axis for the first subplot to be produced, which is for the 
# # integrated quadrupole ratio of low magnetic field simulations
# ax1 = fig.add_subplot(121)

# # Plot the integrated quadrupole ratios as a function of the sub-cube size for
# # simulations with b = 0.1
# plt.plot(iter_array, int_quad_mean_arr[0],color_list[0]+symbol_list[0],label = '{}'.format(short_M[0]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[1],color_list[1]+symbol_list[1],label = '{}'.format(short_M[1]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[2],color_list[2]+symbol_list[2],label = '{}'.format(short_M[2]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[3],color_list[3]+symbol_list[3],label = '{}'.format(short_M[3]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[4],color_list[4]+symbol_list[4],label = '{}'.format(short_M[4]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[5],color_list[5]+symbol_list[5],label = '{}'.format(short_M[5]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[6],color_list[6]+symbol_list[6],label = '{}'.format(short_M[6]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[7],color=color_list[7],marker=symbol_list[7],label= '{}'.format(short_M[7]),ms=5,ls='None')

# # Set the x axis limits to the minimum and maximum sub-cube sizes
# ax1.set_xlim([np.min(iter_array), np.max(iter_array)])

# # Force the legend to appear on the plot
# plt.legend(loc = 1, fontsize = 9, numpoints=1)

# # Add a label to the y-axis
# plt.ylabel('Int Quad Ratio', fontsize = 20)

# # Create an axis for the second subplot to be produced, which is for the 
# # integrated quadrupole ratio of high magnetic field simulations. Make the y axis
# # limits the same as for the low magnetic field plot
# ax2 = fig.add_subplot(122, sharey = ax1)

# # Plot the integrated quadrupole ratio as a function of the sub-cube size for 
# # simulations with b = 1
# plt.plot(iter_array, int_quad_mean_arr[8] ,color_list[0]+symbol_list[0],label = '{}'.format(short_M[8]) ,ms=5)
# plt.plot(iter_array, int_quad_mean_arr[9] ,color_list[1]+symbol_list[1],label = '{}'.format(short_M[9]) ,ms=5)
# plt.plot(iter_array, int_quad_mean_arr[10],color_list[2]+symbol_list[2],label = '{}'.format(short_M[10]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[11],color_list[3]+symbol_list[3],label = '{}'.format(short_M[11]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[12],color_list[4]+symbol_list[4],label = '{}'.format(short_M[12]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[13],color_list[5]+symbol_list[5],label = '{}'.format(short_M[13]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[14],color_list[6]+symbol_list[6],label = '{}'.format(short_M[14]),ms=5)
# plt.plot(iter_array, int_quad_mean_arr[15],color=color_list[7],marker=symbol_list[7],label = '{}'.format(short_M[15]),ms=5,ls='None')

# # Set the x axis limits to the minimum and maximum sub-cube sizes
# ax2.set_xlim([np.min(iter_array), np.max(iter_array)])

# # Make the y axis tick labels invisible
# plt.setp( ax2.get_yticklabels(), visible=False)

# # Add a label to the x-axis
# plt.figtext(0.5, 0.0, 'Sub-cube Size [pixels]', ha = 'center', \
# 	va = 'bottom', fontsize = 20)

# # Force the legend to appear on the plot
# plt.legend(loc = 4, fontsize = 9, numpoints=1)

# # Add some text to the figure, to label the left plot as figure a
# plt.figtext(0.15, 0.95, 'a) Quad Ratio, low B', fontsize = 18)

# # Add some text to the figure, to label the left plot as figure b
# plt.figtext(0.58, 0.95, 'b) Quad Ratio, high B', fontsize = 18)

# # Save the figure using the given filename and format
# plt.savefig(simul_loc + 'Publication_Plots/fig_subcube3.eps', format = 'eps')

# # Print a message to the screen to show that the plot of the integrated 
# # quadrupole ratio as a function of the sub-cube size has been saved
# print 'Plot of the integrated quadrupole ratio as a function of sub-cube saved b=1'

# # Close the figure, now that it has been saved.
# plt.close()