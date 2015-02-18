#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the radially averaged structure function of the  #
# synchrotron intensity maps. These maps have been influenced by observational #
# effects, such as noise and angular resolution. The code then produces plots  #
# of these structure functions for each simulation, where each plot contains   #
# multiple plots representing the structure function obtained when the         #
# observational effect has a different strength.                               #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 15/1/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the function that calculates the structure function and the function 
# that calculates the radially averaged structure function.
from sf_fft import sf_fft
from sfr import sfr

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 25

# Create a string for the directory that contains the simulated synchrotron
# intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/Ultimate_Output/'

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

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a string that determines what observational effect will be studied
# String can be one of the following:
# spec - Study how statistics change as spectral resolution is varied
# noise - Study how statistics change as noise level is varied
# res - Study how statistics change as the spatial resolution is varied
obs_effect = 'res'

# Create a variable that controls how many data points are being used for the
# free parameter
free_num = 20

# Create a variable that controls how many sub-channels are used if we are 
# studying the effect of spectral resolution.
sub_channels = 20

# Depending on what observational effect is being studied, create an array of 
# values over which we will iterate. This array represents the values of the 
# free parameter related to the observational effect 
if obs_effect == 'spec':
	# Create an array of spectral channel width values (in MHz), over which
	# to iterate. Each value represents the width of a spectral channel that 
	# is centred on 1.4 GHz.
	iter_array = np.linspace(0.1, 10, free_num)

	# Create a string that will specify which folder the synchrotron maps to
	# use are in
	map_direc = 'Spec_Res_Gam2_Correct/'

	# Create a string to be used in the titles of any plots that are made 
	# against spectral channel width
	title_string = 'Spectral Resolution'

	# Create a string to be used in legends involving spectral channel width
	leg_string = 'SpecRes = ' 
elif obs_effect == 'noise':
	# Create an array of values that will be used to determine the standard
	# deviation of the Gaussian distribution from which noise values are 
	# generated. The standard deviation will be calculated by multiplying the
	# median synchrotron intensity by the values in this array.
	iter_array = np.linspace(0.02, 0.5, free_num)

	# Create a string that will specify which folder the synchrotron maps to
	# use are in
	map_direc = 'Noise_Gam2_Correct/'

	# Create a string to be used in the titles of any plots that are made 
	# against noise standard deviation
	title_string = 'Noise StandDev'

	# Create a string to be used in legends involving spectral channel width
	leg_string = 'Noise = ' 
elif obs_effect == 'res':
	# Create an array of values that represent the standard deviation of the 
	# Gaussian used to smooth the synchrotron maps. All values are in pixels.
	iter_array = np.linspace(1.0, 50.0, free_num)

	# Create a string that will specify which folder the synchrotron maps to
	# use are in
	map_direc = 'Ang_Res_Gam2_Astropy/'

	# Create a string to be used in the titles of any plots that are made 
	# against angular resolution
	title_string = 'Angular Resolution'

	# Create a string to be used in legends involving angular resolution
	leg_string = 'AngRes = ' 

# Create a new string representing the directory in which all plots should
# be saved, and where all synchrotron maps are stored
data_loc = simul_loc + map_direc

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(short_simul)):
	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(short_simul[j])

	# Open the FITS files that contain the simulated synchrotron intensity maps
	# for lines of sight along the z axis and x axis
	sync_fits_z = fits.open(data_loc + short_simul[j] + '_' + obs_effect + '_z.fits')
	sync_fits_x = fits.open(data_loc + short_simul[j] + '_' + obs_effect + '_x.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensity maps for different strengths of the observational
	# effect.
	sync_data_z = sync_fits_z[0].data
	sync_data_x = sync_fits_x[0].data

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Create an empty array, where each entry specifies the calculated value
	# of the structure function of the synchrotron intensity image, 
	# of the corresponding simulation, for a particular value of the free 
	# parameter related to the observational effect being studied. Each row 
	# corresponds to a value of the free parameter, and each column 
	# corresponds to a radial value. There is one array for a line of sight
	# along the z axis, and another for a line of sight along the x axis.
	sf_z_arr = np.zeros((len(iter_array),num_bins))
	sf_x_arr = np.zeros((len(iter_array),num_bins))

	# Create an empty array, where each entry specifies the radius values
	# used to calculate the structure function of the synchrotron intensity 
	# image, of the corresponding simulation, for a particular value of the 
	# free parameter related to the observational effect being studied. Each
	# row corresponds to a value of the free parameter, and each column 
	# corresponds to a radial value. There is one array for a line of sight 
	# along the z axis, and another for a line of sight along the x axis.
	rad_z_arr = np.zeros((len(iter_array),num_bins))
	rad_x_arr = np.zeros((len(iter_array),num_bins))

	# Loop over the various values of the free parameter related to the 
	# observational effect being studied, to calculate the structure function
	# for the synchrotron map observed for each value of the free parameter
	for i in range(len(iter_array)):
		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity maps, for the lines of sight along the x and z axes. Note 
		# that no_fluct = True is set, because we are not subtracting the mean
		# from the synchrotron maps before calculating the structure function.
		strfn_z = sf_fft(sync_data_z[i], no_fluct = True)
		strfn_x = sf_fft(sync_data_x[i], no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins, for lines of sight along the x and z axes.
		rad_sf_z = sfr(strfn_z, num_bins, verbose = False)
		rad_sf_x = sfr(strfn_x, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function for lines
		# of sight along the x and z axes.
		sf_z = rad_sf_z[1]
		sf_x = rad_sf_x[1]

		# Extract the radius values used to calculate this structure function,
		# for lines of sight along the x and z axes.
		sf_rad_arr_z = rad_sf_z[0]
		sf_rad_arr_x = rad_sf_x[0]

		# Store the values for the radially averaged structure function in the 
		# corresponding array, for each line of sight
		sf_z_arr[i] = sf_z
		sf_x_arr[i] = sf_x

		# Store the radius values used to calculate the structure function in
		# the corresponding array
		rad_z_arr[i] = sf_rad_arr_z
		rad_x_arr[i] = sf_rad_arr_x

		# When the code reaches this part, the structure function has been 
		# successfully calculated, and the code then goes on to calculate the 
		# structure function for the next slice of the data

	# When the code reaches this part, structure functions have been calculated
	# for all slices of the data cube, for the current simulation.
	# Print a message to the screen to inform the user that the structure 
	# functions have been calculated
	print 'All structure functions calculated for this simulation'

	# Create a figure to display a plot of the structure functions for different
	# values of the observational parameter, for a line of sight along the z
	# axis
	fig1 = plt.figure()

	# Create an axis for this figure
	ax1 = fig1.add_subplot(111)

	# Plot the structure function for various values of the observational effect
	# being studied.
	plt.plot(rad_z_arr[0], sf_z_arr[0],'b-o',label = leg_string\
		+'{}'.format(iter_array[0]))
	plt.plot(rad_z_arr[4], sf_z_arr[4],'r-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[4]))
	plt.plot(rad_z_arr[8], sf_z_arr[8],'g-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[8]))
	plt.plot(rad_z_arr[12], sf_z_arr[12],'k-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[12]))
	plt.plot(rad_z_arr[16], sf_z_arr[16],'m-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[16]))

	# Set the x-axis of the plot to be logarithmically scaled
	plt.xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation [pixels]', fontsize = 20)

	# Set the y-axis of the plot to be logarithmically scaled
	plt.yscale('log')

	# Add a label to the y-axis
	plt.ylabel('Structure Function', fontsize = 20)

	# Add a title to the plot
	plt.title('SF {} Gam{} z'.format(short_simul[j], gamma), fontsize = 20)

	# Force the legend to appear on the plot
	plt.legend(loc = 4)

	# Save the figure using the given filename and format
	plt.savefig(data_loc + 'SF_{}_{}_gam{}_z.png'.format(short_simul[j], \
		obs_effect,gamma), format = 'png')

	# Close the figure, now that it has been saved.
	plt.close()

	# Create a figure to display a plot of the structure functions for different
	# values of the observational parameter, for a line of sight along the x
	# axis
	fig2 = plt.figure()

	# Create an axis for this figure
	ax2 = fig2.add_subplot(111)

	# Plot the structure function for various values of the observational effect
	# being studied.
	plt.plot(rad_x_arr[0], sf_x_arr[0],'b-o',label = leg_string\
		+'{}'.format(iter_array[0]))
	plt.plot(rad_x_arr[4], sf_x_arr[4],'r-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[4]))
	plt.plot(rad_x_arr[8], sf_x_arr[8],'g-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[8]))
	plt.plot(rad_x_arr[12], sf_x_arr[12],'k-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[12]))
	plt.plot(rad_x_arr[16], sf_x_arr[16],'m-o',\
		label= leg_string +'{0:.2f}'.format(iter_array[16]))

	# Set the x-axis of the plot to be logarithmically scaled
	plt.xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation [pixels]', fontsize = 20)

	# Set the y-axis of the plot to be logarithmically scaled
	plt.yscale('log')

	# Add a label to the y-axis
	plt.ylabel('Structure Function', fontsize = 20)

	# Add a title to the plot
	plt.title('SF {} Gam{} x'.format(short_simul[j], gamma), fontsize = 20)

	# Force the legend to appear on the plot
	plt.legend(loc = 4)

	# Save the figure using the given filename and format
	plt.savefig(data_loc + 'SF_{}_{}_gam{}_x.png'.format(short_simul[j], \
		obs_effect,gamma), format = 'png')

	# Close the figure, now that it has been saved.
	plt.close()

	# Print a message to the screen to show that the plots of the structure
	# functions have been saved for this simulation
	print 'Plots of the structure functions saved for {}'.format(short_simul[j])

# When the code reaches this point, plots have been produced for every 
# simulation.
# Print a message to the screen to show that the code has finished
print 'All structure functions plotted successfully'