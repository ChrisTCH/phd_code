#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and produces histograms of the synchrotron intensity for all of #
# the values of gamma used to construct the synchrotron intensity maps.        #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 5/2/2015                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/Sync_Inten_PDF/'

# Loop over the simulations, as we need to produce a histogram for each one
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting histogram for simulation {}'.format(simul_arr[j])

	# Open the FITS files that contain the simulated synchrotron intensity maps
	sync_fits = fits.open(data_loc + 'synint_p1-4.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensities observed for different values of gamma, the power
	# law index of the cosmic ray electrons emitting the synchrotron emission.
	sync_data = sync_fits[0].data

	# Create a figure in which to plot the histograms
	fig = plt.figure(3*j)

	# Add axes to the figure
	ax = fig.add_subplot(111)

	# Next, we want to produce a histogram for each value of gamma, so loop over
	# the values of gamma
	for i in range(len(gamma_arr)):
		# Plot a histogram of the synchrotron intensity values for this value
		# of gamma
		plt.hist((sync_data[i]).flatten(), bins = 50, alpha = 0.3, label =\
		 'Gam{}'.format(gamma_arr[i]))

	# Place a legend on the plot
	plt.legend(loc = 1)

	# Put an label on the x-axis
	plt.xlabel('Sync Inten [arb.]')

	# Add a title to the histogram
	plt.title('Sync Inten PDFs {} zLOS'.format(short_simul[j]))

	# Save the figure using the given filename and format
	plt.savefig(save_loc + 'Sync_PDF_{}_z.png'.format(short_simul[j]), format = 'png')

	# Close the current figure so that it doesn't take up memory
	plt.close()

	# Create a figure to plot histograms for low values of gamma
	fig_low = plt.figure(3*j + 1)

	# Add axes to the figure
	ax_low = fig_low.add_subplot(111)

	# Next, we want to produce a histogram for the four lowest values of gamma
	for i in range(4):
		# Plot a histogram of the synchrotron intensity values for this value
		# of gamma
		plt.hist((sync_data[i]).flatten(), bins = 50, alpha = 0.3, label =\
		 'Gam{}'.format(gamma_arr[i]))

	# Place a legend on the plot
	plt.legend(loc = 1)

	# Put an label on the x-axis
	plt.xlabel('Sync Inten [arb.]')

	# Add a title to the histogram
	plt.title('Sync Inten PDFs {} zLOS'.format(short_simul[j]))

	# Save the figure using the given filename and format
	plt.savefig(save_loc + 'Sync_PDF_{}_lowGam_z.png'.format(short_simul[j]), format = 'png')

	# Close the current figure so that it doesn't take up memory
	plt.close()

	# Create a figure to plot histograms for high values of gamma
	fig_high = plt.figure(3*j + 2)

	# Add axes to the figure
	ax_high = fig_high.add_subplot(111)

	# Next, we want to produce a histogram for the three high values of gamma
	for i in (4,5,6):
		# Plot a histogram of the synchrotron intensity values for this value
		# of gamma
		plt.hist((sync_data[i]).flatten(), bins = 50, alpha = 0.3, label =\
		 'Gam{}'.format(gamma_arr[i]))

	# Place a legend on the plot
	plt.legend(loc = 1)

	# Put an label on the x-axis
	plt.xlabel('Sync Inten [arb.]')

	# Add a title to the histogram
	plt.title('Sync Inten PDFs {} zLOS'.format(short_simul[j]))

	# Save the figure using the given filename and format
	plt.savefig(save_loc + 'Sync_PDF_{}_highGam_z.png'.format(short_simul[j]), format = 'png')

	# Close the current figure so that it doesn't take up memory
	plt.close()

	# Print a line to the screen to show that the figure has been saved
	print 'Synchrotron PDF saved for {}'.format(short_simul[j])

# When the code reaches this point, the everything has been saved, so print 
# a message stating this to the screen
print 'All PDFs produced successfully'