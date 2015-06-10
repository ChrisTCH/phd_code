#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# fields, and produces histograms of the synchrotron emissivity for all of the #
# simulations. The Gaussian distribution that has the same mean and standard   #
# deviation as the synchrotron emissivity distribution is over-plotted, to     #
# study any deviation from Gaussianity.                                        #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 23/4/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, # scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

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

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can be 'x', 'y', or 'z'
line_o_sight = 'z'

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/Sync_Emis_PDF/'

# Loop over the simulations, as we need to produce a histogram for each one
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting histogram for simulation {}'.format(simul_arr[j])

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

	# Depending on the line of sight, the strength of the magnetic field 
	# perpendicular to the line of sight is calculated in different ways
	if line_o_sight == 'z':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

	elif line_o_sight == 'y':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_z_data, 2.0) )

	elif line_o_sight == 'x':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the y and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_y_data, 2.0) + np.power(mag_z_data, 2.0) )

	# Calculate the result of raising the perpendicular magnetic field strength
	# to the power of gamma, for these slices. This is the synchrotron 
	# emissivity
	mag_perp_gamma = np.power(mag_perp, gamma)

	# Flatten out the synchrotron emissivity values, so that we can calculate
	# statistics and histograms for the emissivity
	sync_emis_flat = mag_perp_gamma.flatten()

	# Calculate the mean of the synchrotron emissivity values
	emis_mean = np.mean(sync_emis_flat, dtype = np.float64)

	# Calculate the standard deviation of the synchrotron emissivity values
	emis_stdev = np.std(sync_emis_flat, dtype = np.float64)

	# Calculate the minimum and maximum emissivity values to be used in 
	# constructing the plot and histogram of emissivity values
	# emis_min = np.log10(np.min(sync_emis_flat))
	# emis_max = np.log10(np.max(sync_emis_flat))
	emis_min = np.min(sync_emis_flat)
	emis_max = np.max(sync_emis_flat)

	# Create an array of values that extend from the minimum synchrotron 
	# emissivity to the maximum. This will be used to plot the Gaussian that
	# best matches the synchrotron emissivity histogram.
	# gauss_emis = np.logspace(emis_min, emis_max, 100)
	gauss_emis = np.linspace(emis_min, emis_max, 100)

	# Calculate the values for the Gaussian that best approximates the 
	# synchrotron emissivity distribution
	best_gauss = np.exp(-np.power(gauss_emis - emis_mean, 2.0)/\
		(2.0*np.power(emis_stdev,2.0))) / (np.sqrt(2 * np.pi) * emis_stdev)

	# Construct an array that specifies the bin edges to use when creating
	# the histogram. These bin edges should be equally spaced logarithmically
	# bin_edges = np.logspace(emis_min, emis_max, num = 51)
	bin_edges = np.linspace(emis_min, emis_max, num = 51)

	# Create a figure in which to plot the histograms
	fig = plt.figure(j)

	# Add axes to the figure
	ax = fig.add_subplot(111)

	# Make a histogram of the synchrotron intensity values for this value
	# of gamma
	hist_val, edges = np.histogram(sync_emis_flat, bins = bin_edges, density = True)

	# Calculate the centre of each bin, to use in the plotting process
	bin_centres = ((edges + np.roll(edges,-1))[0:-1])/2.0

	# Plot the normalised histogram as a function of the bin centre location
	plt.plot(bin_centres, hist_val, label = 'sync emis hist')

	# Plot the best-fitting Gaussian on the plot
	plt.plot(gauss_emis, best_gauss, label = 'best fit gauss')

	# Put a legend on the plot
	plt.legend(loc = 1)

	# Put an label on the x-axis
	plt.xlabel('Sync Emissivity [arb.]')

	# Set the x-axis of the plot to be logarithmically scaled
	# plt.xscale('log')

	# Add a title to the histogram
	plt.title('Sync Emis PDFs {} {}LOS'.format(short_simul[j],line_o_sight))

	# Save the figure using the given filename and format
	plt.savefig(save_loc + 'Sync_Emis_PDF_{}_{}_nolog.png'.format(short_simul[j],\
	 line_o_sight), format = 'png')

	# Close the current figure so that it doesn't take up memory
	plt.close()

	# Close all of the FITS files, to save memory
	mag_x_fits.close()
	mag_y_fits.close()
	mag_z_fits.close()

	# Print a line to the screen to show that the figure has been saved
	print 'Synchrotron PDF saved for {}'.format(short_simul[j])

# When the code reaches this point, the everything has been saved, so print 
# a message stating this to the screen
print 'All PDFs produced successfully'