#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the observed synchrotron emission maps for a #
# cube that is saturated with a uniform, isotropic distribution of cosmic rays #
# with power spectrum index gamma. Various values of gamma are used. The       #
# produced maps are stored in a FITS file, and plots comparing the calculated  #
# maps to those produced by Blakesley Burkhart are produced.                   #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 16/9/2014                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import mat2FITS_Image to convert arrays to FITS format, and mat_plot to 
# produce images of matrices
from mat2FITS_Image import mat2FITS_Image
from mat_plot import mat_plot
from fractalcube import fractalcube

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
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

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can be 'x', 'y', or 'z'
line_o_sight = 'y'

# Iterate over the simulations, to produce synchrotron intensity maps for each
# simulation
for i in range(len(simul_arr)): 
	# Print a message to show what simulation calculations are being performed
	# for
	print 'Simulations starting for {}'.format(simul_arr[i])

	# Create a string for the full directory path to use in calculations
	data_loc =  simul_loc + simul_arr[i]

	# # Open the FITS file that contains the synchrotron maps that Blakesley made
	# Bl_sync_fits = fits.open(data_loc + 'b.1p2_synint_p1-4.fits')

	# # Extract the data for the synchrotron maps that Blakesley made
	# Bl_sync_data = Bl_sync_fits[0].data 

	# Create an array that specifies the gamma values that were used to produce
	# these synchrotron emission maps
	gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])
	# gamma_arr = np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
	 
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

	# ------ Use this code to test with fractal data

	# # Create a cube of fractal data, which is meant to represent the x component
	# # of the magnetic field
	# mag_x_data = fractalcube(3.0, seed = 6, size = 512)

	# # # slicing(mag_x_data, xlabel = 'x', ylabel = 'y',\
	# # #  zlabel = 'z', title = 'x-comp B Field')

	# # Create a cube of fractal data, which is meant to represent the y component 
	# # of the magnetic field
	# mag_y_data = fractalcube(3.0, seed = 8, size = 512)

	# # # slicing(mag_y_data, xlabel = 'x', ylabel = 'y',\
	# # #  zlabel = 'z', title = 'y-comp B Field')

	# ------ End fractal data generation

	# Print a message to the screen to show that the data has been loaded
	print 'Magnetic field components loaded successfully'

	# Depending on the line of sight, the strength of the magnetic field 
	# perpendicular to the line of sight is calculated in different ways
	if line_o_sight == 'z':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the z axis, we need to integrate along axis 0. (Numpy convention is 
		# that axes are ordered as (z, y, x))
		int_axis = 0
	elif line_o_sight == 'y':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_z_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the y axis, we need to integrate along axis 1.
		int_axis = 1
	elif line_o_sight == 'x':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the y and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_y_data, 2.0) + np.power(mag_z_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the x axis, we need to integrate along axis 2.
		int_axis = 2

	# Create a Numpy array to hold the calculated synchrotron emission maps
	sync_arr = np.zeros((len(gamma_arr), np.shape(mag_perp)[1],\
	 np.shape(mag_perp)[2]))

	# Loop over the array of gamma values, calculating the observed synchrotron
	# emission map for each one
	for j in range(len(gamma_arr)):
		# Calculate the result of raising the perpendicular magnetic field strength
		# to the power of gamma
		mag_perp_gamma = np.power(mag_perp, gamma_arr[j])

		# Integrate the perpendicular magnetic field strength raised to the power
		# of gamma along the required axis, to calculate the observed synchrotron 
		# map. This integration is performed by the trapezoidal rule. To normalise 
		# the calculated synchrotron map, divide by the number of pixels along the 
		# z-axis. Note the array is ordered by (z,y,x)!
		# NOTE: Set dx to whatever the pixel spacing is
		sync_arr[j] = np.trapz(mag_perp_gamma, dx = 1.0, axis = int_axis) /\
		 np.shape(mag_perp)[0]

	# Print a message to the screen stating that the synchrotron maps have been
	# produced
	print 'Synchrotron maps calculated'

	# Now that the synchrotron maps have been produced, we need to save the 
	# produced maps as a FITS file

	# To do this, we need to make a FITS header, so that anyone using the FITS
	# file in the future will know what gamma values were used

	# Create a primary HDU to contain the synchrotron data
	pri_hdu = fits.PrimaryHDU(sync_arr)

	# Add a header keyword to the HDU header, specifying the reference pixel
	# along the third axis
	pri_hdu.header['CRPIX3'] = 1

	# Add a header keyword to the HDU header, specifying the value of gamma
	# at the reference pixel
	pri_hdu.header['CRVAL3'] = 1.0

	# Add a header keyword to the HDU header, specifying the increment in gamma
	# along each slice of the array
	pri_hdu.header['CDELT3'] = 0.5

	# Add a header keyword to the HDU header, describing what the third axis is
	pri_hdu.header['CTYPE3'] = 'Gamma   '

	# Save the produced synchrotron maps as a FITS file
	mat2FITS_Image(sync_arr, pri_hdu.header, data_loc + 'synint_p1-4' +\
	 line_o_sight + '.fits')

	# Close all of the fits files, to save memory
	mag_x_fits.close()
	mag_y_fits.close()
	mag_z_fits.close()

	# Print a message to state that the FITS file was saved successfully
	print 'FITS file of synchrotron maps saved successfully {}'.format(simul_arr[i])

	# # Now we wish to compare our calculated maps with Blakesley's. To do this,
	# # we will cycle through all of the synchrotron maps that have been calculated,
	# # and for each a plot will be produced of the difference between my map
	# # of synchrotron emission and Blakesley's. I will also print out the maximum 
	# # difference between the maps.
	# for i in range(len(gamma_arr)):
	# 	# Calculate the difference between my synchrotron map and Blakesley's
	# 	sync_diff = sync_arr[i] - Bl_sync_data[i]

	# 	# Print the maximum absolute difference between the calculated maps to
	# 	# the screen
	# 	print 'Gamma = {} - Maximum difference is: {}'.format(gamma_arr[i],\
	# 	 np.max(np.abs(sync_diff)) )

	# 	# Create an image of the difference between the synchrotron maps
	# 	mat_plot(sync_diff, data_loc + 'CH-BB_sync_diff_p{}.png'.format(gamma_arr[i]),\
	# 	 'png', cmap ='hot', origin = 'lower', xlabel = 'X Axis [pixels]',\
	# 	  ylabel = 'Y Axis [pixels]', title = 'CH-BB Sync Diff Gamma {}'.\
	# 	  format(gamma_arr[i]))

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All synchrotron maps calculated successfully'