#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates a map of projected magnetic field strength.  #
# The produced map is stored in a FITS file.                                   #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 8/12/2014                                                        #
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
# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/', 'c512b3p.01/', 'c512b5p.01/']

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the map of the 
# projected magnetic field. This can be 'x', 'y', or 'z'
line_o_sight = 'x'

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

	# Print a message to the screen to show that the data has been loaded
	print 'Magnetic field components loaded successfully'

	# Calculate the magnitude of the magnetic field, which is just the square
	# root of the sum of the x, y and z component magnitudes squared.
	mag_amp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0)\
	+ np.power(mag_z_data, 2.0) )

	# Depending on the line of sight, the integration of the magnetic field 
	# amplitude is performed in different ways
	if line_o_sight == 'z':
		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the map of projected magnetic field. 
		# Since the line of sight is the z axis, we need to integrate along 
		# axis 0. (Numpy convention is that axes are ordered as (z, y, x))
		int_axis = 0
	elif line_o_sight == 'y':
		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the map of projected magnetic field. 
		# Since the line of sight is the y axis, we need to integrate along axis 1.
		int_axis = 1
	elif line_o_sight == 'x':
		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the map of projected magnetic field.
		# Since the line of sight is the x axis, we need to integrate along axis 2.
		int_axis = 2

	# Integrate the magnetic field strength along the required axis. This 
	# integration is performed by the trapezoidal rule. To normalise 
	# the calculated map, divide by the number of pixels along the 
	# z-axis. Note the array is ordered by (z,y,x)!
	# NOTE: Set dx to whatever the pixel spacing is
	mag_proj = np.trapz(mag_amp, dx = 1.0, axis = int_axis) /\
	 np.shape(mag_amp)[0]

	# Now that the projected magnetic field map has been produced, we need to 
	# save the produced map as a FITS file

	# To do this, we need to make a FITS header, so that anyone using the FITS
	# file in the future will know what gamma values were used

	# Create a primary HDU to contain the map of projected magnetic field
	pri_hdu = fits.PrimaryHDU(mag_proj)

	# Save the produced map of the projected magnetic field as a FITS file
	mat2FITS_Image(mag_proj, pri_hdu.header, data_loc + 'mag_proj_' +\
	 line_o_sight + '.fits')

	# Print a message to state that the FITS file was saved successfully
	print 'FITS file of projected magnetic field saved successfully'

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All projected magnetic field calculated successfully'