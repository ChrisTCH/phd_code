#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the observed synchrotron emission maps for a #
# cube that is saturated with a uniform, isotropic distribution of cosmic rays #
# with power spectrum index gamma. Various values of gamma are used, and the   #
# angles that the line of sight makes relative to the z axis of the cube can   #
# be altered. The produced maps are stored in FITS files, one for each         #
# rotation angle.                                        					   #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 4/11/2014                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.ndimage to handle rotation of data cubes.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage

# Import mat2FITS_Image to convert arrays to FITS format
from mat2FITS_Image import mat2FITS_Image
from fractalcube import fractalcube

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
spec_loc = 'c512b5p.01/'

# # Create a variable that controls the angle between the line of sight and the
# # z axis of the data cube. This needs to be a decimal. The rotation from the
# # z axis to the new line of sight is performed around the y axis.
# rotate_angle = 0.0

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc

# Create an array that specifies the gamma values that were used to produce
# these synchrotron emission maps
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# ------ Use this code to test with fractal data

# # Create a cube of fractal data, which is meant to represent the x component
# # of the magnetic field. 5.0 is added to give a mean field in the x direction.
# mag_x_data = fractalcube(3.0, seed = 6, size = 512) + 5.0

# # Create a cube of fractal data, which is meant to represent the y component 
# # of the magnetic field
# mag_y_data = fractalcube(3.0, seed = 8, size = 512)

# # Create a cube of fractal data, which is meant to represent the z component 
# # of the magnetic field
# mag_z_data = fractalcube(3.0, seed = 10, size = 512)

# ------ End fractal data generation

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

# Create an array that specifies the rotation angles relative to the z axis of
# the MHD cubes, of the synchrotron maps to be used
rot_ang_arr = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,\
	80.0, 90.0]) 

# Calculate the rotation angle in radians
rad_angle_arr = np.deg2rad(rot_ang_arr)

# Loop over all of the required angles, to calculate the synchrotron map for
# each one
for j in range(len(rad_angle_arr)):
	# Calculate the strength of the component of the magnetic field in the 
	# horizontal direction, as viewed from the new line of sight
	mag_horiz = mag_x_data * np.cos(rad_angle_arr[j]) + mag_z_data *np.sin(rad_angle_arr[j])

	# Calculate the magnitude of the magnetic field perpendicular to the new line
	# of sight, which is just the square root of the sum of the horizontal and 
	# y magnitudes squared
	mag_perp = np.sqrt( np.power(mag_horiz, 2.0) + np.power(mag_y_data, 2.0) )

	# Calculate the shape of the input data cubes
	shape_input = np.shape(mag_perp)

	# Calculate the length of each side of the sub-cube that will be extracted.
	# This length is chosen so that the sub-cube formed is the biggest possible
	# sub-cube that exists for every line of sight.
	sub_cube_length = np.floor( shape_input[0] / np.sqrt(2.0) )

	# Print a message to show what the sub-cube length should be
	print 'Sub-cube length should be {}'.format(sub_cube_length)

	# Rotate the cube containing values for the perpendicular component of the 
	# magnetic field, so that the new line of sight is aligned with the z-axis of 
	# the new cube. This is done so that we can integrate along the line of sight
	# more easily.
	rotated_mag_perp = ndimage.interpolation.rotate(mag_perp, -1.0 * rot_ang_arr[j],\
		axes = [2,0], reshape = True, order = 3)

	# Print a message to show that the perpendicular component of the magnetic
	# field has been rotated to make integration easier
	print 'Perpendicular component of the magnetic field has been rotated'

	# Calculate the shape of the rotated cube
	shape_rotate = np.shape(rotated_mag_perp)

	# Print the shape of the rotated cube to the screen
	print 'Shape of rotated cube is {}'.format(shape_rotate)

	# Calculate the minimum index of the rotated cube, to be included when 
	# extracting the sub-cube
	ind_min = int(shape_rotate[0] / 2.0 - sub_cube_length / 2.0)

	# Calculate the maximum index of the rotated cube, to not be included when
	# extracting the sub-cube
	ind_max = int(shape_rotate[0] / 2.0 + sub_cube_length / 2.0)

	# Calculate the minimum index of the rotated cube, to be included when 
	# extracting the sub-cube, for the y axis
	ind_min_y = int(shape_rotate[1] / 2.0 - sub_cube_length / 2.0)

	# Calculate the maximum index of the rotated cube, to not be included when
	# extracting the sub-cube, for the y axis
	ind_max_y = int(shape_rotate[1] / 2.0 + sub_cube_length / 2.0)

	# Extract the sub-cube that will be used to produce the maps of synchrotron
	# emission. This sub-cube is taken from the rotated cube of the component of
	# the magnetic field perpendicular to the new line of sight
	sub_mag_perp = rotated_mag_perp[ind_min:ind_max,ind_min_y:ind_max_y,ind_min:ind_max]

	# Print the shape of the obtained sub-cube to the screen
	print 'Shape of the sub-cube is {}'.format(np.shape(sub_mag_perp))

	# Create a Numpy array to hold the calculated synchrotron emission maps
	sync_arr = np.zeros((len(gamma_arr), np.shape(sub_mag_perp)[1],\
	 np.shape(sub_mag_perp)[2]))

	# Loop over the array of gamma values, calculating the observed synchrotron
	# emission map for each one
	for i in range(len(gamma_arr)):
		# Calculate the result of raising the perpendicular magnetic field strength
		# to the power of gamma
		mag_perp_gamma = np.power(sub_mag_perp, gamma_arr[i])

		# Integrate the perpendicular magnetic field strength raised to the power
		# of gamma along the required axis, to calculate the observed synchrotron 
		# map. This integration is performed by the trapezoidal rule. To normalise 
		# the calculated synchrotron map, divide by the number of pixels along the 
		# z-axis. Note the array is ordered by (z,y,x)!
		# NOTE: Set dx to whatever the pixel spacing is
		sync_arr[i] = np.trapz(mag_perp_gamma, dx = 1.0, axis = 0) /\
		 np.shape(sub_mag_perp)[0]

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
	mat2FITS_Image(sync_arr, pri_hdu.header, data_loc + 'synint_p1-4_' +\
	 'rot_{}'.format(rot_ang_arr[j]) + '.fits')

	# Print a message to state that the FITS file was saved successfully
	print 'FITS file of synchrotron maps saved successfully'

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All synchrotron maps calculated successfully'