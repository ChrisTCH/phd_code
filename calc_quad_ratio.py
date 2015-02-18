#------------------------------------------------------------------------------#
#                                                                              #
# This code is a function that calculates the magnitude and axis of the        #
# quadrupole ratio of a two-dimensional structure function. It utilises the    #
# calc_multipole_2D code to calculate the real and imaginary parts of the      #
# quadrupole moment, and then uses these to calculate the modulus and argument #
# of the quadrupole ratio. The modulus and argument are returned to the        #
# calling function, along with the radius values used in the calculation.      #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 30/1/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, sys to exit if a problem occurs
import numpy as np
import sys

# Import calc_multipole_2D to calculate the required multipole moments
from calc_multipole_2D import calc_multipole_2D

# Define the function calc_quad_ratio, which calculates the quadrupole ratio
# (quadrupole moment divided monopole moment) of a two-dimensional structure 
# function
def calc_quad_ratio(image, num_bins = 15):
	'''
	Description
		This function calculates the quadrupole ratio (quadrupole moment divided
		by the monopole moment) of a given two-dimensional, normalised structure
		function, where the zero-spacing pixel is in the centre of the image. 
		If the image is not 2-dimensional, then an error occurs. It returns the
		radius values used in the calculation, as well as the magnitude and 
		axis of anisotropy calculated at each radius value.  
	
	Required Input
		image: A numpy array containing a two-dimensional, normalised structure
			   function. Must be two-dimensional, otherwise an error will occur.
		num_bins: The number of radial bins to use when calculating the 
				  quadrupole ratio for the image. Must be an integer.
	Output
		mag_arr: A numpy array with num_bins entries. Each entry of the array
				 specifies the magnitude of the quadrupole ratio for the 
				 corresponding entry of rad_arr.
		ang_arr: A numpy array with num_bins entries. Each entry of the array
				 specifies the the angle that the axis of anisotropy makes with
				 the positive x-axis, in degrees, for the corresponding entry 
				 of rad_arr.
		rad_arr: A numpy array with num_bins entries. Each entry of the array
				 specifies the radius value (in pixels) of that bin.
	'''

	# Calculate the shape of the input array
	im_shape = np.shape(image)

	# Calculate the number of dimensions in the input array
	num_dims = np.shape(im_shape)[0]

	# Check to see if the array is not 2-dimensional
	if num_dims != 2:
		# In this case the given array is not 2-dimensional, and so the 
		# calculation performed will not work. Print an error message to the
		# screen.
		print'calc_quad_ratio: Given image is {} dimensional'.format(num_dims)

		# Exit the function, since we should not proceed if the given array is
		# not 2-dimensional
		sys.exit()

	# Calculate the monopole for the normalised structure function
	monopole_arr, mono_rad_arr = calc_multipole_2D(image, order = 0,\
	 num_bins = num_bins)

	# Calculate the quadrupole for the normalised structure function
	quadpole_arr, quad_rad_arr = calc_multipole_2D(image, order = 2,\
	 num_bins = num_bins)

	# Calculate the quadrupole ratio from the quadrupole and monopole ratios
	quad_mono = quadpole_arr / monopole_arr

	# Calculate the modulus of the quadrupole ratio for each radius value
	mag_arr = np.absolute(quad_mono)

	# Calculate the anisotropy angle, which describes the axis along which the
	# structures are elongated, for each radius value. In degrees.
	ang_arr = 0.5 * np.rad2deg(np.arctan2(-np.imag(quad_mono), -np.real(quad_mono)))

	# Create the radius array to return. This is the same as the radius array
	# returned when the monopole moment was calculated, which should also be
	# the same as the radius array calculated for the quadrupole moment
	rad_arr = mono_rad_arr

	# We have now finished calculating the quadrupole ratio, so return the
	# magnitude and axis arrays for the quadrupole ratio, and the radius array
	return mag_arr, ang_arr, rad_arr