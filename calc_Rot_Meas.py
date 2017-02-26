#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of Stokes Q   #
# and U for an image, and produces an array of the observed rotation measure  #
# at each pixel. The arrays provided must be three dimensional, with the      #
# first axis being wavelength squared. The rotation measure array is returned #
# to the caller.                                                              #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 26/10/2016                                                      #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Rot_Meas, which will calculate the observed 
# rotation measure from the Stokes Q and U values at each pixel.
def calc_Rot_Meas(Sto_Q, Sto_U, wave_sq_space = 1.0):
	'''
	Description
	    This function calculates the observed rotation measure at each point
	    of an image, when given the Stokes Q and U values at each pixel
	    for the image. This function should also be provided the spacing between
	    the wavelength slices of the Stokes Q and U images, to ensure that
	    the derivative of polarisation angle with respect to wavelength squared
	    is calculated correctly.
	    
	Required Input
	    Sto_Q - A Numpy array containing the value of Stokes Q at each
	            pixel of the image, for various wavelengths. The array must
	            conform to the convention that the first dimension represents
	            the square of the wavelength, the second dimension represent 
	            the y-axis, and the third dimension represents the x-axis. Each
	            entry of the array must be a float.
	    Sto_U - A Numpy array containing the Stokes U value at each point of
	            the image. Each entry of the array must be a float. Array must
	            have the same size as the Stokes Q array.
	    wave_sq_space - The wavelength squared spacing between adjacent slices 
	            of the Stokes Q and arrays, in m^2. 
	               
	Output
	    rot_meas - A Numpy array containing the value of the observed rotation
	            measure at each point. This array has the same size as the 
	            Stokes arrays. Units of rad m^-2.
	    unwound_angle - A Numpy array containing the unwound polarisation angle 
	            values, in radians. This array has the same size as the Stokes
	            arrays.
	'''
	
	# Calculate the observed polarisation angle (in radians)
	polar_angle = 0.5 * np.arctan2(Sto_U, Sto_Q)

	# Create a separate array, that will be used to store the unwound angles
	unwound_angle = np.copy(polar_angle)

	# As the polarisation angle is defined to be between -pi/2 and +pi/2, there
	# will be places where the polarisation angle has a discontinuity, because
	# the angle switches from being close to +pi/2, to close to -pi/2, or vice
	# versa. We need to find these locations in the cube, and correct for them,
	# so that the correct derivative is calculated.
	# Iterate over the slices of the polarisation angle array, except the final
	# slice
	for i in range(np.shape(polar_angle)[0] - 1):
		# To use differences in polarisation angle to determine if we need to
		# adjust the polarisation angle, we cannot be at the first slice of the
		# polarisation angle cube
		if i > 0:
			# Calculate the difference in polarisation angle in radians between 
			# adjacent slices.
			polar_diff_prev = unwound_angle[i] - unwound_angle[i-1]
			polar_diff_next = unwound_angle[i+1] - unwound_angle[i]

			# Calculate the absolute difference in polarisation angle between 
			# the current slice and the next slice
			abs_diff = np.absolute(polar_angle[i+1] - polar_angle[i])

			# Find any locations where the differences in polarisation angle
			# between this slice, and the slices on either side, have opposite
			# signs
			disc_pix = polar_diff_prev * polar_diff_next < 0

			# Find any locations where the change in polarisation angle is
			# more than pi/2.0
			disc_pix = np.logical_or(disc_pix, abs_diff > np.pi/2.0)

			# Find pixels in the image of the current slice with a difference
			# in polarisation angles that is large and positive, as in this
			# case we need to subtract pi
			angle_sub = polar_diff_next >= np.pi/2.0

			# Find pixels in the image of the current slice with a difference
			# in polarisation angles that is large and negative, as in this
			# case we need to add pi
			angle_add = polar_diff_next <= -1.0 * np.pi/2.0

		else:
			# Calculate the absolute difference in polarisation angle between 
			# the current slice and the next slice
			abs_diff = np.absolute(polar_angle[i+1] - polar_angle[i])

			# Find the pixels in the image at this slice for which this 
			# difference is more than 2.6 radians (using a number less than pi 
			# radians to account for the discrete nature of the measurements). 
			# At these pixels, there is likely a discontinuity in the 
			# polarisation angle
			disc_pix = abs_diff > np.pi/2.0

			# Find pixels in the image of the current slice with a polarisation
			# angle that is positive and large, as in this case we need to
			# add pi
			angle_add = polar_angle[i] >= 0

			# Find pixels in the image of the current slice with a polarisation 
			# angle that is negative and large, as in this case we need to 
			# subtract pi
			angle_sub = polar_angle[i] < 0

		# Find the pixels that have a discontinuity in the polarisation angle,
		# and have a positive polarisation angle
		pos_disc = np.logical_and(disc_pix, angle_add)

		# Find the pixels that have a discontinuity in the polarisation angle,
		# and have a negative polarisation angle
		neg_disc = np.logical_and(disc_pix, angle_sub)

		# For the pixels that have a discontinuity and a positive polarisation
		# angle, and pi to that pixel for all remaining slices in the array
		unwound_angle[i+1:,pos_disc] = unwound_angle[i+1:,pos_disc] + np.pi

		# For the pixels that have a discontinuity and a negative polarisation
		# angle, subtract pi from that pixel for all remaining slices in the 
		# array
		unwound_angle[i+1:,neg_disc] = unwound_angle[i+1:,neg_disc] - np.pi

	# Calculate the derivative of the polarisation angle with respect to the
	# square of the wavelength, to find the rotation measure, in rad m^-2
	rot_meas = np.gradient(unwound_angle, wave_sq_space, axis = 0)
	
	# Return the polarisation angle to the calling function
	return rot_meas, unwound_angle