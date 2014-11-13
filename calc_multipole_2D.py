#------------------------------------------------------------------------------#
#                                                                              #
# This code is a function that calculates the multipoles of a 2-dimensional    #
# image by integrating over polar angle for certain bins defined by radius. It #
# returns the radius values used in the calculation, as well as the multipole  #
# calculated at each radius value.                                             #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 20/10/2014                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, sys to exit if a problem occurs
import numpy as np
import sys

# Define the function calc_multipole_2D, which calculates a multipole of a given
# 2D image.
def calc_multipole_2D(image, order = 0, num_bins = 15):
	'''
	Description
		This function calculates a multipole of an image by integrating over 
		polar angle for certain bins defined by radius. If the image is not
		2-dimensional, then an error occurs. It returns the radius values used
		in the calculation, as well as the multipole calculated at each radius
		value.  
	
	Required Input
		image: A numpy array containing an image. Must be two-dimensional, 
			   otherwise an error will occur.
		order: The order of the multipole to calculate, as an integer. 0 is the 
			   default, which calculates the monopole. 1 calculates the dipole,
			   2 calculates the quadrupole, 4 calculates the octopole, and so
			   on.
		num_bins: The number of radial bins to use when calculating the 
				  multipole for the image. Must be an integer.
	Output
		multi_arr: A numpy array with num_bins entries. Each entry of the array
				   specifies the value of the multipole for the corresponding
				   entry of rad_arr.
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
		print'calc_multipole_2D: Given image is {} dimensional'.format(num_dims)

		# Exit the function, since we should not proceed if the given array is
		# not 2-dimensional
		sys.exit()

	# Calculate the length of the first dimension of the data (y-axis)
	N1 = im_shape[0]

	# Calculate the length of the second dimension of the data (x-axis)
	N2 = im_shape[1]

	# Check to see if the length of the first dimension is even or odd
	if (N1 % 2) == 0:
		# In this case the length of the first dimension is even
		# Record this in a boolean variable, which is True if the length is even
		isEven1 = True
	else:
		# In this case the length of the first dimension is odd
		# Record this in a boolean variable
		isEven1 = False

	# Check to see if the length of the second dimension is even or odd
	if (N2 % 2) == 0:
		# In this case the length of the second dimension is even
		# Record this in a boolean variable, which is True if the length is even
		isEven2 = True
	else:
		# In this case the length of the second dimension is odd
		# Record this in a boolean variable
		isEven2 = False

	# Create an array that specifies the index value along the first dimension
	# of the data, where index 0 is in the centre of the array.
	# This array needs to be constructed differently depending on whether the
	# length along the first dimension is even or odd.
	if isEven1 == True:
		# In this case the length along the first axis is even
		# Construct an array of decimals that specifies how far away each pixel
		# is from the centre of the axis
		N1_indices = np.linspace(-N1/2.0 + 0.5, N1/2.0 - 0.5, num = N1)
	else:
		# In this case the length along the first axis is odd
		# Construct an array of decimals that specifies how far away each pixel
		# is from the centre of the axis
		N1_indices = np.linspace(-(N1-1.0)/2.0, (N1-1.0)/2.0, num = N1)

	# Create an array that specifies the index value along the second dimension
	# of the data, where index 0 is in the centre of the array.
	# This array needs to be constructed differently depending on whether the
	# length along the second dimension is even or odd.
	if isEven2 == True:
		# In this case the length along the second axis is even
		# Construct an array of decimals that specifies how far away each pixel
		# is from the centre of the axis
		N2_indices = np.linspace(-N2/2.0 + 0.5, N2/2.0 - 0.5, num = N2)
	else:
		# In this case the length along the second axis is odd
		# Construct an array of decimals that specifies how far away each pixel
		# is from the centre of the axis
		N2_indices = np.linspace(-(N2-1.0)/2.0, (N2-1.0)/2.0, num = N2)

	# Use meshgrid to obtain the (x,y) pixel co-ordinates of every point in the 
	# data. i_mat specifies the index value along the first dimension, and j_mat
	# specifies the index value along the second dimension, relative to the 
	# centre of the array.
	i_mat, j_mat = np.meshgrid(N2_indices, N1_indices)

	# Calculate the distance to each entry in the array, from the centre
	r_mat = np.sqrt(np.power(i_mat, 2.0) + np.power(j_mat, 2.0))

	# Now that all of the required distances have been calculated, remove any
	# degenerate dimensions in the distance array
	r_mat = np.squeeze(r_mat)

	# Calculate the polar angle phi for every pixel in the data
	# These angles are in radians, and lie between [-pi, pi]
	phi_mat = np.arctan2(j_mat, i_mat)

	# Now that all of the polar angles have been calculated, remove any
	# degenerate dimensions in the polar angle array
	phi_mat = np.squeeze(phi_mat)

	# Flatten the array of radius values, so that we can properly sort data
	# points into bins that are equally spaced in r
	flat_r_mat = r_mat.flatten()

	# Flatten the array of polar angle values, so that we can associate a polar
	# angle to each radius value
	flat_phi_mat = phi_mat.flatten()

	# Flatten the image, so that we can associate a radius and polar angle to
	# each image value
	flat_image = image.flatten()

	# We need to sort the polar angle values from smallest to largest, so that
	# the integration over polar angle is performed correctly. Find the index 
	# order that sorts the polar angle values.
	sort_arr = np.argsort(flat_phi_mat)

	# Create an array of polar angle values which has been sorted from smallest
	# to largest
	flat_phi_mat = flat_phi_mat[sort_arr]

	# Sort the radius array, so that the radius values correspond to the 
	# correct values of polar angle
	flat_r_mat = flat_r_mat[sort_arr]

	# Sort the image values, so that the image values correspond to the correct
	# pixel co-ordinates
	flat_image = flat_image[sort_arr]

	# We need to check if any of the radius values are less than or equal to
	# zero, since we are about to compute a logarithm.
	# Calculate the minimum of the radius array
	min_rad = np.min(flat_r_mat)

	# Check to see if the minimum is less than or equal to zero
	if min_rad <= 0:
		# In this case the minimum is less than or equal to zero, so find the
		# indices of the radius array where the values are less than or equal to
		# zero.
		min_index = np.argmin(flat_r_mat)

		# Change the minimum value to a small, positive number
		flat_r_mat[min_index] = 1.0e-8

	# Calculate the minimum and maximum radius values to be used in calculating
	# the integral over polar angle
	r_min = np.log10(2.0)
	r_max = np.log10(max([N1-1, N2-1]) / 2.0 )

	# Construct an array that specifies the bin edges to use when performing
	# the integration over polar angle. These bins are equally spaced
	# logarithmically.
	bin_edges = np.linspace(r_min, r_max, num = num_bins + 1)

	# Now we need to figure out which elements of the radius value array are in 
	# which bin
	bin_indices = np.digitize(np.log10(flat_r_mat), bin_edges)

	# Create an array of zeroes to hold all of the calculated radius values
	rad_arr = np.zeros(num_bins)

	# Create an array of zeroes to hold all of the multipole values
	multi_arr = np.zeros(num_bins)

	# We now need to cycle through each of the bins, to calculate the value of
	# the integral for that bin
	for i in range(1, num_bins + 1):
		# Find all of the radius values that are in this bin
		r_i_bin = flat_r_mat[bin_indices == i]

		# Check to see if this bin is empty
		if len(r_i_bin) > 0:
			# In this case the bin is not empty, so find the polar angle values
			# corresponding to the radius values
			phi_i_bin = flat_phi_mat[bin_indices == i]

			# Find all of the image values corresponding to the radius values
			image_i_bin = flat_image[bin_indices == i]

			# Calculate the integrand of the integral that gives the multipole
			integrand_i_bin = np.exp(-1.0j * order * phi_i_bin) * image_i_bin

			# Calculate the result of integrating over the polar angle phi by
			# using the trapz function of Numpy. This automatically takes into
			# account the value of phi for each pixel, so we don't need to 
			# worry about spacings, or whether pixels have the same value of
			# phi. The value of the multipole for this radius is stored in the
			# corresponding array.
			multi_arr[i - 1] = np.trapz(integrand_i_bin,phi_i_bin)/(2.0 * np.pi)

			# Calculate the average radius value for this bin, and store it in
			# the radius array
			rad_arr[i - 1] = np.mean(r_i_bin, dtype = np.float64)

	# It is possible that the calculated multipole has an imaginary part, due
	# to numerical imprecision, so extract the real part
	multi_arr = np.real(multi_arr)

	# We have now finished calculating the multipole, so return the multipole
	# array and radius array
	return multi_arr, rad_arr