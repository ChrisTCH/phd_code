#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the sfr.pro IDL code written by Alexy       #
# Chepurnov, and available at http://www.astro.wisc.edu/~lazarian/code.html.   #
# This function averages the calculated structure function or correlation      #
# function over distance in the image or datacube, to produce a                #
# one-dimensional correlation function or structure function.                  #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alexy Chepurnov)          #
# Start Date: 3/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, and sys to exit if a problem occurs
import numpy as np
import sys

# Define the function sfr, which calculates a radially averaged correlation 
# function or structure function.
def sfr(sf, nb):
	'''
	Description
		This function calculates the radially averaged correlation or 
		structure function. It returns a list of radii, as well as the value of
		the radially averaged correlation or structure function at these radii.
	
	Required Input
		sf: A numpy array containing the correlation or structure function
			that is to be radially averaged. 
		nb: The number of bins to use when calculating the radially averaged
			correlation or structure function.
	
	Output
		sf_r: A numpy array with three rows. The first row contains the radius
			values used to calculate the radially averaged correlation or 
			structure function, the second row contains the radially averaged
			correlation or structure function corresponding to each radius 
			value, and the third row will hold the number of data points used
			to calculate that part of the function.
	'''
 
	# Calculate the length of the first dimension of the data
	N1 = (np.shape(sf))[0]

	# Initialise variables to hold the lengths of the second and third 
	# dimensions, if they are present.
	N2 = 1
	N3 = 1

	# Check to see if the data for which a spectrum is being taken has 2 or
	# more dimensions
	if len(np.shape(sf)) >= 2:
		# In this case, there are two or more dimensions, so extract the length
		# of the second dimension
		N2 = (np.shape(sf))[1]
	
	# Check to see if the data for which a spectrum is being taken has 3 or
	# more dimensions
	if len(np.shape(sf)) >= 3:
		# In this case, there are three or more dimensions, so extract the 
		# length of the third dimension
		N3 = (np.shape(sf))[2]
	
	# Check to see if the data for which a spectrum is being taken has four or
	# more dimensions
	if len(np.shape(sf)) >= 4:
		# In this case there are four or more dimensions, so print an error
		# message to the screen.
		print 'Well, please no more than 3 dimensions !'

		# Stop the function from continuing on, since this function cannot
		# handle radial averaging for data that is four 
		# dimensional or higher.
		sys.exit()

	# Create an array of zeroes, whose shape is the same as the shape of the 
	# input correlation or structure function, possibly with some extra
	# degenerate dimensions
	r_set = np.zeros((N1, N2, N3))

	# Create an array that specifies the index value along the first dimension
	# of the data 
	N1_indices = np.linspace(0, N1 - 1, num = N1)

	# Create an array that specifies the index value along the second dimension
	# of the data 
	N2_indices = np.linspace(0, N2 - 1, num = N2)

	# Create an array that specifies the index value along the third dimension
	# of the data 
	N3_indices = np.linspace(0, N3 - 1, num = N3)

	# Use meshgrid to obtain the (x,y,z) = (i,j,k) pixel index co-ordinates
	# of every point in the data. i_mat specifies the index value along the
	# first dimension, j_mat specifies the index value along the second
	# dimension, and k_mat specifies the index value along the third dimension
	i_mat, j_mat, k_mat = np.meshgrid(N1_indices, N2_indices, N3_indices,\
	 indexing = 'ij')

	# Calculate the distance to each entry in the array
	r_set = np.sqrt(np.power(i_mat, 2.0) + np.power(j_mat, 2.0) +\
	 np.power(k_mat, 2.0))

	# # Loop over the entries in the first dimension of the array, to calculate
	# # the distance to each entry in the array
	# for i in range(N1):
	# 	# Loop over the entries in the second dimension of the array
	# 	for j in range(N2): 
	# 		# Loop over the entries in the third dimension of the array
	# 		for k in range(N3):
	# 			# Calculate the distance to this particular array entry
	# 			r_set[i,j,k] = np.sqrt(np.power(i, 2.0) + np.power(j, 2.0) +\
	# 			np.power(k, 2.0))

	# Now that all of the required distances have been calculated, remove any
	# degenerate dimensions in the distance array
	r_set = np.squeeze(r_set)

	# Flatten the array of radius values, so that we can properly sort data
	# points into bins that are equally spaced in r
	flat_r_set = r_set.flatten()

	# Set the first element of the flattened array to a very small value, to
	# prevent divide by zero errors when calculating the logarithm of the
	# array
	flat_r_set[0] = 1.0e-8

	# Flatten the input data, so that each data point is matched with the 
	# corresponding value of r
	flat_sf = sf.flatten()

	# Calculate the minimum and maximum radius values to be used in calculating
	# the radially averaged functions
	r_min = 0.0
	r_max = np.log10(max([N1-1, N2-1, N3-1]))

	# Construct an array that specifies the bin edges to use when performing
	# a radial average. These bins are equally spaced logarithmically.
	bin_edges = np.linspace(r_min, r_max, num = nb + 1)

	# Construct a histogram of the radius values being used to calculate the
	# function, just so we can get the bin edges of the histogram
	# rlist, bin_edges = np.histogram(np.log10(flat_r_set), bins = nb, \
	# range = (r_min, r_max))

	# Now we need to figure out which elements of the radius value array are in 
	# which bin
	bin_indices = np.digitize(np.log10(flat_r_set), bin_edges)

	# Create an array of zeros, which has three rows. The first row contains the
	# radius values used to calculate the radially averaged correlation or 
	# structure function, the second row contains the radially averaged
	# correlation or structure function corresponding to each radius 
	# value, and the third row will hold the number of data points used
	# to calculate that part of the function.
	sf_r = np.zeros((3,nb))
	
	# Set the third row of the array to one for now
	sf_r[2,:] = 1.0

	# We now need to cycle through each of the bins, to calculate the value of
	# the function for that bin
	for i in range(1, nb + 1):
		# Find all of the radius values that are in this bin
		r_i_bin = flat_r_set[bin_indices == i]

		# Check to see if this bin is empty
		if len(r_i_bin) > 0:
			# Calculate the number of elements in this bin
			sf_r[2,i - 1] = len(r_i_bin)

			# Since this bin is not empty, it makes sense to calculate the 
			# average of all radius values in this bin
			sf_r[0,i - 1] = np.sum(r_i_bin, dtype = np.float64) / sf_r[2,i - 1]

			# Calculate the radially averaged function for these radius values
			sf_r[1,i - 1] = np.sum(flat_sf[bin_indices == i], dtype =\
			 np.float64) / sf_r[2,i - 1]

			# Print a message to the screen to show what is happening
			print 'Sfr: Radial average for bin {} complete'.format(i)
			
	# The radially averaged function, and radius value for each data point have
	# now been calculated, so return them to the calling function
	return sf_r