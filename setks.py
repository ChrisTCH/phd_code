#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the setks.pro IDL code written by Alexy     #
# Chepurnov, and available at http://www.astro.wisc.edu/~lazarian/code.html.   #
# This function creates the arrays of k values to be used when calculating the #
# spectrum of a 2D image or 3D data cube.                                      #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alexy Chepurnov)          #
# Start Date: 3/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, and sys to exit if a problem occurs
import numpy as np
import sys

# Define the function setks, which constructs an array of k values to be used
# when calculating the spectrum of an image or data cube.
def setks(sizeden, no_pi = False, anti_alias = False):
	'''
	Description
		This function constructs an array of k (wavenumber) values to be used
		when calculating the spectrum of a 2D image or 3D data cube. 
	
	Required Input
		sizeden: A numpy array specifying the length of each dimension of the
				 image or data cube for which a spectrum is being calculated.
				 There should be three or less entries in this array.
		no_pi: A boolean value. If True, then k=1/lambda is used, and if False,
			   then k = 2 * pi / lambda is used as the definition of the 
			   wavenumber.
		anti_alias: A boolean value. If True, then anti-aliasing is used to 
					construct the array of k values. If False, then 
					anti-aliasing is not used. 
	
	Output
		k_set: The array of k values to be used in constructing the spectrum.
	'''
	
	# Extract the length of the first dimension of the image or data cube for
	# which a spectrum is being calculated.
	# NOTE: We may need to work backwards, to make sure that we get the
	# axes around the right way. i.e. we match the axes of the image and k
	# array correctly
	N1 = sizeden[0]

	# Initialise variables to hold the lengths of the second and third 
	# dimensions, if they are present.
	N2 = 1
	N3 = 1

	# Check to see if the data for which a spectrum is being taken has 2 or
	# more dimensions
	if len(sizeden) >= 2: 
		# In this case, there are two or more dimensions, so extract the length
		# of the second dimension
		N2 = sizeden[1]

	# Check to see if the data for which a spectrum is being taken has 3 or
	# more dimensions
	if len(sizeden) >= 3:
		# In this case, there are three or more dimensions, so extract the 
		# length of the third dimension
		N3 = sizeden[2]

	# Check to see if the data for which a spectrum is being taken has four or
	# more dimensions    
	if len(sizeden) >= 4:
		# In this case there are four or more dimensions, so print an error
		# message to the screen.
		print 'Well, please no more than 3 dimensions !'    
		
		# Stop the function from continuing on, since this function cannot
		# handle constructing the wavenumber values for data that is four 
		# dimensional or higher.
		sys.exit()

	# Check to see whether the definition of the wavenumber will include 
	# the factor of 2 * pi or not
	if no_pi == True:
		# In this case the definition of the wavenumber is k = 1/lambda, so
		# a factor of 2 * pi is not required.
		# NOTE: Python's floats have double point precision, so this is the 
		# same as the IDL code, which specifies everything to double precision.
		C = 1.0
	else:
		# In this case the definition of the wavenumber is k = 2*pi/lambda, so
		# a factor of 2 * pi is required.
		C = 2.0 * np.pi

	# Create an empty array to hold all of the k values that are to be 
	# calculated
	k_set = np.zeros((N1,N2,N3))

	# Create some variables that will be used in calculating the k values 
	# for the array
	nyq1 = N1/2.0
	nyq1_1 = nyq1 - 1.0
	
	nyq2 = N2/2.0
	nyq2_1 = nyq2 - 1.0
	
	nyq3 = N3/2.0
	nyq3_1 = nyq3 - 1.0

	# Create an array of values related to the wavenumber in the direction of
	# the first dimension
	i1k = np.concatenate((np.linspace(0,nyq1_1,nyq1), np.linspace(nyq1,1,nyq1)))

	# Check to see if the data for which a spectrum is being taken has 2 or
	# more dimensions
	if len(sizeden) >= 2:
		# Create an array of values related to the wavenumber in the direction of
		# the second dimension
		i2k = np.concatenate((np.linspace(0,nyq2_1,nyq2),\
		 np.linspace(nyq2,1,nyq2)))
	else:
		# In this case the data is one-dimensional, so just make i2k an array
		# whose only entry is a zero
		i2k = np.array([0])

	# Check to see if the data for which a spectrum is being taken has 3 or
	# more dimensions
	if len(sizeden) >= 3:
		# Create an array of values related to the wavenumber in the direction of
		# the third dimension
		i3k = np.concatenate((np.linspace(0,nyq3_1,nyq3), np.linspace(nyq3,1,nyq3)))
	else:
		# In this case the data is two-dimensional, so just make i3k an array
		# whose only entry is a zero
		i3k = np.array([0])

	# Use meshgrid to obtain a 3D grid specifying the components of the 
	# wavevector at each entry in the array
	i1k_mat, i2k_mat, i3k_mat = np.meshgrid(i1k, i2k, i3k, indexing = 'ij')

	# Now that all of i1k, i2k, i3k matrices have been calculated, enter the
	# wavenumber array for this wavevector array 
	k_set = C * np.sqrt(np.power(i1k_mat, 2.0) + np.power(i2k_mat, 2.0) +\
	 np.power(i3k_mat, 2.0))

	# # Start calculating k values by looping over the third dimension of the data
	# # This loop iterates from 0 to N3 - 1, inclusive.
	# for i3 in range(N3):
	# 	# Create a new variable, related to the wavenumber in the direction of 
	# 	# the third dimension.
	# 	i3k = i3

	# 	# Check to see if the iteration along the third dimension is past the
	# 	# halfway point
	# 	if i3 >= nyq3:
	# 		# In this case we are past the halfway point, so calculate i3k
	# 		# to take this into account
	# 		i3k = N3 - i3

	# 	# Now loop over the second dimension, from 0 to N2 - 1 inclusive.    
	# 	for i2 in range(N2):
	# 		# Create a new variable, related to the wavenumber in the direction
	# 		# of the second dimension
	# 		i2k = i2

	# 		# Check to see if the iteration along the second dimension is past
	# 		# the halfway point
	# 		if i2 >= nyq2: 
	# 			# In this case we are past the halfway point, so calculate i2k
	# 			# to take this into account
	# 			i2k = N2 - i2

	# 		# Now loop over the first dimension, from 0 to N1 - 1 inclusive.
	# 		for i1 in range(N1):
	# 			# Create a new variable, related to the wavenumber in the 
	# 			# direction of the first dimension
	# 			i1k = i1

	# 			# Check to see if the iteration along the first dimension is
	# 			# past the halfway point
	# 			if i1 >= nyq1:
	# 				# In this case we are past the halfway point, so calculate
	# 				# i1k to take this into account
	# 				i1k = N1 - i1
				
	# 			# Now that all of i1k, i2k, i3k have been calculated, enter the
	# 			# wavenumber for this entry of the array
	# 			k_set[i1,i2,i3] = C * np.sqrt(np.power(i1k, 2.0) +\
	# 			np.power(i2k, 2.0) + np.power(i3k, 2.0))

	# When the code reaches this point, all of the k values have been calculated
	# Check to see if the k values are to be anti-aliased
	if anti_alias == True:
		# In this case the k values need to be anti-aliased, which involves 
		# setting certain elements to zero if they are too big
		k_set[k_set >= max([nyq1,nyq2,nyq3])] = 0.0

	# Reshape the final array of k values, so that it has the same number of
	# dimensions as the data array for which a spectrum is being calculated.
	k_set = np.squeeze(k_set)

	# Return the final array of k values to the calling function
	return k_set