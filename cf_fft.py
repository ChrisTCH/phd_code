#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the cf_fft.pro IDL code written by Alex     #
# Lazarian, and available at http://www.astro.wisc.edu/~lazarian/code.html.    #
# This function calculates the correlation function of an image or data cube,  #
# using a fast fourier transform.                                              #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alex Lazarian)            #
# Start Date: 3/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, and sys to exit if a problem occurs
import numpy as np
import sys

# Define the function cf_fft, which calculates the correlation function of an
# image or data cube.
def cf_fft(field, no_fluct = False, mirror = False):
	'''
	Description
		This function calculates the correlation function of an image or data
		cube, using a fast fourier transform.
	
	Required Input
		field: A numpy array containing an image or data cube. Must be one, two
			   or three dimensional. 
		no_fluct: A boolean value. If False, then the mean value of the data
				  is subtracted from the data before calculating the correlation
				  function. If True, then there is no subtraction of the mean.
		mirror: A boolean value. If True, then the mirror image of the 
				correlation function is returned. If False, then nothing happens
	
	Output
		acf: A numpy array with the same shape as the input image or data cube.
			 This array gives the values for the auto-correlation function of
			 the data.
	'''

	# Determine the shape of the input data
	sizefield = np.shape(field)

	# Calculate the length of the first dimension of the data
	N1 = sizefield[0]

	# Initialise variables to hold the lengths of the second and third 
	# dimensions, if they are present.
	N2 = 1
	N3 = 1

	# Check to see if the data has 2 or more dimensions
	if len(sizefield) >= 2:
		# In this case, there are two or more dimensions, so extract the length
		# of the second dimension
		N2 = sizefield[1]

	# Check to see if the data has 3 or more dimensions
	if len(sizefield) >= 3:
		# In this case, there are three or more dimensions, so extract the 
		# length of the third dimension
		N3 = sizefield[2]

	# Check to see if the data has four or more dimensions    
	if len(sizefield) >= 4:
		# In this case there are four or more dimensions, so print an error
		# message to the screen.
		print 'Well, please no more than 3 dimensions !'
		
		# Stop the function from continuing on, since this function cannot
		# handle mirroring the correlation function for data that is four 
		# dimensional or higher.
		sys.exit()

	# Check to see whether the mean should be subtracted from the data before
	# calculating the correlation function
	if no_fluct == False:
		# In this case we need to subtract the mean of the data before 
		# calculating the correlation function of the data
		field1 = field - np.mean(field)
	else:
		# In this case we do not subtract the mean of the data before 
		# calculating the correlation function, so do nothing to the data
		field1 = field

	# Calculate the fourier transform of the data
	fftfield = np.fft.fftn(field1)

	# Multiply the fourier transform of the data by it's complex conjugate
	ps = fftfield * np.conj(fftfield)

	# Perform an inverse fourier transform on the result obtained by multiplying
	# the fourier transform with its conjugate
	acf = np.fft.ifftn(ps)

	# Check to see if the mirror image of the auto-correlation function needs
	# to be returned.
	if mirror == True:
		# Let's do here the trick of producing the mirror images
		# Create some variables that will be used to create the mirror image
		nyq1 = N1/2.0
		nyq2 = N2/2.0
		nyq3 = N3/2.0
		
		# Loop over the third dimension, to reorder elements in the array
		for i3 in range(N3):
			# Create a new variable, to handle indexing of the mirrored array
			i3k = i3

			# Check to see if the iteration along the third dimension is past 
			# the halfway point
			if i3 >= nyq3:
				# In this case we are past the halfway point, so calculate i3k
				# to take this into account
				i3k = N3 - i3
			
			# Loop over the second dimension, to reorder elements
			for i2 in range(N2):
				# Create a new variable, to handle indexing of the 
				# mirrored array
				i2k = i2

				# Check to see if the iteration along the second dimension is 
				# past the halfway point
				if i2 >= nyq2:
					# In this case we are past the halfway point, so calculate 
					# i2k to take this into account
					i2k = N2 - i2
				
				# Loop over the first dimension, to reorder elements
				for i1 in range(N1):
					# Create a new variable, to handle indexing of the 
					# mirrored array
					i1k = i1

					# Check to see if the iteration along the first dimension is
					# past the halfway point
					if i1 >= nyq1:
						# In this case we are past the halfway point, so 
						# calculate i1k to take this into account
						i1k = N1 - i1
						
					# Now that i1k, i2k and i3k have been determined, update
					# entries of the auto-correlation data to make it appear
					# mirrored
					acf[i1,i2,i3] = acf[i1k,i2k,i3k]
	
	# Due to numerical imprecision, there may be small imaginary parts in
	# every entry of the produced array. We are only interested in the real 
	# part, so extract that from the data
	acf = np.real(acf)

	# Return the auto-correlation function to the caller
	return acf