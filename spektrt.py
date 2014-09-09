#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the spektrt.pro IDL code written by Alex    #
# Lazarian, and available at http://www.astro.wisc.edu/~lazarian/code.html.    #
# This function calculates the spectrum of a 2D image or 3D data cube.         #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alex Lazarian)            #
# Start Date: 3/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling
import numpy as np

# Import the setks function to calculate the wavenumbers used in calculating
# the spectrum
from setks import setks

# Define the function spektrt, which calculates the spectrum of an image or 
# data cube.
def spektrt(source, nb, no_pi = False):
	'''
	Description
		This function calculates the spectrum of a 2D image or 3D data cube. It
		depends upon the setks function to determine the wavenumber values at
		which the spectrum is calculated.
	
	Required Input
		source: The image or data cube for which a spectrum is to be calculated.
				Must be one, two or three dimensional. Must be a numpy array.
		nb: The number of bins to use when calculating the spectrum.
		no_pi: A boolean value. If True, then k=1/lambda is used, and if False,
			   then k = 2 * pi / lambda is used as the definition of the 
			   wavenumber.
	
	Output
		pk: A numpy array with three rows. The first row contains the k values
			used to calculate the power spectrum, the second row contains the
			the power spectrum value for corresponding to each k value, and the 
			third row will hold the number of data points used to calculate that
			part of the spectrum.
	'''

	# Take a fast fourier transform of the input image or data cube
	den = np.fft.fftn(source)

	# For all dimensions of the image, extract the smallest length of any 
	# dimension
	up = min(np.shape(den))

	# Check to see whether the wavenumber is defined as k = 1/lambda, or as
	# k = 2 * pi / lambda
	if no_pi == True:
		# In this case we are using the k = 1/lambda definition.
		# Run the setks function to calculate the k values for the array
		k_set = setks(np.shape(den), no_pi = no_pi)
		
		# Set the logarithm of the minimum and maximum k values that are to be
		# used in calculating the spectrum
		k_min = 0.0
		k_max = np.log10(up/2.0)
	else:
		# In this case we are using the k = 2 * pi/lambda definition.
		# Run the setks function to calculate the k values for the array
		k_set = setks(np.shape(den))

		# Set the logarithm of the minimum and maximum k values that are to be 
		# used in calculating the spectrum 
		k_min = np.log10(2.0 * np.pi)
		k_max = np.log10(np.pi *up)

	# Flatten the array of k values, so that we can properly sort data points
	# into bins that are equally spaced in k
	flat_k_set = k_set.flatten()

	# Flatten the fourier transform of the input data, so that each data 
	# point is matched with the corresponding value of k
	flat_den = den.flatten()

	# Set the zero frequency value of the k array to some small number
	flat_k_set[0] = 0.0001

	# Construct a histogram of the k values being used to calculate the
	# spectrum, just so we can get the bin edges of the histogram
	klist, bin_edges = np.histogram(np.log10(flat_k_set), bins = nb, range =\
	(k_min, k_max))

	# Now we need to figure out which elements of the k value array are in 
	# which bin
	bin_indices = np.digitize(np.log10(flat_k_set), bin_edges)

	# Create an array of zeros, which has three rows. The first row contains the
	# k values used to calculate the power spectrum, the second row contains the
	# the power spectrum value for corresponding to each k value, and the 
	# third row will hold the number of data points used to calculate that part
	# of the spectrum.
	pk = np.zeros((3,nb))
	
	# Set the third row of the array to one for now
	pk[2,:] = 1.0
	
	# We now need to cycle through each of the bins, to calculate the value of
	# the power spectrum for that bin
	for i in range(1, nb + 1):
		# Find all of the k values that are in this bin
		k_i_bin = flat_k_set[bin_indices == i]

		# Check to see if this bin is empty
		if len(k_i_bin) > 0:
			# Calculate the number of elements in this bin
			pk[2,i - 1] = len(k_i_bin)

			# Since this bin is not empty, it makes sense to calculate the 
			# average of all k values in this bin
			pk[0,i - 1] = np.sum(k_i_bin) / pk[2,i - 1]

			# Calculate the power spectrum for these k values
			pk[1,i - 1] = np.sum(np.power(np.abs(flat_den[bin_indices == i]),\
			 2.0)) / pk[2,i - 1]

			# Print a message to the screen to show what is happening
			print 'Spektrt: Power spectrum for bin {} complete'.format(i)
	
	# The power spectrum, and k value for each power spectrum data point have
	# now been calculated, so return them to the calling function
	return pk