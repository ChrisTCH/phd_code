#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the fractalcube.pro IDL code written by     #
# Alexy Chepurnov, and available at                                            #
# http://www.astro.wisc.edu/~lazarian/code.html. This function generates a     #
# three dimensional fBm fractal cloud with fractal index beta.                 #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alexy Chepurnov)          #
# Start Date: 4/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, and sys to exit if a problem occurs
import numpy as np
import sys

# Define the function fractalcube, which generates a three dimensional fractal
# cloud with fractal index beta.
def fractalcube(beta, seed = 1, size = 128):
	'''
	Description
		This function generates a three dimensional fractal fBm cloud with
		fractal index beta.
	
	Required Input
		beta: A float value. Specifies the exponent to use in the spectral 
			  density function.
		seed: A positive integer, used as a seed value in generating random 
			  numbers. The default value is 1.
		size: A integer, representing the length of one side of the produced
			  data cube in pixels. This must be even. If an even number is not
			  provided, then the next largest even number is used. The default
			  value is 128. Do not use very large values, as this will greatly
			  increase the computation time.
	
	Output
		frac_cube: A numpy array. This is the generated three dimensional 
				   fractal cube, which has fractal index beta, and size pixels
				   along each dimension.
	'''

	# In case a decimal value is specified for the seed, convert it to an 
	# integer
	seed = int(seed)

	# Print a message to the screen to state what seed value is being used
	print "Fractalcube: Seed value used is {}".format(seed)
	
	# Set the seed for the random number generator
	np.random.seed(seed)

	# In case a decimal value is specified for the size, take the floor (lowest
	# integer) to convert it to an integer.
	size = int(np.floor(size))

	# Check to see if the given size is even
	if ( (size % 2) == 1):
		# In this case the size specified is odd, so print an error message
		# to the screen.
		print "Fractalcube: WARNING: Size must be even! Set to {}".\
		format(size + 1.0)
		
		# Add one to the given size value, to make sure it is even
		ndim = size + 1
	else:
		# In this case, the given size is even, so nothing needs to be done
		ndim = size

		# Print a message to the screen saying what size value is being used
		print 'Fractalcube: Size value used is {}'.format(size) 

	# Create an empty data array that will hold all of the values for the
	# three dimensional fractal cube.
	data = np.zeros((ndim,ndim,ndim), dtype = np.complex128)
	
	# Convert the size variable to a float, to be used in calculations
	fndim = float(ndim)

	# *********  1. half *************************
	# Here we construct half of the data cube that will be used to produce
	# the final fractal cube

	# Create an array of values related to the frequency in the direction of 
	# the first dimension, for half of the array
	fix = np.linspace(0, ndim/2.0, ndim/2.0 + 1)

	# Create an array of values related to the frequency in the direction of
	# the second dimension
	fiy = np.concatenate((np.linspace(0, ndim/2.0 - 1, ndim/2.0),\
	 np.linspace(ndim/2.0, 1, ndim/2.0)))

	# Create an array of values related to the frequency in the direction of
	# the third dimension
	fiz = np.concatenate((np.linspace(0, ndim/2.0 - 1, ndim/2.0),\
	 np.linspace(ndim/2.0, 1, ndim/2.0)))

	# Use meshgrid to obtain a 3D grid specifying the frequency components 
	# at each entry in the array
	fix_mat, fiy_mat, fiz_mat = np.meshgrid(fix, fiy, fiz, indexing = 'ij')

	# Randomly generate a matrix of values, where each value is between 0 and
	# 2 * pi
	phi = 2.0 * np.pi * np.random.uniform(size = (ndim/2.0 + 1, ndim, ndim))

	# Calculate the total frequency at each entry of the array
	frequenz = np.sqrt(fix_mat*fix_mat + fiy_mat*fiy_mat + fiz_mat*fiz_mat)
	
	# Set the zero index entry of the frequency array to a small non-zero
	# number, to prevent a divide by zero error
	frequenz[0,0,0] = 1.0e-6

	# Calculate the amplitude for each entry of the array
	amp = np.sqrt( np.power(1.0/(frequenz/fndim), beta) )

	# Calculate the data value for each entry of the array, using the amplitude
	# and argument phi calculated above. Note that this is a complex number.
	data[0:ndim/2.0 + 1,:,:] = amp*np.cos(phi) + amp*np.sin(phi) * 1.0j

	# # Start by looping over half of the first dimension of the cube
	# for ix in range(ndim/2 + 1):
	# 	# Loop over all entries in the second dimension
	# 	for iy in range(ndim):
	# 		# Loop over all entries in the third dimension
	# 		for iz in range(ndim):
	# 			# Randomly generate a decimal between 0 and 2 * pi
	# 			phi = 2.0 * np.pi * np.random.uniform()
				
	# 			# Calculate the minimum value of ix and ndim - ix. This
	# 			# represents the frequency in the x direction
	# 			fix = min([float(ix), float(ndim-ix)])

	# 			# Calculate the minimum value of iy and ndim - iy. This
	# 			# represents the frequency in the y direction
	# 			fiy = min([float(iy), float(ndim-iy)])

	# 			# Calculate the minimum value of iz and ndim - iz. This
	# 			# represents the frequency in the z direction
	# 			fiz = min([float(iz), float(ndim-iz)])
				
	# 			# Calculate the total frequency at this entry of the array,
	# 			# but only allow the smallest possible value to be 1.0e-6
	# 			frequenz = max([np.sqrt(fix*fix + fiy*fiy + fiz*fiz), 1.0e-6])
				
	# 			# Calculate the amplitude for this entry of the array
	# 			amp = np.sqrt( np.power(1.0/(frequenz/fndim), beta) )

	# 			# Calculate the data value for this entry of the array, using
	# 			# the amplitude and argument phi calculated above. Note that 
	# 			# this is a complex number.
	# 			data[ix,iy,iz] = amp*np.cos(phi) + amp*np.sin(phi) * 1.0j

	# *********  2. half *************************
	# Here we fill in the second half of the data cube, to form the final cube
	# that will be used to produce the fractal cube

	# Start by looping over half of the first dimension of the cube
	for ix in range(ndim/2 + 1):
		# Loop over all entries in the second dimension
		for iy in range(ndim):
			# Loop over all entries in the third dimension
			for iz in range(ndim):
				# Set the current data point to be equal to the complex
				# conjugate of the point on the opposite side of the cube
				data[(ndim-ix) % ndim,(ndim-iy) % ndim,(ndim-iz) % ndim] =\
				np.conj(data[ix,iy,iz])

	# *********  real points *************************
	# Here we force some of the data points in the data array to be real
	# valued, but still expressed as complex numbers (i.e. with zero imaginary
	# part). These points correspond to the corner of the data cube, the 
	# halfway point of the edges, the middle of the faces, and the centre
	# of the cube.

	data[0,0,0] = 0.0 + 0.0j
	data[0,0,ndim/2] = abs(data[0,0,ndim/2]) + 0.0j
	data[0,ndim/2,0] = abs(data[0,ndim/2,0]) + 0.0j
	data[ndim/2,0,0] = abs(data[ndim/2,0,0]) + 0.0j
	data[0,ndim/2,ndim/2] = abs(data[0,ndim/2,ndim/2]) + 0.0j
	data[ndim/2,0,ndim/2] = abs(data[ndim/2,0,ndim/2]) + 0.0j
	data[ndim/2,ndim/2,0] = abs(data[ndim/2,ndim/2,0]) + 0.0j
	data[ndim/2,ndim/2,ndim/2] = abs(data[ndim/2,ndim/2,ndim/2]) + 0.0j

	# Now that the data cube has been constructed, perform an inverse fourier
	# transform of the data. 
	# NOTE: In the IDL convention for inverse FFT, there is no normalisation
	# factor, but the Python convention involves dividing by the number of
	# data points. To ensure the same output as the IDL code, the result
	# of the inverse FFT is multiplied by the number of data points, to
	# undo the effect of the normalisation.
	ffdata = np.fft.ifftn(data) * np.size(data)

	# Now rearrange the order of the array, so that the bottom corner of the
	# inverse transformed data cube is placed in the centre of the cube 
	ffdata = np.fft.fftshift(ffdata)

	# Calculate the unbiased sample variance of the inverse transformed data
	var = np.var(ffdata, ddof = 1)

	# Normalise the fractal data cube using the calculated variance
	frac_cube = ffdata / np.sqrt(var)
	
	# Due to numerical imprecision, there are very small imaginary parts in
	# every entry of the produced array. We are only interested in the real 
	# part, so extract that from the data
	frac_cube = np.real(frac_cube)

	# Return the produced fractal data cube to the caller
	return frac_cube