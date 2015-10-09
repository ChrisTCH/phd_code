#-----------------------------------------------------------------------------#
#                                                                             #
# This code describes a function that is designed to open FITS file images    #
# provided to the function, and produce FITS files and images of local        #
# statistics calculated for the diagnostic quantities, e.g. the skewness of   #
# the polarisation gradient, sparsely sampled across the image. This is       #
# performed for each input FITS file that is given. The calculation of these  #
# quantities will occur in separate functions. The images will then be saved  #
# as FITS files using the given save filename and directory.                  #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 9/9/2015                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script, scipy.stats for calculating statistical quantities
import numpy as np
from astropy.io import fits
from scipy import stats

# Import utility functions
from mat2FITS_Image import mat2FITS_Image

# Define the function that can be imported by a calling code to calculate
# local statistics at pixels distributed on a grid throughout the input image,
# and then save the resultant map as a FITS file
def calc_sparse_stats(input_files, output_filenames, stat_list, box_halfwidths):
	'''
	Description
        This function calculates local statistics for some pixels in an image,
        that are sparsely distributed throughout the image, using the data in
        surrounding pixels. The selected pixels will be located on a rectangular
        grid, and the spacing between the pixels is determined by the size of 
        the box used to calculate statistics, so that the input map is
        sampled as a rate greater than te Nyquist frequency. The statistics that
        are calculated can be specified in the stat_list variable. All of these 
        statistics will be calculated for each input image. Multiple input files
        can be provided, and statistics will be calculated for each image. Maps 
        of the produced statistics will be saved using the given output 
        filenames.
        
    Required Input
        input_files - An array of strings, where each string specifies the 
        			  directory and filename of the FITS image to calculate 
        			  statistics for. Local statistics are calculated for each
        			  file provided in this array.
        output_filenames - An array of strings, where each string specifies the 
        				   directory and filename to use when saving a FITS 
        				   file image of the produced statistic map. An image is
        				   saved for each input file, and each statistic. This
        				   array needs to be the same length as input_files.
        				   Each FITS file that is saved will have the type of 
        				   statistic appended to the output filename provided.
        				   The given strings should not have the .fits extension
        				   present, as this is added in the function.
        stat_list - An array of strings, where each string specifies a statistic
        			that is to be calculated locally for each input image.
        			Allowed values are 'skewness' and 'kurtosis'.
        box_halfwidths - The half-widths of the box to use when calculating the
        				 local statistics, in pixels. This is an array of
        				 positive integers greater than or equal to 1. This 
        				 array must have the same length as input_files. Pixels 
        				 selected for the evaluation of statistics are separated
        				 by half of the box half-width.
                   
    Output
        0 - An error occurred, possibly due to inappropriate input values
        1 - Function ran to completion, and saved FITS images of the calculated
        	statistics maps successfully.
	'''

	# Check to see that the number of output filenames matches the number of
	# input filenames
	if len(input_files) != len(output_filenames):
		# In this case the number of output filenames does not match the number
		# input filenames, so the code cannot proceed. Print an error message
		# to say what has happened
		print 'ERROR: Number of input files and output filenames are different'

		# The code should not proceed in this case, so return 0
		return 0

	# Check to see that the number of box halfwidth values equals the number
	# of input filenames
	if len(input_files) != len(box_halfwidths):
		# In this case the number of box halfwidths to use does not match the
		# number of input filenames, so the code cannot proceed. Print an 
		# error message to say what has happened.
		print 'ERROR: Number of input files and box halfwidths are different'

		# The code should not proceed in this case, so return 0
		return 0

	# Create a list of valid strings that can be given to the stat_list variable
	valid_stats = ['skewness', 'kurtosis'] 

	# Check to see that valid statistics were provided
	if len(list(set(valid_stats).intersection(stat_list))) == 0:
		# In this case the array of strings given to the function does not
		# provide any valid statistics to calculate, so print an error message
		# to the screen
		print 'ERROR: No valid statistics provided to the function'

		# The code should not proceed in this case, so return 0
		return 0

	# Check to see that the values given for the box halfwidths are positive
	# integers greater than or equal to 1
	if np.any(box_halfwidths < 1):
		# In this case a value given for the box halfwidth is invalid, so
		# print an error message to the screen
		print 'ERROR: At least one of the values for box half-width is invalid'

		# The code should not proceed in this case, so return to 0
		return 0

	# Loop over the given input files, so that we can calculate statistics for
	# each one
	for i in range(len(input_files)):
		# Print a message to show that calculations are starting for the current
		# input file
		print 'Calculations starting for {}'.format(input_files[i])

		# Open the CGPS FITS file for the current resolution
		fits_file = fits.open(input_files[i])

		# Obtain the header of the primary HDU for the data
		fits_hdr = fits_file[0].header

		# Extract the data from the FITS file, which is held in the primary HDU
		fits_data = fits_file[0].data

		# Print a message to the screen saying that the data was successfully 
		# extracted
		print 'CGPS data successfully extracted from the FITS file.'

		# Extract the size of each pixel from the header. This is the length of 
		# each side of the pixel (assumed to be square), in degrees. 
		pix_size_deg = fits_hdr['CDELT2']

		# Extract the number of pixels along the horizontal axis of the image
		num_pix_horiz = fits_hdr['NAXIS1']

		# Extract the number of pixels along the vertical axis of the image
		num_pix_vert = fits_hdr['NAXIS2']

		# Create a new header object, that will become the header for the 
		# statistics map that is produced
		stats_hdr = fits_hdr

		# Calculate the spacing between the grid points that will be used to
		# calculate local statistics. The floor is taken just to make sure it
		# is an integer.
		grid_space = int(np.floor(box_halfwidths[i]/2))

		# Calculate the number of grid points to place along the horizontal axis
		# A ceiling is taken to make sure we get the right integer number of 
		# grid points to use
		num_grid_horiz = int(np.ceil(num_pix_horiz / grid_space))

		# Calculate the number of grid points to place along the vertical axis
		# A ceiling is taken to make sure we get the right integer number of 
		# grid points to use
		num_grid_vert = int(np.ceil(num_pix_vert / grid_space))

		# Construct an array, where each entry gives the index value along the
		# horizontal axis of the grid points
		horiz_grid_indices = grid_space * np.array(range(num_grid_horiz))

		# Construct an array, where each entry gives the index value along the
		# vertical axis of the grid points
		vert_grid_indices = grid_space * np.array(range(num_grid_vert))

		# At this point the grid has been constructed, and we just need to 
		# modify some of the FITS header values so that the constructed
		# statistics maps can be saved
		# Change the NAXIS keywords, that specify how many pixels are along
		# each axis of the image
		stats_hdr['NAXIS1'] = num_grid_horiz
		stats_hdr['NAXIS2'] = num_grid_vert

		# Change the CDELT keywords, that specify the spacing between pixels
		# in degrees
		stats_hdr['CDELT1'] = stats_hdr['CDELT1'] * grid_space
		stats_hdr['CDELT2'] = stats_hdr['CDELT2'] * grid_space

		# Change the CRPIX keywords, that specify which pixel is the reference
		# pixel in the image
		stats_hdr['CRPIX1'] = np.floor(num_grid_horiz / 2.0) + 1
		stats_hdr['CRPIX2'] = np.floor(num_grid_vert / 2.0) + 1

		# Create a dictionary that will hold the arrays corresponding to each 
		# statistic
		stat_dict = {}

		# Create new numpy arrays, that have the same size as the input array. 
		# These will be used to store the calculated local statistics. The 
		# arrays are stored in a dictionary, with each statistic having its own 
		# array
		for stat in stat_list:
			# Assign an empty array to the current statistic. Note that the data
			# type is 32-bit float, since all of the input arrays have that data
			# type
			stat_dict[stat] = np.zeros((num_grid_vert,num_grid_horiz)\
				, dtype = np.float32)

		# Loop over all of the pixels at which we will evaluate statistics,
		# starting by looping over each row
		for j in range(num_grid_vert):
			# Loop over all of the pixels in this row
			for k in range(num_grid_horiz):
				# Extract the value of the data at the current pixel
				pix_value =fits_data[vert_grid_indices[j],horiz_grid_indices[k]]

				# Calculate the index of the uppermost pixels to be included in 
				# the local box. Note that the minus sign is because Python 
				# indexes so that the pixels highest in the image have the 
				# lowest index value
				upper_index = vert_grid_indices[j] - box_halfwidths[i]

				# Calculate the index of the lowermost pixels to be included in 
				# the local box
				lower_index = vert_grid_indices[j] + box_halfwidths[i]

				# Calculate the index of the rightmost pixels to be included in 
				# the local box
				right_index = horiz_grid_indices[k] + box_halfwidths[i]

				# Calculate the index of the leftmost pixels to be included in 
				# the local box
				left_index = horiz_grid_indices[k] - box_halfwidths[i]

				# Check that the index of the uppermost pixel to include in the
				# local box is valid, i.e. not negative
				if upper_index < 0:
					# In this case the value of the upper index doesn't make 
					# sense, so reset it to zero
					upper_index = 0

				# Check that the index of the leftmost pixel to include in the 
				# local box is valid, i.e. not negative
				if left_index < 0:
					# In this case the value of the left index doesn't make 
					# sense, so reset it to zero
					left_index = 0

				# Check that the index of the lowermost pixel to include in the
				# local box is valid, i.e. that it does not extend beyond the
				# boundaries of the image
				if lower_index >= num_pix_vert:
					# In this case the value of the lower index doesn't make 
					# sense, so reset it to the largest possible index for the 
					# vertical axis
					lower_index = num_pix_vert - 1

				# Check that the index of the rightmost pixel to include in the
				# local box is valid, i.e. that it does not extend beyond the
				# boundaries of the image
				if right_index >= num_pix_horiz:
					# In this case the value of the lower index doesn't make 
					# sense, so reset it to the largest possible index for the 
					# horizontal axis
					right_index = num_pix_horiz - 1

				# Extract the data in the local box around the current
				# pixel we are studying
				local_data = fits_data[upper_index:lower_index+1,\
				 left_index:right_index+1]

				# Flatten the local data into a one-dimensional array
				local_data_flat = local_data.flatten()

				# Find where all of the NaN values are in the local data
				NaN_position = np.isnan(local_data_flat)

				# Extract the values in the local data that are not NaN
				local_data_no_nan =\
				 local_data_flat[np.logical_not(NaN_position)]

				# Check that most of the data in the box is not NaN
				if (len(local_data_flat)/2.0)>len(local_data_no_nan):
					# In this case more than half of the data is NaN, so
					# record NaN for any statistics that are to be calculated
					for stat in stat_list:
						(stat_dict[stat])[j,k] = float('nan')
				else:
					# Check to see if the skewness of the local area needs to be
					# calculated
					if 'skewness' in stat_list:
						# Calculate the skewness of the local data that has had 
						# the NaN values removed, and store it in the 
						# corresponding array
						(stat_dict['skewness'])[j,k] =\
						 stats.skew(local_data_no_nan)

					elif 'kurtosis' in stat_list:
						# Calculate the kurtosis of the local data that has had 
						# the NaN values removed, and store it in the 
						# corresponding array
						(stat_dict['kurtosis'])[j,k] =\
						 stats.kurtosis(local_data_no_nan)

				# When the code reaches this point, all of the required 
				# statistics have been calculated for this pixel

			# When the code reaches this point, statistics have been calculated 
			# for all of the pixels in this row

			# Every 20 rows, print out a message to show where the code is up to
			if (j+1) % 20 == 0:
				# Another 20 rows have been completed, so print a message
				print '{} rows calculated'.format(j+1)

		# When the code reaches this point, statistics have been calculated for
		# all of the pixels, for this particular value of the final resolution

		# Next, we want to save the produced maps of the local statistics

		# Loop over all of the statistics that were calculated
		for stat in stat_list:
			# Convert the matrix of values for this statistic into a FITS file, 
			# and save the result using the same header as the input CGPS data. 
			# The FITS file is saved in the same location as the CGPS data
			stat_FITS = mat2FITS_Image(stat_dict[stat], stats_hdr,\
			 output_filenames[i] + '_{}_sparse'.format(stat) + '.fits')

			# Print a message to the screen, to show that a FITS file was 
			# produced for the current statistic
			print 'FITS file saved for {}'.format(stat)

		# At this point, all FITS files have been saved for the required 
		# statistics, so print a message to the screen about this
		print 'All FITS files saved for {}'.format(input_files[i])

	# When the code reaches this point, all of the FITS files have been saved
	# Print a message stating this
	print 'All FITS files saved successfully'

	# Since the code has run successfully, return 1
	return 1