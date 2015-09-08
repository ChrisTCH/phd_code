#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives an array of data for #
# a group of points on an image, the longitude and latitude of the points,    #
# and the edges of the angular separation bins that are to be used to         #
# calculate the structure function of the data. The computed structure        #
# function is returned as an array to the calling function.                   #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 7/8/2014                                                        #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, Eli Bressert's fast angular separation code
import numpy as np
from fast_angular_sep_Eli import separation

# Define the function calc_Struc_Func, which will calculate the structure 
# function of the given data. 
def calc_Struc_Func(data_array, lon_coords, lat_coords, bin_edges,\
 ang_sep_centres):
    '''
    Description
        This function calculates the structure function for the given data 
        from an image, which is returned as an array. The structure function is
        calculated using a set of discrete points, rather than every pixel in
        the image.
        
    Required Input
        data_array - A Numpy array containing the values of the data
                     for the selected pixels. 
        lon_coords - A Numpy array specifying the Galactic longitude of
                     each of the selected pixels. Must have the same length
                     as data_array. Values in degrees.
        lat_coords - A Numpy array specifying the Galactic latitude of
                     each of the selected pixels. Must have the same length
                     as data_array. Values in degrees.
        bin_edges - A Numpy array specifying the edges of the angular separation
                    bins to be used in calculating the structure function.
        ang_sep_centres - A Numpy array specifying the centres of the angular
                          separation bins to be used in calculating the 
                          structure function. Length must be one less than 
                          that of bin_edges.
                   
    Output
        struc_func - A Numpy array specifying the structure function for the
                     data. Has the same length as ang_sep_centres. 
    '''

    # Firstly, we need to create a dictionary that will be used in constructing the
    # structure function. This dictionary will have a key for each angular
    # separation bin, and each key will have an array in which a running total
    # of the values required to calculate the structure function is stored, along
    # with a number specifying how many pixels were used in the calculation.
    # After these numbers have been finalised, dividing the running total by
    # the number of pixels will produce the required structure function.
    data_dict = {}
    
    # Now that the empty dictionary has been created, we need to create a
    # key for each angular separation bin, and assign each key an array.
    for sep in ang_sep_centres:
        # Assign a Numpy array to the dictionary, for this angular
        # separation. The key used to access each array is the centre of the
        # angular separation bin.
        data_dict['{}'.format(sep)] = np.array([0.0,0.0])

    # To calculate angular separations, it is necessary to combine the 
    # latitude and longitude arrays into one array, and to convert the units
    # from degrees to radians. Latitude values must be in the first column,
    # and longitude values must be in the second column.
    rad_coord_array = np.deg2rad(np.column_stack([lat_coords,lon_coords]))

    # Iterate over the selected pixels, so that we can calculate the 
    # angular separation between pixels, and the quantity required to calculate
    # the structure function.
    for i in range(len(data_array)):
    	# Calculate the square of the difference in the data, between the 
        # current pixel and all other pixels. This is required to calculate the
        # structure function.
    	data_diff_array = np.power(data_array[i] - data_array, 2.0)

    	# Extract the latitude and longitude of the currently selected pixel,
    	# as an array, in radians. This just extracts the i-th row of the 
    	# coordinate array.
    	pix_lat_lon = rad_coord_array[i]

    	# Tile this row of the coordinate array to create a new array that
    	# has the same shape as the coordinate array, where every row is the
    	# same.
    	pix_lat_lon = np.tile(pix_lat_lon,(len(lat_coords) ,1))

    	# Calculate the angular separation between this pixel and all other
    	# pixels, in radians.
    	rad_ang_sep = separation(pix_lat_lon,rad_coord_array)

    	# Convert the angular separations into degrees
    	deg_ang_sep = np.rad2deg(rad_ang_sep)

    	# Find out which angular separation bin each pixel belongs to, using
    	# the digitize function of Numpy. This returns the bin index that the
    	# pixel belongs to. The bin index values start at 1. If a pixel does
    	# not fit into a bin, then it is given an index outside the acceptable
    	# range of values, and is not included in the later averaging.
    	bin_allocation = np.digitize(deg_ang_sep, bin_edges)

        # Now that we know which bin each pixel belongs to, it is possible to 
        # allocate the quantities needed to calculate structure function 
        # to each bin.
        # Loop over the values in the angular separation bin allocation array,
        # so that calculated quantities can be allocated to the correct 
        # angular separation key in the dictionary
        for j in range(1,len(bin_edges)):
            # Calculate the number of pixels in the j-th bin
            num_pix_j = len(data_diff_array[bin_allocation == j])

            # Add the sum of the relevant quantity for points in the j-th
            # bin to the running total for the data.
            data_dict['{}'.format(ang_sep_centres[j - 1])][0] +=\
            np.sum(data_diff_array[bin_allocation == j], dtype = np.float64)

            # Add the number of pixels for the j-th bin to the running total
            # for the data
            data_dict['{}'.format(ang_sep_centres[j - 1])][1] += num_pix_j

    	# # Now that we know which bin each pixel belongs to, it is possible to 
    	# # average the quantities needed to calculate structure functions 
    	# # within each bin. 
    	# # Perform a bin average for each angular separation bin, for the
    	# # data, and store the result in the i-th column of the array
    	# data_SF_mat[:,i] = np.array([np.mean(data_diff_array[\
    	# 	bin_allocation == j]) for j in range(1,len(bin_edges))])

    	# Print a message to the screen after a certain number of pixels have
    	# been processed, so that it is clear what the program is doing.
    	if (i+1) % 500 == 0:
    		# A certain number of pixels have been processed, so print a
    		# message to the screen.
    		print "{} pixels have been processed.".format(i+1)

    # Now that the structure function dictionary has been filled with
    # data for every pixel, create an array that contains information on the 
    # amplitude of the structure function for each angular separation. This is 
    # done by dividing the running total for the quantity by the total number
    # of pixel pairs for that bin, for each key in a dictionary.
    struc_func = np.array([data_dict['{}'.format(sep)][0]\
     / data_dict['{}'.format(sep)][1] for sep in ang_sep_centres])

    # The structure functions has now been successfully calculated, so print
    # a message to the screen stating this.
    print 'Structure function successfully calculated.'

    # Now return the structure function to the calling function
    return struc_func