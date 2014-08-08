#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives an array of data for #
# a group of points on an image, the longitude and latitude of the points,    #
# and the edges of the angular separation bins that are to be used to         #
# calculate the structure function of the data. The computed structure        #
# function is returned as an array to the calling function.                   #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 7/8/2014                                                        #
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

    # Firstly, we need to create an array that will be used in constructing the
    # structure function. This array will have a row for each angular
    # separation bin, and a column for every pixel. After the array has
    # been filled, averaging over the columns will produce the required 
    # structure function.
    data_SF_mat = np.zeros((len(ang_sep_centres),len(data_array)))
    
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
    	# average the quantities needed to calculate structure functions 
    	# within each bin. 
    	# Perform a bin average for each angular separation bin, for the
    	# data, and store the result in the i-th column of the array
    	data_SF_mat[:,i] = np.array([np.mean(data_diff_array[\
    		bin_allocation == j]) for j in range(1,len(bin_edges))])

    	# Print a message to the screen after a certain number of pixels have
    	# been processed, so that it is clear what the program is doing.
    	if (i+1) % 500 == 0:
    		# A certain number of pixels have been processed, so print a
    		# message to the screen.
    		print "{} pixels have been processed.".format(i+1)

    # For some pixels, there may be NaN values for certain angular separations,
    # if there are no other pixels that are separated from the pixel by the
    # angular separation. These NaN values need to be ignored, so that they
    # do not affect the averaging process. To achieve this, the data matrix will
    # be masked, so that NaN values are ignored.
    data_SF_mat = np.ma.masked_array(data_SF_mat, np.isnan(data_SF_mat))

    # Now that the structure function matrix has been filled with
    # data for every pixel, average over the columns over the matrix, to
    # calculate the structure function for the observations and noise.
    struc_func = np.mean(data_SF_mat, axis = 1)

    # The structure functions has now been successfully calculated, so print
    # a message to the screen stating this.
    print 'Structure function successfully calculated.'

    # Now return the structure function to the calling function
    return struc_func