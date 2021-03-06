#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the        #
# observed Stokes Q and U for a group of points on an image, the noise in     #
# Stokes Q and U these points, the longitude and latitude of the points, and  #
# the edges of the angular separation bins that are to be used to calculate   #
# the structure functions of the observed, noise, and true complex            #
# polarisation and polarised intensity. All computed structure functions are  #
# returned as arrays to the calling function.                                 #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 31/7/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, Eli Bressert's fast angular separation code
import numpy as np
from fast_angular_sep_Eli import separation

# Define the function calc_Polar_Struc, which will calculate the structure 
# functions of complex polarisation and polarised intensity for the 
# observations, the noise, and the true values, from the observed and noise
# values of Stokes Q and U.
def calc_Polar_Struc(obs_Sto_Q, obs_Sto_U, noise_Sto_Q, noise_Sto_U,\
 lon_coords, lat_coords, bin_edges, ang_sep_centres):
    '''
    Description
        This function calculates the structure functions for the complex
        polarisation and polarised intensity for the observations, the noise,
        and the true values of an image. Each of these structure functions is
        returned as a separate array. The structure functions are calculated
        using a set of discrete points, rather than every pixel in the image.
        
    Required Input
        obs_Sto_Q - A Numpy array containing the values of the observed 
                      Stokes Q for the selected pixels. 
        obs_Sto_U - A Numpy array containing the values of the observed
                      Stokes U for the selected pixels. Must have
                      the same length as obs_Sto_Q.
        noise_Sto_U - A Numpy array containing the values of the noise in 
                        Stokes Q for the selected pixels. Must 
                        have the same length as obs_Sto_Q.
        noise_Sto_Q - A Numpy array containing the values of the noise in 
                        Stokes U for the selected pixels. Must 
                        have the same length as obs_Sto_Q.
        lon_coords - A Numpy array specifying the Galactic longitude of
                     each of the selected pixels. Must have the same length
                     as obs_Sto_Q. Values in degrees.
        lat_coords - A Numpy array specifying the Galactic latitude of
                     each of the selected pixels. Must have the same length
                     as obs_Sto_Q. Values in degrees.
        bin_edges - A Numpy array specifying the edges of the angular separation
                    bins to be used in calculating the structure function.
        ang_sep_centres - A Numpy array specifying the centres of the angular
                          separation bins to be used in calculating the 
                          structure functions. Length must be one less than 
                          that of bin_edges.
                   
    Output
        obs_compl_P_struc - The structure function for the observed complex
                            polarisation. Has the same length as ang_sep_centres
        obs_P_inten_struc - The structure function for the observed polarised
                            intensity. Has the same length as ang_sep_centres
        noise_compl_P_struc - The structure function for the noise in complex
                              polarisation. Has the same length as 
                              ang_sep_centres.
        noise_P_inten_struc - The structure function for the noise in polarised
                              intensity. Has the same length as ang_sep_centres.
        true_compl_P_struc - The structure function for the true complex
                             polarisation. Has the same length as 
                             ang_sep_centres.
        true_P_inten_struc - The structure functions for the true polarised
                             intensity. Has the same length as ang_sep_centres.
    '''

    # Calculate the observed complex polarisation at the pixel locations
    obs_compl_P = obs_Sto_Q + 1.0j * obs_Sto_U

    # Calculate the observed polarised intensity at the pixel locations
    obs_P_inten = np.sqrt(np.power(obs_Sto_Q,2.0) + np.power(obs_Sto_U,2.0) )

    # Calculate the complex polarisation noise for the selected pixels
    noise_compl_P = noise_Sto_Q + 1.0j * noise_Sto_U

    # Calculate the polarised intensity noise for the selected pixels 
    noise_P_inten = np.sqrt(np.power(noise_Sto_Q,2.0) + np.power(noise_Sto_U,2.0) )
    
    # Firstly, we need to create dictionaries that will be used in constructing the
    # structure functions. These dictionaries will have a key for each angular
    # separation bin, and each key will have an array in which a running total
    # of the values required to calculate the structure function is stored, along
    # with a number specifying how many pixels were used in the calculation.
    # After these numbers have been finalised, dividing the running total by
    # the number of pixels will produce the required structure functions.
    obs_compl_P_dict = {}
    obs_P_inten_dict = {}
    noise_compl_P_dict = {}
    noise_P_inten_dict = {}

    # Now that the empty dictionaries have been created, we need to create a
    # key for each angular separation bin, and assign each key an array.
    # This needs to be performed on all dictionaries.
    for sep in ang_sep_centres:
        # Assign a Numpy array to each dictionary, for this angular
        # separation. The key used to access each array is the centre of the
        # angular separation bin.
        obs_compl_P_dict['{}'.format(sep)] = np.array([0.0,0.0])
        obs_P_inten_dict['{}'.format(sep)] = np.array([0.0,0.0])
        noise_compl_P_dict['{}'.format(sep)] = np.array([0.0,0.0])
        noise_P_inten_dict['{}'.format(sep)] = np.array([0.0,0.0])

    # To calculate angular separations, it is necessary to combine the 
    # latitude and longitude arrays into one array, and to convert the units
    # from degrees to radians. Latitude values must be in the first column,
    # and longitude values must be in the second column.
    rad_coord_array = np.deg2rad(np.column_stack([lat_coords,lon_coords]))

    # Iterate over the selected pixels, so that we can calculate the 
    # angular separation between pixels, and quantities required to calculate
    # the structure functions.
    for i in range(len(obs_compl_P)):
    	# Calculate the square of the difference in the observed complex 
    	# polarisation, between the current pixel and all other pixels. This
    	# is required to calculate the corresponding structure function.
    	obs_compl_P_diff = np.power(np.abs(obs_compl_P[i] - obs_compl_P), 2.0)

    	# Calculate the square of the difference in the noise complex
    	# polarisation, between the current pixel and all other pixels. This 
    	# is required to calculate the corresponding noise structure function.
    	noise_compl_P_diff = np.power(np.abs(noise_compl_P[i] - noise_compl_P),\
    	2.0)

    	# Calculate the square of the difference in the observed polarised 
    	# intensity, between the current pixel and all other pixels. This is
    	# required to calculate the corresponding observed structure function.
    	obs_P_inten_diff = np.power(obs_P_inten[i] - obs_P_inten, 2.0)

    	# Calculate the square of the difference in the noise polarised
    	# intensity, between the current pixel and all other pixels. This is
    	# required to calculate the corresponding noise structure function.
    	noise_P_inten_diff = np.power(noise_P_inten[i] - noise_P_inten, 2.0)

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
    	# allocate the quantities needed to calculate structure functions 
    	# to each bin.
        # Loop over the values in the angular separation bin allocation array,
        # so that calculated quantities can be allocated to the correct 
        # angular separation key in each dictionary
        for j in range(1,len(bin_edges)):
            # Calculate the number of pixels in the j-th bin
            num_pix_j = len(obs_compl_P_diff[bin_allocation == j])

            # Add the sum of the relevant quantity for points in the j-th
            # bin to the running total for the observed complex polarisation.
            obs_compl_P_dict['{}'.format(ang_sep_centres[j - 1])][0] +=\
            np.sum(obs_compl_P_diff[bin_allocation == j], dtype = np.float64)

            # Add the number of pixels for the j-th bin to the running total
            # for the observed complex polarisation
            obs_compl_P_dict['{}'.format(ang_sep_centres[j - 1])][1] += num_pix_j

            # Add the sum of the relevant quantity for points in the j-th
            # bin to the running total for the noise complex polarisation.
            noise_compl_P_dict['{}'.format(ang_sep_centres[j - 1])][0] +=\
            np.sum(noise_compl_P_diff[bin_allocation == j], dtype = np.float64)

            # Add the number of pixels for the j-th bin to the running total
            # for the noise complex polarisation
            noise_compl_P_dict['{}'.format(ang_sep_centres[j - 1])][1] += num_pix_j

            # Add the sum of the relevant quantity for points in the j-th
            # bin to the running total for the observed polarisation intensity.
            obs_P_inten_dict['{}'.format(ang_sep_centres[j - 1])][0] +=\
            np.sum(obs_P_inten_diff[bin_allocation == j], dtype = np.float64)

            # Add the number of pixels for the j-th bin to the running total
            # for the observed polarisation intensity
            obs_P_inten_dict['{}'.format(ang_sep_centres[j - 1])][1] += num_pix_j

            # Add the sum of the relevant quantity for points in the j-th
            # bin to the running total for the noise polarisation intensity.
            noise_P_inten_dict['{}'.format(ang_sep_centres[j - 1])][0] +=\
            np.sum(noise_P_inten_diff[bin_allocation == j], dtype = np.float64)

            # Add the number of pixels for the j-th bin to the running total
            # for the noise polarisation intensity
            noise_P_inten_dict['{}'.format(ang_sep_centres[j - 1])][1] += num_pix_j

            # # Append the calculated quantities for the observed complex 
            # # polarisation and for the j-th bin to the corresponding key
            # # in the corresponding dictionary
            # obs_compl_P_dict['{}'.format(ang_sep_centres[j - 1])] = \
            # np.append(obs_compl_P_dict['{}'.format(ang_sep_centres[j - 1])],\
            #     obs_compl_P_diff[bin_allocation == j])

            # # Append the calculated quantities for the noise complex 
            # # polarisation and for the j-th bin to the corresponding key
            # # in the corresponding dictionary
            # noise_compl_P_dict['{}'.format(ang_sep_centres[j - 1])] = \
            # np.append(noise_compl_P_dict['{}'.format(ang_sep_centres[j - 1])],\
            #     noise_compl_P_diff[bin_allocation == j])

            # # Append the calculated quantities for the observed polarised 
            # # intensity and for the j-th bin to the corresponding key
            # # in the corresponding dictionary
            # obs_P_inten_dict['{}'.format(ang_sep_centres[j - 1])] = \
            # np.append(obs_P_inten_dict['{}'.format(ang_sep_centres[j - 1])],\
            #     obs_P_inten_diff[bin_allocation == j])

            # # Append the calculated quantites for the noise polarised 
            # # intensity and for the j-th bin to the corresponding key
            # # in the corresponding dictionary
            # noise_P_inten_dict['{}'.format(ang_sep_centres[j - 1])] = \
            # np.append(noise_P_inten_dict['{}'.format(ang_sep_centres[j - 1])],\
            #     noise_P_inten_diff[bin_allocation == j])

    	# # Perform a bin average for each angular separation bin, for the
    	# # observed complex polarisation, and store the result in the i-th
    	# # column of the corresponding array
    	# obs_compl_P_mat[:,i] = np.array([np.mean(obs_compl_P_diff[\
    	# 	bin_allocation == j]) for j in range(1,len(bin_edges))])

    	# # Perform a bin average for each angular separation bin, for the 
    	# # noise complex polarisation, and store the result in the i-th
    	# # column of the corresponding array
    	# noise_compl_P_mat[:,i] = np.array([np.mean(noise_compl_P_diff[\
    	# 	bin_allocation == j]) for j in range(1,len(bin_edges))])

    	# # Perform a bin average for each angular separation bin, for the
    	# # observed polarised intensity, and store the result in the i-th
    	# # column of the corresponding array
    	# obs_P_inten_mat[:,i] = np.array([np.mean(obs_P_inten_diff[\
    	# 	bin_allocation == j]) for j in range(1,len(bin_edges))])

    	# # Perform a bin average for each angular separation bin, for the
    	# # noise polarised intensity, and store the result in the i-th
    	# # column of the corresponding array
    	# noise_P_inten_mat[:,i] = np.array([np.mean(noise_P_inten_diff[\
    	# 	bin_allocation == j]) for j in range(1,len(bin_edges))])

    	# Print a message to the screen after a certain number of pixels have
    	# been processed, so that it is clear what the program is doing.
    	if (i+1) % 500 == 0:
    		# A certain number of pixels have been processed, so print a
    		# message to the screen.
    		print "{} pixels have been processed.".format(i+1)

    # # For some pixels, there may be NaN values for certain angular separations,
    # # if there are no other pixels that are separated from the pixel by the
    # # angular separation. These NaN values need to be ignored, so that they
    # # do not affect the averaging process. To achieve this, all matrices will
    # # be masked, so that NaN values are ignored.
    # obs_compl_P_mat = np.ma.masked_array(obs_compl_P_mat, np.isnan(obs_compl_P_mat))
    # obs_P_inten_mat = np.ma.masked_array(obs_P_inten_mat, np.isnan(obs_P_inten_mat))
    # noise_compl_P_mat = np.ma.masked_array(noise_compl_P_mat,\
    # np.isnan(noise_compl_P_mat))
    # noise_P_inten_mat = np.ma.masked_array(noise_P_inten_mat,\
    # np.isnan(noise_P_inten_mat))

    # Now that all of the structure function dictionaries have been filled with
    # data for every pixel, create arrays that contain information on the 
    # amplitude of the structure function for each angular separation, to
    # calculate the structure functions for the observations and noise. This is 
    # done by dividing the running total for the quantity by the total number
    # of pixel pairs for that bin, for each key in a dictionary.
    obs_compl_P_struc = np.array([obs_compl_P_dict['{}'.format(sep)][0]\
     / obs_compl_P_dict['{}'.format(sep)][1] for sep in ang_sep_centres])
    obs_P_inten_struc = np.array([obs_P_inten_dict['{}'.format(sep)][0]\
     / obs_P_inten_dict['{}'.format(sep)][1] for sep in ang_sep_centres])
    noise_compl_P_struc = np.array([noise_compl_P_dict['{}'.format(sep)][0]\
     / noise_compl_P_dict['{}'.format(sep)][1] for sep in ang_sep_centres])
    noise_P_inten_struc = np.array([noise_P_inten_dict['{}'.format(sep)][0]\
     / noise_P_inten_dict['{}'.format(sep)][1] for sep in ang_sep_centres])
    
    # To calculate the true complex polarisation structure function, we simply
    # need to subtract the corresponding noise structure function from the 
    # observed structure function.
    true_compl_P_struc = obs_compl_P_struc - noise_compl_P_struc
    
    # To calculate the true polarised intensity structure function, we need
    # to subtract the noise structure function from the observed structure
    # function, and we also need to subtract an extra term.
    # FIXME: NEED TO ADD IN THE TERM THAT HAS BEEN IGNORED
    true_P_inten_struc = obs_P_inten_struc - noise_P_inten_struc

    # The structure functions have now been successfully calculated, so print
    # a message to the screen stating this.
    print 'All structure functions successfully calculated.'

    # Now return all of the structure functions to the calling function
    return obs_compl_P_struc, obs_P_inten_struc, noise_compl_P_struc,\
    noise_P_inten_struc, true_compl_P_struc, true_P_inten_struc