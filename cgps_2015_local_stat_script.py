#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to open the FITS files containing the    #
# smoothed CGPS data, and produce FITS files and images of local statistics   #
# calculated for the diagnostic quantities, e.g. the skewness of the          #
# polarisation gradient. This is performed for each final angular resolution  #
# that was used to smooth the data. The calculation of these quantities will  #
# occur in separate functions. The images will then be saved as FITS files in #
# the same directory as the CGPS data.                                        #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 25/8/2015                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script, scipy.stats for calculating statistical quantities
import numpy as np
import aplpy
from astropy.io import fits
from scipy import stats

# Import utility functions
from mat2FITS_Image import mat2FITS_Image
from fits2aplpy import fits2aplpy

# Import the function that will calculate all of the local statistics
#from calc_local_stats import calc_local_stats         # Cython version
#from calc_local_stats_purepy import calc_local_stats  # Python version
from calc_sparse_stats import calc_sparse_stats       # Sparse Python version

# Create a string object which stores the directory of the CGPS data
data_loc = '/Users/chrisherron/Documents/PhD/CGPS_2015/'

# Create a string that will be used to determine what quantity will be read 
# through the script and analysed, e.g. the polarisation gradient, or 
# polarisation intensity. FITS files for the chosen quantity will be loaded
# by the script. Could be 'Polar_Grad' or 'Polar_Inten', for example
data_2_load = 'Polar_Grad'

# Create a string that will be used to control what FITS files are used
# to perform calculations, and that will be appended into the filename of 
# anything produced in this script. This is either 'high_lat' or 'plane'
save_append = 'plane'

# Create an array that specifies all of the final resolution values that were 
# used to create mosaics. This code will calculate quantities for each of the
# resolutions given in this array
final_res_array = np.array([75, 90, 105, 120, 135, 150, 165, 180, 195, 210,\
 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 450,\
 480, 510, 540, 570, 600, 630, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140,\
 1200])
#final_res_array = np.array([75])

# Create a list that specifies all of the files for which we want to calculate
# local statistics
input_files = [data_loc + '{}_{}_smoothed/'.format(data_2_load, save_append)\
+ '{}_{}_smooth2_{}.fits'.format(data_2_load, save_append, res) for res in final_res_array]

# Create a list that specifies the filenames to use to save all of the 
# produced files
output_files = [data_loc + '{}_{}_smooth2_{}'.format(data_2_load, save_append,\
 res) for res in final_res_array]

# Create a list of strings, that will be used to control what local statistics
# are calculated for the input data. For each statistic chosen, FITS files of
# the locally calculated statistic will be produced for each final resolution
# value, and saved in the same directory as the CGPS data.
# Valid statistics include 'skewness', 'kurtosis'
stat_list = ['skewness']

# Specify the size of each pixel in the CGPS survey, in degrees
pix_size_deg = 4.9999994 * np.power(10.0, -3.0)

# Determine the half width of the box used to calculate the local skewness 
# around each pixel. This half width depends upon the angular resolution of
# the image being considered, so that we always have a certain number of 
# beamwidths within the box. This is given as the largest integer that fits 
# the required number of beamwidths.
box_halfwidths = np.asarray([int(np.ceil(10 * res/(pix_size_deg * 3600)))\
 for res in final_res_array])

# Run the function that calculates the local statistics for each image. This
# will calculate all statistics for each value of the final resolution, and then
# save the resulting map as a FITS file.
calc_sparse_stats(input_files, output_files, stat_list, box_halfwidths)