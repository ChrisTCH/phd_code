#------------------------------------------------------------------------------#
#                                                                              #
# This is a script which is designed to open the FITS files containing the     #
# line of sight averaged sonic and Alfvenic Mach numbers, and the polarisation #
# gradient images, for Blakesley's simulations, and produce FITS files and     #
# images of local statistics calculated for the diagnostic quantities, e.g.    #
# the skewness of the polarisation gradient. This is performed for each        #
# simulation that Blakesley provided me. The calculation of these quantities   #
# will occur in separate functions. The images will then be saved as FITS      #
# files in the same directory as the simulations.                              #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 27/5/2016                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script, scipy.stats for calculating statistical quantities
import numpy as np
import aplpy
from astropy.io import fits
from scipy import stats

# Import utility functions
from mat2FITS_Image import mat2FITS_Image

# Import the function that will calculate all of the local statistics
from calc_sparse_stats import calc_sparse_stats       # Sparse Python version

# Create a variable that specifies the number of pixels across the width of the
# evaluation box, divided by two
half_width = 120

# Create a string for the directory that contains the simulated Mach number
# maps to use. 
data_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# The directories end in:
# b.1p.1_Oct_Burk
# b.1p.01_Oct_Burk
# b.1p2_Aug_Burk
# b1p.1_Oct_Burk
# b1p.01_Oct_Burk
# b1p2_Aug_Burk
# c512b.1p.0049
# c512b.1p.0077
# c512b.1p.025
# c512b.1p.05
# c512b.1p.7
# c512b1p.0049
# c512b1p.0077
# c512b1p.025
# c512b1p.05
# c512b1p.7
# c512b3p.01
# c512b5p.01
# c512b5p2

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/', 'c512b3p.01/', 'c512b5p.01/']

# Create strings that will be used to determine what quantities will be read 
# through the script and analysed, e.g. the polarisation gradient, or 
# polarisation intensity. FITS files for the chosen quantity will be loaded
# by the script. Could be 'polar_grad_x.fits' or 'los_av_sonic_x.fits', for
# example
sonic_image_str = 'los_av_sonic_x.fits'
alf_image_str = 'los_av_alf_x.fits'
grad_image_str = 'polar_grad_x.fits'

# Create a list of strings, that will be used to control what local statistics
# are calculated for the input data. For each statistic chosen, FITS files of
# the locally calculated statistic will be produced.
# Do this once for the sonic Mach number image, once for the Alfvenic Mach
# number image, and once for the polarisation gradient.
# Valid statistics include 'mean', 'stdev', skewness', 'kurtosis'
stat_list_sonic = ['stdev', 'skewness']
stat_list_alf = ['stdev', 'skewness']
stat_list_grad = ['skewness']

# Create a list that specifies all of the files for which we want to calculate
# local statistics.
# Do this once for the sonic Mach number image, once for the Alfvenic Mach
# number image, and once for the polarisation gradient.
input_files_sonic = [data_loc + '{}'.format(sim) + sonic_image_str for sim in simul_arr]
input_files_alf = [data_loc + '{}'.format(sim) + alf_image_str for sim in simul_arr]
input_files_grad = [data_loc + '{}'.format(sim) + grad_image_str for sim in simul_arr]

# Create a list that specifies the filenames to use to save all of the 
# produced files.
# Do this once for the sonic Mach number image, once for the Alfvenic Mach
# number image, and once for the polarisation gradient.
output_files_sonic = [data_loc + '{}'.format(sim) + 'los_av_sonic_x_{}'.format(half_width) for sim in simul_arr]
output_files_alf = [data_loc + '{}'.format(sim) + 'los_av_alf_x_{}'.format(half_width) for sim in simul_arr]
output_files_grad = [data_loc + '{}'.format(sim) + 'polar_grad_x_{}'.format(half_width) for sim in simul_arr]

# Determine the half width of the box used to calculate the local statistics 
# around each pixel. 
box_halfwidths = np.asarray([half_width] * len(simul_arr))

# Run the function that calculates the local statistics for each image. This
# will calculate all statistics for each simulation, and then
# save the resulting map as a FITS file.
calc_sparse_stats(input_files_sonic, output_files_sonic, stat_list_sonic, box_halfwidths, all_fits_info = False)
calc_sparse_stats(input_files_alf, output_files_alf, stat_list_alf, box_halfwidths, all_fits_info = False)
calc_sparse_stats(input_files_grad, output_files_grad, stat_list_grad, box_halfwidths, all_fits_info = False)