#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to read in the Stokes Q and U data for   #
# the SGPS test region, and produces a plot of the structure functions for    #
# the polarised intensity, and for the complex polarisation vector. Any       #
# difference between these structure functions should be related to whether   #
# the observed depolarisation of diffuse emission is being caused by effects  #
# intrinsic to the source or between the source and the observer.             #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 16/7/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
from astropy.io import fits
from astropy import wcs
from calc_Polar_Struc import calc_Polar_Struc
from scat_plot2 import scat_plot2

# Create a string object which stores the directory of the SGPS data
data_loc = '/Users/chrisherron/Documents/PhD/SGPS_Data/'

#--------------------- EXTRACT DATA FOR SGPS TEST REGION ---------------------

# Open the SGPS data FITS file
sgps_fits = fits.open(data_loc + 'sgps_iqup.imcat.fits')

# Print the information about the data file. This should show that there is only
# a primary HDU, which contains all of the image data
sgps_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the primary HDU
sgps_hdr = sgps_fits[0].header

# Delete all of the history cards in the header
del sgps_hdr[41:]
# Delete all of the information about the third and fourth axes of the array,
# which will be irrelevant in produced plots
del sgps_hdr[22:30]

# Print the header information to the screen (WARNING: The header for the sgps
# data is very long if the history cards are not removed)
print sgps_hdr
# Print a blank line to make the script output easier to read
print ''

# Extract the data from the FITS file, which is held in the primary HDU
sgps_data = sgps_fits[0].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'SGPS data successfully extracted from the FITS file.'

# Check the shape of the data array.
# NOTE: The shape of the array should be (1, 4, 553, 1142), with the first 
# dimension being meaningless, the second being the Stokes parameters and
# the polarised intensity, the third dimension corresponds to the y-axis,
# which is Galactic Latitude, and the fourth dimension is Galactic Longitude,
# which goes along the x-axis.
print 'The shape of the data array is: {}.'.format(sgps_data.shape) 

# Create a temporary array where the redundant first dimension is removed
temp_arr = sgps_data[0]

# The elements in the second dimension of the sgps data array are Stokes I,
# Stokes Q, polarised intensity, and Stokes U, in that order.
# Extract these slices from the temporary data array
Sto_I = temp_arr[0]
Sto_Q = temp_arr[1]
polar_inten = temp_arr[2]
Sto_U = temp_arr[3]

# Print a message to the screen to show that everything is going smoothly
print 'Stokes parameters successfully extracted from data.'

#------------------ EXTRACT SIMULATED STOKES Q AND U DATA ---------------------

# Open the simulated Stokes Q data FITS file
simul_Q_fits = fits.open(data_loc + 'sgps_imgen_final_Q.fits')

# Open the simulated Stokes U data FITS file
simul_U_fits = fits.open(data_loc + 'sgps_imgen_final_U.fits')

# Extract the Stokes Q data (the noise data)
simul_Q = simul_Q_fits[0].data

# Extract the Stokes U data (the noise data)
simul_U = simul_U_fits[0].data

# The shapes of these arrays should both be (553,1142). Print a message to the
# screen to check that this is the case.
print 'The shape of the simulated Q array is: {}.'.format(simul_Q.shape)
print 'The shape of the simulated U array is: {}.'.format(simul_U.shape)

#--------------------- GENERATE POINTS IN THE IMAGE AREA ----------------------

# Obtain the number of pixels along the x axis for the SGPS data
num_x = sgps_hdr['NAXIS1']

# Obtain the number of pixels along the y axis for the SGPS data
num_y = sgps_hdr['NAXIS2']

# Create a wcs conversion for the SGPS data
sgps_wcs = wcs.WCS(sgps_hdr, naxis=[1,2])

# Set a variable that controls how many pixels will be randomly chosen in the
# SGPS image in order to construct the structure function
num_pix = 40000

# Randomly generate the x-axis pixel locations for the required number of
# pixels.
x_coords = np.random.randint(0, num_x, num_pix)

# Randomly generate the y-axis pixel locations for the required number of 
# pixels
y_coords = np.random.randint(0, num_y, num_pix)

# Find the longitude and latitude values for every randomly chosen pixel in
# the SGPS data, in degrees.
lon_coords, lat_coords = sgps_wcs.all_pix2world(x_coords, y_coords, 0)

# Print a message to the screen to show that the longitude and latitude values
# have been successfully calculated for the randomly chosen pixels
print 'Galactic coordinates successfully calculated for randomly chosen pixels.'

#------------------- CALCULATE OBSERVED AND NOISE STOKES Q AND U -------------

# Calculate the observed Stokes Q values at the selected pixel locations
# Note that this is a one-dimensional array, whose length is equal to the
# number of pixel locations being used to calculate the structure function.
obs_Sto_Q = Sto_Q[y_coords,x_coords]

# Calculate the observed Stokes U values at the selected pixel locations
obs_Sto_U = Sto_U[y_coords,x_coords]

# Calculate the noise in Stokes Q for the selected pixel locations
noise_Sto_Q = simul_Q[y_coords,x_coords]

# Calculate the noise in Stokes U for the selected pixel locations
noise_Sto_U = simul_U[y_coords,x_coords]

#---------------------- CREATE BINS IN ANGULAR SEPARATION ----------------------

# To create bins that cover a certain range of angular separations, extract the
# angular separation between adjacent pixels, in degrees.
pix_sep = sgps_hdr['CDELT2']

# Set a variable that controls how many bins are used to calculate the structure
# functions. Note that the number of bin edges is one more than this.
num_bin = 40

# Set a variable that controls the maximum angular separation that should be
# probed by the structure function, in degrees.
max_ang_sep = 10 * num_bin * pix_sep

# Construct an array that specifies the bin edges for angular separation
# Note that the bin edges need to be logarithmically separated. The minimum
# angular separation used is the size of an individual pixel.
bin_edges = np.logspace(np.log10(5.0 * pix_sep), np.log10(max_ang_sep), num_bin + 1.0)

# Now create an array to specify what the characteristic value of the angular
# separation is for each bin of interest.
# First calculate the exponent values used to produce the bin edges
log_edges = np.linspace(np.log10(5.0 * pix_sep), np.log10(max_ang_sep), num_bin + 1.0)

# Calculate the values half-way between the bin edges on a log scale
log_centres = log_edges[:-1] + np.ediff1d(log_edges)/2.0

# Calculate the centres of each angular separation bin. These should be 
# properly logarithmically spaced. These are in degrees.
ang_sep_centres = np.power(10.0, log_centres)

# Print a message to the screen to show that the bins for angular separation 
# have been correctly produced
print 'Angular separation bins successfully produced.'

#-------------------- RUN STRUCTURE FUNCTION CALCULATIONS ----------------------

# Pass the observed and noise values for the complex polarisation and 
# polarised intensity data to the function that calculates the structure
# function of the complex polarisation and the polarised intensity. Also pass
# the arrays containing the Galactic coordinates, and the bin edges for the
# angular separation. The structure function arrays have the same length as
# the angular separation centres array, i.e. num_bin.
obs_compl_P_struc, obs_P_inten_struc, noise_compl_P_struc, noise_P_inten_struc,\
true_compl_P_struc, true_P_inten_struc = calc_Polar_Struc(obs_Sto_Q, obs_Sto_U,\
 noise_Sto_Q, noise_Sto_U, lon_coords, lat_coords,bin_edges, ang_sep_centres)

#-------------------------- PLOT STRUCTURE FUNCTIONS ---------------------------

# Create a plot showing the structure functions for the observed complex
# polarisation vector and the polarised intensity. This plot is automatically
# saved.
scat_plot2(ang_sep_centres, obs_compl_P_struc, ang_sep_centres,\
	obs_P_inten_struc, data_loc + 'sgps_obs_struc_func_new2.png', 'png', x_label = \
	'Angular Separation [deg]', y_label = 'Structure Function Value'\
	+ ' [Jy^2/beam^2]', title = 'Observed Structure Functions', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 4)

# Create a plot showing the structure functions for the noise complex
# polarisation vector and the polarised intensity. This plot is automatically 
# saved.
scat_plot2(ang_sep_centres, noise_compl_P_struc, ang_sep_centres,\
	noise_P_inten_struc, data_loc + 'sgps_noise_struc_func_new2.png', 'png',x_label=\
	'Angular Separation [deg]', y_label = 'Structure Function Value'\
	+ ' [Jy^2/beam^2]', title = 'Noise Structure Functions', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = False, loc = 1)

# Create a plot showing the structure functions for the true complex 
# polarisation vector and the polarised intensity. This plot is automatically
# saved.
scat_plot2(ang_sep_centres, true_compl_P_struc, ang_sep_centres,\
	true_P_inten_struc, data_loc + 'sgps_true_struc_func_new2.png', 'png',x_label=\
	'Angular Separation [deg]', y_label = 'Structure Function Value'\
	+ ' [Jy^2/beam^2]', title = 'True Structure Functions', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 4)

# The code has now finished, so print a message to the screen stating that
# everything functioned as expected
print 'Structure functions calculated successfully.'
