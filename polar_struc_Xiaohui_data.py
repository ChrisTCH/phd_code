#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to read in the Stokes Q and U data for   #
# S-PASS data, provided by Xiaohui, and produces a plot of the structure      #
# functions for the polarised intensity, and for the complex polarisation     #
# vector. Any difference between these structure functions should be related  #
# to whether the observed depolarisation of diffuse emission is being caused  #
# by effects intrinsic to the source or between the source and the observer.  #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 6/8/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
from astropy.io import fits
from calc_Polar_Struc import calc_Polar_Struc
from scat_plot2 import scat_plot2

# Create a string object which stores the directory of the spass data
data_loc = '/Users/chrisherron/Documents/PhD/SGPS_Data/'

#--------------------- EXTRACT DATA FOR S-PASS --------------------------------

# Open the S-PASS data FITS file
spass_fits = fits.open(data_loc + 'g22.spass13cm.sf.iuqpipa.fits')

# Print the information about the data file. This should show that there are
# five HDUs, which contains all of the image data
spass_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the Stokes U HDU
spass_hdr = spass_fits[2].header

# Print the header information to the screen 
print spass_hdr
# Print a blank line to make the script output easier to read
print ''

# Extract the Stokes U data from the FITS file, which is held in the second HDU
spass_U_data = spass_fits[2].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'S-PASS Stokes U data successfully extracted from the FITS file.'

# Extract the Stokes Q data from the FITS file, which is held in the third HDU
spass_Q_data = spass_fits[3].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'S-PASS Stokes Q data successfully extracted from the FITS file.'

# Check the shape of the data array.
# NOTE: The shape of the array should be (201, 481), with the first dimension
# corresponding to the y-axis, which is Galactic Latitude, and the second
# dimension corresponding to Galactic Longitude, which goes along the x-axis.
print 'The shape of the data array is: {}.'.format(spass_U_data.shape) 

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

# Obtain the number of pixels along the x axis for the spass data
num_x = spass_hdr['NAXIS1']

# Obtain the number of pixels along the y axis for the spass data
num_y = spass_hdr['NAXIS2']

# Set a variable that controls how many pixels will be randomly chosen in the
# spass image in order to construct the structure function
num_pix = 30000

# Randomly generate the x-axis pixel locations for the required number of
# pixels.
x_coords = np.random.randint(0, num_x, num_pix)

# Randomly generate the y-axis pixel locations for the required number of 
# pixels
y_coords = np.random.randint(0, num_y, num_pix)

# Calculate the observed Stokes Q values at the selected pixel locations
# Note that this is a one-dimensional array, whose length is equal to the
# number of pixel locations being used to calculate the structure function.
obs_Sto_Q = spass_Q_data[y_coords,x_coords]

# Calculate the observed Stokes U values at the selected pixel locations
obs_Sto_U = spass_U_data[y_coords,x_coords]

# Find all NaN values in the Stokes Q array
Nan_Q = np.isnan(obs_Sto_Q)

# Find all NaN values in the Stokes U array
Nan_U = np.isnan(obs_Sto_U)

# Combine the NaN arrays, so that we can eliminate all randomly generated
# pixels that landed on a NaN value.
Nan_all = np.logical_or(Nan_Q, Nan_U)

# Find which pixels are not located on NaN values
All_good = np.logical_not(Nan_all)

# Only continue operations with pixels that are not located on NaN values
# Find the x and y pixel locations for the good pixels
x_coords = x_coords[All_good]
y_coords = y_coords[All_good]

# Find the corresponding observed Stokes Q and Stokes U values for these pixels
obs_Sto_Q = obs_Sto_Q[All_good]
obs_Sto_U = obs_Sto_U[All_good]

# Print the number of remaining pixels to the screen
print 'Number of pixels used to calculate structure functions: {}'.format(len(x_coords))

# Extract values from the S-PASS header that are required to calculate the 
# Galactic longitude and latitude of points
crval1 = spass_hdr['CRVAL1']
crval2 = spass_hdr['CRVAL2']
crpix1 = spass_hdr['CRPIX1']
crpix2 = spass_hdr['CRPIX2']
cdelt1 = spass_hdr['CDELT1']
cdelt2 = spass_hdr['CDELT2']

# Find the longitude and latitude values for every randomly chosen pixel in
# the S-PASS data, in degrees.
lon_coords = crval1 + (x_coords - crpix1) * cdelt1
lat_coords = crval2 + (y_coords - crpix2) * cdelt2

# Print a message to the screen to show that the longitude and latitude values
# have been successfully calculated for the randomly chosen pixels
print 'Galactic coordinates successfully calculated for randomly chosen pixels.'

#----------------------------- CALCULATE NOISE VALUES -------------------------

# Calculate the noise in Stokes Q for the selected pixel locations
noise_Sto_Q = simul_Q[y_coords,x_coords]

# Calculate the noise in Stokes U for the selected pixel locations
noise_Sto_U = simul_U[y_coords,x_coords]

#---------------------- CREATE BINS IN ANGULAR SEPARATION ----------------------

# To create bins that cover a certain range of angular separations, extract the
# angular separation between adjacent pixels, in degrees.
pix_sep = spass_hdr['CDELT2']

# Set a variable that controls how many bins are used to calculate the structure
# functions. Note that the number of bin edges is one more than this.
num_bin = 40

# Set a variable that controls the maximum angular separation that should be
# probed by the structure function, in degrees.
max_ang_sep = 10 * num_bin * pix_sep

# Construct an array that specifies the bin edges for angular separation
# Note that the bin edges need to be logarithmically separated. The minimum
# angular separation used is the size of an individual pixel.
bin_edges = np.logspace(np.log10(2.0 * pix_sep), np.log10(max_ang_sep), num_bin + 1.0)

# Now create an array to specify what the characteristic value of the angular
# separation is for each bin of interest.
# First calculate the exponent values used to produce the bin edges
log_edges = np.linspace(np.log10(2.0 * pix_sep), np.log10(max_ang_sep), num_bin + 1.0)

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

# Multiply the angular separation centres array by 60, to convert it from 
# degrees to arcminutes.
ang_sep_centres = 60.0 * ang_sep_centres

# Create a plot showing the structure functions for the observed complex
# polarisation vector and the polarised intensity. This plot is automatically
# saved.
scat_plot2(ang_sep_centres, obs_compl_P_struc, ang_sep_centres,\
	obs_P_inten_struc, data_loc + 'spass_obs_struc_func3.png', 'png', x_label = \
	'Angular Separation [arcmin]', y_label = 'Structure Function Value'\
	+ ' [mK^2]', title = 'Observed Structure Functions - Xiaohui', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 4)

# Create a plot showing the structure functions for the noise complex
# polarisation vector and the polarised intensity. This plot is automatically 
# saved.
scat_plot2(ang_sep_centres, noise_compl_P_struc, ang_sep_centres,\
	noise_P_inten_struc, data_loc + 'spass_noise_struc_func3.png', 'png',x_label=\
	'Angular Separation [arcmin]', y_label = 'Structure Function Value'\
	+ ' [mK^2]', title = 'Noise Structure Functions - Xiaohui', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = False, loc = 1)

# Create a plot showing the structure functions for the true complex 
# polarisation vector and the polarised intensity. This plot is automatically
# saved.
scat_plot2(ang_sep_centres, true_compl_P_struc, ang_sep_centres,\
	true_P_inten_struc, data_loc + 'spass_true_struc_func3.png', 'png',x_label=\
	'Angular Separation [arcmin]', y_label = 'Structure Function Value'\
	+ ' [mK^2]', title = 'True Structure Functions - Xiaohui', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 4)

# The code has now finished, so print a message to the screen stating that
# everything functioned as expected
print 'Structure functions calculated successfully.'
