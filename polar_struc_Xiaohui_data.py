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
num_pix = 40000

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
bin_edges = np.logspace(np.log10(pix_sep), np.log10(1.3 * max_ang_sep), num_bin + 1.0)

# Now create an array to specify what the characteristic value of the angular
# separation is for each bin of interest.
# First calculate the exponent values used to produce the bin edges
log_edges = np.linspace(np.log10(pix_sep), np.log10(1.3 * max_ang_sep), num_bin + 1.0)

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

# Create an array of the centres of the angular separation bins for Xiaohui's
# data
Xiao_sep_centre = np.array([0.0509694666, 0.0634814799, 0.0742559731, 0.0868752003,\
	0.101612106, 0.118684702, 0.138772532, 0.162279263, 0.189662904, 0.221812427,\
	0.259352148, 0.30336377, 0.354609728, 0.414519817, 0.48473534, 0.566673517,\
	0.662482619, 0.774623811, 0.905546486, 1.05879116, 1.23788464, 1.44734359,\
	1.69196451, 1.97814715, 2.31270528, 2.7037313, 3.16044497, 3.6946733,\
	4.31970072, 5.05465794, 5.90662146, 6.89912844, 8.05670452, 9.40233803,\
	10.9931259, 12.8478184, 15.0008411, 17.4840393, 20.2194672, 22.9411488])

# Create an array of the structure function values for polarised intensity
# in Xiaohui's data
Xiao_P_inten_struc = np.array([50.0531883, 69.0552216, 87.6915207, 100.143974,\
	130.076874, 164.537781, 204.685684, 245.515167, 297.188019, 337.048004,\
	383.43811, 436.84787, 485.785461, 523.864136, 565.291016, 604.150146,\
	644.604858, 679.76239, 717.844055, 758.189819, 787.752686, 824.260681,\
	850.419678, 890.345764, 938.202698, 996.339111, 1034.32324, 1054.50586,\
	1036.23254, 1059.30005, 1135.37634, 1166.05127, 1224.03967, 1279.48291,\
	1308.26099, 1270.27515, 1274.49341, 1111.00708, 956.980957, 1401.15259])

# Create an array of the structure function values for complex polarisation
# in Xiaohui's data
Xiao_compl_P_struc = np.array([125.300171, 199.189987, 241.001129, 279.267853,\
	371.725677, 491.87793, 629.569092, 769.859192, 937.650452, 1104.00574,\
	1298.42188, 1493.79932, 1685.49878, 1858.52112, 1987.25549, 2122.81128,\
	2247.81616, 2363.0835, 2434.49316, 2508.55518, 2559.38452, 2592.87744,\
	2618.47168, 2625.9021, 2632.91895, 2733.70557, 2810.39917, 2825.35547,\
	2754.96045, 2814.19189, 2984.6377, 3152.83813, 3430.25537, 3660.80957,\
	3571.3916, 3160.2793, 3019.51978, 2955.30054, 2786.69067, 3375.07056])

# Multiply the angular separation centres array by 60, to convert it from 
# degrees to arcminutes.
ang_sep_centres = 60.0 * ang_sep_centres
# Do the same for Xiaohui's array of angular separation bins
Xiao_sep_centre = 60.0 * Xiao_sep_centre

# Create a plot showing the structure functions for the observed complex
# polarisation vector, for my method of calculating the structure function and
# for Xiaohui's data. This plot is automatically saved.
scat_plot2(ang_sep_centres, obs_compl_P_struc, Xiao_sep_centre,\
	Xiao_compl_P_struc, data_loc + 'spass_compl_P_struc_func_compare2.png',\
	'png', x_label = 'Angular Separation [arcmin]', y_label =\
	'Structure Function Value'+ ' [mK^2]', title =\
	'Observed Complex Polarisation Structure Functions - S-PASS', col1 = 'b',\
	col2 = 'r', label1 = 'My Method', label2 ='Xiaohui Data',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 4)

# Create a plot showing the structure functions for the observed polarisation
# intensity, for my method of calculating the structure function and
# for Xiaohui's data. This plot is automatically saved.
scat_plot2(ang_sep_centres, obs_P_inten_struc, Xiao_sep_centre,\
	Xiao_P_inten_struc, data_loc + 'spass_P_inten_struc_func_compare2.png',\
	'png', x_label = 'Angular Separation [arcmin]', y_label =\
	'Structure Function Value'+ ' [mK^2]', title =\
	'Observed Polarisation Intensity Structure Functions - S-PASS', col1 = 'b',\
	col2 = 'r', label1 = 'My Method', label2 ='Xiaohui Data',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 4)

# Create a plot showing the structure functions for the noise complex
# polarisation vector and the polarised intensity. This plot is automatically 
# saved.
scat_plot2(ang_sep_centres, noise_compl_P_struc, ang_sep_centres,\
	noise_P_inten_struc, data_loc + 'spass_noise_struc_func5.png', 'png',x_label=\
	'Angular Separation [arcmin]', y_label = 'Structure Function Value'\
	+ ' [mK^2]', title = 'Noise Structure Functions - Xiaohui', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 1)

# Create a plot showing the structure functions for the true complex 
# polarisation vector and the polarised intensity. This plot is automatically
# saved.
scat_plot2(ang_sep_centres, true_compl_P_struc, ang_sep_centres,\
	true_P_inten_struc, data_loc + 'spass_true_struc_func5.png', 'png',x_label=\
	'Angular Separation [arcmin]', y_label = 'Structure Function Value'\
	+ ' [mK^2]', title = 'True Structure Functions - Xiaohui', col1 = 'b',\
	col2 = 'r', label1 = 'Complex Polarisation', label2 ='Polarised Intensity',\
	marker1 = 'o', marker2 = 'x', log_x = True, log_y = True, loc = 4)

# The code has now finished, so print a message to the screen stating that
# everything functioned as expected
print 'Structure functions calculated successfully.'
