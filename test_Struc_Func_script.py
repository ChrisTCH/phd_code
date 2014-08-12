#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to test the calculation of the structure #
# function. This test is carried out by calculating the structure function of #
# Gaussian noise, which should be a horizontal line of constant amplitude.    #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 7/8/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
from astropy.io import fits
from astropy import wcs
from calc_Struc_Func import calc_Struc_Func
from scat_plot import scat_plot

# Create a string object which stores the directory of the SGPS data
data_loc = '/Users/chrisherron/Documents/PhD/SGPS_Data/'

#--------------------- EXTRACT DATA FOR TEST REGION ---------------------------

# Open the data FITS file
data_fits = fits.open(data_loc + 'struc_func_test_noise2.fits')

# Print the information about the data file. This should show that there is only
# a primary HDU, which contains all of the image data
data_fits.info()
# Print a blank line to make the script output easier to read
print ''

# Obtain the header of the primary HDU
data_hdr = data_fits[0].header

# Print the header information to the screen 
print data_hdr
# Print a blank line to make the script output easier to read
print ''

# Extract the data from the FITS file, which is held in the primary HDU
data = data_fits[0].data

# Print a message to the screen saying that the data was successfully 
# extracted
print 'Test data successfully extracted from the FITS file.'

# Check the shape of the data array.
# NOTE: The shape of the array should be (553, 1142), with the first 
# dimension corresponding to the y-axis, which is Galactic Latitude, and the
# second dimension corresponding to Galactic Longitude, which goes along the
# x-axis.
print 'The shape of the data array is: {}.'.format(data.shape) 

#--------------------- GENERATE POINTS IN THE IMAGE AREA ----------------------

# Obtain the number of pixels along the x axis for the test data
num_x = data_hdr['NAXIS1']

# Obtain the number of pixels along the y axis for the test data
num_y = data_hdr['NAXIS2']

# Create a wcs conversion for the test data
data_wcs = wcs.WCS(data_hdr, naxis=[1,2])

# Set a variable that controls how many pixels will be randomly chosen in the
# test image in order to construct the structure function
num_pix = 30000

# Randomly generate the x-axis pixel locations for the required number of
# pixels.
x_coords = np.random.randint(0, num_x, num_pix)

# Randomly generate the y-axis pixel locations for the required number of 
# pixels
y_coords = np.random.randint(0, num_y, num_pix)

# Find the longitude and latitude values for every randomly chosen pixel in
# the test data, in degrees.
lon_coords, lat_coords = data_wcs.all_pix2world(x_coords, y_coords, 0)

# Print a message to the screen to show that the longitude and latitude values
# have been successfully calculated for the randomly chosen pixels
print 'Galactic coordinates successfully calculated for randomly chosen pixels.'

#---------------------------- CALCULATE DATA VALUES ----------------------------

# Calculate the data values at the selected pixel locations
# Note that this is a one-dimensional array, whose length is equal to the
# number of pixel locations being used to calculate the structure function.
data_array = data[y_coords,x_coords]

#---------------------- CREATE BINS IN ANGULAR SEPARATION ----------------------

# To create bins that cover a certain range of angular separations, extract the
# angular separation between adjacent pixels, in degrees.
pix_sep = data_hdr['CDELT2']

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

# Pass the data values to the function that calculates the structure
# function. Also pass the arrays containing the Galactic coordinates, and the
# bin edges for the angular separation. The structure function arrays have the
# same length as the angular separation centres array, i.e. num_bin.
test_struc = calc_Struc_Func(data_array, lon_coords, lat_coords,\
	bin_edges, ang_sep_centres)

#-------------------------- PLOT STRUCTURE FUNCTION ---------------------------

# Create a plot showing the structure function for the data.
# This plot is automatically saved.
scat_plot(ang_sep_centres, test_struc, data_loc + 'test_struc_func_new2.png',\
	'png', x_label = 'Angular Separation [deg]', y_label =\
	'Structure Function Value [Jy^2/beam^2]', title =\
	'Test Structure Function - Gaussian Noise',log_x = True, log_y = True)

# The code has now finished, so print a message to the screen stating that
# everything functioned as expected
print 'Structure function calculated successfully.'
