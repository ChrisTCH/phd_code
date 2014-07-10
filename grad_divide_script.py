#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to read in data for my polarisation      #
# gradient map, and Bryan's polarisation gradient map, divide one by the      #
# other, and convert the resulting matrix into a fits file and an image.      #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 8/7/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
import aplpy
from astropy.io import fits
from astropy import wcs
from mat2FITS_Image import mat2FITS_Image
from fits2aplpy import fits2aplpy

# Create a string object which stores the directory of the SGPS data
data_loc = '/Users/chrisherron/Documents/PhD/SGPS_Data/'

# Open up the FITS file containing my polarisation gradient map
my_grad = fits.open(data_loc + 'sgps_polar_grad.fits')

# Open up Bryan's FITS file containing his polarisation gradient map
B_grad = fits.open(data_loc + 'sgps_Bryan_grad.fits')

# Obtain the header for my FITS file
my_hdr = my_grad[0].header

# Obtain the header for Bryan's FITS file
B_hdr = B_grad[0].header

# Obtain the data for my FITS file
my_data = my_grad[0].data

# Obtain the data for Bryan's FITS file
B_data = B_grad[0].data

# Obtain the number of pixels along the x axis for my FITS file
my_num_x = my_hdr['NAXIS1']

# Obtain the number of pixels along the y axis for my FITS file
my_num_y = my_hdr['NAXIS2']

# Obtain the number of pixels along the x axis for Bryan's FITS file
B_num_x = B_hdr['NAXIS1']

# Obtain the number of pixels along the y axis for Bryan's FITS file
B_num_y = B_hdr['NAXIS2']

# Create a wcs conversion for my FITS file
my_wcs = wcs.WCS(my_hdr)

# Create a wcs conversion for Bryan's FITS file
B_wcs = wcs.WCS(B_hdr)

# Create an array representing the pixel identification number of every pixel
# along the x axis, for my FITS file.
my_x_array = np.arange(0,my_num_x)

# Create an array representing the pixel identification number of every pixel
# along the y axis, for my FITS file.
my_y_array = np.arange(0,my_num_y)

# Create an array representing the pixel identification number of every pixel
# along the x axis, for Bryan's FITS file.
B_x_array = np.arange(0,B_num_x)

# Create an array representing the pixel identification number of every pixel
# along the y axis, for Bryan's FITS file.
B_y_array = np.arange(0,B_num_y)

# Create arrays specifying the coordinate values at each pixel for my FITS file
my_xx_mat, my_yy_mat = np.meshgrid(my_x_array, my_y_array, indexing = 'xy')

# Check the shape of the resultant matrices
print 'Shape of my coordinate matrix: {}'.format(np.shape(my_xx_mat))

# Create arrays specifying the coordinate values at each pixel for Bryan's FITS
# file
B_xx_mat, B_yy_mat = np.meshgrid(B_x_array, B_y_array, indexing = 'xy')

# Check the shape of the resultant matrices
print "Shape of Bryan's coordinate matrix: {}".format(np.shape(B_xx_mat))

# Find the longitude and latitude values for every pixel along the x or y
# axis of my FITS file
my_lon_mat, my_lat_mat = my_wcs.wcs_pix2world(my_xx_mat, my_yy_mat, 0)

# Find the longitude and latitude values for every pixel along the x or y
# axis of Bryan's FITS file
B_lon_mat, B_lat_mat = B_wcs.wcs_pix2world(B_xx_mat, B_yy_mat, 0)

# Create an empty matrix with the same shape as my FITS data
division_mat = np.zeros(np.shape(my_data))

# Cycle through the x axis values of the array, to calculate the result of
# dividing the values at corresponding positions in the matrices
for i in range(len(my_x_array)):
    # Cycle through the y axis values of the array
    for j in range(len(my_y_array)):
        # Find the longitude value at this pixel
        current_lon = my_lon_mat[j,i]
        
        # Find the pixel index for Bryan's data that has the same longitude
        # Note that the every pixel of the longitude matrix is unique, due
        # to the wcs transformation, and the fact that the sky is curved. Hence
        # only the pixel where the longitude values are equal needs to be found.
        y_loc, x_loc = np.where(B_lon_mat == current_lon)

        # Convert x_loc and y_loc from arrays to numbers
        x_loc = x_loc[0]
        y_loc = y_loc[0]
        
        # Find the value of the polarisation gradient at this pixel
        B_value = B_data[y_loc, x_loc]
        
        # Calculate the result of dividing my polarisation gradient data by
        # Bryan's
        division_mat[j,i] = my_data[j,i] / B_value
    
    # Print a message to the screen to show when 50 columns have been computed
    if (i+1)%50 == 0:
        # The number of columns completed is a multiple of 50, so print a 
        # message to the screen
        print '{} columns completed'.format(i+1)
        
# When the code reaches this point, both for loops have completed, and the
# division matrix is filled with values.

# Convert the matrix of divided polarisation gradient values into a FITS file,
# using the header information of my data. Also save the FITS file that is
# produced by the function.
division_FITS = mat2FITS_Image(division_mat, my_hdr,\
data_loc + 'sgps_divided_polar.fits')

# Print a message to the screen to show that the FITS file was produced and
# saved successfully.
print 'FITS file successfully saved for the divided polarisation gradients.'

# Create an image of the divided polarisation gradient values
# using aplpy and the produced FITS file. This image is automatically
# saved using the given filename.
fits2aplpy(division_FITS, data_loc + 'sgps_divided_polar.png', \
colour = 'hot')

# Print a message to the screen to show that the image of the divided 
# polarisation gradient values has been successfully produced and saved.
print 'Image of the divided polarisation gradients successfully saved.\n'