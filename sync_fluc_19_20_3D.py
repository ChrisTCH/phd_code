#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates equations 19 and 20 of Lazarian and Pogosyan #
# 2012, to check if these equations are correct. This script compares the 3D   #
# correlation functions, and does not perform any radial averaging.            #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 19/9/2014                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the functions that calculate the structure and correlation functions
# using FFT, as well as the function that calculates the radially averaged 
# structure or correlation functions.
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from slicing import slicing
from fractalcube import fractalcube

# Set a variable to hold the number of bins to use in calculating the 
# correlation functions
num_bins = 15

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# The two directories end in b.1p2_Aug_Burk and b1p2_Aug_Burk.
spec_loc = 'b1p2_Aug_Burk/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc
 
# # Open the FITS file that contains the x-component of the simulated magnetic
# # field
# mag_x_fits = fits.open(data_loc + 'magx.fits')

# # Extract the data for the simulated x-component of the magnetic field
# mag_x_data = mag_x_fits[0].data

# # Extract the first octant of the x-component of the magnetic field data, to
# # greatly speed up the processing time
# mag_x_data = mag_x_data[0:256, 0:256, 0:256]

# # Open the FITS file that contains the y-component of the simulated magnetic 
# # field
# mag_y_fits = fits.open(data_loc + 'magy.fits')

# # Extract the data for the simulated y-component of the magnetic field
# mag_y_data = mag_y_fits[0].data

# # Extract the first octant of the y-component of the magnetic field data, to
# # greatly speed up the processing time
# mag_y_data = mag_y_data[0:256, 0:256, 0:256]

# ------ Use this code to test with fractal data

# Create a cube of fractal data, which is meant to represent the x component
# of the magnetic field
mag_x_data = fractalcube(3.0, seed = 6, size = 256)

# # slicing(mag_x_data, xlabel = 'x', ylabel = 'y',\
# #  zlabel = 'z', title = 'x-comp B Field')

# Create a cube of fractal data, which is meant to represent the y component 
# of the magnetic field
mag_y_data = fractalcube(3.0, seed = 8, size = 256)

# # slicing(mag_y_data, xlabel = 'x', ylabel = 'y',\
# #  zlabel = 'z', title = 'y-comp B Field')

# ------ End fractal data generation

# Print a message to the screen to show that the data has been loaded
print 'Magnetic field components loaded successfully'

# Calculate the magnitude of the magnetic field perpendicular to the line of 
# sight, which is just the square root of the sum of the x and y component
# magnitudes squared.
mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

# Calculate the average of the x-component of the magnetic field squared
mag_x_mean_sq = np.mean( np.power(mag_x_data, 2.0), dtype = np.float64)

# Calculate the average of the y-component of the magnetic field squared
mag_y_mean_sq = np.mean( np.power(mag_y_data, 2.0), dtype = np.float64)

# Print a message to show that the perpendicular component of the magnetic
# field has been calculated
print 'Perpendicular component of the magnetic field calculated'

# ---------------- Normalised correlation x-comp B field ----------------------

# Calculate the correlation function for the x-component of the magnetic field
x_corr = cf_fft(mag_x_data, no_fluct = True)

# Print a message to show that the correlation function of the x-component of
# the magnetic field has been calculated
print 'Correlation function of the x-component of the magnetic field calculated'

# Calculate the normalised correlation function for the x-component of the
# magnetic field. This is equation 13 of Lazarian and Pogosyan 2012.
c_1 = x_corr / mag_x_mean_sq

# Print a message to show that c_1 has been calculated
print 'Normalised correlation function for the x-component of the magnetic'\
+ ' has been calculated'

# Print out the shape of the c_1 array
print 'Shape of c_1 is: {}'.format(np.shape(c_1))

# Print out the maximum and minimum value of c_1
print 'Maximum value of c_1: {}'.format(np.max(c_1))
print 'Minimum value of c_1: {}'.format(np.min(c_1))

# slicing(c_1, xlabel = 'x', ylabel = 'y',\
#  zlabel = 'z', title = 'Norm Corr x-comp B Field')

# ---------------- Normalised correlation y-comp B field ----------------------

# Calculate the correlation function for the y-component of the magnetic field
y_corr = cf_fft(mag_y_data, no_fluct = True)

# Print a message to show that the correlation function of the y-component of
# the magnetic field has been calculated
print 'Correlation function of the y-component of the magnetic field calculated'

# Calculate the normalised correlation function for the y-component of the
# magnetic field. This is equation 14 of Lazarian and Pogosyan 2012.
c_2 = y_corr / mag_y_mean_sq

# Print a message to show that c_2 has been calculated
print 'Normalised correlation function for the y-component of the magnetic'\
+ ' has been calculated'

# Print out the maximum and minimum value of c_2
print 'Maximum value of c_2: {}'.format(np.max(c_2))
print 'Minimum value of c_2: {}'.format(np.min(c_2))

# Calculate the right hand side of equation 19 of Lazarian and Pogosyan 2012
RHS_19 = 0.5 * ( np.power(c_1, 2.0) + np.power(c_2, 2.0) )

# Calculate the right hand side of equation 20 of Lazarian and Pogosyan 2012
RHS_20 = 0.4 * ( np.power(c_1, 2.0) + np.power(c_2, 2.0) ) + 3.0/40.0 *\
 ( np.power(c_1, 4.0) + np.power(c_2, 4.0) ) + 1.0/20.0 * np.power(c_1, 2.0) *\
  np.power(c_2, 2.0)

# -------------------------- Modified Equation 19 -----------------------------

# # Calculate auxiliary quantities that are required to calculate the modified
# # version of equation 19, which is on page 97 of PhD Logbook 3

# # Calculate the square of the mean of the x-component of the magnetic field
# # squared
# mag_x_sq_mean_sq = np.power( np.mean( np.power(mag_x_data,2.0), dtype = np.float64 ), 2.0)

# # Calculate the square of the mean of the y-component of the magnetic field
# # squared
# mag_y_sq_mean_sq = np.power( np.mean( np.power(mag_y_data,2.0), dtype = np.float64 ), 2.0)

# # Calculate the mean of the x-component of the magnetic field raised to the
# # power of 4
# mag_x_mean_four = np.mean( np.power(mag_x_data, 4.0), dtype = np.float64 )

# # Calculate the mean of the y-component of the magnetic field raised to the
# # power of 4
# mag_y_mean_four = np.mean( np.power(mag_y_data, 4.0), dtype = np.float64 )

# # Calculate the correlation function for the x-component of the magnetic
# # field squared
# x2_corr = cf_fft(np.power(mag_x_data,2.0), no_fluct = True)

# # Print a message to show that the correlation function of the x-component of
# # the magnetic field squared has been calculated
# print 'Correlation function of the x-component of the magnetic field squared'\
# + ' calculated'

# # Calculate the correlation function for the y-component of the magnetic 
# # field squared
# y2_corr = cf_fft(np.power(mag_y_data,2.0), no_fluct = True)

# # Print a message to show that the correlation function of the y-component of
# # the magnetic field squared has been calculated
# print 'Correlation function of the y-component of the magnetic field squared'\
# + ' calculated'

# # Calculate the modified version of equation 19, which is on page 97 of PhD
# # Logbook 3
# RHS_19 = (x2_corr - mag_x_sq_mean_sq + y2_corr -\
#  mag_y_sq_mean_sq) / (mag_x_mean_four - mag_x_sq_mean_sq + mag_y_mean_four -\
#   mag_y_sq_mean_sq)

# ------------------ Modified Equation 19 - No Assumptions --------------------

# # Calculate the mean of the x-component of the magnetic field squared times
# # the y-component of the magnetic field squared
# mag_x2_y2_mean = np.mean(np.power(mag_x_data,2.0) * np.power(mag_y_data,2.0), dtype = np.float64)

# # Calculate the correlation function for the x-component of the magnetic field
# # squared and the y-component of the magnetic field squared
# x2_y2_corr = cf_fft(np.power(mag_x_data,2.0), np.power(mag_y_data,2.0),\
# 	no_fluct = True )

# # Print a message to show that the correlation function of the squared x and y
# # components of the magnetic field has been calculated
# print 'Correlation function of the squared x and y-components of the magnetic'\
# + ' field calculated'

# # Calculate the correlation function for the y-component of the magnetic field
# # squared and the x-component of the magnetic field squared
# y2_x2_corr = cf_fft(np.power(mag_y_data,2.0), np.power(mag_x_data,2.0),\
# 	no_fluct = True )

# # Print a message to show that the correlation function of the squared y and x
# # components of the magnetic field has been calculated
# print 'Correlation function of the squared y and x-components of the magnetic'\
# + ' field calculated'

# # Calculate the modified version of equation 19, which is on page 96 of PhD
# # Logbook 3, and does not assume that there is no correlation between the 
# # squared x and y components of the magnetic field
# RHS_19 = (x2_corr - mag_x_sq_mean_sq + y2_corr -\
#  mag_y_sq_mean_sq + x2_y2_corr + y2_x2_corr - 2.0 *\
#   mag_x_mean_sq * mag_y_mean_sq) / (mag_x_mean_four + 2.0 * mag_x2_y2_mean\
#    - mag_x_sq_mean_sq + mag_y_mean_four - mag_y_sq_mean_sq - 2.0 *\
#     mag_x_mean_sq * mag_y_mean_sq)

# # Print a message to show that the right hand sides of the equations have
# # been calculated correctly
# print 'The right hand sides of equations 19 and 20 have been calculated'

# -------------- Normalised correlation B_perp gamma = 2 ----------------------

# Calculate the result of raising the magnetic field strength perpendicular 
# to the line of sight to the power of gamma = 2
mag_perp_gamma_2 = np.power(mag_perp, 2.0)

# Calculate the square of the mean of the perpendicular component of the 
# magnetic field raised to the power of gamma = 2
mag_sq_mean_gamma_2 = np.power(np.mean(mag_perp_gamma_2, dtype =np.float64),2.0)

# Calculate the mean of the squared perpendicular component of the magnetic
# field raised to the power of gamma = 2
mag_mean_sq_gamma_2 = np.mean( np.power(mag_perp_gamma_2, 2.0),dtype=np.float64)

# Calculate the correlation function for the perpendicular component of the
# magnetic field, when raised to the power of gamma = 2
perp_gamma_2_corr = cf_fft(mag_perp_gamma_2, no_fluct = True)

# Print a message to show that the correlation function of the perpendicular 
# component of the magnetic field has been calculated for gamma = 2
print 'Correlation function of the perpendicular component of the magnetic'\
+ ' field calculated for gamma = 2'

# Calculate the normalised correlation function for the magnetic field
# perpendicular to the line of sight, for gamma = 2. This is the left hand
# side of equation 19, and the right hand side of equation 15.
mag_gamma_2_norm_corr = (perp_gamma_2_corr - mag_sq_mean_gamma_2)\
 / (mag_mean_sq_gamma_2 - mag_sq_mean_gamma_2)

# Print a message to show that the normalised correlation function for 
# gamma = 2 has been calculated
print 'The normalised correlation function for gamma = 2 has been calculated'

# ------------- Normalised correlation B_perp gamma = 4 -----------------------

# Calculate the result of raising the magnetic field strength perpendicular 
# to the line of sight to the power of gamma = 4
mag_perp_gamma_4 = np.power(mag_perp, 4.0)

# Calculate the square of the mean of the perpendicular component of the 
# magnetic field raised to the power of gamma = 4
mag_sq_mean_gamma_4 = np.power(np.mean(mag_perp_gamma_4, dtype =np.float64),2.0)

# Calculate the mean of the squared perpendicular component of the magnetic
# field raised to the power of gamma = 4
mag_mean_sq_gamma_4 = np.mean( np.power(mag_perp_gamma_4, 2.0),dtype=np.float64)

# Calculate the correlation function for the perpendicular component of the
# magnetic field, when raised to the power of gamma = 4
perp_gamma_4_corr = cf_fft(mag_perp_gamma_4, no_fluct = True)

# Print a message to show that the correlation function of the perpendicular 
# component of the magnetic field has been calculated for gamma = 2
print 'Correlation function of the perpendicular component of the magnetic'\
+ ' field calculated for gamma = 4'

# Calculate the normalised correlation function for the magnetic field
# perpendicular to the line of sight, for gamma = 4. This is the left hand
# side of equation 20, and the right hand side of equation 15.
mag_gamma_4_norm_corr = (perp_gamma_4_corr - mag_sq_mean_gamma_4)\
 / (mag_mean_sq_gamma_4 - mag_sq_mean_gamma_4)

# Print a message to show that the normalised correlation function for 
# gamma = 4 has been calculated
print 'The normalised correlation function for gamma = 4 has been calculated'

# -------------------- Plots of LHS and RHS Equation 19 -----------------------

# Calculate the maximum difference between the left and right hand sides of 
# equation 19
diff_19 = np.max(RHS_19 - mag_gamma_2_norm_corr)

# Print the maximum difference to the screen
print 'Maximum difference RHS - LHS for Eq 19: {}'.format(diff_19)

# Calculate the maximum difference between the left and right hand sides of 
# equation 20
diff_20 = np.max(RHS_20 - mag_gamma_4_norm_corr)

# Print the maximum difference to the screen
print 'Maximum difference RHS - LHS for Eq 20: {}\n'.format(diff_20)

# Run the slicing program on the difference between the left and right hand 
# sides of equation 19, so that we can see the differences
slicing(RHS_19 - mag_gamma_2_norm_corr, xlabel = 'x', ylabel = 'y',\
 zlabel = 'z', title = 'Difference RHS - LHS Equation 19')

# # Run the slicing program on the difference between the left and right hand 
# # sides of equation 20, so that we can see the differences
# slicing(RHS_20 - mag_gamma_4_norm_corr, xlabel = 'x', ylabel = 'y',\
#  zlabel = 'z', title = 'Difference RHS - LHS Equation 20')