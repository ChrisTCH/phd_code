#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates equation 19 of Lazarian and Pogosyan 2012,   #
# to check if this equation is correct. Two plots are produced, comparing the  #
# left and right hand sides of equation 19 for two simulations; one with a low #
# magnetic field, and the other with a high magnetic field. These plots are    #
# designed to be publication quality.                                          #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 23/1/2015                                                        #
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

# Set a variable to hold the number of bins to use in calculating the 
# correlation functions
num_bins = 25

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data sets to use in calculations.
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
spec_loc1 = 'b.1p.01_Oct_Burk/'
spec_loc2 = 'b1p.01_Oct_Burk/'

# Create a string for the full directory path to use in calculations, for both
# the simulations being considered here
data_loc1 = simul_loc + spec_loc1
data_loc2 = simul_loc + spec_loc2
 
# Open the FITS files that contain the x-component of the simulated magnetic
# field, for the two simulations
mag_x_fits1 = fits.open(data_loc1 + 'magx.fits')
mag_x_fits2 = fits.open(data_loc2 + 'magx.fits')

# Extract the data for the simulated x-component of the magnetic field, for the 
# two simulations being considered
mag_x_data1 = mag_x_fits1[0].data
mag_x_data2 = mag_x_fits2[0].data

# Open the FITS file that contains the y-component of the simulated magnetic 
# field, for the two simulations
mag_y_fits1 = fits.open(data_loc1 + 'magy.fits')
mag_y_fits2 = fits.open(data_loc2 + 'magy.fits')

# Extract the data for the simulated y-component of the magnetic field, for the
# two simulations
mag_y_data1 = mag_y_fits1[0].data
mag_y_data2 = mag_y_fits2[0].data

# Print a message to the screen to show that the data has been loaded
print 'Magnetic field components loaded successfully'

# Calculate the magnitude of the magnetic field perpendicular to the line of 
# sight, which is just the square root of the sum of the x and y component
# magnitudes squared. Do this for both simulations.
mag_perp1 = np.sqrt( np.power(mag_x_data1, 2.0) + np.power(mag_y_data1, 2.0) )
mag_perp2 = np.sqrt( np.power(mag_x_data2, 2.0) + np.power(mag_y_data2, 2.0) )

# Calculate the average of the x-component of the magnetic field squared, for
# both simulations
mag_x_mean_sq1 = np.mean( np.power(mag_x_data1, 2.0), dtype = np.float64 )
mag_x_mean_sq2 = np.mean( np.power(mag_x_data2, 2.0), dtype = np.float64 )

# Calculate the average of the y-component of the magnetic field squared, for
# both simulations
mag_y_mean_sq1 = np.mean( np.power(mag_y_data1, 2.0), dtype = np.float64 )
mag_y_mean_sq2 = np.mean( np.power(mag_y_data2, 2.0), dtype = np.float64 )

# Print a message to show that the perpendicular component of the magnetic
# field has been calculated
print 'Perpendicular component of the magnetic field calculated'

# ---------------- Normalised correlation x-comp B field ----------------------

# Calculate the correlation function for the x-component of the magnetic field,
# for both simulations
x_corr1 = cf_fft(mag_x_data1, no_fluct = True)
x_corr2 = cf_fft(mag_x_data2, no_fluct = True)

# Print a message to show that the correlation function of the x-component of
# the magnetic field has been calculated
print 'Correlation function of the x-component of the magnetic field calculated'

# Calculate the radially averaged correlation function for the x-component
# of the magnetic field, for both simulations
x_rad_av_corr1 = sfr(x_corr1, num_bins, verbose = False)
x_rad_av_corr2 = sfr(x_corr2, num_bins, verbose = False)

# Extract the radius values used to calculate the radially averaged 
# correlation function, for both simulations
radius_array1 = x_rad_av_corr1[0]
radius_array2 = x_rad_av_corr2[0]

# Calculate the normalised radially averaged correlation function for the 
# x-component of the magnetic field. This is equation 13 of Lazarian and 
# Pogosyan 2012. Do this for both simulations.
c_1_1 = x_rad_av_corr1[1] / mag_x_mean_sq1
c_1_2 = x_rad_av_corr2[1] / mag_x_mean_sq2

# Print a message to show that c_1 has been calculated
print 'Normalised correlation function for the x-component of the magnetic'\
+ ' has been calculated'

# ---------------- Normalised correlation y-comp B field ----------------------

# Calculate the correlation function for the y-component of the magnetic field,
# for both simulations
y_corr1 = cf_fft(mag_y_data1, no_fluct = True)
y_corr2 = cf_fft(mag_y_data2, no_fluct = True)

# Print a message to show that the correlation function of the y-component of
# the magnetic field has been calculated
print 'Correlation function of the y-component of the magnetic field calculated'

# Calculate the radially averaged correlation function for the y-component
# of the magnetic field, for both simulations
y_rad_av_corr1 = sfr(y_corr1, num_bins, verbose = False)
y_rad_av_corr2 = sfr(y_corr2, num_bins, verbose = False)

# Calculate the normalised radially averaged correlation function for the 
# y-component of the magnetic field. This is equation 14 of Lazarian and
# Pogosyan 2012. Do this for both simulations.
c_2_1 = y_rad_av_corr1[1] / mag_y_mean_sq1
c_2_2 = y_rad_av_corr2[1] / mag_y_mean_sq2

# Print a message to show that c_2 has been calculated
print 'Normalised correlation function for the y-component of the magnetic'\
+ ' has been calculated'

# Calculate the right hand side of equation 19 of Lazarian and Pogosyan 2012,
# for both simulations
RHS_19_1 = 0.5 * ( np.power(c_1_1, 2.0) + np.power(c_2_1, 2.0) )
RHS_19_2 = 0.5 * ( np.power(c_1_2, 2.0) + np.power(c_2_2, 2.0) )

# -------------- Normalised correlation B_perp gamma = 2 ----------------------

# Calculate the result of raising the magnetic field strength perpendicular 
# to the line of sight to the power of gamma = 2, for both simulations
mag_perp_gamma_2_1 = np.power(mag_perp1, 2.0)
mag_perp_gamma_2_2 = np.power(mag_perp2, 2.0)

# Calculate the square of the mean of the perpendicular component of the 
# magnetic field raised to the power of gamma = 2, for both simulations
mag_sq_mean_gamma_2_1 = np.power(np.mean(mag_perp_gamma_2_1, dtype = np.float64), 2.0)
mag_sq_mean_gamma_2_2 = np.power(np.mean(mag_perp_gamma_2_2, dtype = np.float64), 2.0)

# Calculate the mean of the squared perpendicular component of the magnetic
# field raised to the power of gamma = 2, for both simulations
mag_mean_sq_gamma_2_1 = np.mean( np.power(mag_perp_gamma_2_1, 2.0), dtype = np.float64 )
mag_mean_sq_gamma_2_2 = np.mean( np.power(mag_perp_gamma_2_2, 2.0), dtype = np.float64 )

# Calculate the correlation function for the perpendicular component of the
# magnetic field, when raised to the power of gamma = 2, for both simulations
perp_gamma_2_corr1 = cf_fft(mag_perp_gamma_2_1, no_fluct = True)
perp_gamma_2_corr2 = cf_fft(mag_perp_gamma_2_2, no_fluct = True)

# Print a message to show that the correlation function of the perpendicular 
# component of the magnetic field has been calculated for gamma = 2
print 'Correlation function of the perpendicular component of the magnetic'\
+ ' field calculated for gamma = 2'

# Calculate the radially averaged correlation function for the perpendicular
# component of the magnetic field, raised to the power of gamma = 2
perp_gamma_2_rad_corr1 = (sfr(perp_gamma_2_corr1, num_bins, verbose = False))[1]
perp_gamma_2_rad_corr2 = (sfr(perp_gamma_2_corr2, num_bins, verbose = False))[1]

# Calculate the normalised correlation function for the magnetic field
# perpendicular to the line of sight, for gamma = 2. This is the left hand
# side of equation 19, and the right hand side of equation 15. Do this for 
# both simulations.
mag_gamma_2_norm_corr1 = (perp_gamma_2_rad_corr1 - mag_sq_mean_gamma_2_1)\
 / (mag_mean_sq_gamma_2_1 - mag_sq_mean_gamma_2_1)
mag_gamma_2_norm_corr2 = (perp_gamma_2_rad_corr2 - mag_sq_mean_gamma_2_2)\
 / (mag_mean_sq_gamma_2_2 - mag_sq_mean_gamma_2_2)

# Print a message to show that the normalised correlation function for 
# gamma = 2 has been calculated
print 'The normalised correlation function for gamma = 2 has been calculated'

# -------------------- Plots of LHS and RHS Equation 19 ------------------------

# Here we want to produce one plot with two subplots. One subplot should compare
# the LHS and RHS for the low magnetic field simulation, and the other subplot
# should compare the LHS and RHS for the high magnetic field simulation

# Create a figure to hold both of the subplots
fig = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the low 
# magnetic field simulation
ax1 = fig.add_subplot(121)

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array1, RHS_19_1, 'b-o') 
plt.plot(radius_array1, mag_gamma_2_norm_corr1, 'r-o')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000, len(radius_array2)), np.zeros(np.shape(radius_array1)), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Add a label to the x-axis
# plt.xlabel('Radial Separation [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Normalized Correlation Function', fontsize = 20)

# Create an axis for the first subplot to be produced, which is for the high
# magnetic field simulation. Make the y axis limits the same as for the low
# magnetic field plot
ax2 = fig.add_subplot(122, sharey = ax1)

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array2, RHS_19_2, 'b-o', label = 'RHS Eq. 19') 
plt.plot(radius_array2, mag_gamma_2_norm_corr2, 'r-o', label = 'LHS Eq. 19')

# Plot a faded dashed line to represent the line y = 0
plt.plot(np.linspace(0,1000, len(radius_array2)), np.zeros(np.shape(radius_array2)), 'k--', alpha = 0.5)

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Radial Separation [pixels]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend()

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.09, 0.95, 'a) Sim 3: b.1p.01', fontsize = 18)

# Add some text to the figure, to label the right plot as figure b
plt.figtext(0.55, 0.95, 'b) Sim 11: b1p.01', fontsize = 18)

# Make sure that all of the labels are clearly visible in the plot
#plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig2.eps', format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()