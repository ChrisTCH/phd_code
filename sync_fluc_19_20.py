#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates equations 19 and 20 of Lazarian and Pogosyan #
# 2012, to check if these equations are correct.                               #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 8/9/2014                                                         #
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
num_bins = 15

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# The two directories end in b.1p2_Aug_Burk and b1p2_Aug_Burk.
spec_loc = 'b1p2_Aug_Burk/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc
 
# Open the FITS file that contains the x-component of the simulated magnetic
# field
mag_x_fits = fits.open(data_loc + 'magx.fits')

# Extract the data for the simulated x-component of the magnetic field
mag_x_data = mag_x_fits[0].data

# Open the FITS file that contains the y-component of the simulated magnetic 
# field
mag_y_fits = fits.open(data_loc + 'magy.fits')

# Extract the data for the simulated y-component of the magnetic field
mag_y_data = mag_y_fits[0].data

# Print a message to the screen to show that the data has been loaded
print 'Magnetic field components loaded successfully'

# Calculate the magnitude of the magnetic field perpendicular to the line of 
# sight, which is just the square root of the sum of the x and y component
# magnitudes squared.
mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

# Calculate the average of the x-component of the magnetic field squared
mag_x_mean_sq = np.mean( np.power(mag_x_data, 2.0) )

# Calculate the average of the y-component of the magnetic field squared
mag_y_mean_sq = np.mean( np.power(mag_y_data, 2.0) )

# Print a message to show that the perpendicular component of the magnetic
# field has been calculated
print 'Perpendicular component of the magnetic field calculated'

# ---------------- Normalised correlation x-comp B field ----------------------

# Calculate the correlation function for the x-component of the magnetic field
x_corr = cf_fft(mag_x_data, no_fluct = True)

# Print a message to show that the correlation function of the x-component of
# the magnetic field has been calculated
print 'Correlation function of the x-component of the magnetic field calculated'

# Calculate the radially averaged correlation function for the x-component
# of the magnetic field
x_rad_av_corr = sfr(x_corr, num_bins)

# Extract the radius values used to calculate the radially averaged 
# correlation function
radius_array = x_rad_av_corr[0]

# Calculate the normalised radially averaged correlation function for the 
# x-component of the magnetic field. This is equation 13 of Lazarian and 
# Pogosyan 2012.
c_1 = x_rad_av_corr[1] / mag_x_mean_sq

# Print a message to show that c_1 has been calculated
print 'Normalised correlation function for the x-component of the magnetic'\
+ ' has been calculated'

# ---------------- Normalised correlation y-comp B field ----------------------

# Calculate the correlation function for the y-component of the magnetic field
y_corr = cf_fft(mag_y_data, no_fluct = True)

# Print a message to show that the correlation function of the y-component of
# the magnetic field has been calculated
print 'Correlation function of the y-component of the magnetic field calculated'

# Calculate the radially averaged correlation function for the y-component
# of the magnetic field
y_rad_av_corr = sfr(y_corr, num_bins)

# Calculate the normalised radially averaged correlation function for the 
# y-component of the magnetic field. This is equation 14 of Lazarian and
# Pogosyan 2012.
c_2 = y_rad_av_corr[1] / mag_y_mean_sq

# Print a message to show that c_2 has been calculated
print 'Normalised correlation function for the y-component of the magnetic'\
+ ' has been calculated'

# Calculate the right hand side of equation 19 of Lazarian and Pogosyan 2012
RHS_19 = 0.5 * ( np.power(c_1, 2.0) + np.power(c_2, 2.0) )

# Calculate the right hand side of equation 20 of Lazarian and Pogosyan 2012
RHS_20 = 0.4 * ( np.power(c_1, 2.0) + np.power(c_2, 2.0) ) + 3.0/40.0 *\
 ( np.power(c_1, 4.0) + np.power(c_2, 4.0) ) + 1.0/20.0 * np.power(c_1, 2.0) *\
  np.power(c_2, 2.0)

# Print a message to show that the right hand sides of the equations have
# been calculated correctly
print 'The right hand sides of equations 19 and 20 have been calculated'

# -------------- Normalised correlation B_perp gamma = 2 ----------------------

# Calculate the result of raising the magnetic field strength perpendicular 
# to the line of sight to the power of gamma = 2
mag_perp_gamma_2 = np.power(mag_perp, 2.0)

# Calculate the square of the mean of the perpendicular component of the 
# magnetic field raised to the power of gamma = 2
mag_sq_mean_gamma_2 = np.power(np.mean(mag_perp_gamma_2), 2.0)

# Calculate the mean of the squared perpendicular component of the magnetic
# field raised to the power of gamma = 2
mag_mean_sq_gamma_2 = np.mean( np.power(mag_perp_gamma_2, 2.0) )

# Calculate the correlation function for the perpendicular component of the
# magnetic field, when raised to the power of gamma = 2
perp_gamma_2_corr = cf_fft(mag_perp_gamma_2, no_fluct = True)

# Print a message to show that the correlation function of the perpendicular 
# component of the magnetic field has been calculated for gamma = 2
print 'Correlation function of the perpendicular component of the magnetic'\
+ ' field calculated for gamma = 2'

# Calculate the radially averaged correlation function for the perpendicular
# component of the magnetic field, raised to the power of gamma = 2
perp_gamma_2_rad_corr = (sfr(perp_gamma_2_corr, num_bins))[1]

# Calculate the normalised correlation function for the magnetic field
# perpendicular to the line of sight, for gamma = 2. This is the left hand
# side of equation 19, and the right hand side of equation 15.
mag_gamma_2_norm_corr = (perp_gamma_2_rad_corr - mag_sq_mean_gamma_2)\
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
mag_sq_mean_gamma_4 = np.power(np.mean(mag_perp_gamma_4), 2.0)

# Calculate the mean of the squared perpendicular component of the magnetic
# field raised to the power of gamma = 4
mag_mean_sq_gamma_4 = np.mean( np.power(mag_perp_gamma_4, 2.0) )

# Calculate the correlation function for the perpendicular component of the
# magnetic field, when raised to the power of gamma = 4
perp_gamma_4_corr = cf_fft(mag_perp_gamma_4, no_fluct = True)

# Print a message to show that the correlation function of the perpendicular 
# component of the magnetic field has been calculated for gamma = 2
print 'Correlation function of the perpendicular component of the magnetic'\
+ ' field calculated for gamma = 4'

# Calculate the radially averaged correlation function for the perpendicular
# component of the magnetic field, raised to the power of gamma = 4
perp_gamma_4_rad_corr = (sfr(perp_gamma_4_corr, num_bins))[1]

# Calculate the normalised correlation function for the magnetic field
# perpendicular to the line of sight, for gamma = 4. This is the left hand
# side of equation 19, and the right hand side of equation 15.
mag_gamma_4_norm_corr = (perp_gamma_4_rad_corr - mag_sq_mean_gamma_4)\
 / (mag_mean_sq_gamma_4 - mag_sq_mean_gamma_4)

# Print a message to show that the normalised correlation function for 
# gamma = 4 has been calculated
print 'The normalised correlation function for gamma = 4 has been calculated'

# -------------------- Plots of LHS and RHS Equation 19 -----------------------

# Create a figure to display a plot comparing the left and right hand sides of 
# Equation 19
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array, RHS_19, 'b-o', label = 'RHS Eq. 19') 
plt.plot(radius_array, mag_gamma_2_norm_corr, 'r-o', label = 'Norm Corr B Perp')

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax1.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Normalised Correlation Function', fontsize = 20)

# Add a title to the plot
plt.title('Comparison LHS and RHS Eq. 19', fontsize = 20)

# Force the legend to appear on the plot
plt.legend()

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_19_Norm_Corr_Comp_3.png', format = 'png')

# Create a figure to display a plot showing the difference between the left
# and right hand sides of Equation 19
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the left and right hand sides of equation 19 on the same plot
plt.plot(radius_array, RHS_19 - mag_gamma_2_norm_corr, 'b-o') 

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax2.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Difference between LHS and RHS', fontsize = 20)

# Add a title to the plot
plt.title('Difference LHS and RHS Eq. 19', fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_19_Side_Diff_3.png', format = 'png')

# Print a message to show that the plots were successfully created for gamma = 2
print 'Comparison plots created for gamma = 2, equation 19'

# -------------------- Plots of LHS and RHS Equation 20 -----------------------

# Create a figure to display a plot of the left and right hand sides of 
# Equation 20
fig3 = plt.figure()

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the left and right hand sides of equation 20 on the same plot
plt.plot(radius_array, RHS_20, 'b-o', label = 'RHS Eq. 20') 
plt.plot(radius_array, mag_gamma_4_norm_corr, 'r-o', label = 'Norm Corr B Perp')

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax3.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Normalised Correlation Function', fontsize = 20)

# Add a title to the plot
plt.title('Comparison LHS and RHS Eq. 20', fontsize = 20)

# Force the legend to appear on the plot
plt.legend()

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_20_Norm_Corr_Comp_3.png', format = 'png')

# Create a figure to display a plot showing the difference between the left
# and right hand sides of Equation 20
fig4 = plt.figure()

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the left and right hand sides of equation 20 on the same plot
plt.plot(radius_array, RHS_20 - mag_gamma_4_norm_corr, 'b-o') 

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax4.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Difference between LHS and RHS', fontsize = 20)

# Add a title to the plot
plt.title('Difference LHS and RHS Eq. 20', fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_20_Side_Diff_3.png', format = 'png')

# Print a message to show that the plots were successfully created for gamma = 4
print 'Comparison plots created for gamma = 4, equation 20'