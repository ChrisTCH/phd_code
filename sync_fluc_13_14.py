#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates equations 13 and 14 of Lazarian and Pogosyan #
# 2012, to check if these equations are correct.                               #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 11/9/2014                                                        #
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

# Calculate the average of the x-component of the magnetic field squared
mag_x_mean_sq = np.mean( np.power(mag_x_data, 2.0) )

# Calculate the average of the y-component of the magnetic field squared
mag_y_mean_sq = np.mean( np.power(mag_y_data, 2.0) )

# Print a message to the screen to show that the data has been loaded
print 'Magnetic field components loaded successfully'

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
# x-component of the magnetic field. This is the left hand side of equation 13
# of Lazarian and Pogosyan 2012.
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
# y-component of the magnetic field. This is the left hand side of equation 14
# of Lazarian and Pogosyan 2012.
c_2 = y_rad_av_corr[1] / mag_y_mean_sq

# Print a message to show that c_2 has been calculated
print 'Normalised correlation function for the y-component of the magnetic'\
+ ' has been calculated'

# --------------- Structure function x-comp B field ---------------------------

# Calculate the structure function for the x-component of the magnetic field
x_sf = sf_fft(mag_x_data, no_fluct = True)

# Print a message to show that the structure function of the x-component of
# the magnetic field has been calculated
print 'Structure function of the x-component of the magnetic field calculated'

# Calculate the radially averaged structure function for the x-component
# of the magnetic field
x_rad_av_sf = sfr(x_sf, num_bins)

# Calculate the right hand side of equation 13 of Lazarian and Pogosyan 2012
# NOTE: The equation as specified in the paper is wrong. The structure function
# needs to be divided by the average of the x-component of the magnetic field
# squared.
RHS_13 = 1.0 - 0.5 * x_rad_av_sf[1] / mag_x_mean_sq

# Print a message to show that the right hand side of equation 13 has been
# calculated
print 'The right hand side of equation 13 has been calculated'

# --------------- Structure function y-comp B field ---------------------------

# Calculate the structure function for the y-component of the magnetic field
y_sf = sf_fft(mag_y_data, no_fluct = True)

# Print a message to show that the structure function of the y-component of
# the magnetic field has been calculated
print 'Structure function of the y-component of the magnetic field calculated'

# Calculate the radially averaged structure function for the y-component
# of the magnetic field
y_rad_av_sf = sfr(y_sf, num_bins)

# Calculate the right hand side of equation 14 of Lazarian and Pogosyan 2012
# NOTE: The equation as specified in the paper is wrong. The structure function
# needs to be divided by the average of the y-component of the magnetic field
# squared.
RHS_14 = 1.0 - 0.5 * y_rad_av_sf[1] / mag_y_mean_sq

# Print a message to show that the right hand side of equation 14 has been
# calculated
print 'The right hand side of equation 14 has been calculated'

# -------------------- Plots of LHS and RHS Equation 13 -----------------------

# Create a figure to display a plot comparing the left and right hand sides of 
# Equation 13
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the left and right hand sides of equation 13 on the same plot
plt.plot(radius_array, RHS_13, 'b-o', label = 'RHS Eq. 13') 
plt.plot(radius_array, c_1, 'r-o', label = 'Norm Corr x-comp B')

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax1.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Normalised Correlation Function', fontsize = 20)

# Add a title to the plot
plt.title('Comparison LHS and RHS Eq. 13', fontsize = 20)

# Force the legend to appear on the plot
plt.legend()

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_13_Norm_Corr_Comp_3.png', format = 'png')

# Create a figure to display a plot showing the difference between the left
# and right hand sides of Equation 13
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the left and right hand sides of equation 13 on the same plot
plt.plot(radius_array, RHS_13 - c_1, 'b-o') 

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax2.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Difference between LHS and RHS', fontsize = 20)

# Add a title to the plot
plt.title('Difference LHS and RHS Eq. 13', fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_13_Side_Diff_3.png', format = 'png')

# Print a message to show that the plots were successfully created for Eq 13
print 'Comparison plots created for equation 13'

# -------------------- Plots of LHS and RHS Equation 14 -----------------------

# Create a figure to display a plot of the left and right hand sides of 
# Equation 14
fig3 = plt.figure()

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the left and right hand sides of equation 14 on the same plot
plt.plot(radius_array, RHS_14, 'b-o', label = 'RHS Eq. 14') 
plt.plot(radius_array, c_2, 'r-o', label = 'Norm Corr y-comp B')

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax3.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Normalised Correlation Function', fontsize = 20)

# Add a title to the plot
plt.title('Comparison LHS and RHS Eq. 14', fontsize = 20)

# Force the legend to appear on the plot
plt.legend()

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_14_Norm_Corr_Comp_3.png', format = 'png')

# Create a figure to display a plot showing the difference between the left
# and right hand sides of Equation 14
fig4 = plt.figure()

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the left and right hand sides of equation 14 on the same plot
plt.plot(radius_array, RHS_14 - c_2, 'b-o') 

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax4.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Difference between LHS and RHS', fontsize = 20)

# Add a title to the plot
plt.title('Difference LHS and RHS Eq. 14', fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Eq_14_Side_Diff_3.png', format = 'png')

# Print a message to show that the plots were successfully created for Eq 14
print 'Comparison plots created for equation 14'