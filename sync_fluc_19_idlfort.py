#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, as well as correlation functions and structure functions    #
# calculated in IDL and Fortran, and calculates equation 19 of Lazarian and    #
# Pogosyan 2012, to check if this equation is correct.                         #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 3/10/2014                                                        #
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
from line_plots import line_plots
from fractalcube import fractalcube

# Set a variable to hold the number of bins to use in calculating the 
# correlation functions
num_bins = 15

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# The two directories end in b.1p2_Aug_Burk and b1p2_Aug_Burk.
spec_loc = 'b.1p2_Aug_Burk/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc
 
# Open the FITS file that contains the x-component of the simulated magnetic
# field
mag_x_fits = fits.open(data_loc + 'magx.fits')

# Extract the data for the simulated x-component of the magnetic field
mag_x_data = mag_x_fits[0].data

# # Extract the first octant of the x-component of the magnetic field data, to
# # greatly speed up the processing time
# mag_x_data = mag_x_data[0:256, 0:256, 0:256]

# Open the FITS file that contains the y-component of the simulated magnetic 
# field
mag_y_fits = fits.open(data_loc + 'magy.fits')

# Extract the data for the simulated y-component of the magnetic field
mag_y_data = mag_y_fits[0].data

# # Extract the first octant of the y-component of the magnetic field data, to
# # greatly speed up the processing time
# mag_y_data = mag_y_data[0:256, 0:256, 0:256]

# ------ Use this code to test with fractal data

# # Create a cube of fractal data, which is meant to represent the x component
# # of the magnetic field
# mag_x_data = fractalcube(3.0, seed = 6, size = 256)

# # # slicing(mag_x_data, xlabel = 'x', ylabel = 'y',\
# # #  zlabel = 'z', title = 'x-comp B Field')

# # Create a cube of fractal data, which is meant to represent the y component 
# # of the magnetic field
# mag_y_data = fractalcube(3.0, seed = 8, size = 256)

# # # slicing(mag_y_data, xlabel = 'x', ylabel = 'y',\
# # #  zlabel = 'z', title = 'y-comp B Field')

# ------ End fractal data generation

# Print a message to the screen to show that the data has been loaded
print 'Magnetic field components loaded successfully'

# Calculate the magnitude of the magnetic field perpendicular to the line of 
# sight, which is just the square root of the sum of the x and y component
# magnitudes squared.
mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

# Calculate the average of the x-component of the magnetic field squared
mag_x_mean_sq = np.mean( np.power(mag_x_data, 2.0), dtype = np.float64 )

# Calculate the average of the y-component of the magnetic field squared
mag_y_mean_sq = np.mean( np.power(mag_y_data, 2.0), dtype = np.float64 )

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

# Calculate the radially averaged correlation function for the perpendicular
# component of the magnetic field, raised to the power of gamma = 2
perp_gamma_2_rad_corr = (sfr(perp_gamma_2_corr, num_bins))[1]

# Calculate the normalised correlation function for the magnetic field
# perpendicular to the line of sight, for gamma = 2. This is the left hand
# side of equation 19, and the right hand side of equation 15.
mag_gamma_2_norm_corr = (perp_gamma_2_rad_corr - mag_sq_mean_gamma_2)\
 / (mag_mean_sq_gamma_2 - mag_sq_mean_gamma_2)

# # Radially average the left hand side of equation 19, to compare to the 
# # calculated right hand side
# mag_gamma_2_norm_corr = (sfr(mag_gamma_2_norm_corr_3D, num_bins))[1]

# Print a message to show that the normalised correlation function for 
# gamma = 2 has been calculated
print 'The normalised correlation function for gamma = 2 has been calculated'

# Print a message to show that the left and right hand sides of equation 19 have
# been calculated correctly
print 'The left and right hand sides of equation 19 have been calculated'

# ------------------- Calculating the LHS and RHS for IDL ---------------------

# Open the FITS file containing the IDL calculation of the correlation function
# for the x component of the magnetic field. This is already radially averaged.
# b.1p2_IDL_bx_corr.fits for actual data, frac_IDL_x_corr.fits for fractal
idl_corr_x = fits.open(data_loc + 'b.1p2_IDL_bx_corr.fits')

# Extract the data for the correlation function from this file
idl_corr_x = (idl_corr_x[0].data)[:,1]

# Open the FITS file containing the IDL calculation of the correlation function
# for the y component of the magnetic field. This is already radially averaged.
# b.1p2_IDL_by_corr.fits for actual data, frac_IDL_y_corr.fits for fractal
idl_corr_y = fits.open(data_loc + 'b.1p2_IDL_by_corr.fits')

# Extract the data for the correlation function from this file
idl_corr_y = (idl_corr_y[0].data)[:,1]

# Open the FITS file containing the IDL calculation of the correlation function
# for the perpendicular component of the magnetic field squared. This is already
# radially averaged.
# b.1p2_IDL_bperp2_corr.fits for actual data, frac_IDL_perp_corr.fits for fractal
idl_corr_perp2 = fits.open(data_loc + 'b.1p2_IDL_bperp2_corr.fits')

# Extract the data for the correlation function from this file
idl_corr_perp2 = (idl_corr_perp2[0].data)[:,1]

# Calculate the normalised radially averaged correlation function for the x
# component of the magnetic field, using the IDL calculation
c_1_idl = idl_corr_x / mag_x_mean_sq

# Calculate the normalised radially averaged correlation function for the y
# component of the magnetic field, using the IDL calculation
c_2_idl = idl_corr_y / mag_y_mean_sq

# Calculate the right hand side of equation 19 of Lazarian and Pogosyan 2012,
# using the IDL calculation
RHS_19_idl = 0.5 * ( np.power(c_1_idl, 2.0) + np.power(c_2_idl, 2.0) )

# Calculate the left hand side of equation 19, using the IDL calculation
LHS_19_idl = (idl_corr_perp2 - mag_sq_mean_gamma_2)\
 / (mag_mean_sq_gamma_2 - mag_sq_mean_gamma_2)

# Print a message to show that the left and right hand sides of equation 19 have
# been calculated correctly for IDL
print 'The left and right hand sides of equation 19 have been calculated for IDL'

# ----------------- Calculating the LHS and RHS for Fortran -------------------

# Open the FITS file containing the Fortran calculation of the structure 
# function for the x component of the magnetic field. This is already radially
# averaged.
fort_struc_x = fits.open(data_loc + 'b.1p2_magx_corr.fits')

# Get the data for the second order structure function from this file
fort_struc_x = (fort_struc_x[0].data)[:,1]

# Open the FITS file containing the Fortran calculation of the structure 
# function for the y component of the magnetic field. This is already radially
# averaged.
fort_struc_y = fits.open(data_loc + 'b.1p2_magy_corr.fits')

# Get the data for the second order structure function from this file
fort_struc_y = (fort_struc_y[0].data)[:,1]

# Open the FITS file containing the Fortran calculation of the structure 
# function for the perpendicular component of the magnetic field squared. This
# is already radially averaged.
fort_struc_perp = fits.open(data_loc + 'glo_dens_corr.fits')

# Get the data for the second order structure function from this file
fort_struc_perp = (fort_struc_perp[0].data)[:,1]

# Calculate the normalised radially averaged correlation function for the x
# component of the magnetic field using the structure function, using the 
# Fortran calculation
c_1_fort = 1.0 - 0.5 * fort_struc_x / mag_x_mean_sq

# Calculate the normalised radially averaged correlation function for the y
# component of the magnetic field using the structure function, using the 
# Fortran calculation
c_2_fort = 1.0 - 0.5 * fort_struc_y / mag_y_mean_sq

# Calculate the radially averaged correlation for the perpendicular 
# component of the magnetic field squared, from the corresponding structure
# function
fort_corr_perp2 = mag_mean_sq_gamma_2 - 0.5 * fort_struc_perp

# Calculate the right hand side of equation 19 of Lazarian and Pogosyan 2012,
# using the Fortran calculation
RHS_19_fort = 0.5 * ( np.power(c_1_fort, 2.0) + np.power(c_2_fort, 2.0) )

# Calculate the left hand side of equation 19, using the Fortran calculation
LHS_19_fort = (fort_corr_perp2 - mag_sq_mean_gamma_2)\
 / (mag_mean_sq_gamma_2 - mag_sq_mean_gamma_2)

# Print a message to show that the left and right hand sides of equation 19 have
# been calculated correctly for Fortran
print 'The left and right hand sides of equation 19 have been calculated for Fortran'

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

# # Calculate the radially averaged correlation function for the x-component
# # of the magnetic field squared
# x2_rad_av_corr = sfr(x2_corr, num_bins)

# # Calculate the correlation function for the y-component of the magnetic 
# # field squared
# y2_corr = cf_fft(np.power(mag_y_data,2.0), no_fluct = True)

# # Print a message to show that the correlation function of the y-component of
# # the magnetic field squared has been calculated
# print 'Correlation function of the y-component of the magnetic field squared'\
# + ' calculated'

# # Calculate the radially averaged correlation function for the y-component
# # of the magnetic field squared
# y2_rad_av_corr = sfr(y2_corr, num_bins)

# Calculate the modified version of equation 19, which is on page 97 of PhD
# Logbook 3
# RHS_19 = (x2_rad_av_corr[1] - mag_x_sq_mean_sq + y2_rad_av_corr[1] -\
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

# # Calculate the radially averaged cross-correlation function for the squared
# # x and y components of the magnetic field.
# x2_y2_rad_av_corr = sfr(x2_y2_corr, num_bins)

# # Calculate the correlation function for the y-component of the magnetic field
# # squared and the x-component of the magnetic field squared
# y2_x2_corr = cf_fft(np.power(mag_y_data,2.0), np.power(mag_x_data,2.0),\
# 	no_fluct = True )

# # Print a message to show that the correlation function of the squared y and x
# # components of the magnetic field has been calculated
# print 'Correlation function of the squared y and x-components of the magnetic'\
# + ' field calculated'

# # Calculate the radially averaged cross-correlation function for the squared
# # y and x components of the magnetic field.
# y2_x2_rad_av_corr = sfr(y2_x2_corr, num_bins)

# # Calculate the modified version of equation 19, which is on page 96 of PhD
# # Logbook 3, and does not assume that there is no correlation between the 
# # squared x and y components of the magnetic field
# RHS_19 = (x2_rad_av_corr[1] - mag_x_sq_mean_sq + y2_rad_av_corr[1] -\
#  mag_y_sq_mean_sq + x2_y2_rad_av_corr[1] + y2_x2_rad_av_corr[1] - 2.0 *\
#   mag_x_mean_sq * mag_y_mean_sq) / (mag_x_mean_four + 2.0 * mag_x2_y2_mean\
#    - mag_x_sq_mean_sq + mag_y_mean_four - mag_y_sq_mean_sq - 2.0 *\
#     mag_x_mean_sq * mag_y_mean_sq)

# -------------------- Plots of LHS and RHS Equation 19 -----------------------

# Create a dictionary that will hold all of the information on things to plot
plot_dict = {}

# Add an entry to the dictionary for the LHS calculated by Python
plot_dict['LHS_Py'] = [radius_array, mag_gamma_2_norm_corr, 'b-']
# Add an entry to the dictionary for the RHS calculated by Python
plot_dict['RHS_Py'] = [radius_array, RHS_19, 'r-']

# Add an entry to the dictionary for the LHS calculated by IDL
plot_dict['LHS_IDL'] = [radius_array, LHS_19_idl, 'g-']
# Add an entry to the dictionary for the RHS calculated by IDL
plot_dict['RHS_IDL'] = [radius_array, RHS_19_idl, 'k-']

# Add an entry to the dictionary for the LHS calculated by IDL
plot_dict['LHS_Fort'] = [np.array(range(len(LHS_19_fort))), LHS_19_fort, 'm-']
# Add an entry to the dictionary for the RHS calculated by IDL
plot_dict['RHS_Fort'] = [np.array(range(len(LHS_19_fort))), RHS_19_fort, 'c-']

# Plot all of the left and right hand sides
line_plots(plot_dict, data_loc + 'Eq19_idlfort_comp6.png', 'png', xlabel =\
 'Radial separation [pixels]', ylabel = 'Normalised Correlation Function',\
 title = 'Comparison LHS and RHS Eq. 19', linewidth = 1, markersize = 6,\
  log_x = True, log_y = False,\
  loc = 1)

# Create a dictionary to hold the information on the relative difference 
# between the LHS and RHS, for each code language. Tests whether the LHS and RHS
# are found to be the same in a certain code language.
eq_comp_dict = {}

# Add an entry to this dictionary to compare the relative difference for Python
eq_comp_dict['Python'] = [radius_array, np.abs(RHS_19 - mag_gamma_2_norm_corr)\
/ np.abs(mag_gamma_2_norm_corr), 'b-' ]

# Add an entry to this dictionary to compare the relative difference for IDL
eq_comp_dict['IDL'] = [radius_array, np.abs(RHS_19_idl - LHS_19_idl)\
/ np.abs(LHS_19_idl), 'r-' ]

# Add and entry to this dictionary to compare the relative difference for Fortran
eq_comp_dict['Fortran'] = [np.array(range(len(LHS_19_fort))), \
np.abs(RHS_19_fort - LHS_19_fort) / np.abs(LHS_19_fort), 'g-' ]

# Plot all of the relative differences between the left and right hand sides
line_plots(eq_comp_dict, data_loc + 'Eq19_idlfort_reldiff3.png', 'png', xlabel =\
 'Radial separation [pixels]', ylabel = 'Relative difference',\
 title = 'Comparison LHS and RHS Eq. 19', linewidth = 1, markersize = 6,\
  log_x = True, log_y = False,\
  loc = 1, ymin = -1.0, ymax = 1.0)

# Create a dictionary to plot the relative difference between all RHS terms, and
# all LHS terms. Tests whether the code languages are giving the same result.
lang_dict = {}

# Add an entry to compare the Python and IDL RHSs
lang_dict['RHS Py-IDL'] = [radius_array, np.abs(RHS_19 - RHS_19_idl) /\
 np.abs(RHS_19), 'b-']

# # Add an entry to compare the Python and Fortran RHSs
# lang_dict['RHS Py-Fort'] = [radius_array, np.abs(RHS_19 - RHS_19_fort) /\
#  RHS_19, 'r-']

# Add an entry to compare the Python and IDL LHSs
lang_dict['LHS Py-IDL'] = [radius_array, np.abs(mag_gamma_2_norm_corr -\
 LHS_19_idl) / np.abs(mag_gamma_2_norm_corr), 'g-']

# # Add an entry to compare the Python and Fortran LHSs
# lang_dict['LHS Py-Fort'] = [radius_array, np.abs(mag_gamma_2_norm_corr -\
#  LHS_19_fort) / mag_gamma_2_norm_corr, 'k-']

# Plot the relative differences between the code languages
line_plots(lang_dict, data_loc + 'Eq19_idlfort_codediff3.png', 'png', xlabel =\
 'Radial separation [pixels]', ylabel = 'Relative difference',\
 title = 'Comparison LHS and RHS Eq. 19 Codes', linewidth = 1, markersize = 6,\
  log_x = True, log_y = False,\
  loc = 1, ymin = -1.0, ymax = 1.0)

# # Create a figure to display a plot comparing the left and right hand sides of 
# # Equation 19
# fig1 = plt.figure()

# # Create an axis for this figure
# ax1 = fig1.add_subplot(111)

# # Plot the left and right hand sides of equation 19 on the same plot
# plt.plot(radius_array, RHS_19, 'b-o', label = 'RHS Eq. 19') 
# plt.plot(radius_array, mag_gamma_2_norm_corr, 'r-o', label = 'Norm Corr B Perp')

# # Make the x axis of the plot logarithmic
# ax1.set_xscale('log')

# # Make the y axis of the plot logarithmic
# #ax1.set_yscale('log')

# # Add a label to the x-axis
# plt.xlabel('Radial Separation R', fontsize = 20)

# # Add a label to the y-axis
# plt.ylabel('Normalised Correlation Function', fontsize = 20)

# # Add a title to the plot
# plt.title('Comparison LHS and RHS Eq. 19', fontsize = 20)

# # Force the legend to appear on the plot
# plt.legend()

# # Save the figure using the given filename and format
# plt.savefig(data_loc + 'Eq_19_Corr_Comp_3.png', format = 'png')

# # Create a figure to display a plot showing the difference between the left
# # and right hand sides of Equation 19
# fig2 = plt.figure()

# # Create an axis for this figure
# ax2 = fig2.add_subplot(111)

# # Plot the left and right hand sides of equation 19 on the same plot
# plt.plot(radius_array, RHS_19 - mag_gamma_2_norm_corr, 'b-o') 

# # Make the x axis of the plot logarithmic
# ax2.set_xscale('log')

# # Make the y axis of the plot logarithmic
# #ax2.set_yscale('log')

# # Add a label to the x-axis
# plt.xlabel('Radial Separation R', fontsize = 20)

# # Add a label to the y-axis
# plt.ylabel('Difference between LHS and RHS', fontsize = 20)

# # Add a title to the plot
# plt.title('Difference LHS and RHS Eq. 19', fontsize = 20)

# # Save the figure using the given filename and format
# plt.savefig(data_loc + 'Eq_19_Side_Diff_3.png', format = 'png')

# # Print a message to show that the plots were successfully created for gamma = 2
# print 'Comparison plots created for gamma = 2, equation 19'

# -------------------------- Plots of c_1 and c_2 -----------------------------

# # Create a figure to display a plot of c_1 and c_2
# fig5 = plt.figure()

# # Create an axis for this figure
# ax5 = fig5.add_subplot(111)

# # Plot the radially averaged c_1 and c_2 on the same plot
# plt.plot(radius_array, c_1, 'b-o', label = 'c_1') 
# plt.plot(radius_array, c_2, 'r-o', label = 'c_2')

# # Make the x axis of the plot logarithmic
# ax5.set_xscale('log')

# # Make the y axis of the plot logarithmic
# #ax1.set_yscale('log')

# # Add a label to the x-axis
# plt.xlabel('Radial Separation R', fontsize = 20)

# # Add a label to the y-axis
# plt.ylabel('Normalised Correlation Function', fontsize = 20)

# # Add a title to the plot
# plt.title('Comparison c_1 and c_2', fontsize = 20)

# # Force the legend to appear on the plot
# plt.legend()

# # Save the figure using the given filename and format
# plt.savefig(data_loc + 'c1_c2_Comp_1.png', format = 'png')