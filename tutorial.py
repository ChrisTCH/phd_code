#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python version of the tutorial.pro IDL code written by        #
# Alex Lazarian, and available at                                              #
# http://www.astro.wisc.edu/~lazarian/code.html. This is a script designed to  #
# test the various functions available on this website, and to demonstrate how #
# they work.                                                                   #
#                                                                              #
# Author: Chris Herron (adapted from code written by Alex Lazarian)            #
# Start Date: 4/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, sys to exit if a problem occurs, and
# matplotlib for plotting
import numpy as np
import sys
import matplotlib.pyplot as plt

# Import all of the functions required to calculate the power spectrum, 
# correlation function and structure function
from fractalcube import fractalcube
from slicing import slicing
from spektrt import spektrt
from cf_fft import cf_fft
from sf_fft import sf_fft
from sfr import sfr
from setks import setks
from mat_plot import mat_plot

# Print a message to the screen to inform the user that the tutorial is starting
print 'Tutorial: Starting...'

# First we create some fractal data cubes using the function fractalcube in
# "fractalcube.py". beta = 3.7 corresponds to a power spectrum with index -3.7,
# and beta=5.2 corresponds to a power spectrum with index -5.2

# Generate a three dimensional fractal cube with power spectrum index of -3.7
# This cube will be generated to have 128 pixels on each side (the default)
rho1 = fractalcube(3.7, seed = 7.0)

# Print a message to inform the user that the first fractal cube has been made
print 'Tutorial: The first fractal cube has been made'

# Generate a three dimensional fractal cube with power spectrum index of -2.5
# This cube will be generated to have 128 pixels on each side (the default)
rho2 = fractalcube(2.5, seed = 5.0)

# Print a message to the screen to show that the interactive slicing procedure
# is about to start. This allows the data cubes to be visualised.
print 'Tutorial: You are before the slicing procedure \n'

# # Run the slicing function on the rho1 data cube.
# slicing(rho1, 'x', 'y', 'z', 'Fractal Cube: Index -3.7')
# # Run the slicing function on the rho2 data cube.
# slicing(rho2, 'x', 'y', 'z', 'Fractal Cube: Index -2.5')

# Print a message to the screen to show that the interactive slicing procedure
# has completed.
print 'Tutorial: You are after the slicing procedure'

# Integrate the cubes in the z direction using Numpy's sum function. The x-axis
# has index 0, the y axis axis has index 1, and the z axis has index 2.
int1 = np.sum(rho1, axis = 2)
int2 = np.sum(rho2, axis = 2)

# Visualize the resulting 2D maps using the mat_plot function. This function 
# will save an image of the 2D map to the specified filename.

# Save an image of the integrated rho1 fractal cube data
mat_plot(int1, 'frac_cube_2D_3.7.png', 'png', cmap =\
'hot', xlabel = 'x-axis', ylabel = 'y-axis', title = 'Integrated fractal cube:'\
 + ' Index = -3.7')

# Save an image of the integrated rho2 fractal cube data
mat_plot(int2, 'frac_cube_2D_2.5.png', 'png', cmap =\
'hot', xlabel = 'x-axis', ylabel = 'y-axis', title = 'Integrated fractal cube:'\
 + ' Index = -2.5')

# Print a message to the screen to show that the 2D maps were saved successfully
print 'Tutorial: You are after the visualization procedure'

# Now we wish to compute the power spectrum of the (3D) fields, which uses the
# spektrt.py function.
# The 15 is the number of points in k after binning, and the no_pi = True option
# means that we use k=1/lambda instead of k=2 pi/lambda.

# Calculate the power spectrum for rho1
ps1 = spektrt(rho1, 15, no_pi = True)
# Calculate the power spectrum for rho2
ps2 = spektrt(rho2, 15, no_pi = True)

# Print the values of the power spectrum for rho1
print 'Tutorial: Values of rho1 spectrum are ps1 = {}'.format(ps1)

# Now we wish to create plots of the power spectra for rho1 and rho2 

# Create a figure to display the image
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the spectra for rho1 and rho2 on the same plot
plt.plot(ps1[0], ps1[1], 'b-o', label = 'Index -3.7') 
plt.plot(ps2[0], ps2[1], 'r-o', label = 'Index -2.5')

# Make the x axis of the plot logarithmic
ax1.set_xscale('log')

# Make the y axis of the plot logarithmic
ax1.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Wavenumber k', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Power Spectrum', fontsize = 20)

# Add a title to the plot
plt.title('Power Spectra of 3D Fractal Cubes', fontsize = 20)

# Force the legend to appear on the plot
plt.legend()

# Save the figure using the given filename and format
plt.savefig('Frac_cube_3D_spec.png', format = 'png')

# Print a message to the screen saying that the plot of the spectra was
# successfully saved
print 'Tutorial: Plot of power spectra saved successfully for 3D fields'
 
# To check the prescribed spectral indices we can use a linear fit to the
# logarithms (they are power-laws)

# Calculate the gradient of the linear fit to the power spectrum of rho1,
# and print the result to the screen. The result should be -3.7.
print 'Spectral index rho1: {}'.format( (np.polyfit(np.log10(ps1[0]),\
 np.log10(ps1[1]), 1))[0] )

# Calculate the gradient of the linear fit to the power spectrum of rho2,
# and print the result to the screen. The result should be -2.5.
print 'Spectral index rho2: {}'.format( (np.polyfit(np.log10(ps2[0]),\
 np.log10(ps2[1]), 1))[0] )

# Calculate the power spectra for the 2D fields (obtained from the 3D field by
# summing along the z axis)
# Calculate the power spectrum for the integrated rho1 data
psint1 = spektrt(int1, 15, no_pi = True)
# Calculate the power spectrum for the integrated rho2 data
psint2 = spektrt(int2, 15, no_pi = True)

# Now we wish to create plots of the power spectra for int1 and int2 

# Create a figure to display the image
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the spectra for rho1 and rho2 on the same plot
plt.plot(psint1[0], psint1[1], 'b-o', label = 'Index -3.7') 
plt.plot(psint2[0], psint2[1], 'r-o', label = 'Index -2.5')

# Make the x axis of the plot logarithmic
ax2.set_xscale('log')

# Make the y axis of the plot logarithmic
ax2.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Wavenumber k', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Power Spectrum', fontsize = 20)

# Add a title to the plot
plt.title('Power Spectra of 2D Fractal Cubes (z-integrated)', fontsize = 20)

# Force the legend to appear on the plot
plt.legend()

# Save the figure using the given filename and format
plt.savefig('Frac_cube_2D_spec.png', format = 'png')

# Print a message to the screen saying that the plot of the spectra was
# successfully saved for the 2D fields
print 'Tutorial: Plot of power spectra saved successfully for 2D fields'

# Calculate the gradient of the linear fit to the power spectrum of int1,
# and print the result to the screen.
print 'Spectral index int1: {}'.format( (np.polyfit(np.log10(psint1[0]),\
 np.log10(psint1[1]),1))[0] )

# Calculate the gradient of the linear fit to the power spectrum of int2,
# and print the result to the screen.
print 'Spectral index int2: {}'.format( (np.polyfit(np.log10(psint2[0]),\
 np.log10(psint2[1]),1))[0] )

#**********************************************************************

# Here we will calculate the structure functions and correlation functions
# of the fractal cube data using fast fourier transforms.

# Print a message to the screen to inform the user what is happening
print 'Tutorial: Computing structure functions and correlation functions'\
+ ' with FFT and inverse FFT'

# (You can find the routines to do this directly in configuration space
# in sfunc*.pro, you can figure out the differences)

# Calculate the structure function (SF) of int1, the 2D image found by 
# integrating rho1 along the z direction
sf_int1_2D = sf_fft(int1)
# Calculate the correlation function (CF) of int1
cf_int1_2D = cf_fft(int1)

# Calculate the structure function (SF) of int2, the 2D image found by 
# integrating rho2 along the z direction
sf_int2_2D = sf_fft(int2)
# Calculate the correlation function (CF) of int2
cf_int2_2D = cf_fft(int2)

# Print the shape of the returned structure function for int1, to demonstrate
# what is returned by the structure and correlation functions
print 'Tutorial: The shape of the structure function for int1 is = {}'\
.format(np.shape(sf_int1_2D))

# Create an image of the 2D correlation function for int1
mat_plot(cf_int1_2D, 'int1_2D_corr.png', 'png', cmap =\
'hot', xlabel = 'x-axis separation', ylabel = 'y-axis separation', title =\
 '2D Correlation Function for int1')

# Create an image of the 2D correlation function for int2
mat_plot(cf_int2_2D, 'int2_2D_corr.png', 'png', cmap =\
'hot', xlabel = 'x-axis separation', ylabel = 'y-axis separation', title =\
 '2D Correlation Function for int2')

# Print a message to show that the plots of the correlation functions have
# been produced successfully
print 'Tutorial: Plots of the correlation functions for int1 and int2 '\
+ 'produced successfully'

# The results are in 2D, in fact I used these 2D maps to plot the isocountours
# that you took to Northwestern U with Alex. sf_fft, cf_fft work in 3D as well.

# To average in R you can use the function sfr (works also in 3D)

# Calculate the radially averaged structure function for int1. The 15 is the 
# number of bins to use in the radial averaging process.
sf_int1_r = sfr(sf_int1_2D, 15)

# Calculate the radially averaged correlation function for int1.
cf_int1_r = sfr(cf_int1_2D, 15)
# Calculate the radially averaged correlation function for int2.
cf_int2_r = sfr(cf_int2_2D, 15)

# The sfr function returns a matrix, where the first row of the matrix,
# sf_int1_r[0] returns the values of R (radius) used to perform the radial
# averaging, and the second row, sf_int1_r[1], returns the average of the
# structure function values for that value of R.

# Print out the radially averaged structure function for int1, to show what
# the values are.
print 'Tutorial: The values of the radius (top row) and of the radially'\
+ ' averaged structure function (second row) for int1 are sf_int1_r = {}'\
.format(sf_int1_r)

# Create a plot showing the radially averaged structure function and correlation
# function for int1, and the correlation function for int2, on the same plot
# Create a figure to display the image
fig5 = plt.figure()

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot the radially averaged structure function for int1
plt.plot(sf_int1_r[0], sf_int1_r[1], 'b-o', label = 'SF int1')
# Plot the radially averaged correlation functions for int1 and int2 on the plot 
plt.plot(cf_int1_r[0], cf_int1_r[1], 'r-o', label = 'CF int1')
plt.plot(cf_int2_r[0], cf_int2_r[1], 'g-o', label = 'CF int2')

# Make the x axis of the plot logarithmic
ax5.set_xscale('log')

# Make the y axis of the plot logarithmic
#ax5.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Structure or Correlation Function', fontsize = 20)

# Add a title to the plot
plt.title('Radially averaged 2D Structure and Correlation Functions', fontsize = 20)

# Force the legend to appear on the plot
plt.legend()

# Change the figure size to fit everything on the image
plt.tight_layout()

# Save the figure using the given filename and format
plt.savefig('Frac_cube_2D_CF_SF.png', format = 'png')

# Print a message to inform the user that the plot was created successfully
print 'Tutorial: Plot of the radially averaged structure and correlation'\
+ ' functions produced successfully'

# Print a message to inform the user that the tutorial is complete
print 'Tutorial: Tutorial complete!'