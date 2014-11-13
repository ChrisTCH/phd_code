#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the observed synchrotron emission maps for a #
# cube that is saturated with a uniform, isotropic distribution of cosmic rays #
# with power spectrum index gamma. Various values of gamma are used, and the   #
# depth of the cube being used to calculate the synchrotron maps is altered.   #
# For each synchrotron map calculated, the structure function and quadrupole / #
# monopole ratio are calculated, to see how the depth along the line of sight  #
# affects these quantities.                                                    #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 7/11/2014                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the functions that calculate the structure and correlation functions
# using FFT, the function that calculates the radially averaged 
# structure or correlation functions, and the function that calculates
# multipoles of 2D images
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 25

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# The directories end in:
# b.1p.1_Oct_Burk
# b.1p.01_Oct_Burk
# b.1p2_Aug_Burk
# b1p.1_Oct_Burk
# b1p.01_Oct_Burk
# b1p2_Aug_Burk
# c512b.1p.0049
# c512b.1p.05
# c512b.1p.7
# c512b1p.0049
# c512b1p.05
# c512b1p.7
# c512b3p.01
# c512b5p.01
# c512b5p2
spec_loc = 'b1p2_Aug_Burk/'

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + spec_loc

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can be 'x', 'y', or 'z'
line_o_sight = 'z'

# Create an array that specifies the gamma values that were used to produce
# these synchrotron emission maps
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Select the index corresponding to the desired value of gamma to use
gam_index = 2

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

# Open the FITS file that contains the z-component of the simulated magnetic 
# field
mag_z_fits = fits.open(data_loc + 'magz.fits')

# Extract the data for the simulated z-component of the magnetic field
mag_z_data = mag_z_fits[0].data

# Print a message to the screen to show that the data has been loaded
print 'Magnetic field components loaded successfully'

# Depending on the line of sight, the strength of the magnetic field 
# perpendicular to the line of sight is calculated in different ways
if line_o_sight == 'z':
	# Calculate the magnitude of the magnetic field perpendicular to the line of
	# sight, which is just the square root of the sum of the x and y component
	# magnitudes squared.
	mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the synchrotron maps. Since the line of sight
	# is the z axis, we need to integrate along axis 0. (Numpy convention is 
	# that axes are ordered as (z, y, x))
	int_axis = 0
elif line_o_sight == 'y':
	# Calculate the magnitude of the magnetic field perpendicular to the line of
	# sight, which is just the square root of the sum of the x and z component
	# magnitudes squared.
	mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_z_data, 2.0) )

	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the synchrotron maps. Since the line of sight
	# is the y axis, we need to integrate along axis 1.
	int_axis = 1
elif line_o_sight == 'x':
	# Calculate the magnitude of the magnetic field perpendicular to the line of
	# sight, which is just the square root of the sum of the y and z component
	# magnitudes squared.
	mag_perp = np.sqrt( np.power(mag_y_data, 2.0) + np.power(mag_z_data, 2.0) )

	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the synchrotron maps. Since the line of sight
	# is the x axis, we need to integrate along axis 2.
	int_axis = 2

# Calculate the shape of the array for the magnitude of the magnetic field 
# perpendicular to the line of sight.
b_perp_shape = np.shape(mag_perp)

# Calculate the index that refers to the last slice of the data cube
max_index = b_perp_shape[int_axis]

# Calculate the minimum index (smallest depth) to be used in the study
min_index = max_index - 300

# Create an array of values representing the number of slices along the line of
# sight that are to be included when calculating the synchrotron map
depth_arr = np.linspace(min_index, max_index, num = 31)

# Create an array of zeroes, which will hold the radially averaged structure
# functions calculated for the synchrotron data. This array is 2 dimensional, 
# with the same number of rows as depth values, the number of columns
# is equal to the number of bins being used to calculate the correlation 
# functions.
sf_mat = np.zeros((len(depth_arr), num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# each structure function. This array has the same shape as the array holding
# the radially averaged structure functions
rad_arr = np.zeros((len(depth_arr), num_bins))

# Create an array of zeroes, which will hold the values of the slope of the
# radially averaged structure functions - 1. There is one such value for each
# depth value of interest.
m_arr = np.zeros(len(depth_arr))

# Create an array of zeroes, which will hold the values of the residuals from
# fitting a line to the radially averaged structure functions on small scales.
# There is one for each value of depth.
residual_arr = np.zeros(len(depth_arr))

# Create an array of zeroes, which will hold the values of the multipole
# ratios calculated for the synchrotron data. This array is 2 dimensional, 
# with the same number of rows as depth values, the number of columns is equal
# to the number of bins being used to calculate the multipole ratios.
quad_mat = np.zeros((len(depth_arr), num_bins))

# Create an array of zeroes, which will hold the radius values used to calculate
# each multipole ratio. This array has the same shape as the array holding
# the values of the multipole ratios.
multi_rad_arr = np.zeros((len(depth_arr), num_bins))

# Loop over the depth values, as we will produce a synchrotron map for each one
for i in range(len(depth_arr)):
	# Depending on which axis we are integrating along, slice the data cube
	# in a different way
	if int_axis == 0:
		# Selecting the relevant slices of the data cube for this depth, z axis
		mag_perp_slice = mag_perp[0:depth_arr[i],:,:]
	elif int_axis == 1:
		# Selecting the relevant slices of the data cube for this depth, y axis
		mag_perp_slice = mag_perp[:,0:depth_arr[i],:]
	elif int_axis == 2:
		# Selecting the relevant slices of the data cube for this depth, x axis
		mag_perp_slice = mag_perp[:,:,0:depth_arr[i]]

	# Calculate the result of raising the perpendicular magnetic field strength
	# to the power of gamma, for these slices
	mag_perp_gamma = np.power(mag_perp_slice, gamma_arr[gam_index])

	# Integrate the perpendicular magnetic field strength raised to the power
	# of gamma along the required axis, to calculate the observed synchrotron 
	# map for these slices. This integration is performed by the trapezoidal 
	# rule. To normalise the calculated synchrotron map, divide by the number 
	# of pixels along the integration axis. Note the array is ordered by(z,y,x)!
	# NOTE: Set dx to whatever the pixel spacing is
	sync_arr = np.trapz(mag_perp_gamma, dx = 1.0, axis = int_axis) /\
	 np.shape(mag_perp_slice)[int_axis]

	# Print a message to the screen stating that the synchrotron map has been
	# produced
	print 'Synchrotron map calculated for depth = {}'.\
	format(np.shape(mag_perp_slice)[int_axis])

	# Calculate the 2D structure function for the synchrotron intensity map. 
	# Note that no_fluct = True is set, because we are not subtracting the 
	# mean from the synchrotron maps before calculating the structure function
	strfn = sf_fft(sync_arr, no_fluct = True)

	# Radially average the calculated 2D structure function, using the 
	# specified number of bins
	rad_sf = sfr(strfn, num_bins, verbose = False)

	# Insert the calculated radially averaged structure function
	# into the matrix that stores all of the calculated structure functions
	sf_mat[i] = rad_sf[1]

	# Insert the radius values used to calculate this structure function
	# into the matrix that stores the radius values
	rad_arr[i] = rad_sf[0]

	# Print a message to show that the structure function has been calculated
	print 'Radially averaged structure function calculated for'\
	+ ' depth = {}'.format(depth_arr[i])

	# Calculate the spectral index of the structure function calculated for
	# this depth. Note that only the first third of the structure
	# function is used in the calculation, as this is the part that is 
	# close to a straight line.
	spec_ind_data = np.polyfit(np.log10(rad_arr[i,0:np.ceil(num_bins/3.0)]),\
		np.log10(sf_mat[i,0:np.ceil(num_bins/3.0)]), 1, full = True)

	# Extract the returned coefficients from the polynomial fit
	coeff = spec_ind_data[0]

	# Calculate the value of m, and store it in the corresponding array
	m_arr[i] = coeff[0] - 1

	# Extract the sum of the residuals from the polynomial fit, and store it
	# in the corresponding array
	residual_arr[i] = spec_ind_data[1]

	# Calculate the normalised 2D structure function for this synchrotron
	# intensity map. Note that no_fluct = True is set, because we are
	# not subtracting the mean from the synchrotron maps before calculating
	# the structure function. We are also calculating the normalised structure
	# function, which only takes values between 0 and 2.
	norm_strfn = sf_fft(sync_arr, no_fluct = True, normalise = True)

	# Shift the 2D structure function so that the zero radial separation
	# entry is in the centre of the image.
	norm_strfn = np.fft.fftshift(norm_strfn)

	# Calculate the monopole for the normalised structure function
	monopole_arr, mono_rad_arr = calc_multipole_2D(norm_strfn, order = 0,\
	 num_bins = num_bins)

	# Calculate the quadrupole for the normalised structure function
	quadpole_arr, quad_rad_arr = calc_multipole_2D(norm_strfn, order = 2,\
	 num_bins = num_bins)

	# Insert the calculated multipole ratios into the matrix that stores all
	# of the calculated multipole
	quad_mat[i] = quadpole_arr / monopole_arr

	# Insert the radius values used to calculate this multipole ratio
	# into the matrix that stores the radius values
	multi_rad_arr[i] = mono_rad_arr

	# Print a message to show that the multipole ratio has been calculated
	print 'Multipoles calculated for {}'.format(depth_arr[i])

# When the code reaches this point, all structure functions and multipole
# moments have been calculated

# Create a figure on which to plot the values of m as a function of depth
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the values of m as a function of depth
plt.plot(depth_arr, m_arr, 'b-o')

# Add a label to the x-axis
plt.xlabel('Slice Depth [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1 (m)', fontsize = 20)

# Add a title to the plot
plt.title('Sync Int SF Slope vs Depth Gam {}'.format(gamma_arr[gam_index]), fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'SynInt_SF_Slope_Depth_Gam{}.png'.\
	format(gamma_arr[gam_index]), format = 'png')

# Print a message to the screen to show that the plot of the structure function
# slope as a function of depth has been created
print 'Plot of the structure function slope vs depth saved'

# Create a figure on which to plot the residuals as a function of depth
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the values of the residuals as a function of depth
plt.plot(depth_arr, residual_arr, 'b-o')

# Add a label to the x-axis
plt.xlabel('Slice Depth [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Sync Int SF Residual vs Depth Gam {}'.format(gamma_arr[gam_index]), fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'SynInt_SF_Resid_Depth_Gam{}.png'.\
	format(gamma_arr[gam_index]), format = 'png')

# Print a message to the screen to show that the plot of the structure function
# slope as a function of depth has been created
print 'Plot of the residuals vs depth saved'

# Create a figure to display a plot comparing the radially
# averaged structure functions for some of the synchrotron maps
fig3 = plt.figure()

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot some of the radially averaged structure functions 
plt.plot(rad_arr[0], sf_mat[0], 'b-o', label ='Depth = {}'.format(depth_arr[0]))
plt.plot(rad_arr[3], sf_mat[3], 'b--o', label ='Depth = {}'.format(depth_arr[3]))
plt.plot(rad_arr[6], sf_mat[6], 'r-o', label ='Depth = {}'.format(depth_arr[6]))
plt.plot(rad_arr[9], sf_mat[9], 'r--o', label ='Depth = {}'.format(depth_arr[9]))
plt.plot(rad_arr[12], sf_mat[12], 'g-o', label ='Depth = {}'.format(depth_arr[12]))
plt.plot(rad_arr[15], sf_mat[15], 'g--o', label ='Depth = {}'.format(depth_arr[15]))
plt.plot(rad_arr[18], sf_mat[18], 'c-o', label ='Depth = {}'.format(depth_arr[18]))
plt.plot(rad_arr[21], sf_mat[21], 'c--o', label ='Depth = {}'.format(depth_arr[21]))
plt.plot(rad_arr[24], sf_mat[24], 'm-o', label ='Depth = {}'.format(depth_arr[24]))
plt.plot(rad_arr[27], sf_mat[27], 'm--o', label ='Depth = {}'.format(depth_arr[27]))

# Make the x axis of the plot logarithmic
ax3.set_xscale('log')

# Make the y axis of the plot logarithmic
ax3.set_yscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Structure Function', fontsize = 20)

# Add a title to the plot
plt.title('Sync Int Str Fun Gamma {}'.format(gamma_arr[gam_index]), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 4)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'SynInt_SF_Depth_Gam{}.png'.\
	format(gamma_arr[gam_index]), format = 'png')

# Print a message to the screen to show that the plot of all of the synchrotron
# structure functions has been saved
print 'Plot of the radially averaged structure functions'\
+ ' for synchrotron intensity saved'

# Create a figure to display a plot comparing the quadrupole / monopole
# ratios for some of the synchrotron maps
fig4 = plt.figure()

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot some of the quadrupole / monopole ratios 
plt.plot(multi_rad_arr[0], quad_mat[0], 'b-o', label ='Depth = {}'.format(depth_arr[0]))
plt.plot(multi_rad_arr[3], quad_mat[3], 'b--o', label ='Depth = {}'.format(depth_arr[3]))
plt.plot(multi_rad_arr[6], quad_mat[6], 'r-o', label ='Depth = {}'.format(depth_arr[6]))
plt.plot(multi_rad_arr[9], quad_mat[9], 'r--o', label ='Depth = {}'.format(depth_arr[9]))
plt.plot(multi_rad_arr[12], quad_mat[12], 'g-o', label ='Depth = {}'.format(depth_arr[12]))
plt.plot(multi_rad_arr[15], quad_mat[15], 'g--o', label ='Depth = {}'.format(depth_arr[15]))
plt.plot(multi_rad_arr[18], quad_mat[18], 'c-o', label ='Depth = {}'.format(depth_arr[18]))
plt.plot(multi_rad_arr[21], quad_mat[21], 'c--o', label ='Depth = {}'.format(depth_arr[21]))
plt.plot(multi_rad_arr[24], quad_mat[24], 'm-o', label ='Depth = {}'.format(depth_arr[24]))
plt.plot(multi_rad_arr[27], quad_mat[27], 'm--o', label ='Depth = {}'.format(depth_arr[27]))

# Make the x axis of the plot logarithmic
ax4.set_xscale('log')

# Add a label to the x-axis
plt.xlabel('Radial Separation R', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Quad / Monopole', fontsize = 20)

# Add a title to the plot
plt.title('Quad / Monopole Gamma {}'.format(gamma_arr[gam_index]), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc = 4)

# Save the figure using the given filename and format
plt.savefig(data_loc + 'Quad_Mono_DepthComp_Gam{}.png'.\
	format(gamma_arr[gam_index]), format = 'png')

# Print a message to the screen to show that the plot of all of the synchrotron
# structure functions has been saved
print 'Plot of the radially averaged structure functions'\
+ ' for synchrotron intensity saved'

# Close the figures so that they don't stay in memory
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)