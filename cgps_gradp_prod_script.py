#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the observed polarisation gradient image of  #
# the simulation for a particular line of sight. This assumes that there is no #
# polarised emission from within the cube, that the observing wavelength is    #
# unity, and that the polarisation amplitude is unity everywhere.              #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 26/5/2016                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import mat2FITS_Image to convert arrays to FITS format
from mat2FITS_Image import mat2FITS_Image

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
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

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/', 'c512b3p.01/', 'c512b5p.01/']

# Create an array, where each entry specifies the pressure of the corresponding
# simulation in the list of simulation directories
press_arr = np.array([0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0, 0.01, 0.01])

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can be 'x', 'y', or 'z'. The mean magnetic field is along the 
# x axis.
line_o_sight = 'x'

# Depending on the line of sight, the integral to calculate the rotation
# measure is performed in different ways.
if line_o_sight == 'z':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the rotation measure. Since the line of sight
	# is the z axis, we need to integrate along axis 0. (Numpy convention is 
	# that axes are ordered as (z, y, x))
	int_axis = 0
elif line_o_sight == 'y':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the rotation measure. Since the line of sight
	# is the y axis, we need to integrate along axis 1.
	int_axis = 1
elif line_o_sight == 'x':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate the rotation measure. Since the line of sight
	# is the x axis, we need to integrate along axis 2.
	int_axis = 2

# Create a variable that specifies the size of a pixel in parsecs
dl = 0.15

# Create a variable that specifies the density scaling in cm^-3
n_e = 0.1

# Create a variable that specifies the mass density scaling in kg m^-3
rho_0 = n_e * 1.67 * np.power(10.0, -21.0)

# Create a variable that specifies the permeability of free space
mu_0 = 4.0 * np.pi * np.power(10.0, -7.0)

# Iterate over the simulations, to produce synchrotron intensity maps for each
# simulation
for i in range(len(simul_arr)): 
	# Print a message to show what simulation calculations are being performed
	# for
	print 'Simulations starting for {}'.format(simul_arr[i])

	# Create a string for the full directory path to use in calculations
	data_loc =  simul_loc + simul_arr[i]

	# Open the density file that contains the number density for the simulation
	dens_fits = fits.open(data_loc + 'dens.fits')

	# Extract the data for the simulated density
	dens_data = dens_fits[0].data

	# Scale the density to units of cm^-3
	dens_data = n_e * dens_data

	# Depending on the line of sight, open the magnetic field data for the 
	# component of the magnetic field parallel to the line of sight
	if line_o_sight == 'z':
		# Open the FITS file that contains the z-component of the simulated magnetic 
		# field
		mag_fits = fits.open(data_loc + 'magz.fits')
	elif line_o_sight == 'y':
		# Open the FITS file that contains the y-component of the simulated magnetic 
		# field
		mag_fits = fits.open(data_loc + 'magy.fits')
	elif line_o_sight == 'x':
		# Open the FITS file that contains the x-component of the simulated magnetic
		# field
		mag_fits = fits.open(data_loc + 'magx.fits')

	# Extract the data for the simulated component of the magnetic field
	mag_data = mag_fits[0].data
	
	# Create a variable that specifies the velocity scaling in m s^-1
	v_0 = 10.15 * np.power(10.0,3.0) / np.sqrt(press_arr[i])

	# Calculate the magnetic field scaling for this simulation in micro Gauss
	B_0 = np.sqrt(mu_0 * rho_0 * np.power(v_0,2.0)) / np.power(10.0,-10.0)

	# Scale the magnetic field to physical units of micro Gauss
	mag_data = B_0 * mag_data

	# Print a message to the screen to show that the data has been loaded
	print 'All required data loaded successfully'

	# Integrate the product of the electron density and the magnetic field
	# strength along the line of sight to calculate the rotation measure
	# map. This integration is performed by the trapezoidal rule. Note the
	# array is ordered by (z,y,x)!
	RM_image = 0.81 * np.trapz(dens_data * mag_data, dx = dl, axis = int_axis)

	# Print a message to the screen stating that the rotation measure map has been
	# produced
	print 'Rotation Measure map calculated'

	# Calculate the gradient of the rotation measure
	dRM_dy, dRM_dx = np.gradient(RM_image, dl)
	grad_RM = np.sqrt(np.power(dRM_dx,2.0) + np.power(dRM_dy,2.0))

	# Calculate the polarisation gradient from the rotation measure
	gradP = 2.0 * grad_RM

	# Print a message to the screen stating that the gradient map has been
	# produced
	print 'Polarisation gradient map calculated'

	# Now that the polarisation gradient map has been produced, we need to save the 
	# produced map as a FITS file

	# To do this, we need to make a FITS header, so that anyone using the FITS
	# file in the future will know what gamma values were used

	# Create a primary HDU to contain the rotation measure data
	pri_hdu_RM = fits.PrimaryHDU(RM_image)

	# Create a primary HDU to contain the polarisation gradient data
	pri_hdu_gradP = fits.PrimaryHDU(gradP)

	# Save the produced rotation measure map as a FITS file
	mat2FITS_Image(RM_image, pri_hdu_RM.header, data_loc + 'rot_meas_' +\
	 line_o_sight + '.fits', clobber = True)

	# Save the produced polarisation gradient map as a FITS file
	mat2FITS_Image(gradP, pri_hdu_gradP.header, data_loc + 'polar_grad_' +\
	 line_o_sight + '.fits', clobber = True)

	# Close all of the fits files, to save memory
	mag_fits.close()
	dens_fits.close()

	# Print a message to state that the FITS file was saved successfully
	print 'FITS file of gradient map saved successfully {}'.format(simul_arr[i])

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All gradient maps calculated successfully'