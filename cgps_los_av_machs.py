#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated velocities,   #
# and calculates the line of sight averaged sonic and Alfvenic Mach numbers of #
# the simulation for a particular line of sight.                               #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 27/5/2016                                                        #
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

# Create an array, where each entry specifies the initial mean magnetic field of
# the corresponding simulation to study 
mag_arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,\
	1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 5.0])

# Create an array, where each entry specifies the initial sound speed of the
# corresponding simulation to study. This should only be used when the sound
# speed is being calculated from the initial condition of the simulation, so
# that an assumption is made that the sound speed at each pixel does not change
# as the simulation evolves. If the assumption being made is that the 
# simulations are isobaric, so that the pressure at each pixel is the same as 
# the simulations evolve, then the sound speed needs to be calculated at each 
# pixel.
# Calculate the initial sound speed (assumed to be equal to the sound speed
# at any later time) for each simulation, assuming a uniform density of 1
c_s_arr = np.sqrt(press_arr)

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the Mach number
# maps. This can be 'x', 'y', or 'z'. The mean magnetic field is along the 
# x axis.
line_o_sight = 'x'

# Depending on the line of sight, the axis along which we average is different 
if line_o_sight == 'z':
	# Construct a variable which tells the script which axis we need to 
	# average along to calculate the Mach number maps. Since the line of sight
	# is the z axis, we need to average along axis 0. (Numpy convention is 
	# that axes are ordered as (z, y, x))
	int_axis = 0
elif line_o_sight == 'y':
	# Construct a variable which tells the script which axis we need to 
	# average along to calculate the Mach number maps. Since the line of sight
	# is the y axis, we need to average along axis 1.
	int_axis = 1
elif line_o_sight == 'x':
	# Construct a variable which tells the script which axis we need to 
	# average along to calculate the Mach number maps. Since the line of sight
	# is the x axis, we need to average along axis 2.
	int_axis = 2

# Iterate over the simulations, to produce Mach number maps for each
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

	# Find any index values where the density array is less than or equal to 
	# zero
	invalid_dens = dens_data <= 0.0

	# Set the invalid values to a very small, positive number
	dens_data[invalid_dens] = 1.0e-4

	# Load magnetic fields

	# Open the FITS file that contains the z-component of the simulated magnetic 
	# field
	mag_z_fits = fits.open(data_loc + 'magz.fits')

	# Open the FITS file that contains the y-component of the simulated magnetic 
	# field
	mag_y_fits = fits.open(data_loc + 'magy.fits')

	# Open the FITS file that contains the x-component of the simulated magnetic
	# field
	mag_x_fits = fits.open(data_loc + 'magx.fits')

	# Extract the data for the z-component of the magnetic field
	mag_z_data = mag_z_fits[0].data

	# Extract the data for the y-component of the magnetic field
	mag_y_data = mag_y_fits[0].data

	# Extract the data for the x-component of the magnetic field
	mag_x_data = mag_x_fits[0].data
	
	# Print a message to the screen to show that the data has been loaded
	print 'Magnetic field components loaded successfully'

	# Load velocities

	# Open the FITS file that contains the z-component of the simulated velocity 
	# field
	vel_z_fits = fits.open(data_loc + 'velz.fits')

	# Open the FITS file that contains the y-component of the simulated velocity 
	# field
	vel_y_fits = fits.open(data_loc + 'vely.fits')

	# Open the FITS file that contains the x-component of the simulated velocity
	# field
	vel_x_fits = fits.open(data_loc + 'velx.fits')

	# Extract the data for the z-component of the velocity field
	vel_z_data = vel_z_fits[0].data

	# Extract the data for the y-component of the velocity field
	vel_y_data = vel_y_fits[0].data

	# Extract the data for the x-component of the velocity field
	vel_x_data = vel_x_fits[0].data

	# Print a message to the screen to show that the data has been loaded
	print 'Velocity field components loaded successfully'

	# Calculate the magnitude of the magnetic field at each pixel
	mag_amp_arr = np.sqrt( np.power(mag_x_data,2.0) + np.power(mag_y_data,2.0)\
	+ np.power(mag_z_data,2.0) )

	# Calculate the magnitude of the velocity field at each pixel
	vel_amp_arr = np.sqrt( np.power(vel_x_data,2.0) + np.power(vel_y_data,2.0)\
	+ np.power(vel_z_data,2.0) )

	# Calculate the Alfven velocity at every pixel for this simulation
	v_alf = mag_amp_arr / np.sqrt(dens_data)

	# Calculate the sonic Mach number for the simulation (when c_s calculated 
	# once for each simulation, assuming c_s is not time dependent, and is
	# uniform)
	sonic_mach = vel_amp_arr / c_s_arr[i]

	# Calculate the Alfvenic Mach number for the simulation
	alf_mach = vel_amp_arr / v_alf

	# Average the sonic Mach number along the line of sight. Note the
	# array is ordered by (z,y,x)!
	los_sonic = np.mean(sonic_mach, axis = int_axis, dtype = np.float64)

	# Average the Alfvenic Mach number along the line of sight. Note the
	# array is ordered by (z,y,x)!
	los_alf = np.mean(alf_mach, axis = int_axis, dtype = np.float64)

	# Print a message to the screen stating that the Mach number maps have been
	# produced
	print 'Mach number maps calculated'

	# Now that the Mach number maps have been produced, we need to save the 
	# produced map as a FITS file

	# To do this, we need to make a FITS header, so that anyone using the FITS
	# file in the future will know what gamma values were used

	# Create a primary HDU to contain the sonic Mach number data
	pri_hdu_sonic = fits.PrimaryHDU(los_sonic)

	# Create a primary HDU to contain the Alfvenic Mach data
	pri_hdu_alf = fits.PrimaryHDU(los_alf)

	# Save the produced sonic Mach number map as a FITS file
	mat2FITS_Image(los_sonic, pri_hdu_sonic.header, data_loc + 'los_av_sonic_' +\
	 line_o_sight + '.fits', clobber = True)

	# Save the produced Alfvenic Mach number map as a FITS file
	mat2FITS_Image(los_alf, pri_hdu_alf.header, data_loc + 'los_av_alf_' +\
	 line_o_sight + '.fits', clobber = True)

	# Close all of the fits files, to save memory
	mag_x_fits.close()
	mag_y_fits.close()
	mag_z_fits.close()
	vel_x_fits.close()
	vel_y_fits.close()
	vel_z_fits.close()
	dens_fits.close()

	# Print a message to state that the FITS file was saved successfully
	print 'FITS file of Mach number maps saved successfully {}'.format(simul_arr[i])

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All Mach number maps calculated successfully'