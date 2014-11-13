#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths and velocities, and calculates the sonic and Alfvenic Mach   #
# numbers of the snapshot of the turbulence. The calculated values are printed #
# to the screen and associated with the corresponding simulation.              #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 12/11/2014                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.ndimage to handle rotation of data cubes.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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

# Create an empty array, which will hold the calculated sonic Mach number for 
# each simulation
sonic_mach_arr = np.zeros(len(simul_arr))

# Create an empty array, which will hold the calculated Alfvenic Mach number for
# each simulation
alf_mach_arr = np.zeros(len(simul_arr))

# Loop over the simulations, as we need to calculate the sonic and Alfvenic 
# Mach number for each one
for i in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[i]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[i])

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

	# Open the FITS file that contains the x-component of the simulated velocity
	# field
	vel_x_fits = fits.open(data_loc + 'velx.fits')

	# Extract the data for the simulated x-component of the velocity field
	vel_x_data = vel_x_fits[0].data

	# Open the FITS file that contains the y-component of the simulated velocity 
	# field
	vel_y_fits = fits.open(data_loc + 'vely.fits')

	# Extract the data for the simulated y-component of the velocity field
	vel_y_data = vel_y_fits[0].data

	# Open the FITS file that contains the z-component of the simulated velocity 
	# field
	vel_z_fits = fits.open(data_loc + 'velz.fits')

	# Extract the data for the simulated z-component of the velocity field
	vel_z_data = vel_z_fits[0].data

	# Print a message to the screen to show that the data has been loaded
	print 'Velocity field components loaded successfully'

	# Open the FITS file that contains the density of the simulation
	dens_fits = fits.open(data_loc + 'dens.fits')

	# Extract the data for the simulated density
	dens_data = dens_fits[0].data

	# Print a message to the screen to show that the data has been loaded
	print 'Density loaded successfully'

	# Calculate the magnitude of the magnetic field at each pixel
	mag_amp_arr = np.sqrt( np.power(mag_x_data,2.0) + np.power(mag_y_data,2.0)\
	+ np.power(mag_z_data,2.0) )

	# Calculate the magnitude of the velocity field at each pixel
	vel_amp_arr = np.sqrt( np.power(vel_x_data,2.0) + np.power(vel_y_data,2.0)\
	+ np.power(vel_z_data,2.0) )

	# Calculate the isothermal speed of sound at each pixel, for this simulation
	c_s = np.sqrt(press_arr[i] / dens_data)

	# Calculate the Alfven velocity at every pixel for this simulation
	v_alf = mag_amp_arr / np.sqrt(4.0 * np.pi * dens_data)

	# Calculate the sonic Mach number for the simulation
	sonic_mach = np.mean(vel_amp_arr / c_s, dtype = np.float64)

	# Calculate the Alfvenic Mach number for the simulation
	alf_mach = np.mean(vel_amp_arr / v_alf, dtype = np.float64)

	# Close all of the fits files, to save memory
	mag_x_fits.close()
	mag_y_fits.close()
	mag_z_fits.close()
	vel_x_fits.close()
	vel_y_fits.close()
	vel_z_fits.close()
	dens_fits.close()
