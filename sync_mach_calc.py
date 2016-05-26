#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths and velocities, and calculates the sonic and Alfvenic Mach   #
# numbers of the snapshot of the turbulence. The calculated values are printed #
# to the screen and associated with the corresponding simulation. The mean     #
# magnetic field perpendicular to the line of sight is also calculated.        #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 12/11/2014                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, astropy.io for fits manipulation
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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

# Create an empty array, which will hold the calculated sonic Mach number for 
# each simulation
sonic_mach_arr = np.zeros(len(simul_arr))

# Create an empty array, which will hold the calculated Alfvenic Mach number for
# each simulation
alf_mach_arr = np.zeros(len(simul_arr))

# Create an empty array, which will hold the calculated ratio of the random
# magnetic field divided by the regular magnetic field for each simulation.
mean_ratio_arr = np.zeros(len(simul_arr))

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

	# Find any index values where the density array is less than or equal to 
	# zero
	invalid_dens = dens_data <= 0.0

	# Set the invalid values to a very small, positive number
	dens_data[invalid_dens] = 1.0e-4

	# Print a message to the screen to show that the data has been loaded
	print 'Density loaded successfully'

	# Calculate the magnitude of the magnetic field at each pixel
	mag_amp_arr = np.sqrt( np.power(mag_x_data,2.0) + np.power(mag_y_data,2.0)\
	+ np.power(mag_z_data,2.0) )

	# Calculate the magnitude of the velocity field at each pixel
	vel_amp_arr = np.sqrt( np.power(vel_x_data,2.0) + np.power(vel_y_data,2.0)\
	+ np.power(vel_z_data,2.0) )

	# # Calculate the isothermal speed of sound at each pixel, for this simulation
	# c_s = np.sqrt(press_arr[i] / dens_data)

	# Calculate the Alfven velocity at every pixel for this simulation
	v_alf = mag_amp_arr / np.sqrt(dens_data)

	# # Calculate the sonic Mach number for the simulation (when c_s calculated 
	# # at each pixel, assuming isobaric simulations)
	# sonic_mach = np.mean(vel_amp_arr / c_s, dtype = np.float64)

	# Calculate the sonic Mach number for the simulation (when c_s calculated 
	# once for each simulation, assuming c_s is not time dependent, and is
	# uniform)
	sonic_mach = np.mean(vel_amp_arr / c_s_arr[i], dtype = np.float64)

	# Calculate the Alfvenic Mach number for the simulation
	alf_mach = np.mean(vel_amp_arr / v_alf, dtype = np.float64)

	# Enter the calculated sonic Mach number into the corresponding entry of
	# the array
	sonic_mach_arr[i] = sonic_mach

	# Enter the calculated Alfvenic Mach number into the corresponding entry
	# of the array
	alf_mach_arr[i] = alf_mach

	# Calculate the mean magnetic field strength in the x direction
	mean_Bx = np.mean(mag_x_data, dtype = np.float64)

	# Calculate the mean magnetic field strength in the y direction
	mean_By = np.mean(mag_y_data, dtype = np.float64)

	# Calculate the mean magnetic field strength in the z direction
	mean_Bz = np.mean(mag_z_data, dtype = np.float64)

	# Calculate the random contribution to the magnetic field component in the
	# x direction
	random_Bx = mag_x_data - mean_Bx

	# Calculate the random contribution to the magnetic field component in the
	# y direction
	random_By = mag_y_data - mean_By

	# Calculate the random contribution to the magnetic field component in the
	# z direction
	random_Bz = mag_z_data - mean_Bz

	# Calculate the magnitude of the random contribution to the magnetic field
	# at each pixel
	random_B = np.sqrt( np.power(random_Bx,2.0) + np.power(random_By,2.0)\
	 + np.power(random_Bz,2.0))

	# Calculate the magnitude of the regular component of the magnetic field
	regular_B = np.sqrt( np.power(mean_Bx,2.0) + np.power(mean_By,2.0)\
	 + np.power(mean_Bz,2.0))

	# Calculate the ratio of the random magnetic field strength to the regular
	# magnetic field strength
	ratio_random_regular = random_B / regular_B

	# Calculate the mean of the ratio of the random magnetic field divided by
	# the total magnetic field
	mean_ratio_arr[i] = np.mean(ratio_random_regular, dtype = np.float64)

	# Close all of the fits files, to save memory
	mag_x_fits.close()
	mag_y_fits.close()
	mag_z_fits.close()
	vel_x_fits.close()
	vel_y_fits.close()
	vel_z_fits.close()
	dens_fits.close()

# Print out the sonic Mach number array to the screen
print 'Sonic Mach number array: {}'.format(sonic_mach_arr)

# Print out the Alfvenic Mach number array to the screen
print 'Alfvenic Mach number array: {}'.format(alf_mach_arr)

# Print out the mean ratio of the random B field to the regular B field array
# to the screen
print 'Mean Ratio Random B / Regular B array: {}'.format(mean_ratio_arr)

# Use a for loop to loop over the different simulations
for i in range(len(simul_arr)):
	# Print a line to the screen specifying the initial pressure and magnetic
	# field strength, and the sonic and Alfvenic Mach number for the simulation
	print 'Pressure: {}, B field: {}, Sonic Mach: {}, Alf Mach: {}, Mean random B ratio: {}'.format(\
		press_arr[i], mag_arr[i], sonic_mach_arr[i], alf_mach_arr[i], mean_ratio_arr[i])

	# # Print a line to the screen specifying the initial pressure and magnetic 
	# # field strength, and the sonic Mach number
	# print 'Pressure: {}, B field: {}, Sonic Mach: {}'.format(\
	# 	press_arr[i], mag_arr[i], sonic_mach_arr[i])

# Create a figure that will hold a plot of the ratio of the random magnetic 
# field strength to the regular magnetic field strength against the Alfvenic
# Mach number
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the skewness as a function of sonic Mach number for each gamma
plt.scatter(alf_mach_arr, mean_ratio_arr)

# Add a label to the x-axis
plt.xlabel('Alfvenic Mach Number', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Mean Ratio Random to Regular B', fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/' + 'B_ratio_Alf_mach.png', format = 'png')

# Print a message to the screen to show that the plot of the ratio of the 
# random B field to the regular B field as a function of the Alfvenic Mach 
# number has been saved
print 'Plot of the random B ratio as a function of Alfvenic Mach number saved'

# Close the figure, now that it has been saved.
plt.close()