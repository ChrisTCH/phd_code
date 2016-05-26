#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the observed synchrotron emission maps for a #
# cube that is saturated with a uniform, isotropic distribution of cosmic rays #
# with power spectrum index gamma. Various values of gamma are used. The       #
# produced maps are stored in a FITS file. This code is to be used with        #
# simulations produced by Christoph Federrath.                                 #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 14/1/2016                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, and h5py for HDF5 manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py

# Import mat2FITS_Image to convert arrays to FITS format, and mat_plot to 
# produce images of matrices
from mat2FITS_Image import mat2FITS_Image
from mat_plot import mat_plot

# Create a string for the directory that contains the simulations
simul_loc = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create a string for the specific simulated data set to use in calculations.
# The directories end in:
# 512sM5Bs5886_20 (Solenoidal turbulence, timestep 20)
# 512sM5Bs5886_25 (Solenoidal turbulence, timestep 25)
# 512sM5Bs5886_30 (Solenoidal turbulence, timestep 30)
# 512sM5Bs5886_35 (Solenoidal turbulence, timestep 35)
# 512sM5Bs5886_40 (Solenoidal turbulence, timestep 40)
# 512m075M5Bs5887_20 (zeta = 0.75, timestep 20)
# 512m075M5Bs5887_25 (zeta = 0.75, timestep 25)
# 512m075M5Bs5887_30 (zeta = 0.75, timestep 30)
# 512m075M5Bs5887_35 (zeta = 0.75, timestep 35)
# 512m075M5Bs5887_40 (zeta = 0.75, timestep 40)
# 512mM5Bs5887_20 (zeta = 0.5, timestep 20)
# 512mM5Bs5887_25 (zeta = 0.5, timestep 25)
# 512mM5Bs5887_30 (zeta = 0.5, timestep 30)
# 512mM5Bs5887_35 (zeta = 0.5, timestep 35)
# 512mM5Bs5887_40 (zeta = 0.5, timestep 40)
# 512m025M5Bs5887_20 (zeta = 0.25, timestep 20)
# 512m025M5Bs5887_25 (zeta = 0.25, timestep 25)
# 512m025M5Bs5887_30 (zeta = 0.25, timestep 30)
# 512m025M5Bs5887_35 (zeta = 0.25, timestep 35)
# 512m025M5Bs5887_40 (zeta = 0.25, timestep 40)
# 512cM5Bs5886_20 (Compressive turbulence, timestep 20)
# 512cM5Bs5886_25 (Compressive turbulence, timestep 25)
# 512cM5Bs5886_30 (Compressive turbulence, timestep 30)
# 512cM5Bs5886_35 (Compressive turbulence, timestep 35)
# 512cM5Bs5886_40 (Compressive turbulence, timestep 40)

# Create a variable that specifies the timestep number
timestep = 40

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['512sM5Bs5886_{}/'.format(timestep), '512m075M5Bs5887_{}/'.format(timestep),\
 '512mM5Bs5887_{}/'.format(timestep), '512m025M5Bs5887_{}/'.format(timestep),\
 '512cM5Bs5886_{}/'.format(timestep)]

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Iterate over the simulations, to produce synchrotron intensity maps for each
# simulation
for i in range(len(simul_arr)): 
	# Print a message to show what simulation calculations are being performed
	# for
	print 'Simulations starting for {}'.format(simul_arr[i])

	# Create a string for the full directory path to use in calculations
	data_loc =  simul_loc + simul_arr[i]
	 
	# Open the HDF5 file that contains the x-component of the simulated magnetic
	# field
	mag_x_hdf = h5py.File(data_loc + 'OUT_hdf5_plt_cnt_00{}_magx'.format(timestep), 'r')

	# Extract the data for the simulated x-component of the magnetic field
	mag_x_data = mag_x_hdf['magx']

	# Open the HDF5 file that contains the y-component of the simulated magnetic 
	# field
	mag_y_hdf = h5py.File(data_loc + 'OUT_hdf5_plt_cnt_00{}_magy'.format(timestep), 'r')

	# Extract the data for the simulated y-component of the magnetic field
	mag_y_data = mag_y_hdf['magy']

	# Open the HDF5 file that contains the z-component of the simulated magnetic 
	# field
	mag_z_hdf = h5py.File(data_loc + 'OUT_hdf5_plt_cnt_00{}_magz'.format(timestep), 'r')

	# Extract the data for the simulated z-component of the magnetic field
	mag_z_data = mag_z_hdf['magz']

	# Print a message to the screen to show that the data has been loaded
	print 'Magnetic field components loaded successfully'

	# Depending on the line of sight, the strength of the magnetic field 
	# perpendicular to the line of sight is calculated in different ways
	if 'z' in line_o_sight:
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the z axis, we need to integrate along axis 1. (Numpy convention is 
		# that axes are ordered as (z, y, x))
		int_axis = 0

		# Create a Numpy array to hold the calculated synchrotron emission map
		sync_arr_z = np.zeros((np.shape(mag_perp)[1],np.shape(mag_perp)[2]))

		# Calculate the result of raising the perpendicular magnetic field 
		# strength to the power of gamma
		mag_perp_gamma_z = np.power(mag_perp, gamma)

		# Integrate the perpendicular magnetic field strength raised to the power
		# of gamma along the required axis, to calculate the observed synchrotron 
		# map. This integration is performed by the trapezoidal rule. To normalise 
		# the calculated synchrotron map, divide by the number of pixels along the 
		# z-axis. Note the array is ordered by (z,y,x)!
		# NOTE: Set dx to whatever the pixel spacing is
		sync_arr_z = np.trapz(mag_perp_gamma_z, dx = 1.0, axis = int_axis) /\
		 np.shape(mag_perp)[0]

		# Now that the synchrotron maps have been produced, we need to save the 
		# produced maps as a FITS file

		# Create a primary HDU to contain the synchrotron data
		pri_hdu = fits.PrimaryHDU(sync_arr_z)

		# Save the produced synchrotron maps as a FITS file
		mat2FITS_Image(sync_arr_z, pri_hdu.header, data_loc + 'synint_z' +\
		  '_gam{}'.format(gamma) + '.fits')

	if 'y' in line_o_sight:
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_z_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the y axis, we need to integrate along axis 1.
		int_axis = 1

		# Create a Numpy array to hold the calculated synchrotron emission map
		sync_arr_y = np.zeros((np.shape(mag_perp)[1],np.shape(mag_perp)[2]))

		# Calculate the result of raising the perpendicular magnetic field 
		# strength to the power of gamma
		mag_perp_gamma_y = np.power(mag_perp, gamma)

		# Integrate the perpendicular magnetic field strength raised to the power
		# of gamma along the required axis, to calculate the observed synchrotron 
		# map. This integration is performed by the trapezoidal rule. To normalise 
		# the calculated synchrotron map, divide by the number of pixels along the 
		# y-axis. Note the array is ordered by (z,y,x)!
		# NOTE: Set dx to whatever the pixel spacing is
		sync_arr_y = np.trapz(mag_perp_gamma_y, dx = 1.0, axis = int_axis) /\
		 np.shape(mag_perp)[0]

		# Now that the synchrotron maps have been produced, we need to save the 
		# produced maps as a FITS file

		# Create a primary HDU to contain the synchrotron data
		pri_hdu = fits.PrimaryHDU(sync_arr_y)

		# Save the produced synchrotron maps as a FITS file
		mat2FITS_Image(sync_arr_y, pri_hdu.header, data_loc + 'synint_y' +\
		  '_gam{}'.format(gamma) + '.fits')
	
	if 'x' in line_o_sight:
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the y and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_y_data, 2.0) + np.power(mag_z_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the x axis, we need to integrate along axis 2.
		int_axis = 2

		# Create a Numpy array to hold the calculated synchrotron emission map
		sync_arr_x = np.zeros((np.shape(mag_perp)[1],np.shape(mag_perp)[2]))

		# Calculate the result of raising the perpendicular magnetic field 
		# strength to the power of gamma
		mag_perp_gamma_x = np.power(mag_perp, gamma)

		# Integrate the perpendicular magnetic field strength raised to the power
		# of gamma along the required axis, to calculate the observed synchrotron 
		# map. This integration is performed by the trapezoidal rule. To normalise 
		# the calculated synchrotron map, divide by the number of pixels along the 
		# x-axis. Note the array is ordered by (z,y,x)!
		# NOTE: Set dx to whatever the pixel spacing is
		sync_arr_x = np.trapz(mag_perp_gamma_x, dx = 1.0, axis = int_axis) /\
		 np.shape(mag_perp)[0]

		# Now that the synchrotron maps have been produced, we need to save the 
		# produced maps as a FITS file

		# Create a primary HDU to contain the synchrotron data
		pri_hdu = fits.PrimaryHDU(sync_arr_x)

		# Save the produced synchrotron maps as a FITS file
		mat2FITS_Image(sync_arr_x, pri_hdu.header, data_loc + 'synint_x' +\
		  '_gam{}'.format(gamma) + '.fits')

	# Print a message to the screen stating that the synchrotron maps have been
	# produced
	print 'Synchrotron maps calculated'

	# Close all of the fits files, to save memory
	mag_x_hdf.close()
	mag_y_hdf.close()
	mag_z_hdf.close()

	# Print a message to state that the FITS file was saved successfully
	print 'FITS file of synchrotron maps saved successfully {}'.format(simul_arr[i])

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All synchrotron maps calculated successfully'