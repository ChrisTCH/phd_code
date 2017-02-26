#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths and density and calculates the observed Stokes Q and U       #
# image of the simulation for any line of sight, any frequency, and for the    #
# cases where emission comes from within the cube, or the cube is backlit by   #
# polarised emission.                                                          #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 24/10/2016                                                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import mat2FITS_Image to convert arrays to FITS format
from mat2FITS_Image import mat2FITS_Image

# Import the function that can calculate the Stokes Q and U images/cubes
from calc_StoQU import calc_StoQU

# Create a string for the directory that contains the simulated magnetic field
# and density cubes to use. 
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

# Create a string for the directory that will be used to save all of the output
save_loc = '/Volumes/CAH_ExtHD/Pol_2016/'

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
# c512b.5p.0049
# c512b.5p.0077
# c512b.5p.01
# c512b.5p.025
# c512b.5p.05
# c512b.5p.1
# c512b.5p.7
# c512b.5p2
# c512b1p.0049
# c512b1p.0077
# c512b1p.025
# c512b1p.05
# c512b1p.7

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['b1p2_Aug_Burk/']
# simul_arr = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
# 'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
# 'b.1p2_Aug_Burk/', 'c512b.5p.0049/', 'c512b.5p.0077/', 'c512b.5p.01/',\
# 'c512b.5p.025/', 'c512b.5p.05/', 'c512b.5p.1/', 'c512b.5p.7/', 'c512b.5p2/',\
# 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
# 'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
# 'b1p2_Aug_Burk/']


# Create a list, where each entry is the name of the folder that will be used to
# store the results for each simulation
save_sim_arr = ['b1p2/']
# save_sim_arr = ['b.1p.0049/', 'b.1p.0077/', 'b.1p.01/','b.1p.025/', 'b.1p.05/',\
# 'b.1p.1/', 'b.1p.7/', 'b.1p2/', 'b.5p.0049/', 'b.5p.0077/', 'b.5p.01/',\
# 'b.5p.025/', 'b.5p.05/', 'b.5p.1/', 'b.5p.7/', 'b.5p2/', 'b1p.0049/',\
# 'b1p.0077/', 'b1p.01/', 'b1p.025/', 'b1p.05/', 'b1p.1/', 'b1p.7/', 'b1p2/']


# Create an array, where each entry specifies the pressure of the corresponding
# simulation in the list of simulation directories
press_arr = [2.0]
# press_arr = np.array([0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
# 	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
# 	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0])

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the images or 
# cubes of Stokes Q or U. This can be 'x', 'y', or 'z'. The mean magnetic field 
# is along the x axis.
line_o_sight = 'z'

# Create a string that selects the folder to store Q and U in based on the LOS
los_loc = line_o_sight + '_los/'

# Depending on the line of sight, the observed Stokes Q and U will be different
if line_o_sight == 'z':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate Stokes Q and U. Since the line of sight
	# is the z axis, we need to integrate along axis 0. (Numpy convention is 
	# that axes are ordered as (z, y, x))
	int_axis = 0
elif line_o_sight == 'y':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate Stokes Q and U. Since the line of sight
	# is the y axis, we need to integrate along axis 1.
	int_axis = 1
elif line_o_sight == 'x':
	# Construct a variable which tells the script which axis we need to 
	# integrate along to calculate Stokes Q and U. Since the line of sight
	# is the x axis, we need to integrate along axis 2.
	int_axis = 2

# Create a variable that controls whether the Stokes Q and U maps/cubes are 
# calculated for the case where the simulation cube is backlit by polarised
# emission, or when polarised emission is generated from within the cube.
# This can be 'backlit', 'internal', or both.
emis_mech = ['internal']

# Create a variable that specifies the frequency at which mock observations 
# should be performed in the case where the simulation cube is backlit by
# polarised emission. This needs to be in Hz.
freq_backlit = 1.4 * np.power(10.0, 9.0)

# Create a variable that specifies the range of wavelengths at which mock
# observeations should be performed in the case where the observed emission 
# is generated within the simulation cube. These values are for the wavelength
# squared, to give observing frequencies between 0.5 - 2 GHz.
lambda_sq_arr = np.linspace(0.0225, 0.36, 50, endpoint = True) 

# Calculate the frequencies corresponding to the chosen values of the wavelength
# squared. All values should be given in Hz.
freq_internal = 3.0 * np.power(10.0,8.0) / np.sqrt(lambda_sq_arr)

# Create a variable that specifies the size of a pixel in parsecs
dl = 0.15

# Create a variable that specifies the density scaling in cm^-3
n_e = 0.2

# Create a variable that specifies the mass density scaling in kg m^-3
rho_0 = n_e * 1.67 * np.power(10.0, -21.0)

# Create a variable that specifies the permeability of free space
mu_0 = 4.0 * np.pi * np.power(10.0, -7.0)

# Create a variable that specifies the sound speed in gas with a temperature of
# 8000 K, in m s^-1
c_s = 10.15 * np.power(10.0,3.0)

# Create a variable that specifies the spectral index of the synchrotron 
# emission (physically reasonable values are between 0 and -3)
spec_ind = -1.0

# Iterate over the simulations, to produce Stokes Q and U maps for each
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

	# Open the files that contain the three components of the magnetic field
	magx_fits = fits.open(data_loc + 'magx.fits')
	magy_fits = fits.open(data_loc + 'magy.fits')
	magz_fits = fits.open(data_loc + 'magz.fits')

	# Extract the data for the three components of the magnetic field
	magx_data = magx_fits[0].data
	magy_data = magy_fits[0].data
	magz_data = magz_fits[0].data
	
	# Create a variable that specifies the velocity scaling in m s^-1
	v_0 = c_s / np.sqrt(press_arr[i])

	# Calculate the magnetic field scaling for this simulation in micro Gauss
	B_0 = np.sqrt(mu_0 * rho_0 * np.power(v_0,2.0)) / np.power(10.0,-10.0)

	# Scale the magnetic field components to physical units of micro Gauss
	magx_data = B_0 * magx_data
	magy_data = B_0 * magy_data
	magz_data = B_0 * magz_data

	# Print a message to the screen to show that the data has been loaded
	print 'All required data loaded successfully'

	# Check to see if maps of Stokes Q and U are to be produced for the case
	# when the cube is backlit by polarised emission, or when emission is 
	# generated from within the cube
	if 'backlit' in emis_mech:
		# In this case we need to produce images of Stokes Q and U for the case
		# when the cube is backlit by polarised emission, at a single frequency

		# Run the calc_StoQU function, to calculate the Stokes Q and U images
		StoQ, StoU = calc_StoQU(dens_data, magx_data, magy_data, magz_data,\
		 freq_backlit, dl, int_axis, 'backlit', spec_ind)

		# Now create FITS headers for Stokes Q and U, so that they can be
		# saved as FITS files

		# Create primary HDUs to contain the Stokes Q and U data
		pri_hdu_StoQ = fits.PrimaryHDU(StoQ)
		pri_hdu_StoU = fits.PrimaryHDU(StoU)

		# Add header keywords to describe the line of sight, frequency,
		# spectral index, pixel size, density, velocity scale, and magnetic
		# field scale for Stokes Q
		pri_hdu_StoQ.header['SIM'] = (save_sim_arr[i], 'simulation used')
		pri_hdu_StoQ.header['LOS'] = (int_axis, '0-z, 1-y, 2-x')
		pri_hdu_StoQ.header['FREQ'] = (freq_backlit, 'observing frequency (Hz)')
		pri_hdu_StoQ.header['SPEC-IND'] = (spec_ind, 'spectral index')
		pri_hdu_StoQ.header['PIX-SIZE'] = (dl, 'size of each pixel (pc)')
		pri_hdu_StoQ.header['DENSITY'] = (n_e, 'density scaling (cm^-3)')
		pri_hdu_StoQ.header['VELSCALE'] = (v_0, 'velocity scaling (m s^-1)')
		pri_hdu_StoQ.header['MAGSCALE'] = (B_0, 'magnetic field scaling (mu G)')

		# Add header keywords to describe the line of sight, frequency,
		# spectral index, pixel size, density, velocity scale, and magnetic
		# field scale for Stokes U
		pri_hdu_StoU.header['SIM'] = (save_sim_arr[i], 'simulation used')
		pri_hdu_StoU.header['LOS'] = (int_axis, '0-z, 1-y, 2-x')
		pri_hdu_StoU.header['FREQ'] = (freq_backlit, 'observing frequency (Hz)')
		pri_hdu_StoU.header['SPEC-IND'] = (spec_ind, 'spectral index')
		pri_hdu_StoU.header['PIX-SIZE'] = (dl, 'size of each pixel (pc)')
		pri_hdu_StoU.header['DENSITY'] = (n_e, 'density scaling (cm^-3)')
		pri_hdu_StoU.header['VELSCALE'] = (v_0, 'velocity scaling (m s^-1)')
		pri_hdu_StoU.header['MAGSCALE'] = (B_0, 'magnetic field scaling (mu G)')

		# Save the produced Stokes Q and U images as FITS files
		mat2FITS_Image(StoQ, pri_hdu_StoQ.header, save_loc + save_sim_arr[i] + \
		 los_loc + 'StoQ_' + 'backlit' + '.fits', clobber = True)
		mat2FITS_Image(StoU, pri_hdu_StoU.header, save_loc + save_sim_arr[i] + \
		 los_loc + 'StoU_' + 'backlit' + '.fits', clobber = True)

	elif 'internal' in emis_mech:
		# In this case we need to produce cubes of Stokes Q and U for the case
		# when the emission is generated within the simulation cube, over a
		# range of frequencies

		# Create empty arrays, that will be used to contain the Stokes Q and U
		# maps created at each frequency
		StoQ = np.zeros((len(freq_internal), dens_data.shape[1],\
		 dens_data.shape[2]), dtype = np.float32)
		StoU = np.zeros((len(freq_internal), dens_data.shape[1],\
		 dens_data.shape[2]), dtype = np.float32)

		# Loop over the frequency array, to produce maps of Stokes Q and U at
		# each frequency
		for j in range(len(freq_internal)):
			# Calculate the Stokes Q and U maps that would be observed at this
			# frequency using the calc_StoQU function, and store the results in
			# the Stokes Q and U arrays
			StoQ[j,:,:], StoU[j,:,:] = calc_StoQU(dens_data, magx_data, \
				magy_data, magz_data, freq_internal[j], dl, int_axis,\
				'internal', spec_ind)

			# Check to see if the number of frequencies that Stokes Q and U have
			# been calculated for is divisible by 10
			if (j+1)%10 == 0:
				# Print out a message to say how many frequencies Stokes
				# Q and U have been calculated for
				print 'Stokes Q and U calculated for {} frequencies'.format(j+1)
	
		# When the code reaches this stage, maps of Stokes Q and U have been
		# produced for every frequency

		# Now create FITS headers for Stokes Q and U, so that they can be
		# saved as FITS files

		# Create primary HDUs to contain the Stokes Q and U data
		pri_hdu_StoQ = fits.PrimaryHDU(StoQ)
		pri_hdu_StoU = fits.PrimaryHDU(StoU)

		# Add header keywords to describe the wavelength squared axis of the 
		# Stokes Q array
		# Specify the reference pixel along the wavelength squared axis
		pri_hdu_StoQ.header['CRPIX3'] = 1

		# Specify the wavelength squared at the reference pixel
		pri_hdu_StoQ.header['CRVAL3'] = lambda_sq_arr[0]

		# Specify the increment in wavelength squared along each slice of the 
		# array
		pri_hdu_StoQ.header['CDELT3'] = lambda_sq_arr[1] - lambda_sq_arr[0]

		# Specify what the third axis is
		pri_hdu_StoQ.header['CTYPE3'] = 'Wavelength Squared (m^2)'

		# Add header keywords to describe the line of sight, spectral index,
		# pixel size, density, velocity scale, and magnetic field scale for 
		# Stokes Q
		pri_hdu_StoQ.header['SIM'] = (save_sim_arr[i], 'simulation used')
		pri_hdu_StoQ.header['LOS'] = (int_axis, '0-z, 1-y, 2-x')
		pri_hdu_StoQ.header['SPEC-IND'] = (spec_ind, 'spectral index')
		pri_hdu_StoQ.header['PIX-SIZE'] = (dl, 'size of each pixel (pc)')
		pri_hdu_StoQ.header['DENSITY'] = (n_e, 'density scaling (cm^-3)')
		pri_hdu_StoQ.header['VELSCALE'] = (v_0, 'velocity scaling (m s^-1)')
		pri_hdu_StoQ.header['MAGSCALE'] = (B_0, 'magnetic field scaling (mu G)')

		# Add header keywords to describe the wavelength squared axis of the 
		# Stokes U array
		# Specify the reference pixel along the wavelength squared axis
		pri_hdu_StoU.header['CRPIX3'] = 1

		# Specify the wavelength squared at the reference pixel
		pri_hdu_StoU.header['CRVAL3'] = lambda_sq_arr[0]

		# Specify the increment in wavelength squared along each slice of the 
		# array
		pri_hdu_StoU.header['CDELT3'] = lambda_sq_arr[1] - lambda_sq_arr[0]

		# Specify what the third axis is
		pri_hdu_StoU.header['CTYPE3'] = 'Wavelength Squared (m^2)'

		# Add header keywords to describe the line of sight, spectral index,
		# pixel size, density, velocity scale, and magnetic field scale for 
		# Stokes U
		pri_hdu_StoU.header['SIM'] = (save_sim_arr[i], 'simulation used')
		pri_hdu_StoU.header['LOS'] = (int_axis, '0-z, 1-y, 2-x')
		pri_hdu_StoU.header['SPEC-IND'] = (spec_ind, 'spectral index')
		pri_hdu_StoU.header['PIX-SIZE'] = (dl, 'size of each pixel (pc)')
		pri_hdu_StoU.header['DENSITY'] = (n_e, 'density scaling (cm^-3)')
		pri_hdu_StoU.header['VELSCALE'] = (v_0, 'velocity scaling (m s^-1)')
		pri_hdu_StoU.header['MAGSCALE'] = (B_0, 'magnetic field scaling (mu G)')

		# Save the produced Stokes Q and U arrays as FITS files
		mat2FITS_Image(StoQ, pri_hdu_StoQ.header, save_loc + save_sim_arr[i] + \
		 los_loc + 'StoQ_' + 'internal' + '.fits', clobber = True)
		mat2FITS_Image(StoU, pri_hdu_StoU.header, save_loc + save_sim_arr[i] + \
		 los_loc + 'StoU_' + 'internal' + '.fits', clobber = True)

	# Close all of the fits files, to save memory
	dens_fits.close()
	magx_fits.close()
	magy_fits.close()
	magz_fits.close()

	# Print a message to state that the FITS files were saved successfully
	print 'FITS files of Stokes Q and U saved successfully {}'.\
	format(save_sim_arr[i])

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All Stokes Q and U maps calculated successfully'