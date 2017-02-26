#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths and density and calculates the observed Stokes Q and U       #
# image of the simulation for any line of sight, any frequency, and for the    #
# cases where emission comes from within the cube, for different spectral      #
# indices. The polarised intensity and polarisation angle are also calculated. #
# This is to test whether the spectral index significantly affects the         #
# observed polarisation.                                                       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 25/10/2016                                                       #
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

# Import the functions that can calculate the polarisation angle and 
# polarisation intensity
from calc_Polar_Angle import calc_Polar_Angle
from calc_Polar_Inten import calc_Polar_Inten

# Create a string for the directory that contains the simulated magnetic field
# and density cubes to use. 
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

# Create a string for the directory that will be used to save all of the output
save_loc = '/Users/chrisherron/Documents/PhD/Pol_2016/Spec_Ind_Study/'

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

# Specify the directory containing the simulation to study
simul = 'b1p2_Aug_Burk/'

# Specify the name of the simulation to be used when saving output
save_sim = 'b1p2'

# Specify the pressure of the simulation
press = 2.0
# press_arr = np.array([0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
# 	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
# 	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0])

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the images or 
# cubes of Stokes Q or U. This can be 'x', 'y', or 'z'. The mean magnetic field 
# is along the x axis.
line_o_sight = 'y'

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

# Create a variable that ensures that the Stokes Q and U maps/cubes are 
# calculated for the case where polarised emission is generated from within the 
# cube. (If the cube is backlit by polarised emission, the spectral index 
# doesn't do anything)
emis_mech = 'internal'

# Create a variable that specifies the frequency at which mock observations 
# should be performed. This needs to be in Hz.
freq = 1.4 * np.power(10.0, 9.0)

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

# Create an array that specifies the spectral indices of the synchrotron 
# emission (physically reasonable values are between 0 and -3)
spec_ind_arr = np.linspace(0.0, -3.0, 7, endpoint = True)

# Print a message to show what simulation calculations are being performed
# for
print 'Simulations starting for {}'.format(simul)

# Create a string for the full directory path to use in calculations
data_loc =  simul_loc + simul

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
v_0 = c_s / np.sqrt(press)

# Calculate the magnetic field scaling for this simulation in micro Gauss
B_0 = np.sqrt(mu_0 * rho_0 * np.power(v_0,2.0)) / np.power(10.0,-10.0)

# Scale the magnetic field components to physical units of micro Gauss
magx_data = B_0 * magx_data
magy_data = B_0 * magy_data
magz_data = B_0 * magz_data

# Print a message to the screen to show that the data has been loaded
print 'All required data loaded successfully'

# Create empty arrays, that will be used to contain the Stokes Q and U
# maps created for each spectral index
StoQ = np.zeros((len(spec_ind_arr), dens_data.shape[1],\
 dens_data.shape[2]), dtype = np.float32)
StoU = np.zeros((len(spec_ind_arr), dens_data.shape[1],\
 dens_data.shape[2]), dtype = np.float32)

# Loop over the spectral index array, to produce maps of Stokes Q and U at
# each spectral index
for j in range(len(spec_ind_arr)):
	# Calculate the Stokes Q and U maps that would be observed for this
	# spectral index using the calc_StoQU function, and store the results in
	# the Stokes Q and U arrays
	StoQ[j,:,:], StoU[j,:,:] = calc_StoQU(dens_data, magx_data, \
		magy_data, magz_data, freq, dl, int_axis, emis_mech, spec_ind_arr[j])

# When the code reaches this stage, maps of Stokes Q and U have been
# produced for every spectral index

# Calculate the polarisation intensity
pol_inten = calc_Polar_Inten(StoQ, StoU)

# Calculate the polarisation angle
pol_angle = calc_Polar_Angle(StoQ, StoU)

# Now create FITS headers for Stokes Q and U, so that they can be
# saved as FITS files

# Create primary HDUs to contain the Stokes Q and U data, as well as polarised
# intensity and the polarisation angle
pri_hdu_StoQ = fits.PrimaryHDU(StoQ)
pri_hdu_StoU = fits.PrimaryHDU(StoU)
pri_hdu_pol_inten = fits.PrimaryHDU(pol_inten)
pri_hdu_pol_angle = fits.PrimaryHDU(pol_angle)

# Add header keywords to describe the spectral index axis of the Stokes Q
# array
# Specify the reference pixel along the spectral index axis
pri_hdu_StoQ.header['CRPIX3'] = 1

# Specify the spectral index at the reference pixel
pri_hdu_StoQ.header['CRVAL3'] = spec_ind_arr[0]

# Specify the increment in spectral index along each slice of the array
pri_hdu_StoQ.header['CDELT3'] = spec_ind_arr[1] - spec_ind_arr[0]

# Specify what the third axis is
pri_hdu_StoQ.header['CTYPE3'] = 'Spec-ind'

# Add header keywords to describe the line of sight, frequency,
# pixel size, density, velocity scale, and magnetic field scale for 
# Stokes Q
pri_hdu_StoQ.header['SIM'] = (save_sim, 'simulation used')
pri_hdu_StoQ.header['LOS'] = (int_axis, '0-z, 1-y, 2-x')
pri_hdu_StoQ.header['FREQ'] = (freq, 'observing frequency (Hz)')
pri_hdu_StoQ.header['PIX-SIZE'] = (dl, 'size of each pixel (pc)')
pri_hdu_StoQ.header['DENSITY'] = (n_e, 'density scaling (cm^-3)')
pri_hdu_StoQ.header['VELSCALE'] = (v_0, 'velocity scaling (m s^-1)')
pri_hdu_StoQ.header['MAGSCALE'] = (B_0, 'magnetic field scaling (mu G)')

# Save the produced Stokes Q and U arrays as FITS files
mat2FITS_Image(StoQ, pri_hdu_StoQ.header, save_loc + save_sim + '_' + \
 line_o_sight + '_StoQ_specind_' + emis_mech + '.fits', clobber = True)
mat2FITS_Image(StoU, pri_hdu_StoQ.header, save_loc + save_sim + '_' + \
 line_o_sight + '_StoU_specind_' + emis_mech + '.fits', clobber = True)

# Save the produced polarised intensity and polarisation angle as FITS files
mat2FITS_Image(pol_inten, pri_hdu_StoQ.header, save_loc + save_sim + '_' + \
 line_o_sight + '_pol_inten_specind_' + emis_mech + '.fits', clobber = True)
mat2FITS_Image(pol_angle, pri_hdu_StoQ.header, save_loc + save_sim + '_' + \
 line_o_sight + '_pol_angle_specind_' + emis_mech + '.fits', clobber = True)

# Close all of the fits files, to save memory
dens_fits.close()
magx_fits.close()
magy_fits.close()
magz_fits.close()

# Print a message to state that the FITS files were saved successfully
print 'FITS files of Stokes Q and U saved successfully {}'.format(save_sim)

# All of the required maps have been saved, so print a message stating that
# the script has finished
print 'All Stokes Q and U maps calculated successfully'