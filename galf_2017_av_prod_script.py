#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of the observed Stokes Q   #
# and U images for band averaged images of sections of the GALFACTS survey,    #
# and produces images of spatial polarisation diagnostics.                     #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 8/2/2017                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import mat2FITS_Image to convert arrays to FITS format
from mat2FITS_Image import mat2FITS_Image

# Import the functions that calculate derivatives of Stokes Q and U
from calc_Sto_1Diff import calc_Sto_1Diff
from calc_Sto_2Diff import calc_Sto_2Diff

# Import the functions that calculate all of the polarisation diagnostics
from calc_Polar_Inten import calc_Polar_Inten
from calc_Polar_Angle import calc_Polar_Angle
from calc_Polar_Grad import calc_Polar_Grad
from calc_Direc_Amp_Max import calc_Direc_Amp_Max
from calc_Direc_Amp_Min import calc_Direc_Amp_Min
from calc_Direc_Max_Ang import calc_Direc_Max_Ang
from calc_Rad_Direc_Amp import calc_Rad_Direc_Amp
from calc_Tang_Direc_Amp import calc_Tang_Direc_Amp
from calc_Rad_Direc_Ang import calc_Rad_Direc_Ang
from calc_Tang_Direc_Ang import calc_Tang_Direc_Ang
from calc_Curv4MaxDirecDeriv import calc_Curv4MaxDirecDeriv

# Create a list of polarisation diagnostics to calculate. Only the diagnostics
# that are on this list will be calculated
diag_list = ['Inten', 'Angle', 'Grad', 'Direc_Amp_Max', 'Direc_Amp_Min',\
'Direc_Max_Ang', 'Rad_Direc_Amp', 'Tang_Direc_Amp', 'Rad_Direc_Ang',\
'Tang_Direc_Ang', 'Curv4MaxDirecDeriv']

# Create a string for the directory that contains the Stokes Q and U maps to use
data_loc = '/Volumes/CAH_ExtHD/GALFACTS_2017/'

# Open the FITS file that contains Stokes Q
StoQ_fits = fits.open(data_loc + 'GALFACTS_S3_average_image_Q.fits')

# Extract the data for Stokes Q
StoQ = StoQ_fits[0].data

# Extract the header for Stokes Q
StoQ_hdr = StoQ_fits[0].header 

# Open the FITS file that contains Stokes U 
StoU_fits = fits.open(data_loc + 'GALFACTS_S3_average_image_U.fits')

# Extract the data for Stokes U
StoU = StoU_fits[0].data

# Print a message to the screen to show that the data has been loaded
print 'All required data loaded successfully'

# Determine the spacing between adjacent pixels, to be used in calculating
# derivatives of quantities. This is in degrees.
pix_size_deg = np.absolute(StoQ_hdr['CDELT1'])

# Now that Stokes Q and U have been loaded, calculate all polarisation
# diagnostics that can be calculated for either propagation mechanism

# Calculate all of the first order spatial derivatives that we need
dQ_dy, dQ_dx, dU_dy, dU_dx = calc_Sto_1Diff(StoQ, StoU, pix_sep = pix_size_deg)

# Calculate all of the second order spatial derivatives that we need
d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2 =\
 calc_Sto_2Diff(dQ_dy, dQ_dx, dU_dy, dU_dx, pix_sep = pix_size_deg)

#---------------------- Polarisation Intensity -------------------------

# Check to see if we need to calculate the polarisation intensity
if 'Inten' in diag_list:
	# Calculate the polarisation intensity for this Stokes Q and U
	polar_inten = calc_Polar_Inten(StoQ, StoU)

	# Save the polarisation intensity
	mat2FITS_Image(polar_inten, StoQ_hdr, data_loc +\
	 'PolarInten_' + 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Polarisation intensity calculated'

#------------------------- Polarisation Angle ------------------------------

# Check to see if we need to calculate the polarisation angle
if 'Angle' in diag_list:
	# Calculate the observed polarisation angle for this Stokes Q and U
	polar_angle = calc_Polar_Angle(StoQ, StoU)

	# Save the polarisation angle array
	mat2FITS_Image(polar_angle, StoQ_hdr, data_loc + 'PolarAngle_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Polarisation angle calculated'

#----------------------- Polarisation Gradient -----------------------------

# Check to see if we need to calculate the polarisation gradient
if 'Grad' in diag_list:
	# Calculate the observed polarisation gradient for this Stokes Q and U
	polar_grad = calc_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx)

	# Save the polarisation gradient array
	mat2FITS_Image(polar_grad, StoQ_hdr, data_loc + 'PolarGrad_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Polarisation gradient calculated'

#------------ Maximum Amplitude of the Directional Derivative --------------

# Check to see if we need to calculate the maximum amplitude of the 
# directional derivative
if 'Direc_Amp_Max' in diag_list:
	# Calculate the observed maximum amplitude of the directional
	# derivative for this Stokes Q and U
	direc_amp_max = calc_Direc_Amp_Max(dQ_dy, dQ_dx, dU_dy, dU_dx)

	# Save the maximum amplitude of the directional derivative array
	mat2FITS_Image(direc_amp_max, StoQ_hdr, data_loc + 'DirecAmpMax_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Maximum amplitude of the directional derivative calculated'

#------------ Minimum Amplitude of the Directional Derivative --------------

# Check to see if we need to calculate the minimum amplitude of the 
# directional derivative
if 'Direc_Amp_Min' in diag_list:
	# Calculate the observed minimum amplitude of the directional
	# derivative for this Stokes Q and U
	direc_amp_min = calc_Direc_Amp_Min(dQ_dy, dQ_dx, dU_dy, dU_dx)

	# Save the minimum amplitude of the directional derivative array
	mat2FITS_Image(direc_amp_min, StoQ_hdr, data_loc + 'DirecAmpMin_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Minimum amplitude of the directional derivative calculated'

#------------ Angle that Maximises the Directional Derivative --------------

# Check to see if we need to calculate the angle that maximises the 
# directional derivative
if 'Direc_Max_Ang' in diag_list:
	# Calculate the observed angle that maximises the directional
	# derivative for this Stokes Q and U
	direc_max_ang = calc_Direc_Max_Ang(dQ_dy, dQ_dx, dU_dy, dU_dx)

	# Save the angle that maximises the directional derivative array
	mat2FITS_Image(direc_max_ang, StoQ_hdr, data_loc + 'DirecMaxAng_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Angle that maximises the directional derivative calculated'

#------------ Maximum Radial Component Directional Derivative --------------

# Check to see if we need to calculate the maximum radial component of the 
# directional derivative
if 'Rad_Direc_Amp' in diag_list:
	# Calculate the observed maximum radial component of the directional
	# derivative for this Stokes Q and U
	rad_direc_amp = calc_Rad_Direc_Amp(StoQ, StoU, dQ_dy,dQ_dx,dU_dy, dU_dx)

	# Save the maximum radial component of the directional derivative array
	mat2FITS_Image(rad_direc_amp, StoQ_hdr, data_loc + 'RadDirecAmp_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print'Maximum radial component of the directional derivative calculated'

#---------- Maximum Tangential Component Directional Derivative ------------

# Check to see if we need to calculate the maximum tangential component of 
# the directional derivative
if 'Tang_Direc_Amp' in diag_list:
	# Calculate the observed maximum tangential component of the directional
	# derivative for this Stokes Q and U
	tang_direc_amp = calc_Tang_Direc_Amp(StoQ,StoU,dQ_dy,dQ_dx,dU_dy,dU_dx)

	# Save the maximum tangential component of the directional derivative 
	# array
	mat2FITS_Image(tang_direc_amp, StoQ_hdr, data_loc + 'TangDirecAmp_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Maximum tangential component of the directional derivative '+\
	 'calculated'

#---- Difference Radial and Tangential Components Directional Derivative ---

# Check to see if we can calculate the difference between the maximum tangential
# and radial components of the directional derivative
if ('Tang_Direc_Amp' in diag_list) and ('Rad_Direc_Amp' in diag_list):
	# Calculate the difference between the maximum values of the radial and 
	# tangential components of the directional derivative
	diff_rad_tang_direc = rad_direc_amp - tang_direc_amp

	# Save the difference between the maximum radial and tangential 
	# components array 
	diff_rad_tang_direc_FITS = mat2FITS_Image(diff_rad_tang_direc, StoQ_hdr,\
	data_loc + 'DiffRadTangDirec_' + 'S3_average' + '.fits', clobber = True)

	# Print a message to the screen to show that the FITS file was produced and
	# saved successfully.
	print 'Difference between the maximum radial and tangential components '+\
	 'of the directional derivative calculated'

#-------- Angle Maximises Radial Component Directional Derivative ----------

# Check to see if we need to calculate the angle that maximises the radial
# component of the directional derivative
if 'Rad_Direc_Ang' in diag_list:
	# Calculate the observed angle that maximises the radial component of 
	# the directional derivative for this Stokes Q and U
	rad_direc_ang = calc_Rad_Direc_Ang(StoQ,StoU,dQ_dy,dQ_dx,dU_dy,dU_dx)

	# Save the angle that maximises the radial component of the directional 
	# derivative array
	mat2FITS_Image(rad_direc_ang, StoQ_hdr, data_loc + 'RadDirecAng_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Angle that maximises the radial component of the directional '+\
	 'derivative calculated'

#-------- Angle Maximises Tangential Component Directional Derivative ------

# Check to see if we need to calculate the angle that maximises the 
# tangential component of the directional derivative
if 'Tang_Direc_Ang' in diag_list:
	# Calculate the observed angle that maximises the tangential component 
	# of the directional derivative for this Stokes Q and U
	tang_direc_ang = calc_Tang_Direc_Ang(StoQ,StoU,dQ_dy,dQ_dx,dU_dy,dU_dx)

	# Save the angle that maximises the tangential component of the 
	# directional derivative array
	mat2FITS_Image(tang_direc_ang, StoQ_hdr, data_loc + 'TangDirecAng_' +\
	 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Angle that maximises the tangential component of the ' +\
	'directional derivative calculated'

#------- Curvature in Direction that Maximises Directional Derivative ------

# Check to see if we need to calculate the curvature in the direction that
# maximises the directional derivative
if 'Curv4MaxDirecDeriv' in diag_list:
	# Calculate the curvature in the direction that maximises the 
	# directional derivative for this Stokes Q and U
	curv4max_direc_deriv = calc_Curv4MaxDirecDeriv(dQ_dy, dQ_dx,\
		dU_dy, dU_dx, d2Q_dy2, d2Q_dx2, d2U_dy2, d2U_dx2, d2Q_dydx,d2U_dydx)

	# Save the curvature in the direction that maximises the directional
	# derivative array
	mat2FITS_Image(curv4max_direc_deriv, StoQ_hdr, data_loc +\
	 'Curv4MaxDirecDeriv_' + 'S3_average' + '.fits', clobber = True)

	# Print a message to say the diagnostic has been calculated
	print 'Curvature in the direction that maximises the directional ' +\
	 'derivative calculated'

#---------------------------------------------------------------------------

# Print a message to say that everything has been calculated
print 'All diagnostics calculated successfully'