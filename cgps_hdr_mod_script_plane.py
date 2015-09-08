#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to open all of the FITS files in a       #
# directory and modify their FITS headers. The modified files are then saved  #
# in a new directory.                                                         #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 19/8/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
from astropy.io import fits
import os
from mat2FITS_Image import mat2FITS_Image

# Create a string which states the location of the CGPS data
cgps_loc = "/Users/chrisherron/Documents/PhD/CGPS_2015/Mosaic_Production/"

# Create a string which states the location of the directory containing all of
# FITS files
direc_name = "Stokes_Q_snapshots_plane/"

# Create a string which states the location of the directory in which all
# output FITS files should be saved
output_direc = "Sto_Q_hdr_fixed_plane/"

# Get a list of all of the files in the specified directory
fits_file_list = os.listdir(cgps_loc + direc_name)

# Delete the first entry, which is not a FITS file (only for Stokes I)
#del fits_file_list[0]

# Print out the list of files, to check they are ordered correctly
print fits_file_list

# Create a variable that contains the required CRVAL1 value
crval1 = 122.750000
# Create a variable that contains the required CRVAL2 value
crval2 = 1.0000000

# Create an array that contains the CRPIX1 values for all the FITS files
# NOTE: This relies on FITS file list being in alphabetical order
# NOTE: This is for ALL snapshots, for Stokes I, Q and U
crpix1_arr = np.array([1713, 1713, 913, 913, 113, 113, -687, -687, -1487,-1487,\
	-11887, -11887, -12687, -12687, -13487, -13487, 14513, 14513, 13713, 13713,\
	12913, 12913, 12113, 12113, 11313, 11313, 10513, 10513, 9713, 9713,\
	8913, 8913, 8113, 8113, 7313, 7313, -2287, -2287, -3087, -3087,\
	-3887, -3887, -4687, -4687, -5487, -5487, -6287, -6287, -7087, -7087,\
	-7887, -7887, -8687, -8687, -9487, -9487, -10287, -10287, -11087, -11087,\
	6513, 6513, 5713, 5713, 4913, 4913, 4113, 4113, 3313, 3313, 2513, 2513])

# Create an array that contains the CRPIX2 values for all the FITS files
# NOTE: This is for Stokes I, Q and U
crpix2_arr = np.array([913, 113, 913, 113, 913, 113, 913, 113, 913, 113,\
	913, 113, 913, 113, 913, 113, 913, 113, 913, 113,\
	913, 113, 913, 113, 913, 113, 913, 113, 913, 113,\
	913, 113, 913, 113, 913, 113, 913, 113, 913, 113,\
	913, 113, 913, 113, 913, 113, 913, 113, 913, 113,\
	913, 113, 913, 113, 913, 113, 913, 113, 913, 113,\
	913, 113, 913, 113, 913, 113, 913, 113, 913, 113, 913, 113])

# Loop over all of the FITS files that need to be modified
for i in range(len(fits_file_list)):
	# Open up the FITS file
	fits_file = fits.open(cgps_loc + direc_name + fits_file_list[i])

	# Obtain the header from the FITS file
	fits_hdr = fits_file[0].header
	# Obtain the data from the FITS file
	fits_data = fits_file[0].data

	# Change the CRVAL1 value to the required value
	fits_hdr['CRVAL1'] = crval1
	# Change the CRVAL2 value to the required value
	fits_hdr['CRVAL2'] = crval2

	# Change the CRPIX1 value to the required value
	fits_hdr['CRPIX1'] = crpix1_arr[i]
	# Change the CRPIX2 value to the required value
	fits_hdr['CRPIX2'] = crpix2_arr[i]

	# Check to see if there is an EQUINOX keyword in the FITS file
	if 'EQUINOX' in fits_hdr:
		# In this case there is an EQUINOX keyword, so delete it
		del fits_hdr['EQUINOX']

	# Save the new FITS file with the modified header
	mat2FITS_Image(fits_data, fits_hdr, cgps_loc + output_direc + fits_file_list[i])

	# Print a message to the screen to say the FITS file has been made
	print "{} made successfully.".format(fits_file_list[i])

	# Close the FITS file to save memory
	fits_file.close()