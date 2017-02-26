#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of the observed Stokes Q   #
# and U images for all wavelengths of a region of the GALFACTS survey, and     #
# reduces it to a smaller image of a region of interest.                       #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 9/2/2017                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, and astropy.io for fits manipulation
import numpy as np
from astropy.io import fits

# Import mat2FITS_Image to convert arrays to FITS format
from mat2FITS_Image import mat2FITS_Image

# Create a string for the directory that contains the Stokes Q and U maps to use
data_loc = '/Volumes/CAH_ExtHD/GALFACTS_2017/'

# Open the FITS file that contains Stokes Q, as a memory map, so that we don't
# load it all in at once.
StoQ_fits = fits.open(data_loc + 'GALFACTS_S1_0263_4023_10chanavg_Q.fits',\
	memmap = True)

# Extract the data for Stokes Q, as a memory map
StoQ = StoQ_fits[0].data

# Extract the header for Stokes Q
StoQ_hdr = StoQ_fits[0].header 

# Change the value for the CRPIX1 keyword, to give the correct reference
# pixel for the new array
StoQ_hdr['CRPIX1'] = -332.00

# Extract the portion of the Stokes Q image that we are interested in
newQ = StoQ[:,:,2909:3983]

# Save the selected region of the Stokes Q image as a new FITS file
mat2FITS_Image(newQ, StoQ_hdr, data_loc +\
	 'GALFACTS_' + 'S1_chanavg_subset_Q' + '.fits', clobber = True)

# Close the Stokes Q fits file, and anything related to it, to free memory
StoQ_fits.close()
del newQ
del StoQ

# Open the FITS file that contains Stokes U, as a memory map, so that we don't
# load it all in at once.
StoU_fits = fits.open(data_loc + 'GALFACTS_S1_0263_4023_10chanavg_U.fits',\
	memmap = True)

# Extract the data for Stokes U, as a memory map
StoU = StoU_fits[0].data

# Extract the header for Stokes U
StoU_hdr = StoU_fits[0].header 

# Change the value for the CRPIX1 keyword, to give the correct reference
# pixel for the new array
StoU_hdr['CRPIX1'] = -332.00

# Extract the portion of the Stokes U image that we are interested in
newU = StoU[:,:,2909:3983]

# Save the selected region of the Stokes U image as a new FITS file
mat2FITS_Image(newU, StoU_hdr, data_loc +\
	 'GALFACTS_' + 'S1_chanavg_subset_U' + '.fits', clobber = True)

# Close the Stokes Q fits file, and anything related to it, to free memory
StoU_fits.close()
del newU
del StoU