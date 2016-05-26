#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to open the tables containing source     #
# location and properties for images of Stokes Q, Stokes U, and the           #
# polarisation gradient, and create a mask over all of the point sources      #
# found in these images. Two masks are made, corresponding to different mask  #
# sizes, and these are saved as FITS files so that they can be used to mask   #
# any desired matrix.                                                         #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 2/11/2015                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
from astropy.io import fits
import math

# Import utility functions
from mat2FITS_Image import mat2FITS_Image

# Import Aegean helpers
from AegeanTools import catalogs, wcs_helpers, fitting

#global constants
fwhm2cc = 1 / (2 * math.sqrt(2 * math.log(2)))
cc2fwhm = (2 * math.sqrt(2 * math.log(2)))

# Create a function that will take a given catalogue of source locations, and
# add masks for each source to a given matrix that is the same shape as the 
# image from which the sources were found
def generate_mask(catalogue, mask1, mask2,\
				image_loc, thresh1, thresh2):
	'''
	Description
        This function takes a list of source locations and fluxes, and masks
        out these sources at two different threshold levels. These masks are 
        returned by the function. If the given matrices are already partly 
        masked, then the masks for the current source locations are added to 
        this matrix.
        
    Required Input
        catalogue - The catalogue of sources to be masked in the image. Should 
                    be produced with Aegean.
        mask1 - The matrix that represents the mask being applied to the image.
                This matrix is generated using the first, smallest flux 
                threshold. This is returned to the caller.
        mask2 - The matrix that represents the mask being applied to the image. 
                This matrix is generated using the second, largest flux 
                threshold. This is returned to the caller.
        image_loc - The directory and filename of the FITS image that was used
                    to generate the source table. Used to obtain the header 
                    information for the image.
        thresh1 - A decimal specifying the flux threshold used to decide the 
                  size of the first series of masks, expressed as a fraction
                  of the peak flux of the source. Must be between 0 and 1.
        thresh2 - A decimal specifying the flux threshold used to decide the 
                  size of the second series of masks, expressed as a fraction
                  of the peak flux of the source. Must be between 0 and 1. 
                  Should be larger than thresh1.
                   
    Output
        mask1 - The boolean matrix of True/False values, where True indicates
                that a pixel should be masked. All sources in catalogue are 
                masked at flux level thresh1, and any prior masks are still 
                present.
        mask2 - The boolean matrix of True/False values, where True indicates
                that a pixel should be masked. All sources in catalogue are 
                masked at flux level thresh2, and any prior masks are still 
                present.
	'''

	# Open the given image, and extract its data and header
	hdulist = fits.open(image_loc)
	data =hdulist[0].data
	header = hdulist[0].header

	# Create a WCS helper for the image, which will convert between sky
	# coordinates and pixels in the matrix
	wcshelper = wcs_helpers.WCSHelper.from_header(header)

	# Find the points in the data that have a finite value
	x, y = np.where(np.isfinite(data))

	# Loop over the sources in the catalogue, and mask them
	for src in catalogue:
		if src.flags == 0:
			# Extract the peak flux of the source
			amp = src.peak_flux
			# Convert the location of the source from sky coordinates to pixels
			xo,yo = wcshelper.sky2pix([src.ra,src.dec])
			_,_,sx,theta = wcshelper.sky2pix_vec([src.ra,src.dec],src.a/3600,src.pa)
			_,_,sy,_ = wcshelper.sky2pix_vec([src.ra,src.dec],src.b/3600,src.pa+90)
			# Create a 2D Gaussian model for the source
			# TODO: understand why xo/yo -1 is needed
			model = fitting.elliptical_gaussian(x,y,amp,xo-1,yo-1,sx*fwhm2cc,sy*fwhm2cc,theta)
			# If the intensity of the model is above a certain fraction of the 
			# peak flux, then mark the pixel as True, meaning that the pixel is to
			# be masked
			mask1[x,y] = np.logical_or(mask1[x,y], (np.abs(model)>np.abs(amp*thresh1)) )
			mask2[x,y] = np.logical_or(mask2[x,y], (np.abs(model)>np.abs(amp*thresh2)) )

	# All the sources have been masked, so return the matrices to the caller
	return mask1, mask2

# Set the flux threshold to use to create the first mask, expressed as a
# fraction of the peak flux of the source
thresh1 = 0.05

# Set the flux threshold to use to create the second mask, expressed as a 
# fraction of the peak flux of the source
thresh2 = 0.000004

# Create a string object which stores the directory of the CGPS data
data_loc = '/Users/chrisherron/Documents/PhD/CGPS_2015/'

# Create a string that will be used to control what Q and U FITS files are used
# to perform calculations, and that will be appended into the filename of 
# anything produced in this script. This is either 'high_lat' or 'plane'
mosaic_area = 'plane'

# Create a string that will be used to create the correct filename when 
# source tables are being read in
save_append = '_src_table_comp.vot'

# Create a string that stores the location of the Stokes Q image
Sto_Q_file = data_loc + 'Sto_Q_' + mosaic_area + '.fits'

# Create a string that stores the location of the Stokes U image
Sto_U_file = data_loc + 'Sto_U_' + mosaic_area + '.fits'

# Create a string that stores the location of the Stokes Q source table.
Sto_Q_table_loc = data_loc + 'Sto_Q_' + mosaic_area + save_append

# Create a string that stores the location of the Stokes U source table.
Sto_U_table_loc = data_loc + 'Sto_U_' + mosaic_area + save_append

# Obtain a source list for Stokes Q, from the catalogue
Q_srclist = catalogs.table_to_source_list(catalogs.load_table(Sto_Q_table_loc))

# Obtain a source list for Stokes U, from the catalogue
U_srclist = catalogs.table_to_source_list(catalogs.load_table(Sto_U_table_loc))

# Open the Stokes Q FITS file, and extract its data and header
Q_hdulist = fits.open(Sto_Q_file)
Q_data = Q_hdulist[0].data
Q_hdr = Q_hdulist[0].header

# Create two empty matrices, one to hold the first mask, and the other to hold
# the second mask
first_mask = np.zeros(np.shape(Q_data), dtype = bool)
second_mask = np.zeros(np.shape(Q_data), dtype = bool)

# Run the generate mask function on Stokes Q, to create a mask that covers 
# all of the sources detected in Stokes Q
first_mask, second_mask = generate_mask(catalogue = Q_srclist, mask1 =\
 first_mask, mask2 = second_mask, image_loc = Sto_Q_file,\
 thresh1 = thresh1, thresh2 = thresh2)

# Run the generate mask function on Stokes U, to create a mask that covers
# all of the sources detected in Stokes U, in addition to Stokes Q
first_mask, second_mask = generate_mask(catalogue = U_srclist, mask1 =\
 first_mask, mask2 = second_mask, image_loc = Sto_U_file,\
 thresh1 = thresh1, thresh2 = thresh2)

# Cast the two masks into floats, so that they can be saved as FITS images
first_mask = first_mask.astype(np.int)
second_mask = second_mask.astype(np.int)

# Save the mask created with the first threshold as a FITS file
first_FITS = mat2FITS_Image(first_mask, Q_hdr,\
    data_loc + 'combined_mask_{}_thr_{}.fits'.format(mosaic_area, thresh1))

# Save the mask created with the second threshold as a FITS file
second_FITS = mat2FITS_Image(second_mask, Q_hdr,\
    data_loc + 'combined_mask_{}_thr_{}.fits'.format(mosaic_area, thresh2))

# At this point the required masks have been saved as FITS files, so print a 
# message saying that the program completed successfully
print 'Masks successfully saved as FITS files'