#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which converts a numpy array into a #
# FITS image. A FITS Header object can be provided by the caller, which will  #
# be used to produce the header of the FITS object. The FITS object can also  #
# be saved using a given filename.                                            #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 10/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, and astropy.io.fits
import numpy as np
from astropy.io import fits

# Define the function mat2FITS_Image, which will convert the given numpy array
# into a FITS image object.
def mat2FITS_Image(array, hdr = None, filename = None, clobber = False):
    '''
    Description
        This function converts a given numpy array into a FITS image object. If
        a FITS Header object is provided, then this will be used to produce the
        header of the FITS file produced. If a filename is provided, then the
        FITS file will be saved using the given filename. The produced FITS
        object is returned to the caller, as a HDU list.
        
    Required Input
        array - The numpy array to be converted to a FITS image object.
        hdr - The FITS Header object to be used in producing the header of the
              the final FITS file. If this is None, then the produced FITS 
              file will not have any information in its header.
        filename - The filename to be used if the FITS file is to be saved. If
                   this is None, then the FITS file is not saved.
        clobber - A boolean value. If False, it will not overwrite a FITS file 
                  that has the same name. If True, it will overwrite.
                   
    Output
        fits_file - The produced FITS image object, which will have a primary
        HDU, and no HDU extensiond. The given array is stored as the data
        attribute of the primary HDU, and the given header (if any) as the
        Header attribute of the primary HDU. 
    '''
    
    # If a header was not provided, then the primary HDU is constructed without
    # a specific header
    if hdr == None:
        # Create a primary HDU to contain the numpy array data provided
        pri_hdu = fits.PrimaryHDU(array)
    else:
        # In this case a header was provided, so create a primary HDU to
        # contain the array and header provided
        pri_hdu = fits.PrimaryHDU(array, header = hdr)
    
    # Create a HDUList object to contain the primary HDU
    fits_file = fits.HDUList([pri_hdu])
    
    # If a filename was provided, then save the FITS file
    if filename is not None:
        # Save the created FITS image object using the given filename
        fits_file.writeto(filename, clobber = clobber)
    
    # Return the created FITS image object to the caller
    return fits_file