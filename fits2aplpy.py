#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which produces an image from a FITS #
# file object that is given to the function. This image is saved using the    #
# given filename.                                                             #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 11/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, astropy.io.fits and aplpy
import numpy as np
from astropy.io import fits
import aplpy

# Define the function fits2aplpy, which will produce an image from the given
# FITS file object
def fits2aplpy(fits_file, filename, dimensions = [0,1], slices = [], \
colour = 'gray', vmin = None, vmax = None, convention = None, stretch = 'linear'):
    '''
    Description
        This function produces an image of a FITS file image object in aplpy.
        The file is then saved using the given filename. The format of the 
        saved image is determined by the extension of the filename.
        
    Required Input
        fits_file - The FITS image object to be used in constructing the image.
                    This may be a string giving the name and location of the
                    FITS file to use, a FITS HDU list, a FITS primary HDU or a
                    FITS Image HDU.
        filename - The filename to be used when saving the image. The image may
                   be saved as a ps, eps, pdf, png or jpeg file, depending on
                   the extension of the filename. 
        dimensions - This is a list containing only two entries, which must be
                     integers. The first entry of the list states what 
                     dimension of the FITS data cube is plotted on the x axis
                     of the image, and likewise the second entry determines the
                     quantity plotted on the y-axis. 0 corresponds to the first
                     dimension of the image, 1 the second dimension, and so on.
                     An error will occur if the given indices are greater than
                     or equal to the number of dimensions in the data cube.
        slices - This is a list containing n-2 integers, where n is the number
                 of dimensions in the image. Each entry specifies the data
                 slice index to use for the remaining dimensions in creating
                 the image.
        colour - This is a string specifying what colour map to use for the 
                 image. If it is 'gray', then a grayscale image is produced.
                 Otherwise, a colour scale image is produced, and the colour
                 map used is that specified by this string. Valid strings are
                 matplotlib colour maps, see 
                 http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        vmin - The minimum value to be present on the colour scale and colour
               bar. Any values below vmin are mapped to the same colour as a 
               value equal to vmin. If vmin == None, then vmin is calculated
               automatically by aplpy.
        vmax - The maximum value to be present on the colour scale and colour
               bar. Any values above vmax are mapped to the same colour as a 
               value equal to vmax. If vmax == None, then vmax is calculated
               automatically by aplpy.
        convention - A string to be used if a FITS header can be interpreted
                     in multiple ways. For a Cartesian CAR projection, this
                     may be 'wells' or 'calabretta'.
        stretch - A string describing the scaling to be applied to the colour
                  map. Default is 'linear'. Other options are 'log', 'sqrt',
                  'arcsinh' and 'power'.
                   
    Output
        fig - This function returns the aplpy FITSFigure instance which is 
              created during the operation of this function. The image produced
              can be updated further via this returned FITSFigure instance.
    '''
    
    # If no convention is required to resolve any ambiguity in the FITS
    # header, then create the disply frame as usual
    if convention == None:
        # Create a display frame for the image
        fig = aplpy.FITSFigure(fits_file, dimensions = dimensions, slices = slices)
    else:
        # In this case, create a display frame using the given convention
        fig = aplpy.FITSFigure(fits_file, dimensions = dimensions, slices = slices,\
         convention = convention)
    
    # Check to see if a grayscale or colour scale image is being made
    if colour == 'gray':
        # Check to see if the grayscale range is being set manually or 
        # automatically
        if vmin == None or vmax == None:
            # In this case the grayscale range is set automatically
            # Make a grayscale version of the image appear on the frame
            fig.show_grayscale(stretch = stretch)
        else:
            # In this case the grayscale range is set manually using the 
            # specified values for vmin and vmax
            # Make a grayscale version of the image appear on the frame
            fig.show_grayscale(vmin = vmin, vmax = vmax, stretch = stretch)
    else:
        # Check to see if the grayscale range is being set manually or 
        # automatically
        if vmin == None or vmax == None:
            # In this case the colour scale range is being set automatically
            # Make a colour scale version of the image appear on the frame, 
            # using the colour map specified by the user
            fig.show_colorscale(cmap = colour, stretch = stretch)
        else:
            # In this case the colour scale range is set manually using the 
            # specified values for vmin and vmax
            # Make a colour scale version of the image appear on the frame, 
            # using the colour map specified by the user
            fig.show_colorscale(vmin = vmin, vmax = vmax, cmap = colour,\
             stretch = stretch)
    
    # Add a co-ordinate grid to the image
    fig.add_grid()
    
    # Add a colour bar to the image
    fig.add_colorbar()
    
    # Save the image using the given filename
    fig.save(filename, dpi = 600, max_dpi = 600)
    
    # Return the figure to the caller
    return fig