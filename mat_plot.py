#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which produces an image from a      #
# given matrix, and then saves the image using the given filename. The imshow #
# function of matplotlib is used to achieve this. Many of the arguments of    #
# this function behave in the same way as the arguments of the imshow         #
# function.                                                                   #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 4/7/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy and matplotlib, and LogNorm for colour scaling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Define the function mat_plot, which will produce an image from the given
# Numpy array, and save it using the given filename.
def mat_plot(array, filename, format, x_ticks = None, y_ticks = None, cmap =\
'hot', norm = None, aspect = None, interpolation = None, vmin = None,\
vmax = None, origin = 'lower', xlabel = '', ylabel = '', title = ''):
    '''
    Description
        This function produces an image of the given array using the matplotlib
        function imshow. Many of the arguments of this function are the same as
        the corresponding imshow arguments. The image is then saved using the
        given filename and format. The filename is assumed to already include
        the extension corresponding to the format.
        
    Required Input
        array - The 2-dimensional Numpy array to be converted into an image. 
                Each entry of this array is assumed to be a float.
        filename - The filename to be used when saving the image. The image may
                   be saved as a ps, eps, pdf, png or jpeg file, depending on
                   the extension of the filename. Must be a string.
        format - The format in which the image will be saved, as a string. e.g.
                 'png'.
        x_ticks - A one-dimensional Numpy array, where each entry specifies the
                  position along the x axis of each pixel in the array. Must 
                  have the same length as the x axis of the array.
        y_ticks - A one-dimensional Numpy array, where each entry specifies the
                  position along the y axis of each pixel in the array. Must
                  have the same length as the y axis of the array.
        cmap - This is a string specifying what colour map to use for the 
                 image. If it is 'gray', then a grayscale image is produced.
                 Otherwise, a colour scale image is produced, and the colour
                 map used is that specified by this string. Valid strings are
                 matplotlib colour maps, see 
                 http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps. The 
                 'hot' colour scheme is used by default.
        norm - A Normalize instance object used to scale the colour map. The
               default is None, corresponding to a linear colour scale. Using
               LogNorm results in a logarithmic colour scaling.
        aspect - A string controlling the axis aspect ratio. If 'auto', then
                 image aspect ratio matches that of the axes. If 'equal', then
                 the axis aspect ratio matches that of the image. This value
                 defaults to None.
        interpolation - A string specifying the interpolation scheme to use
                        when creating the image, if any. Valid strings can be
                        found at http://matplotlib.org/api/pyplot_api.html. No
                        interpolation is used by default.
        vmin - The minimum value to be present on the colour scale and colour
               bar. Any values below vmin are mapped to the same colour as a 
               value equal to vmin. If vmin == None, then vmin is calculated
               automatically by aplpy.
        vmax - The maximum value to be present on the colour scale and colour
               bar. Any values above vmax are mapped to the same colour as a 
               value equal to vmax. If vmax == None, then vmax is calculated
               automatically by aplpy.
        origin - A string stating whether the origin of the image should be in
                 the top left corner ('upper') or the bottom left corner 
                 ('lower') of the image. Default is 'lower'.
        xlabel - The label for the x axis, provided as a string.
        ylabel - The label for the y axis, provided as a string.
        title - The title for the image, provided as a string.
                   
    Output
        The figure produced is saved using the given filename and format. If 
        the image is created successfully, then 1 is returned.
    '''
    
    # Create a figure to display the image
    fig = plt.figure()
    
    # Create an axis for this figure
    ax = fig.add_subplot(111)
    
    # Check to see if the x_ticks and y_ticks arrays have been provided
    if x_ticks != None and y_ticks != None:
        # In this case the ticks have been specified, to augment the axes of the
        # plot to take into account these ticks.
        
        # Use the y_ticks array to find the data values corresponding to the top
        # and bottom of the image
        y_bottom = y_ticks[0]
        y_top = y_ticks[-1]
    
        # Use the x_ticks array to find the data values corresponding to the 
        # left and right of the image
        x_left = x_ticks[0]
        x_right = x_ticks[-1]
    
        # Use imshow to create an image of the array, using the colourmap, 
        # aspect, interpolation, vmin and vmax values that were given to the 
        # function.
        plt.imshow(array, cmap = cmap, norm = norm, aspect = aspect, \
        interpolation = interpolation, vmin = vmin, vmax = vmax, origin =\
        origin, extent = (x_left, x_right, y_bottom, y_top))
    
        # Set the y_ticks of the plot using the given array
        ax.set_yticks(np.linspace(y_bottom, y_top, 5))
        # Set the x_ticks of the plot using the given array
        ax.set_xticks(np.linspace(x_left, x_right, 5))
    else:
        # In this case the ticks have not been specified, so create the image
        # without changing the ticks
        # Use imshow to create an image of the array, using the colourmap, 
        # aspect, interpolation, vmin and vmax values that were given to the 
        # function.
        plt.imshow(array, cmap = cmap, norm = norm, aspect = aspect, \
        interpolation = interpolation, vmin = vmin, vmax = vmax, origin =\
        origin)
    
    # Add a colourbar to the image
    plt.colorbar()
    
    # Add a label to the x-axis
    plt.xlabel(xlabel, fontsize = 20)
    
    # Add a label to the y-axis
    plt.ylabel(ylabel, fontsize = 20)
    
    # Add a title to the plot
    plt.title(title, fontsize = 20)
    
    # Save the figure using the given filename and format
    plt.savefig(filename, format = format)
    
    # Print a message to the screen saying that the image was created
    #print filename + ' created successfully.\n'
    
    # Close the figure to free up memory
    plt.close()
    
    # The image has been saved successfully, so return 1
    return 1