#------------------------------------------------------------------------------#
#                                                                              #
# This piece of code describes a function which produces an image of multiple  #
# data sets plotted against a common variable, and then saves the image using  #
# the given filename. The plot function of matplotlib is used to achieve this. #
# Many of the arguments of this function behave in the same way as the         #
# arguments of the plot function.                                              #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 30/9/2014                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Define the function line_plots, which will produce an image from the given
# dictionary, and save it using the given filename.
def line_plots(plot_dict, filename, format, xlabel = '', ylabel = '',\
 title = '', linewidth = 1, markersize = 6, log_x = False, log_y = False,\
  loc = 1, ymin = None, ymax = None, xmin = None, xmax = None):
    '''
    Description
        This function produces an image of multiple line plots against a common
        variable, using the plot function of matplotlib. Many of the arguments
        of this function are the same as the corresponding plot arguments. The
        image is then saved using the given filename and format. The filename
        is assumed to already include the extension corresponding to the format.
        
    Required Input
        plot_dict - A dictionary that holds the information required to produce
                    all of the plots. Each key:value pair corresponds to one 
                    plot. The key should be a string, labelling the data set.
                    This string is used to produce the legend of the plot. The 
                    corresponding value should be a list, whose first entry is
                    the x data of the data set. The second entry must be the
                    corresponding y data, the third entry should be a format 
                    string specifying the colour, linestyle and marker style
                    for the data set. If the x and y data arrays do not have the
                    same length, then the data set is ignored, but the other 
                    data sets are still plotted.
        filename - The filename to be used when saving the image. The image may
                   be saved as a ps, eps, pdf, png or jpeg file, depending on
                   the extension of the filename. Must be a string.
        format - The format in which the image will be saved, as a string. e.g.
                 'png'.
        xlabel - The label for the x axis, provided as a string.
        ylabel - The label for the y axis, provided as a string.
        title - The title for the image, provided as a string.
        linewidth - The linewidth of the lines (in points), specified as a float
        markersize - The markersize for the markers used in the plots (in
                     points), specified as a float.
        log_x - A boolean value. If True, then the x-axis is logarithmic.
                Otherwise, the x-axis is linear.
        log_y - A boolean value. If True, then the y-axis is logarithmic. 
                Otherwise, the y-axis is linear.
        loc - An integer or a string, specifying where the legend is located. 
              http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
              contains more information. If None, then no legend is plotted.
        ymin, ymax - The minimum and maximum y values to use on the y axis. If 
                     they are None, then the plot default is used to define the
                     y axis limits.
        xmin, xmax - The minimum and maximum x values to use on the x axis. If 
                     they are None, then the plot default is used to define the
                     x axis limits.
                   
    Output
        The figure produced is saved using the given filename and format. If 
        the image is created successfully, then 1 is returned.
    '''
    
    # Create a figure to display the image
    fig = plt.figure()
    
    # Create an axis for this figure
    ax = fig.add_subplot(111)
    
    # Iterate over the elements of the dictionary, to plot all of the 
    # data sets
    for key in plot_dict:
        # Extract all of the plot data for this data set
        plot_data = plot_dict[key]

        # Check to see if the x and y data arrays have the same length
        if len(plot_data[0]) == len(plot_data[1]):
            # The lengths are the same, so plot the data for this data set
            plt.plot(plot_data[0], plot_data[1], plot_data[2], linewidth =\
             linewidth, markersize = markersize, label = key)
        else:
            # In this case, the lengths are not the same, so print an error
            # message
            print 'line_plots ERROR: The {} x and y data does not have the'\
            ' same length'.format(key)

    # Check to see if a legend is being included on the plot
    if loc != None:
        # In this case a legend is being included, so place it on the plot in
        # the specified location
        plt.legend(loc = loc)

    # Check to see if the y-axis limits need to be changed
    if (ymin != None) and (ymax!= None):
        # Set the y axis limits using the given values
        plt.ylim((ymin, ymax))

    # Check to see if the x-axis limits need to be changed
    if (xmin != None) and (xmax!= None):
        # Set the x axis limits using the given values
        plt.xlim((xmin, xmax))

    # Check to see if the x axis of the plot needs to be logarithmic
    if log_x == True:
        # In this case, make the x axis of the plot area logarithmic
        ax.set_xscale('log')

    # Check to see if the y axis of the plot needs to be logarithmic
    if log_y == True:
        # In this case, make the y axis of the plot area logarithmic
        ax.set_yscale('log')
    
    # Add a label to the x-axis
    plt.xlabel(xlabel, fontsize = 20)
    
    # Add a label to the y-axis
    plt.ylabel(ylabel, fontsize = 20)
    
    # Add a title to the plot
    plt.title(title, fontsize = 20)
    
    # Make sure that all of the axis labels are visible
    plt.tight_layout()
    
    # Save the figure using the given filename and format
    plt.savefig(filename, format = format)
    
    # Print a message to the screen saying that the image was created
    print filename + ' created successfully.\n'
    
    # Close the figure to free up memory
    plt.close()
    
    # The image has been saved successfully, so return 1
    return 1