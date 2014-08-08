#------------------------------------------------------------------------------#
#                                                                              #
# This code is a utility function which automatically produces histogram       #
# images using the Bayesian Blocks function of astroML, when supplied with the #
# data to be histogrammed, as well as axis labels and a figure title, and a    #
# filename. The produced plot is saved using this filename, and the given      #
# extension.                                                                   #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 28/7/2014 (based off code written in 2013)                       #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy, matplotlib, for array handling and plotting
import numpy as np
import matplotlib.pyplot as plt
import astroML.plotting as aML

# Define the function bayes_block, which will automatically produce a histogram
# of the supplied data array, and save the file in the specified format.
def bayes_block(x_data, filename, format, x_label = '', title = '',\
log_x = False, log_y = False):
	'''
	Description
	    This function takes the given data and produces a Bayesian Block 
	    histogram of it. The given axis label and title are applied, and then
	    the histogram is saved using the given filename and format.
	
	Required Input
	    x_data: The data array to be graphed. Numpy array or list of floats.
	            This array is flattened to one dimension before creating 
	            the histogram.
	    filename: The filename (including extension) to use when saving the
		      image. Provide as a string.
	    format: The format (e.g. png, jpeg) in which to save the image. This
	            is a string.
	    x_label: String specifying the x-axis label.
	    title: String specifying the title of the graph.
	    log_x: If this is True, then logarithmic binning is used for the
	           histogram, and the x-axis of the saved image is logarithmic.
	           If this is False (default) then linear binning is used.
	    log_y: If this is True, then a logarithmic scale is used for the
	           y-axis of the histogram. If this is False (default), then
	           a linear scale is used.
	
	Output
	    A histogram is automatically saved using the given data and labels,
	    in the specified format. 1 is returned if the code performs to
	    completion.
	'''
	
	# First make a figure object with matplotlib (default size)
	fig = plt.figure()
	# Create an axis object to go with this figure
	ax = fig.add_subplot(111)
	
	# Check to see if the x-axis of the histogram needs to be logarithmic
	if log_x == True:
		# Set the x-axis scale of the histogram to be logarithmic
		ax.set_xscale('log')

    # Make a histogram of the given data, with the specified number of 
    # bins. Note that the data array is flattened to one dimension.
    # Do we need to normalise to account for the bin sizes being different?
	aML.hist(x_data.flatten(), bins = 'blocks', normed = True, log = log_y)    
	
	# Add the specified x-axis label to the plot
	plt.xlabel(x_label)
	# Add a y-axis label to the plot
	plt.ylabel('Counts')
	# Add the specified title to the plot
	plt.title(title)
	
	# Save the figure using the title given by the user
	plt.savefig(filename, format = format)
	
	# Print a message to the screen saying that the image was created
	print filename + ' created successfully.\n'
	
	# Close the figure so that it does not take up memory
	plt.close()
	
	# Now that the graph has been produced, return 1
	return 1