#------------------------------------------------------------------------------#
#																			   #
# This code is a utility function which automatically produces scatter plots   #
# for two datasets when supplied with the data for the x and y axes, a colour  #
# label for each plot, as well as axis labels and a figure title, and a        #
# filename. The produced plot is saved using this filename, and the given      #
# extension.					                                               #
#																			   #
# Author: Chris Herron														   #
# Start Date: 31/7/2014, copied from code written in 2013.                     #
#																			   #
#------------------------------------------------------------------------------#

# First import numpy, matplotlib, for array handling and plotting
import numpy as np
import matplotlib.pyplot as plt

def scat_plot2(x_data1, y_data1, x_data2, y_data2, filename, format, x_label = \
	'', y_label = '', title = '', col1 = 'b', col2 = 'r', label1 = '', label2 =\
	'', marker1 = 'o', marker2 = 'x', log_x = False, log_y = False, loc = 0):
	'''
	Description
		This function takes the given data and produces scatter plots. The 
		given axis labels and titles are applied, and then the scatter plot is
		saved using the given filename and format.
	
	Required Input
		x_data1: The data array of x-axis coordinates for the first dataset.
		         Numpy array or list.
		y_data1: The data array of y-axis coordinates for the first dataset.
		         Must have the same length as x_data1. The corresponding entries
		         of x_data1 and y_data1 specify the coordinates of each point.
		x_data2: The data array of x-axis coordinates for the second dataset.
		         Numpy array or list.
		y_data2: The data array of y-axis coordinates for the second dataset.
		         Must have the same length as x_data2. The corresponding entries
		         of x_data2 and y_data2 specify the coordinates of each point.
		filename: The filename (including extension) to use when saving the
				  image. Provide as a string.
		format: The format (e.g. png, jpeg) in which to save the image. This is
				a string.
		x_label: String specifying the x-axis label.
		y_label: String specifying the y-axis label.
		title: String specifying the title of the graph.
		col1: A string specifying the colour of the markers to be used for 
		      dataset 1.
		col2: A string specifying the colour of the markers to be used for 
			  dataset 2.
		label1: A string to be used in a legend to identify dataset 1.
		label2: A string to be used in a legend to identify dataset 2.
		marker1: A string specifying the marker symbol to use for dataset 1.
		marker2: A string specifying the marker symbol to use for dataset 2.
		log_x: A boolean value controlling whether the x axis of the plot is
			   logarithmic or not. If True, then the x axis is logarithmic.
		log_y: A boolean value controlling whether the y axis of the plot is
			   logarithmic or not. If True, then the y axis is logarithmic.
		loc: A string or integer specifying where the legend should be placed.
		     Defaults to 0, for 'best' placement. 1, 2, 3 and 4 are upper
		     right, upper left, lower left, lower right.
	
	Output
		A scatter plot is automatically saved using the given data and labels,
		in the specified format.
	'''
	
	# Calculate the maximum and minimum y values for dataset 1
	y1_max = np.amax(y_data1)
	y1_min = np.amin(y_data1)

	# Calculate the maximum and minimum y values for dataset 2
	y2_max = np.amax(y_data2)
	y2_min = np.amin(y_data2)

	# Calculate the maximum and minimum x values for dataset 1
	x1_max = np.amax(x_data1)
	x1_min = np.amin(x_data1)

	# Calculate the maximum and minimum x values for dataset 2
	x2_max = np.amax(x_data2)
	x2_min = np.amin(x_data2)

	# Calculate the overall maximum and minimum y values
	abs_y_max = max(y1_max, y2_max)
	abs_y_min = min(y1_min, y2_min)

	# Calculate the overall maximum and minimum x values
	abs_x_max = max(x1_max, x2_max)
	abs_x_min = min(x1_min, x2_min)

	# First make a figure object
	fig = plt.figure()
	# Create an axis object to go with this figure
	ax = fig.add_subplot(111)
	# Make a scatter plot of the x vs y data for dataset 1
	plt.scatter(x_data1, y_data1, c = col1, marker = marker1, label = label1)
	# Make a scatter plot of the x vs y data for dataset 2
	plt.scatter(x_data2, y_data2, c = col2, marker = marker2, label = label2)
	
	# Set the y-axis limits to go between the maximum and minimum
	ax.set_ylim([0.9 * abs_y_min, 1.1 * abs_y_max])
	# Set the x-axis limits to go between the maximum and minimum
	ax.set_xlim([0.9 * abs_x_min, 1.1 * abs_x_max])

	# Check to see if the x axis of the plot needs to be logarithmic
	if log_x == True:
		# In this case, make the x axis of the plot area logarithmic
		ax.set_xscale('log')

	# Check to see if the y axis of the plot needs to be logarithmic
	if log_y == True:
		# In this case, make the y axis of the plot area logarithmic
		ax.set_yscale('log')

	# Add an x-axis label to the plot
	plt.xlabel(x_label)
	# Add a y-axis label to the plot
	plt.ylabel(y_label)
	# Add a title to the plot
	plt.title(title)
	# Add a legend to the plot
	plt.legend(loc = loc)

	# Save the figure using the title given by the user
	plt.savefig(filename, format = format)
	
	# Print a message to the screen saying that the image was created
	print filename + ' created successfully'
	
	# Close the figure so that it does not take up memory
	plt.close()
	
	# Now that the graph has been produced, return 1
	return 1