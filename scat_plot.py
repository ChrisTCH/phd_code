#------------------------------------------------------------------------------#
#																			   #
# This code is a utility function which automatically produces scatter plot	   #
# images using matplotlib when supplied with the data for the x and y axes,    #
# as well as axis labels and a figure title, and a filename. The produced plot #
# is saved using this filename, and the given extension.					   #
#																			   #
# Author: Chris Herron														   #
# Start Date: 31/7/2014, copied from code written in 2013.                     #
#																			   #
#------------------------------------------------------------------------------#

# First import numpy, matplotlib, for array handling and plotting
import numpy as np
import matplotlib.pyplot as plt

def scat_plot(x_data, y_data, filename, format, x_label = '', \
					y_label = '', title = '',log_x = False, log_y = False):
	'''
	Description
		This function takes the given data and produces scatter plots. The 
		given axis labels and titles are applied, and then the scatter plot is
		saved using the given filename and format.
	
	Required Input
		x_data: The data array of x-axis coordinates. Numpy array or list.
		y_data: The data array of y-axis coordinates. Must have the same length
				as x_data. The corresponding entries of x_data and y_data 
				specify the coordinates of each point.
		filename: The filename (including extension) to use when saving the
				  image. Provide as a string.
		format: The format (e.g. png, jpeg) in which to save the image. This is
				a string.
		x_label: String specifying the x-axis label.
		y_label: String specifying the y-axis label.
		title: String specifying the title of the graph.
		log_x: A boolean value controlling whether the x axis of the plot is
			   logarithmic or not. If True, then the x axis is logarithmic.
		log_y: A boolean value controlling whether the y axis of the plot is
			   logarithmic or not. If True, then the y axis is logarithmic.
	
	Output
		A scatter plot is automatically saved using the given data and labels,
		in the specified format.
	'''
	
	# First make a figure object
	fig = plt.figure()
	# Create an axis object to go with this figure
	ax = fig.add_subplot(111)
	# Make a scatter plot of the x vs y data
	plt.scatter(x_data, y_data)
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

	# Set the y-axis limits to go between the maximum and minimum
	ax.set_ylim([0.9 * np.amin(y_data), 1.1 * np.amax(y_data)])
	# Set the x-axis limits to go between the maximum and minimum
	ax.set_xlim([0.9 * np.amin(x_data), 1.1 * np.amax(x_data)])

	# Save the figure using the title given by the user
	plt.savefig(filename, format = format)
	
	# Print a message to the screen saying that the image was created
	print filename + ' created successfully'
	
	# Close the figure so that it does not take up memory
	plt.close()
	
	# Now that the graph has been produced, return 1
	return 1