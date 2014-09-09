#------------------------------------------------------------------------------#
#                                                                              #
# This code is a function that produces an interactive Mayavi scene of a data  #
# cube. The visualisation shows slices through the x, y and z axes of the data #
# which can be moved to investigate a data cube. The cube can be rotated, as   #
# can the slices present in the visualisation.                                 #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 4/9/2014                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# Import the mlab portion of Mayavi, to provide Python scripting of 
# visualisation methods
from mayavi import mlab

def slicing(data, xlabel = '', ylabel = '', zlabel = '', title = ''):
	'''
	Description
		This function generates a Mayavi scene that shows cut planes along the
		x, y and z axes of a data cube. This is an interactive scene, and so 
		the cut planes can be moved through the cube, and the entire cube
		can be rotated.
	
	Required Input
		data: A three dimensional numpy array. Each entry of the array must
			  contain a scalar value. 
		xlabel: A string specifying a label for the x axis of the data cube.
		ylabel: A string specifying a label for the y axis of the data cube.
		zlabel: A string specifying a label for the z axis of the data cube.
		title: A string specifying the title for the visualisation.
	
	Output
		This function returns 1 if it runs to completion. An interactive
		Mayavi scene is produced by the function, allowing slices through
		a three dimensional data cube to be viewed.
	'''
	
	# Create a new Mayavi scene to visualise slicing the data cube
	scene = mlab.figure(size = (800,700))

	# Run a widget that allows you to visualise three dimensional data sets
	# This creates a slice along the x axis that can be interactively varied
	mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),\
        plane_orientation='x_axes', slice_index=0)

	# This creates a slice along the y axis that can be interactively varied,
	# on the same image as the x axis slice
	mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),\
        plane_orientation='y_axes', slice_index=0)

	# This creates a slice along the z axis that can be interactively varied,
	# on the same image as the x and y axis slices
	mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
        plane_orientation='z_axes', slice_index=0)

	# Add axes to the visualisation of the image cube, and label the x, y and
	# z axes
	mlab.axes(xlabel = xlabel, ylabel = ylabel, zlabel = zlabel)

	# Add a title to the visualisation of the image cube
	mlab.title(title, height = 1.0)

	# Make the outline of the image cube appear
	mlab.outline()

	# Make a colourbar for the data
	mlab.scalarbar(orientation = 'vertical')

	# Add a little symbol to show the orientation of the data
	mlab.orientation_axes()

	# Allow interaction with the produced visualisation
	mlab.show()

	# Once the interactive 3D view has been created, return 0 
	return 1