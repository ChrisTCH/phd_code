#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the skewness, kurtosis, radially averaged        #
# structure function, and quadrupole/monopole ratio of synchrotron intensity   #
# maps that are influenced by both noise and angular resolution. Each of these #
# statistics is plotted against the sonic and Alfvenic Mach numbers, to see    #
# which quantities are sensitive tracers of the sonic and Alfvenic Mach        #
# numbers when observational effects are included. Statistics are also plotted #
# against the noise and angular resolution for each simulation, to see how     #
# these effects change the statistics. A 2D plot of each statistic as a        #
# function of both noise and angular resolution is also produced for each      #
# simulation.                                                                  #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 29/4/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, astropy.convolution for convolution functions, 
# scipy.stats for calculating statistical quantities,
# scipy.ndimage for smoothing and convolution
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy import stats
from scipy import ndimage

# Import the functions that calculate the structure and correlation functions
# using FFT, the function that calculates the radially averaged structure or 
# correlation functions, the function that calculates multipoles of 2D images,
# and the function that calculates the magnitude and argument of the quadrupole
# ratio. The function that converts Numpy arrays into FITS files is also 
# imported. The function that creates images of 2D matrices is imported.
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio
from mat2FITS_Image import mat2FITS_Image
from mat_plot import mat_plot

# Define a function that will simplify the production of many of the plots that
# are produced at the end of this code
def slice_plotter(xdata, y_array, slice_data, slices, filename, format = 'png',\
 xlabel = '', ylabel = '', title = '', legend = '{}', row_column = 'row'):
	'''
	Description
        This function produces a plot of slices from the y array against the 
        x data that is given. The slices used can be rows of the array, or the
        columns of the array. The plot produced is saved using the given 
        filename, in the given format.
        
    Required Input
        xdata - A one-dimensional Numpy array representing the quantity that is
        		being plotted against. Each entry of this array is assumed to be
        		a float.
        y_array - A two-dimensional Numpy array that contains the information to
        		  be plotted against the x data. Rows or columns are taken from
        		  this array, and plotted against the x data. 
        slice_data - A one-dimensional Numpy array that contains the values of 
        			 the quantity being used to determine the slices.
        slices - A one-dimensional Numpy array, giving the index values of the 
        		 rows or columns to be used in the plotting.
        filename - The filename to be used when saving the image. The image may
                   be saved as a ps, eps, pdf, png or jpeg file, depending on
                   the extension of the filename. Must be a string.
        format - The format in which the image will be saved, as a string. e.g.
                 'png'.
        xlabel - The label for the x axis, provided as a string.
        ylabel - The label for the y axis, provided as a string.
        title - The title for the image, provided as a string.
        legend - The base string to be used in the legend. This should contain
        		 {}, so that information from slice_data can be added to the 
        		 legend.
        row_column - Either the string 'row' or 'column'. If 'row', then rows 
        			 are taken from the y_array and plotted against the xdata.
        			 If 'column' is selected, then columns from the y_array are 
        			 plotted against the xdata.
                   
    Output
        The figure produced is saved using the given filename and format. If 
        the image is created successfully, then 1 is returned.

	'''

	# Create the figure to hold the plot
	fig = plt.figure(figsize = (10,6))

	# Create an axis for this figure
	ax = fig.add_subplot(111)

	# Plot the data in the y array as a function of the x data, for the slices
	# that were chosen
	for j in slices:
		# Check to see if we plot rows or columns against the x data
		if row_column == 'row':
			# In this case we plot rows against the xdata
			plt.plot(xdata, y_array[j,:], '-o', label =legend.format(slice_data[j]))
		else:
			# In this case we plot columns against the xdata
			plt.plot(xdata, y_array[:,j], '-o', label =legend.format(slice_data[j]))

	# Add a label to the x-axis
	plt.xlabel(xlabel, fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel(ylabel, fontsize = 20)

	# Add a title to the plot
	plt.title(title, fontsize = 20)

	# Shrink the width of the plot axes
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Force the legend to appear on the plot
	ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

	# Save the figure using the given filename and format
	plt.savefig(filename, format = format)

	# Close the figure, so that it is no longer in memory
	plt.close()

	# Now that the plot has been saved, return 1 to signify that the plot was
	# produced and saved successfully
	return 1

#-------------------------------------------------------------------------------

# Define a function that will simplify the production of many of the plots that
# are produced at the end of this code
def sub_plotter(xdata, data_array, slice_data_x, slice_data_y, slices, filename,\
 format = 'png', xlabel = '', ylabel = '', title = '', x_slice_label = '{}',\
 y_slice_label = '{}'):
	'''
	Description
        This function produces a plot of the values of a quantity that are 
        contained on lines through the three-dimensional data array against the 
        x data that is given. The plot produced includes numerous subplots, 
        representing lines through the y array that have different x and y 
        co-ordinates. The plot produced is saved using the given filename, in 
        the given format.
        
    Required Input
        xdata - A one-dimensional Numpy array representing the quantity that is
        		being plotted against in each subplot. Each entry of this array 
        		is assumed to be a float.
        data_array - A three-dimensional Numpy array that contains the 
        		  information to be plotted against the x data. Lines through 
        		  this cube (along its z axis) are extracted, and the data in 
        		  these lines is plotted against the x data.
        slice_data_x - A one-dimensional Numpy array that contains the values of 
        			 the quantity being used to generate the x axis of the data
        			 array.
        slice_data_y - A one-dimensional Numpy array that contains the values of 
        			 the quantity being used to generate the y axis of the data
        			 array.
        slices - A one-dimensional Numpy array, giving the index values along 
        		 x and y axes of the data_array to use when selecting which 
        		 lines to use in making the subplots. The number of entries in 
        		 this array determines how many subplots are in each row and 
        		 column of the produced plot.
        filename - The filename to be used when saving the image. The image may
                   be saved as a ps, eps, pdf, png or jpeg file, depending on
                   the extension of the filename. Must be a string.
        format - The format in which the image will be saved, as a string. e.g.
                 'png'.
        xlabel - The label for the x axis, provided as a string.
        ylabel - The label for the y axis, provided as a string.
        title - The title for the image, provided as a string.
        x_slice_label - The base string to be used in the titles of the upper-
        		 most row of subplots, to label the value of the quantity used
        		 to make the subplots in each column. This should contain {}, so
        		 that information from slice_data_x can be added to the label.
        y_slice_label - The base string to be used in the y-axis labels of the 
        		 right-most column of subplots, to label the value of the 
        		 quantity used to make the subplots in each row. This should 
        		 contain {}, so that information from slice_data_y can be added 
        		 to the label.
                   
    Output
        The figure produced is saved using the given filename and format. If 
        the image is created successfully, then 1 is returned.

	'''

	# Create the figure instance, as well as all of the axes instances for the
	# subplots. The number of subplots is determined by the length of the 
	# slices array, and all subplots have the same x axes, and the same y axes.
	# axarr is a 2D array, where each entry gives the axis instance for the 
	# subplot in the corresponding space in the figure.
	fig, axarr = plt.subplots(len(slices), len(slices), sharex = True, sharey = 'row')

	# Add an overall x axis to the figure, which should represent the quantity
	# being plotted on the x axis of each subplot
	fig.text(0.5, 0.04, xlabel, ha = 'center', va = 'center', fontsize = 20)

	# Add an overall y axis to the figure, which should represent the quantity
	# being plotted on the y axis of each subplot
	fig.text(0.04, 0.5, ylabel, ha = 'center', va = 'center', fontsize = 20,\
		rotation='vertical')

	# Loop over the rows in the figure, to produce all of the subplots for each 
	# row. Only the slice_data_y variable should change going from row to row. 
	# This corresponds to going down the y-axis of the data array.
	for j in range(len(slices)):
		# Loop over the columns in the figure, to produce all of the subplots
		# for each column. Only the slice_data_x variable should change when
		# going across the columns. This corresponds to going across the x-axis
		# of the data array.
		for i in range(len(slices)):
			# Extract the line (the data going along the z axis) for the given
			# x and y co-ordinates of the data array
			stat_list = np.squeeze(data_array[:,slices[j],slices[i]])

			# Produce a scatter plot using the first 8 data points along the 
			# line that has been extracted
			(axarr[j,i]).scatter(xdata[0:8], stat_list[0:8], c='b')

			# Produce a scatter plot using the remaining data points along the
			# line that has been extracted
			(axarr[j,i]).scatter(xdata[8:], stat_list[8:], c='r')

	# Loop over all rows except the bottom row, to make it so that the ticks
	# along the x axis are invisible
	for j in range(len(slices)-1):
		# Make the ticks along the x axis invisible for these subplots
		plt.setp([a.get_xticklabels() for a in axarr[j, :]], visible=False)

	# Loop over all the columns except the right most column, to make it so that
	# all of the ticks along the y axis are invisible
	for i in np.linspace(1, len(slices) - 1, num = len(slices) - 1):
		# Make the ticks along the y axis invisible for these subplots
		plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)

	# Loop over the rows of the figure, so that a y label can be given to all
	# of the right most subplots
	for j in range(len(slices)):
		# Change the co-ordinates of the y axis label, so that it appears on the
		# right of the subplot, not the left
		axarr[j,-1].yaxis.set_label_coords(1.2,0.5)
		
		# Add the value of the quantity used to construct the y axis of the data 
		# array to the y labels of the subplots in the right most columns
		axarr[j,-1].yaxis.set_label_text(y_slice_label.format(slice_data_y[slices[j]]),\
			fontsize = 10)

	# Loop over the columns of the figure, so that an x axis label can be given
	# to each column, via the titles of the upper most subplots
	for i in range(len(slices)):
		# Add the value of the quantity used to construct the x axis of the data 
		# array to the titles of the subplots in the right most columns
		axarr[0,i].set_title(x_slice_label.format(slice_data_x[slices[i]]),\
			fontsize = 10)

	# Adjust the vertical spacing between subplots, so that there is less white
	# space between them
	fig.subplots_adjust(hspace = 0.35)
	
	# Adjust the horizontal spacing between subplots, so that there is less 
	# white space between them
	fig.subplots_adjust(wspace = 0.35)

	# Add an overall title to the figure
	fig.suptitle(title, fontsize = 20)

	# Save the figure using the given filename and format
	plt.savefig(filename, format = format)

	# Close the figure, so that it is no longer in memory
	plt.close()

	# Now that the plot has been saved, return 1 to signify that the plot was
	# produced and saved successfully
	return 1

#-------------------------------------------------------------------------------

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 25

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# The directories end in:
# b.1p.1_Oct_Burk
# b.1p.01_Oct_Burk
# b.1p2_Aug_Burk
# b1p.1_Oct_Burk
# b1p.01_Oct_Burk
# b1p2_Aug_Burk
# c512b.1p.0049
# c512b.1p.0077
# c512b.1p.025
# c512b.1p.05
# c512b.1p.7
# c512b1p.0049
# c512b1p.0077
# c512b1p.025
# c512b1p.05
# c512b1p.7
# c512b3p.01
# c512b5p.01
# c512b5p2

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
simul_arr = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/', 'c512b3p.01/', 'c512b5p.01/']

# Create a list, where each entry is a string describing the initial magnetic
# field and pressure used to run each simulation.
short_simul = ['b.1p.0049', 'b.1p.0077', 'b.1p.01', 'b.1p.025', 'b.1p.05',\
'b.1p.1', 'b.1p.7', 'b.1p2', 'b1p.0049', 'b1p.0077', 'b1p.01', 'b1p.025',\
'b1p.05', 'b1p.1', 'b1p.7', 'b1p2', 'b3p.01', 'b5p.01']

# Create an array, where each entry specifies the pressure of the corresponding
# simulation in the list of simulation directories
press_arr = np.array([0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0,\
	0.0049, 0.0077, 0.01, 0.025, 0.05, 0.1, 0.7, 2.0, 0.01, 0.01])

# Create an array, where each entry specifies the initial mean magnetic field of
# the corresponding simulation to study 
mag_arr = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,\
	1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 5.0])

# Create an array, where each entry specifies the calculated sonic Mach number 
# for each simulation
sonic_mach_arr = np.array([8.85306946, 5.42555035, 5.81776713, 3.71658244,\
 2.75242104, 2.13759125, 0.81017387, 0.44687901, 7.5584105, 6.13642211,\
 5.47297919, 3.63814214, 2.69179409, 2.22693767, 0.83800535, 0.47029213,\
 6.57849578, 7.17334893])

# Create an array, where each entry specifies the calculated Alfvenic Mach 
# number for each simulation
alf_mach_arr = np.array([1.41278383, 1.77294593, 1.75575508, 1.50830194,\
 1.69455875, 1.85993991, 1.74231524, 1.71939152, 0.49665052, 0.50288954,\
 0.51665006, 0.54928564, 0.57584022, 0.67145057, 0.70015313, 0.65195539,\
 0.21894299, 0.14357068])

# Create an array of index values that sorts the sonic Mach number values from
# smallest to largest
sonic_sort = np.argsort(sonic_mach_arr)

# Create an array of index values that sorts the Alfvenic Mach number values 
# from smallest to largest
alf_sort = np.argsort(alf_mach_arr)

# Create an array of the sonic Mach number values, from smallest to largest
sonic_mach_sort = sonic_mach_arr[sonic_sort]

# Create an array of the Alfvenic Mach number values, from smallest to largest
alf_mach_sort = alf_mach_arr[alf_sort]

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a variable that controls how many data points are being used for the
# noise level and angular resolution. MUST BE A MULTIPLE OF 5
free_num = 25

# Create an array of values that specify what slices to use when plotting 
# statistics against one observational effect, when creating images with lots of
# subplots, or when saving fits files
slice_indices = np.linspace(0, free_num, num = 5, endpoint = False)

# Create an array of values that will be used to determine the standard
# deviation of the Gaussian distribution from which noise values are 
# generated. The standard deviation will be calculated by multiplying the
# median synchrotron intensity by the values in this array.
noise_array = np.linspace(0.01, 0.5, free_num)

# Create a label for the x-axis of plots that are made against noise
# standard deviation
noise_xlabel = 'Noise StandDev [perc median inten]'

# Create a string to be used in the titles of any plots that are made 
# against noise standard deviation
noise_title_string = 'Noise StandDev'

# Create a string to be used in legends involving noise level
noise_leg_string = 'Noise = ' 

# Create an array of values that represent the standard deviation of the 
# Gaussian used to smooth the synchrotron maps. All values are in pixels.
angres_array = np.linspace(1.0, 50.0, free_num)

# Create an array of values representing the final angular resolution of
# the image after smoothing. The final resolution is calculated by 
# quadrature from the initial resolution (1 pixel) and the standard 
# deviation of the convolving Gaussian.
final_res = np.sqrt(1.0 + np.power(angres_array,2.0))

# Create a label for the x-axis of plots that are made against angular 
# resolution
angres_xlabel = 'Angular Resolution [pixels]'

# Create a string to be used in the titles of any plots that are made 
# against angular resolution
angres_title_string = 'Angular Resolution'

# Create a string to be used in legends involving angular resolution
angres_leg_string = 'AngRes = ' 

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of the noise and angular resolution being studied. Each 
# slice corresponds to a simulation, each row within a slice corresponds to
# a value of the angular resolution, and each column corresponds to a value of
# the noise level. There is one array for a line of sight along the z axis, and 
# another for a line of sight along the x axis.
# NOTE: We will calculate the biased skewness
skew_z_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))
skew_x_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))

# Create an empty array, where each entry specifies the calculated kurtosis of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of the noise and angular resolution being studied. Each 
# slice corresponds to a simulation, each row within a slice corresponds to
# a value of the angular resolution, and each column corresponds to a value of
# the noise level. There is one array for a line of sight along the z axis, and 
# another for a line of sight along the x axis.
# NOTE: We will calculate the biased Fisher kurtosis
kurt_z_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))
kurt_x_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))

# Create an empty array, where each entry specifies the calculated slope of the
# structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of the noise and angular 
# resolution being studied. Each slice corresponds to a simulation, each row 
# within a slice corresponds to a value of the angular resolution, and each 
# column corresponds to a value of the noise level. There is one array for a 
# line of sight along the z axis, and another for a line of sight along the x 
# axis.
m_z_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))
m_x_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))

# Create an empty array, where each entry specifies the calculated residuals 
# from the linear fit to the structure function of the synchrotron intensity 
# image, of the corresponding simulation for a particular value of the noise and
# angular resolution being studied. Each slice corresponds to a simulation, each
# row within a slice corresponds to a value of the angular resolution, and each 
# column corresponds to a value of the noise level. There is one array for a 
# line of sight along the z axis, and another for a line of sight along the x 
# axis.
residual_z_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))
residual_x_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))

# Create an empty array, where each entry specifies the calculated integrated
# quadrupole ratio modulus of the synchrotron intensity image of the 
# corresponding simulation for a particular value of the noise and angular 
# resolution being studied. Each slice corresponds to a simulation, each row 
# within a slice corresponds to a value of the angular resolution, and each 
# column corresponds to a value of the noise level. There is one array for a 
# line of sight along the z axis, and another for a line of sight along the x 
# axis.
int_quad_z_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))
int_quad_x_arr = np.zeros((len(simul_arr),len(angres_array),len(noise_array)))

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/Obs_Effects_Combined2/'

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for k in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[k]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[k])

	# Open the FITS files that contain the simulated synchrotron intensity maps
	# for lines of sight along the z axis and x axis
	sync_fits_z = fits.open(data_loc + 'synint_p1-4.fits')
	sync_fits_x = fits.open(data_loc + 'synint_p1-4x.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensities observed for different values of gamma, the power
	# law index of the cosmic ray electrons emitting the synchrotron emission.
	sync_data_z = sync_fits_z[0].data
	sync_data_x = sync_fits_x[0].data

	# Extract the synchrotron intensity map for the value of gamma, for
	# lines of sight along the x and z axes
	sync_map_z = sync_data_z[gam_index]
	sync_map_x = sync_data_x[gam_index]

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Loop over the various values of the angular resolution, to calculate the 
	# various statistics for the synchrotron map observed for each value of the
	# angular resolution
	for j in range(len(angres_array)):
		# Create empty arrays, that will store the synchrotron intensity 
		# maps produced. Each slice of these arrays corresponds to a 
		# different value of the noise. There is one array for a line of 
		# sight along the z axis, and another for a line of sight along the 
		# x axis.
		sync_param_z = np.zeros((free_num, np.shape(sync_map_z)[0], \
			np.shape(sync_map_z)[1]))
		sync_param_x = np.zeros((free_num, np.shape(sync_map_x)[0], \
			np.shape(sync_map_x)[1]))

		# Loop over the various values of the noise, to calculate the various
		# statistics for the synchrotron map observed for each value of the 
		# noise
		for i in range(len(noise_array)):
			# First, we take into account the effect of noise in the telescope. 
			# We start with an array of values that, when multiplied by the 
			# median intensity of the synchrotron map, give the standard 
			# deviation of the Gaussian noise. 

			# Take into account an observing frequency of 1.4 GHz, by 
			# multiplying the extracted synchrotron maps by a gamma dependent 
			# frequency factor
			sync_map_z_f = sync_map_z * np.power(1.4, -(gamma - 1))
			sync_map_x_f = sync_map_x * np.power(1.4, -(gamma - 1))

			# Calculate the standard deviation of the Gaussian noise that will 
			# affect the synchrotron maps. This needs to be done individually 
			# for lines of sight along the z and x axes, because of the lines of
			# sight have different intensity maps.
			noise_stdev_z = noise_array[i] * np.median(sync_map_z_f)
			noise_stdev_x = noise_array[i] * np.median(sync_map_x_f)

			# Create an array of values that are randomly drawn from a Gaussian
			# distribution with the specified standard deviation. This 
			# represents the noise at each pixel of the image. 
			noise_matrix_z = np.random.normal(scale = noise_stdev_z,\
			 size = np.shape(sync_map_z))
			noise_matrix_x = np.random.normal(scale = noise_stdev_x,\
			 size = np.shape(sync_map_x))

			# Add the noise maps onto the synchrotron intensity maps
			sync_map_noise_z = sync_map_z_f + noise_matrix_z
			sync_map_noise_x = sync_map_x_f + noise_matrix_x

			# Now take into account the effect of angular resolution. We start
			# with an array of values that specifies the standard deviation of 
			# the Gaussian to be used to smooth the data.

			# Create a Gaussian kernel to use to smooth the synchrotron map,
			# using the given standard deviation
			gauss_kernel = Gaussian2DKernel(angres_array[j])

			# Smooth the synchrotron maps to the required resolution by 
			# convolution with the above Gaussian kernel.
			sync_map_free_param_z = convolve_fft(sync_map_noise_z,\
			 gauss_kernel, boundary = 'wrap')
			sync_map_free_param_x = convolve_fft(sync_map_noise_x,\
			 gauss_kernel, boundary = 'wrap')

			# Now that the synchrotron map has been produced for this value of 
			# the free parameter, store it in the array that will hold all of 
			# the produced synchrotron maps
			sync_param_z[i] = sync_map_free_param_z
			sync_param_x[i] = sync_map_free_param_x

			# Flatten the synchrotron intensity maps for this value of gamma, for
			# lines of sight along the x and z axes
			flat_sync_z = sync_map_free_param_z.flatten()
			flat_sync_x = sync_map_free_param_x.flatten()

			# Calculate the biased skewness of the synchrotron intensity maps, 
			# for lines of sight along the x and z axes, and store the results 
			# in the corresponding array.
			skew_z_arr[k,j,i] = stats.skew(flat_sync_z)
			skew_x_arr[k,j,i] = stats.skew(flat_sync_x)

			# Calculate the biased Fisher kurtosis of the synchrotron intensity 
			# maps, for lines of sight along the x and z axes, and store the 
			# results in the corresponding array.
			kurt_z_arr[k,j,i] = stats.kurtosis(flat_sync_z)
			kurt_x_arr[k,j,i] = stats.kurtosis(flat_sync_x)

			# Calculate the structure function (two-dimensional) of the 
			# synchrotron intensity maps, for the lines of sight along the x and
			# z axes. Note that no_fluct = True is set, because we are not 
			# subtracting the mean from the synchrotron maps before calculating 
			# the structure function.
			strfn_z = sf_fft(sync_map_free_param_z, no_fluct = True)
			strfn_x = sf_fft(sync_map_free_param_x, no_fluct = True)

			# Radially average the calculated 2D structure function, using the 
			# specified number of bins, for lines of sight along the x and z 
			# axes.
			rad_sf_z = sfr(strfn_z, num_bins, verbose = False)
			rad_sf_x = sfr(strfn_x, num_bins, verbose = False)

			# Extract the calculated radially averaged structure function for 
			# lines of sight along the x and z axes.
			sf_z = rad_sf_z[1]
			sf_x = rad_sf_x[1]

			# Extract the radius values used to calculate this structure 
			# function, for line of sight along the x and z axes.
			sf_rad_arr_z = rad_sf_z[0]
			sf_rad_arr_x = rad_sf_x[0]

			# Calculate the spectral index of the structure function calculated 
			# for this value of gamma. Note that only the first third of the 
			# structure function is used in the calculation, as this is the part
			# that is close to a straight line. Perform a linear fit for a line
			# of sight along the z axis.
			spec_ind_data_z = np.polyfit(np.log10(\
				sf_rad_arr_z[0:np.ceil(num_bins/3.0)]),\
				np.log10(sf_z[0:np.ceil(num_bins/3.0)]), 1, full = True)
			# Perform a linear fit for a line of sight along the x axis
			spec_ind_data_x = np.polyfit(np.log10(\
				sf_rad_arr_x[0:np.ceil(num_bins/3.0)]),\
				np.log10(sf_x[0:np.ceil(num_bins/3.0)]), 1, full = True)

			# Extract the returned coefficients from the polynomial fit, for 
			# lines of sight along the x and z axes
			coeff_z = spec_ind_data_z[0]
			coeff_x = spec_ind_data_x[0]

			# Extract the sum of the residuals from the polynomial fit, for 
			# lines of sight along the x and z axes
			residual_z_arr[k,j,i] = spec_ind_data_z[1]
			residual_x_arr[k,j,i] = spec_ind_data_x[1]

			# Enter the value of m, the slope of the structure function minus 1,
			# into the corresponding array, for lines of sight along the x and z
			# axes
			m_z_arr[k,j,i] = coeff_z[0]-1.0
			m_x_arr[k,j,i] = coeff_x[0]-1.0

			# Calculate the 2D structure function for this slice of the 
			# synchrotron intensity data cube. Note that no_fluct = True is set,
			# because we are not subtracting the mean from the synchrotron maps 
			# before calculating the structure function. We are also calculating
			# the normalised  structure function, which only takes values 
			# between 0 and 2.
			norm_strfn_z = sf_fft(sync_map_free_param_z, no_fluct = True,\
			 normalise = True)
			norm_strfn_x = sf_fft(sync_map_free_param_x, no_fluct = True,\
			 normalise = True)

			# Shift the 2D structure function so that the zero radial separation
			# entry is in the centre of the image. This is done for lines of 
			# sight along the x and z axes
			norm_strfn_z = np.fft.fftshift(norm_strfn_z)
			norm_strfn_x = np.fft.fftshift(norm_strfn_x)

			# Calculate the magnitude and argument of the quadrupole ratio, for 
			# lines of sight along the x and z axes.
			quad_mod_z, quad_arg_z, quad_rad_z = calc_quad_ratio(norm_strfn_z,\
			 num_bins)
			quad_mod_x, quad_arg_x, quad_rad_x = calc_quad_ratio(norm_strfn_x,\
			 num_bins)

			# Integrate the magnitude of the quadrupole/monopole ratio from one 
			# sixth of the way along the radial separation bins, until three 
			# quarters of the way along the radial separation bins. This 
			# integration is performed with respect to log separation (i.e. I am
			# ignoring the fact that the points are equally separated in log 
			# space, to calculate the area under the quadrupole / monopole ratio
			# plot when the x axis is scaled logarithmically). I normalise the 
			# value that is returned by dividing by the number of increments in 
			# log radial separation used in the calculation. This is done for 
			# lines of sight along the x and z axes.
			int_quad_z_arr[k,j,i] = np.trapz(quad_mod_z[np.floor(num_bins/6.0):\
				3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
				 - np.floor(num_bins/6.0))
			int_quad_x_arr[k,j,i] = np.trapz(quad_mod_x[np.floor(num_bins/6.0):\
				3*np.floor(num_bins/4.0)+1], dx = 1.0) / (3*np.floor(num_bins/4.0)\
				 - np.floor(num_bins/6.0))

		# At this point, all of the statistics that need to be calculated 
		# for the noise values have been calculated.

		# For certain values of the angular resolution being applied, we want to
		# save the synchrotron intensity maps that have been produced
		if j in slice_indices:
			# Convert the arrays that show the synchrotron map for each value of
			# the free parameter into a FITS file, and save it.
			mat2FITS_Image(sync_param_z, filename = save_loc + short_simul[k] +\
			 '_Ang{0:.2f}'.format(angres_array[j]) + '_z.fits')
			mat2FITS_Image(sync_param_x, filename = save_loc + short_simul[k] +\
			 '_Ang{0:.2f}'.format(angres_array[j]) + '_x.fits')

		# Replace the array of standard deviations with the array of final
		# resolutions, so that the final resolutions are used in all plots
		angres_array[j] = final_res[j]

	# Close the fits files, to save memory
	sync_fits_z.close()
	sync_fits_x.close()

	# When the code reaches this point, the statistics have been calculated
	# for all noise and angular resolution values, so the code will proceed
	# onto the next simulation

# When the code reaches this point, the statistics have been calculated for
# every simulation and every value of noise and angular resolution, so it is
# time to start plotting

# Loop over the different simulations, to produce plots for each simulation
for k in range(len(simul_arr)):
	#----------------------------- 2D Figures ----------------------------------

	# Create the first figure, which will be a 2D map of how skewness
	# depends on the noise and angular resolution. Do this for lines of sight
	# along the z and x axes
	mat_plot(skew_z_arr[k], save_loc +\
	 '2D_skew_{}_z.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Skewness vs AngRes and Noise {}z'.format(short_simul[k]))
	mat_plot(skew_x_arr[k], save_loc +\
	 '2D_skew_{}_x.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Skewness vs AngRes and Noise {}x'.format(short_simul[k]))

	# Create the second figure, which will be a 2D map of how kurtosis
	# depends on the noise and angular resolution. Do this for lines of sight
	# along the z and x axes
	mat_plot(kurt_z_arr[k], save_loc +\
	 '2D_kurt_{}_z.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Kurtosis vs AngRes and Noise {}z'.format(short_simul[k]))
	mat_plot(kurt_x_arr[k], save_loc +\
	 '2D_kurt_{}_x.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Kurtosis vs AngRes and Noise {}x'.format(short_simul[k]))

	# Create the third figure, which will be a 2D map of how the structure 
	# function slope minus 1 depends on the noise and angular resolution. Do 
	# this for lines of sight along the z and x axes
	mat_plot(m_z_arr[k], save_loc +\
	 '2D_m_{}_z.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'SF Slope vs AngRes and Noise {}z'.format(short_simul[k]))
	mat_plot(m_x_arr[k], save_loc +\
	 '2D_m_{}_x.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'SF Slope vs AngRes and Noise {}x'.format(short_simul[k]))

	# Create the fourth figure, which will be a 2D map of how the residuals 
	# of the linear fit to the structure function depend on the noise and 
	# angular resolution. Do this for lines of sight along the z and x axes
	mat_plot(residual_z_arr[k], save_loc +\
	 '2D_resid_{}_z.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Residuals vs AngRes and Noise {}z'.format(short_simul[k]))
	mat_plot(residual_x_arr[k], save_loc +\
	 '2D_resid_{}_x.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Residuals vs AngRes and Noise {}x'.format(short_simul[k]))

	# Create the fifth figure, which will be a 2D map of how the integrated
	# quadrupole ratio modulus depends on the noise and angular resolution. Do 
	# this for lines of sight along the z and x axes
	mat_plot(int_quad_z_arr[k], save_loc +\
	 '2D_int_quad_{}_z.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Int Quad vs AngRes and Noise {}z'.format(short_simul[k]))
	mat_plot(int_quad_x_arr[k], save_loc +\
	 '2D_int_quad_{}_x.png'.format(short_simul[k]), 'png', x_ticks = noise_array,\
	  y_ticks = angres_array, cmap = 'hot', aspect = 'auto',\
	  interpolation = 'bilinear', origin = 'lower', xlabel = noise_xlabel,\
	  ylabel = angres_xlabel, title = \
	  'Int Quad vs AngRes and Noise {}x'.format(short_simul[k]))

	# After this point, all of the 2D plots have been made for this simulation,
	# so print out a line informing the user of this
	print 'All 2D plots saved for the {} simulation'.format(short_simul[k])

	#------------------------ Plots against Noise level ------------------------

	# Create the first figure in this series, which will be a plot of the 
	# skewness as a function of the noise level, for different angular 
	# resolutions, for this simulation. This is done for lines of sight along
	# the z and x axes.

	# Skewness zLOS
	slice_plotter(noise_array, skew_z_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'Skew_Noise_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Skewness', title =\
	  'Skew vs ' + noise_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Skewness xLOS
	slice_plotter(noise_array, skew_x_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'Skew_Noise_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Skewness', title =\
	  'Skew vs ' + noise_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Create the second figure in this series, which will be a plot of the 
	# kurtosis as a function of the noise level, for different angular 
	# resolutions, for this simulation. This is done for lines of sight along
	# the z and x axes.

	# Kurtosis zLOS
	slice_plotter(noise_array, kurt_z_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'Kurt_Noise_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Kurtosis', title =\
	  'Kurt vs ' + noise_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Kurtosis xLOS
	slice_plotter(noise_array, kurt_x_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'Kurt_Noise_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Kurtosis', title =\
	  'Kurt vs ' + noise_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Create the third figure in this series, which will be a plot of the 
	# structure function slope minus 1 as a function of the noise level, for
	# different angular resolutions, for this simulation. This is done for lines
	# of sight along the z and x axes.

	# m zLOS
	slice_plotter(noise_array, m_z_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'m_Noise_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'SF Slope - 1', title =\
	  'm vs ' + noise_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# m xLOS
	slice_plotter(noise_array, m_x_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'m_Noise_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'SF Slope - 1', title =\
	  'm vs ' + noise_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Create the fourth figure in this series, which will be a plot of the 
	# residuals of the linear fit to the structure function as a function of the
	# noise level, for different angular resolutions, for this simulation. This 
	# is done for lines of sight along the z and x axes.

	# Residuals zLOS
	slice_plotter(noise_array, residual_z_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'resid_Noise_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Residuals', title =\
	  'Residuals vs ' + noise_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Residuals xLOS
	slice_plotter(noise_array, residual_x_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'resid_Noise_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Residuals', title =\
	  'Residuals vs ' + noise_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Create the fifth figure in this series, which will be a plot of the 
	# integrated quadrupole ratio modulus as a function of the noise level, for 
	# different angular resolutions, for this simulation. This is done for lines
	# of sight along the z and x axes.

	# Integrated quadrupole zLOS
	slice_plotter(noise_array, int_quad_z_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'int_quad_Noise_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Int Quad', title =\
	  'Int Quad vs ' + noise_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# Integrated quadrupole xLOS
	slice_plotter(noise_array, int_quad_x_arr[k], angres_array, slice_indices,\
	 filename=save_loc+'int_quad_Noise_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = noise_xlabel, ylabel = 'Int Quad', title =\
	  'Int Quad vs ' + noise_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'AngRes={0:.2f}', row_column = 'row')

	# After this point, all of the plots against noise level have been made for
	# this simulation, so print out a line informing the user of this
	print 'All plots against noise saved for the {} simulation'.format(short_simul[k])

	#-------------------- Plots against Angular Resolution ---------------------

	# Create the first figure in this series, which will be a plot of the 
	# skewness as a function of the angular resolution, for different noise 
	# levels, for this simulation. This is done for lines of sight along
	# the z and x axes.

	# Skewness zLOS
	slice_plotter(angres_array, skew_z_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'Skew_AngRes_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Skewness', title =\
	  'Skew vs ' + angres_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Skewness xLOS
	slice_plotter(angres_array, skew_x_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'Skew_AngRes_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Skewness', title =\
	  'Skew vs ' + angres_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Create the second figure in this series, which will be a plot of the 
	# kurtosis as a function of the angular resolution, for different noise 
	# levels, for this simulation. This is done for lines of sight along
	# the z and x axes.

	# Kurtosis zLOS
	slice_plotter(angres_array, kurt_z_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'Kurt_AngRes_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Kurtosis', title =\
	  'Kurt vs ' + angres_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Kurtosis xLOS
	slice_plotter(angres_array, kurt_x_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'Kurt_AngRes_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Kurtosis', title =\
	  'Kurt vs ' + angres_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Create the third figure in this series, which will be a plot of the 
	# structure function slope minus 1 as a function of the angular resolution,
	# for different noise levels, for this simulation. This is done for lines of
	# sight along the z and x axes.

	# m zLOS
	slice_plotter(angres_array, m_z_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'m_AngRes_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'SF Slope - 1', title =\
	  'm vs ' + angres_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# m xLOS
	slice_plotter(angres_array, m_x_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'m_AngRes_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'SF Slope - 1', title =\
	  'm vs ' + angres_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Create the fourth figure in this series, which will be a plot of the 
	# residuals of the linear fit to the structure function as a function of the
	# angular resolution, for different noise levels, for this simulation. This 
	# is done for lines of sight along the z and x axes.

	# Residuals zLOS
	slice_plotter(angres_array, residual_z_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'resid_AngRes_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Residuals', title =\
	  'Residuals vs ' + angres_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Residuals xLOS
	slice_plotter(angres_array, residual_x_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'resid_AngRes_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Residuals', title =\
	  'Residuals vs ' + angres_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Create the fifth figure in this series, which will be a plot of the 
	# integrated quadrupole ratio modulus as a function of the angular 
	# resolution, for different noise levels, for this simulation. This is done 
	# for lines of sight along the z and x axes.

	# Integrated quadrupole zLOS
	slice_plotter(angres_array, int_quad_z_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'int_quad_AngRes_{}_gam{}_z.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Int Quad', title =\
	  'Int Quad vs ' + angres_title_string + ' Gam{} {} z'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# Integrated quadrupole xLOS
	slice_plotter(angres_array, int_quad_x_arr[k], noise_array, slice_indices,\
	 filename=save_loc+'int_quad_AngRes_{}_gam{}_x.png'.format(short_simul[k],gamma)\
	 , format = 'png', xlabel = angres_xlabel, ylabel = 'Int Quad', title =\
	  'Int Quad vs ' + angres_title_string + ' Gam{} {} x'.format(gamma,\
	 short_simul[k]), legend = 'Noise={0:.2f}', row_column = 'column')

	# After this point, all of the plots against angular resolution have been 
	# made for this simulation, so print out a line informing the user of this
	print 'All plots against angular resolution saved for the {} simulation'.\
	format(short_simul[k])


# When the code reaches this point, the plots have been made for all simulations
# Now we want to make the composite plots, where statistics are plotted against
# a Mach number for different values of noise and angular resolution

#---------------------------- Sonic Mach Number --------------------------------

# Skewness zLOS
sub_plotter(sonic_mach_arr, skew_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'skew_sonic_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Skewness', title = 'Skewness vs Sonic Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Skewness xLOS
sub_plotter(sonic_mach_arr, skew_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'skew_sonic_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Skewness', title = 'Skewness vs Sonic Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Kurtosis zLOS
sub_plotter(sonic_mach_arr, kurt_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'kurt_sonic_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Kurtosis', title = 'Kurtosis vs Sonic Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Kurtosis xLOS
sub_plotter(sonic_mach_arr, kurt_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'kurt_sonic_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Kurtosis', title = 'Kurtosis vs Sonic Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# m zLOS
sub_plotter(sonic_mach_arr, m_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'m_sonic_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'SF Slope - 1', title = 'SF Slope - 1 vs Sonic Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# m xLOS
sub_plotter(sonic_mach_arr, m_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'m_sonic_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'SF Slope - 1', title = 'SF Slope - 1 vs Sonic Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Residuals zLOS
sub_plotter(sonic_mach_arr, residual_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'resid_sonic_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Residuals', title = 'Residuals vs Sonic Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Residuals xLOS
sub_plotter(sonic_mach_arr, residual_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'resid_sonic_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Residuals', title = 'Residuals vs Sonic Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Integrated quadrupole zLOS
sub_plotter(sonic_mach_arr, int_quad_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'int_quad_sonic_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Int Quad', title = 'Int Quad vs Sonic Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Integrated quadrupole xLOS
sub_plotter(sonic_mach_arr, int_quad_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'int_quad_sonic_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Sonic Mach Number', ylabel =\
 'Int Quad', title = 'Int Quad vs Sonic Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# After this point, all of the plots against sonic Mach number have been 
# made, so print out a line informing the user of this
print 'All plots against sonic Mach number saved'

#--------------------------- Alfvenic Mach Number ------------------------------

# Skewness zLOS
sub_plotter(alf_mach_arr, skew_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'skew_alf_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Skewness', title = 'Skewness vs Alf Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Skewness xLOS
sub_plotter(alf_mach_arr, skew_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'skew_alf_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Skewness', title = 'Skewness vs Alf Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Kurtosis zLOS
sub_plotter(alf_mach_arr, kurt_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'kurt_alf_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Kurtosis', title = 'Kurtosis vs Alf Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Kurtosis xLOS
sub_plotter(alf_mach_arr, kurt_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'kurt_alf_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Kurtosis', title = 'Kurtosis vs Alf Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# m zLOS
sub_plotter(alf_mach_arr, m_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'m_alf_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'SF Slope - 1', title = 'SF Slope - 1 vs Alf Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# m xLOS
sub_plotter(alf_mach_arr, m_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'m_alf_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'SF Slope - 1', title = 'SF Slope - 1 vs Alf Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Residuals zLOS
sub_plotter(alf_mach_arr, residual_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'resid_alf_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Residuals', title = 'Residuals vs Alf Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Residuals xLOS
sub_plotter(alf_mach_arr, residual_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'resid_alf_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Residuals', title = 'Residuals vs Alf Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Integrated quadrupole zLOS
sub_plotter(alf_mach_arr, int_quad_z_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'int_quad_alf_AngRes_Noise_gam{}_z.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Int Quad', title = 'Int Quad vs Alf Mach Number Gam{} z'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# Integrated quadrupole xLOS
sub_plotter(alf_mach_arr, int_quad_x_arr, noise_array, angres_array,\
 slice_indices, filename = save_loc+'int_quad_alf_AngRes_Noise_gam{}_x.png'.\
 format(gamma), format = 'png', xlabel = 'Alfvenic Mach Number', ylabel =\
 'Int Quad', title = 'Int Quad vs Alf Mach Number Gam{} x'.format(gamma),\
 x_slice_label = 'Noise = {0:.2f}', y_slice_label = 'Ang = {0:.2f}')

# After this point, all of the plots against Alfvenic Mach number have been 
# made, so print out a line informing the user of this
print 'All plots against Alfvenic Mach number saved'