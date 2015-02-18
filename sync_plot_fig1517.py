#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in an array of synchrotron intensity #
# for one simulation, and produces images of the synchrotron intensity map for #
# different noise levels and angular resolution.                               #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 16/2/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, aplpy for nice
# plots, astropy.io for fits manipulation, astropy.convolution for convolution
# functions,
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import aplpy
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data sets to use in calculations.
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

# Create a string giving the directory of the simulation to use
spec_loc = 'b1p2_Aug_Burk/'

# Create an array that specifies the value of gamma used to produce each 
# synchrotron emissivity cube
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a string that determines what observational effect will be studied
# String can be one of the following:
# noise - Study how statistics change as noise level is varied
# res - Study how statistics change as the spatial resolution is varied
obs_effect = 'noise'

# Create a variable that controls how many data points are being used for the
# free parameter
free_num = 20

# Create a variable that determines how strong the observational effect should
# be in the map that is used in the produced plot. This variable is the index
# of the iter_array to use
iter_index = 10

# Depending on what observational effect is being studied, create an array of 
# values over which we will iterate. This array represents the values of the 
# free parameter related to the observational effect 
if obs_effect == 'noise':
	# Create an array of values that will be used to determine the standard
	# deviation of the Gaussian distribution from which noise values are 
	# generated. The standard deviation will be calculated by multiplying the
	# median synchrotron intensity by the values in this array.
	iter_array = np.linspace(0.02, 0.5, free_num)

	# Create a label for the x-axis of plots that are made against noise
	# standard deviation
	xlabel = 'Noise StandDev [perc median inten]'

	# Create a string to be used in the titles of any plots that are made 
	# against noise standard deviation
	title_string = 'Noise StandDev'

	# Create a string to be used in legends involving spectral channel width
	leg_string = 'Noise = ' 

	# Create a string to be used in sub-plot labels to distinguish between the
	# original synchrotron map and the synchrotron map after an observational
	# effect has been applied
	effect_string = 'After Noise'
elif obs_effect == 'res':
	# Create an array of values that represent the standard deviation of the 
	# Gaussian used to smooth the synchrotron maps. All values are in pixels.
	iter_array = np.linspace(1.0, 50.0, free_num)

	# Create an array of values representing the final angular resolution of
	# the image after smoothing. The final resolution is calculated by 
	# quadrature from the initial resolution (1 pixel) and the standard 
	# deviation of the convolving Gaussian.
	final_res = np.sqrt(1.0 + np.power(iter_array,2.0))

	# Create a label for the x-axis of plots that are made against angular 
	# resolution
	xlabel = 'Angular Resolution [pixels]'

	# Create a string to be used in the titles of any plots that are made 
	# against angular resolution
	title_string = 'Angular Resolution'

	# Create a string to be used in legends involving angular resolution
	leg_string = 'AngRes = ' 

	# Create a string to be used in sub-plot labels to distinguish between the
	# original synchrotron map and the synchrotron map after an observational
	# effect has been applied
	effect_string = 'After Smoothing'

# Create a string for the full directory path to use in the calculation
data_loc = simul_loc + spec_loc

# Open the FITS file that contains the simulated synchrotron intensity
# map for this line of sight
sync_fits = fits.open(data_loc + 'synint_p1-4.fits')

# Extract the data for the simulated synchrotron intensities
sync_data = sync_fits[0].data

# Extract the synchrotron intensity map for the value of gamma
sync_map = sync_data[gam_index]

# Print a message to the screen to show that the data has been loaded
print 'Synchrotron intensities loaded successfully'

# Take into account an observing frequency of 1.4 GHz, by multiplying
# the extracted synchrotron maps by a gamma dependent frequency factor
sync_map = sync_map * np.power(1.4, -(gamma - 1))

# Check to see which observational effect is being studied
if obs_effect == 'noise':
	# In this case, we are taking into account the effect of noise in
	# the telescope. We start with an array of values that, when 
	# multiplied by the median intensity of the synchrotron map, give
	# the standard deviation of the Gaussian noise. 

	# Calculate the standard deviation of the Gaussian noise that will 
	# affect the synchrotron maps.
	noise_stdev = iter_array[iter_index] * np.median(sync_map)

	# Create an array of values that are randomly drawn from a Gaussian
	# distribution with the specified standard deviation. This 
	# represents the noise at each pixel of the image. 
	noise_matrix = np.random.normal(scale = noise_stdev,\
	 size = np.shape(sync_map))

	# Add the noise maps onto the synchrotron intensity maps, to produce
	# the mock 'observed' maps
	sync_map_free_param = sync_map + noise_matrix

elif obs_effect == 'res':
	# In this case, we are taking into account the effect of spatial 
	# resolution. We start with an array of values that specifies the
	# standard deviation of the Gaussian to be used to smooth the data.

	# Create a Gaussian kernel to use to smooth the synchrotron map,
	# using the given standard deviation
	gauss_kernel = Gaussian2DKernel(iter_array[iter_index])

	# Smooth the synchrotron maps to the required resolution by 
	# convolution with the above Gaussian kernel.
	sync_map_free_param = convolve_fft(sync_map, gauss_kernel, boundary = 'wrap')

	# Replace the array of standard deviations with the array of final
	# resolutions, so that the final resolutions are used in all plots
	iter_array[iter_index] = final_res[iter_index]

# Print a message to show that the calculation of the synchrotron map affected
# by observational noise has been completed
print 'Synchrotron map affected by observational effect calculated'

# When the code reaches this point, all required images have been produced,
# so start making plots.

# ------------------------ Images of Sync Intensity ----------------------------

# Here we want to produce one plot with two subplots. The left image should be
# the synchrotron map with nothing done to it, and the right image should be
# the synchrotron map after an observational effect, such as noise or angular
# resolution, has been applied.

# Create a matplotlib figure to hold the APLpy sub-figures that will be made
fig = plt.figure(1, figsize=(9,4), dpi = 300)

# Create a subplot in the figure, for the synchrotron map
ax1 = fig.add_subplot(121)

# Show a grayscale image of the synchrotron intensity map
im1 = plt.imshow(sync_map, cmap = 'gray')

# Add a label to the y-axis
plt.ylabel('Y-axis [pixels]', fontsize = 20)

# Create an extra axes on the right hand side of ax1. The width of this axis is
# 5% of the width of ax1
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes("right", size="5%", pad = 0.05)

# Add a colour bar to the image
plt.colorbar(im1, cax = cax1)

# Create a subplot in the figure, for the synchrotron map affected by an 
# observational effect
ax2 = fig.add_subplot(122)

# Show a grayscale image of this synchrotron intensity map
im2 = plt.imshow(sync_map_free_param, cmap = 'gray')

# Create an extra axes on the right hand side of ax2. The width of this axis is
# 5% of the width of ax2
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes("right", size="5%", pad = 0.05)

# Add a colour bar to the image
plt.colorbar(im2, cax = cax2)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'X-axis [pixels]', ha = 'center', va = 'bottom', fontsize = 20)

# Add some text to the figure, to label the left image as figure a
plt.figtext(0.21, 0.93, 'a) Original', fontsize = 18)

# Add some text to the figure, to label the left image as figure b
plt.figtext(0.61, 0.93, 'b) {}'.format(effect_string), fontsize = 18)

# Depending on the observational effect being studied, change the filename used
# to save the figure
if obs_effect == 'noise':
	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Publication_Plots/fig17.eps', format = 'eps')
elif obs_effect == 'res':
	# Save the figure using the given filename and format
	plt.savefig(simul_loc + 'Publication_Plots/fig15.eps', format = 'eps')

# Print a message saying what observational effect value was used to produce the
# plots
print 'Observational effect value: {}'.format(iter_array[iter_index])

# Close the figure so that it does not stay in memory
plt.close()