#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to open the FITS files containing the    #
# skewness of the polarisation gradient of the CGPS data, for different       #
# evaluation box sizes, and plot the median skewness as a function of         #
# longitude. The plot will be saved in the same directory as the CGPS data.   #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 18/6/2016                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script
import numpy as np
import aplpy
from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt

# Create a string object which stores the directory of the CGPS data
data_loc = '/Users/chrisherron/Documents/PhD/CGPS_2015/'

# Create a string that will be used to determine what quantity will be read 
# through the script and analysed, e.g. the polarisation gradient, or 
# polarisation intensity. FITS files for the chosen quantity will be loaded
# by the script. Could be 'Polar_Grad' or 'Polar_Inten', for example
data_2_load = 'Polar_Grad'

# Create a string that will be used to control what FITS files are used
# to perform calculations, and that will be appended into the filename of 
# anything produced in this script. This is either 'high_lat' or 'plane'
save_append = 'plane_all_mask'

# Set the dpi at which to save the image
save_dpi = 300

# Specify the number of beams that should be across half the width of the
# box used to calculate the statistics around each pixel. This is an array,
# used to specify which skewness maps will be used to make the plot.
num_beams = np.array([20, 40, 60])

# Create an array that specifies the number of beams across the full width
# of the evaluation box
full_width = np.array([40, 80, 120])

# Create a list of strings, that will be used to control what statistics
# maps are used to plot a statistic against longitude.
# Valid statistics include 'mean', 'stdev', skew', 'kurt'
stat = 'skew'
# Full name of the statistic
stat_full = 'skewness'
# Label for y axis
y_label = ' Median Skewness'

# Specify the angular resolution of the polarisation gradient map that is having
# its skewness analysed.
ang_res = 150

# Specify the colour, symbol and linestyle for each plot to be used
symbol_arr = ['b-o', 'r-^', 'g-s']

# Create a list that specifies all of the files for which we want to calculate
# local statistics
input_files = [data_loc + '{}_{}_trunc1_beams{}_{}_sparse/'.format(data_2_load,\
 save_append, beams, stat)+ '{}_{}_smooth2_{}_beams{}_{}_sparse.fits'.\
format(data_2_load, save_append, ang_res, beams, stat_full) for beams in num_beams]

# Create a figure to display a plot of the statistic as a function of longitude
# for each of the beam number values.
fig1 = plt.figure(1, figsize=(9,6), dpi = 300)

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Loop over the different skewness maps that we will analyse
for i in range(len(num_beams)):
	# Print a message to show that calculations are starting for the current
	# input file
	print 'Calculations starting for beams {}'.format(num_beams[i])

	# Open the CGPS FITS file for the current number of beams
	fits_file = fits.open(input_files[i])

	# Obtain the header of the primary HDU for the data
	fits_hdr = fits_file[0].header

	# Extract the data from the FITS file, which is held in the primary HDU
	fits_data = fits_file[0].data

	# Print a message to the screen saying that the data was successfully 
	# extracted
	print 'CGPS data successfully extracted from the FITS file.'

	# Calculate the median of the statistic in each longitude bin
	stat_median = np.median(fits_data, axis = 0)

	# Create a WCS object that will transform between pixel coordinates and 
	# Galactic coordinates
	coord_convert = wcs.WCS(fits_hdr)

	# Create an array that specifies the x and y co-ordinates in pixels for
	# every pixel of the map
	pix_coords_x, pix_coords_y = np.meshgrid(range(fits_hdr['NAXIS1']),\
		range(fits_hdr['NAXIS2']))

	# Obtain the Galactic co-ordinates for every pixel in the map
	Gal_coords_y, Gal_coords_x = coord_convert.wcs_pix2world(pix_coords_x,\
	 pix_coords_y, 0)

	# Reduce the longitude array, so that it has the same length as the statistic
	# array
	longitude_arr = np.mean(Gal_coords_y, axis = 0)

	# Produce a plot of statistic against longitude for this map
	plt.plot(longitude_arr, stat_median, symbol_arr[i], label =\
	 '{} beams'.format(full_width[i]))

# When the for loop has finished, plots have been produced for every map

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 14, numpoints=1)

# Add a label to the y-axis
plt.ylabel(y_label, fontsize = 20)

# Add a label to the x-axis
plt.xlabel('Longitude [degrees]', fontsize = 20)

# Invert the x axis, so that highest values are on the left
ax1.invert_xaxis()

# Save the figure using the given filename and format
plt.savefig(data_loc + '{}_vs_long.eps'.format(stat), dpi = save_dpi, format = 'eps')

# Close the figure
plt.close