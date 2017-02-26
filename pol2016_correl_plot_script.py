#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code is a script that produces correlation plots of           #
# polarisation diagnostics against other quantities for various simulations,  #
# lines of sight, and wavelengths, to see if the correlations that appear in  #
# the produced images are in fact there. The script starts with a function    #
# that is told what needs to be plotted, extracts the required data, and      #
# creates the required plot. The remainder of this script is mostly code that #
# runs this function for different situations.                                #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 25/1/2017                                                       #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, and astropy.io
# for fits manipulation
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Define the function correl_plot, which will extract all of the required 
# data to be used in the plots, and then create and save the produced image.
def correl_plot(sim1, sim2, sim3, sim4, x_quant, y_quant, x_title = '',\
	y_title = '', emis_mech_x = 'backlit', emis_mech_y = 'internal', slice1= 0,\
	cutoff = 0.02, save_name = '', multiplier = None,\
	emis_mech_mult = 'internal'):
	'''
	Description
		This function extracts data for the specified simulations, and produces
		a 4 row by 3 column figure, where each panel shows a correlation plot
		of the specified y axis quantity against the specified x axis quantity.
		It is possible to have plots for two wavelengths in each figure, by 
		specifying the comp_wave keywords. The produced figure is saved using
		the specified filename.
	
	Required Input
		sim1 - A string that specifies the name of the simulation to use in the 
			   first row of the produced plot.
		sim2 - A string that specifies the name of the simulation to use in the 
			   second row of the produced plot.
		sim3 - A string that specifies the name of the simulation to use in the 
			   third row of the produced plot.
		sim4 - A string that specifies the name of the simulation to use in the 
			   fourth row of the produced plot.
		x_quant - A string that specifies the quantity to be plotted on the 
				  x axis of the produced plot. Does not need the .fits extension
		y_quant - A string that specifies the quantity to be plotted on the 
				  y axis of the produced plot. Does not need the .fits extension
		x_title - A string that specifies the title to use on the x axis of the
				  plot.
		y_title - A string that specifies the title to use on the y axis of the
				  plot.
		emis_mech_x - This can be either 'backlit' or 'internal', to specify 
					whether the quantity on the x axis is calculated for the
					case of backlit or internal emission.
		emis_mech_y - This can be either 'backlit' or 'internal', to specify 
					whether the quantity on the y axis is calculated for the
					case of backlit or internal emission.
		slice1 - An integer specifying the slice of the data cube(s) to use for
				 the first data set. Ignored if using backlit emission. 
		cutoff - A decimal between 0 and 0.5, that represents what percent of
				 the top and bottom values should be removed before calculating
				 the colour map.
		save_name - A string that specifies the filename to use when saving the 
					figure.
		multiplier - A string that specifies the quantity that multiplies the 
					 quantity on the x axis before plotting occurs.
		emis_mech_mult - This can be either 'backlit' or 'internal', to specify 
					whether the quantity multiplying the x axis is calculated 
					for the case of backlit or internal emission.
	
	Output
		A figure with 4 rows and 3 columns is saved using the given filename. 
	'''

	# Create a string for the directory that contains the simulated maps to use
	simul_loc = '/Volumes/CAH_ExtHD/Pol_2016/'

	# Create a string for the directory to save plots into
	save_loc = '/Users/chrisherron/Documents/PhD/Pol_2016/Correlation_Plots/'

	# Specify the dpi to use for the saved figures
	save_dpi = 200

	# Specify the format to use for the figures
	save_format = 'png'

	# Create a dictionary, where each entry specifies the name of the simulation
	# to use
	sim_dict = {'sim1':sim1, 'sim2':sim2, 'sim3':sim3, 'sim4':sim4}

	# Create a list, that specifies the different lines of sight
	los_list = ['x_los', 'y_los', 'z_los']

	# Create the figure on which the plots will be made
	fig = plt.figure(1, figsize = (9,11), dpi = save_dpi)

	# Loop over the different rows of the figure
	for i in range(len(sim_dict)):

		# Loop over the different columns of the figure
		for j in range(len(los_list)):
			# Create a string that specifies the directory in which the data
			# is located
			data_loc = simul_loc + sim_dict['sim{}'.format(i+1)] + '/' +\
			 los_list[j] + '/'

			# Create an axis instance for this plot
			ax = fig.add_subplot(len(sim_dict), len(los_list),\
			 i * len(los_list) + j + 1)

			# Open the FITS file containing the data to use on the y axis
			y_fits = fits.open(data_loc + y_quant + '.fits')

			# Open the FITS file containing the data to use on the x axis
			x_fits = fits.open(data_loc + x_quant + '.fits')

			# Extract the data to use on the y axis
			y_data_cube = y_fits[0].data

			# Extract the data to use on the x axis
			x_data_cube = x_fits[0].data

			# Check to see if we need to multiply the values on the x axis
			# by another quantity
			if multiplier != None:
				# Open the FITS file containing the data to use on the x axis
				mult_fits = fits.open(data_loc + multiplier + '.fits')

				# Extract the data to use for the multiplier
				mult_data_cube = mult_fits[0].data

			# Use the given slice number to extract an image from the cube, if
			# we are dealing with the case of internal emission for the 
			# data on the x axis
			if emis_mech_x == 'internal':
				# Extract data for the specified slice of the x data, and
				# flatten the data so that it can be used in a scatter plot
				x_data = x_data_cube[slice1].flatten()
			else:
				# In this case we have backlit emission, and we do not need
				# to extract a slice before flattening the array
				x_data = x_data_cube.flatten()

			# Use the given slice number to extract an image from the cube, if
			# we are dealing with the case of internal emission for the 
			# data on the y axis
			if emis_mech_y == 'internal':
				# Extract data for the specified slice of the y data, and
				# flatten the data so that it can be used in a scatter plot
				y_data = y_data_cube[slice1].flatten()
			else:
				# In this case we have backlit emission, and we do not need
				# to extract a slice before flattening the array
				y_data = y_data_cube.flatten()

			# Check to see if we need to multiply the x axis data by
			# other data
			if multiplier != None:
				if emis_mech_mult == 'internal':
						# Extract data for the specified slice of the multiplier
						# data, and flatten the data so that it can be used in a 
						# scatter plot
						mult_data = mult_data_cube[slice1].flatten()
				else:
						# Extract data for the specified slice of the multiplier
						# data, and flatten the data so that it can be used in a 
						# scatter plot
						mult_data = mult_data_cube.flatten()

			# Check to see if we need to multiply the x axis data by other data
			if multiplier != None:
				# Multiply the x axis data by the other data
				x_data = x_data * mult_data

			# Sort the array of x data values
			sort_x = np.sort(x_data)

			# Find the x data value that corresponds to a certain percentile
			xmax = sort_x[-int(cutoff * len(x_data))]
			xmin = sort_x[int(cutoff * len(x_data))]

			# Sort the array of y data values
			sort_y = np.sort(y_data)

			# Find the y data value that corresponds to a certain percentile
			ymax = sort_y[-int(cutoff * len(y_data))]
			ymin = sort_y[int(cutoff * len(y_data))]

			# Use a 2D histogram to prepare a colour map, that will show how
			# many pixels are in each area of the scatter plot
			heatmap, xedges, yedges = np.histogram2d(x_data, y_data,\
			range = [[xmin,xmax],[ymin,ymax]], bins = 50)

			# Define the extent of the colour map to produce
			extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

			# Create a colour map showing where the most pixels lie on the 
			# scatter plot
			plt.imshow(heatmap.T, cmap = 'viridis', extent=extent,\
			 origin='lower', aspect = 'auto')

			# Set the font size of the tick labels on the axes
			ax.tick_params(axis='both', labelsize = 6)

			# If we are in the top row, add a title to the plot, labelling
			# the line of sight
			if i == 0:
				# Add a title to this axis
				ax.set_title(los_list[j], fontsize = 14)

			# If we are in the left column, add a y axis label to the plot, 
			# specifying the simulation being used
			if j == 0:
				# Add a label to the y axis of this plot
				ax.set_ylabel(sim_dict['sim{}'.format(i+1)], fontsize = 14)

			# Close the FITS files that were being used to make the plot
			x_fits.close()
			y_fits.close()

			# Check to see if we need to multiply the x axis data by other data
			if multiplier != None:
				# Close the FITS file for the multiplier data
				mult_fits.close()

			# The code should now loop around to produce the next plot

	# Add a y axis title to the entire figure
	plt.figtext(0.03, 0.5, y_title, ha = 'left', va = 'center', fontsize = 20,\
	 rotation = 'vertical')

	# Add an x axis title to the entire figure
	plt.figtext(0.5, 0.05, x_title, ha = 'center', va = 'bottom', fontsize = 20)
	
	# Save the figure using the given filename and format
	plt.savefig(save_loc + save_name + '.' + save_format, dpi = save_dpi,\
	 format = save_format)

	# Print a line of text saying that the figure was saved successfully
	print 'Figure saved as {}'.format(save_name)

	# Clear the current figure, now that it has been saved
	plt.clf()

	# Now that the function has finished, return
	return 0


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
# c512b.5p.0049
# c512b.5p.0077
# c512b.5p.01
# c512b.5p.025
# c512b.5p.05
# c512b.5p.1
# c512b.5p.7
# c512b.5p2
# c512b1p.0049
# c512b1p.0077
# c512b1p.025
# c512b1p.05
# c512b1p.7

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study
# simul_arr = ['b1p2/']
# simul_arr = ['b.1p.0049', 'b.1p.0077', 'b.1p.01','b.1p.025', 'b.1p.05',\
# 'b.1p.1', 'b.1p.7', 'b.1p2', 'b1p.0049',\
# 'b1p.0077', 'b1p.01', 'b1p.025', 'b1p.05', 'b1p.1', 'b1p.7', 'b1p2']
# 'b.5p.0049/', 'b.5p.0077/', 'b.5p.01/',\
# 'b.5p.025/', 'b.5p.05/', 'b.5p.1/', 'b.5p.7/', 'b.5p2/',

# Create a list of polarisation diagnostics, for reference
#['Inten', 'Angle', 'RM', 'Grad', 'Direc_Amp_Max', 'Direc_Amp_Min',\
#'Direc_Max_Ang', 'Rad_Direc_Amp', 'Tang_Direc_Amp', 'Rad_Direc_Ang',\
#'Tang_Direc_Ang', 'Wav_Grad', 'Rad_Wav_Grad', 'Tang_Wav_Grad', 'Direc_Curv',\
#'Quad_Curv', 'Curv4MaxDirecDeriv', 'Wav_Curv', 'Mix_Deriv_Max', 'Mix_Max_Ang']

#------------------------- Polarisation Gradient -------------------------------

# Make a scatter plot of the polarisation gradient against the gradient of the 
# Faraday depth, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepthGrad',\
 'PolarGrad_internal', x_title = 'Gradient Faraday Depth', y_title =\
 'Polarisation Gradient', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'PolarGrad_vs_FaradayDepthGrad_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepthGrad',\
 'PolarGrad_internal', x_title = 'Gradient Faraday Depth', y_title =\
 'Polarisation Gradient', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'PolarGrad_vs_FaradayDepthGrad_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the polarisation gradient against the polarisation 
# intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'PolarGrad_internal', x_title = 'Polarisation Intensity', y_title =\
 'Polarisation Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'PolarGrad_vs_PolarInten_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'PolarGrad_internal', x_title = 'Polarisation Intensity', y_title =\
 'Polarisation Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'PolarGrad_vs_PolarInten_long',
 multiplier = None, emis_mech_mult = 'internal')

#------------------ Minimum Amplitude Directional Derivative -------------------

# Make a scatter plot of the minimum amplitude of the directional derivative 
# against the polarisation intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'DirecAmpMin_internal', x_title = 'Polarisation Intensity', y_title =\
 'Min Amp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'DirecAmpMin_vs_PolarInten_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'DirecAmpMin_internal', x_title = 'Polarisation Intensity', y_title =\
 'Min Amp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'DirecAmpMin_vs_PolarInten_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the minimum amplitude of the directional derivative 
# against the difference between the gradient of the perpendicular component
# of the magnetic field and the gradient of the Faraday depth, for internal
# emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'cross_prod_gradFara_gradBPerp',\
 'DirecAmpMin_internal', x_title = 'Cross Prod Gradients', y_title =\
 'Min Amp Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.04, save_name = 'DirecAmpMin_vs_CrossProdGrads_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'cross_prod_gradFara_gradBPerp',\
 'DirecAmpMin_internal', x_title = 'Cross Prod Gradients', y_title =\
 'Min Amp Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.04, save_name = 'DirecAmpMin_vs_CrossProdGrads_long',
 multiplier = None, emis_mech_mult = 'internal')

#--------------- Angle that Maximises the Directional Derivative ---------------

# Make a scatter plot of the angle that maximises the directional derivative 
# against the angle of the gradient of the Faraday depth, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepthGradAng',\
 'DirecMaxAng_internal', x_title = 'Angle Gradient Faraday Depth', y_title =\
 'Ang Max Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'DirecMaxAng_vs_FaradayDepthGradAng_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepthGradAng',\
 'DirecMaxAng_internal', x_title = 'Angle Gradient Faraday Depth', y_title =\
 'Ang Max Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'DirecMaxAng_vs_FaradayDepthGradAng_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the angle that maximises the directional derivative 
# against the angle of the gradient of the perpendicular component of the 
# magnetic field, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_grad_ang',\
 'DirecMaxAng_internal', x_title = 'Ang Grad B Perp', y_title =\
 'Ang Max Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'DirecMaxAng_vs_BPerpGradAng_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_grad_ang',\
 'DirecMaxAng_internal', x_title = 'Ang Grad B Perp', y_title =\
 'Ang Max Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'DirecMaxAng_vs_BPerpGradAng_long',
 multiplier = None, emis_mech_mult = 'internal')

#----- Maximum Amplitude of the Radial Component of Directional Derivative -----

# Make a scatter plot of the maximum amplitude of the radial component of the
# directional derivative against the gradient of polarisation intensity, for
# internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenGrad',\
 'RadDirecAmp_internal', x_title = 'Gradient Polarisation Intensity', y_title =\
 'Max Amp Rad Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.03, save_name = 'RadDirecAmp_vs_PolarIntenGrad_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenGrad',\
 'RadDirecAmp_internal', x_title = 'Gradient Polarisation Intensity', y_title =\
 'Max Amp Rad Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.03, save_name = 'RadDirecAmp_vs_PolarIntenGrad_long',
 multiplier = None, emis_mech_mult = 'internal')

#--- Maximum Amplitude of the Tangential Component of Directional Derivative ---

# Make a scatter plot of the maximum amplitude of the tangential component of 
# the directional derivative against the gradient of the polarisation angle
# multiplied by the polarisation intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarAngleGrad_internal',\
 'TangDirecAmp_internal', x_title = 'Grad Polar Angle x Pol Inten', y_title =\
 'Max Amp Tang Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.03, save_name = 'TangDirecAmp_vs_PolarAngGrad_short',
 multiplier = 'PolarInten_internal', emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarAngleGrad_internal',\
 'TangDirecAmp_internal', x_title = 'Grad Polar Angle x Pol Inten', y_title =\
 'Max Amp Tang Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.03, save_name = 'TangDirecAmp_vs_PolarAngGrad_long',
 multiplier = 'PolarInten_internal', emis_mech_mult = 'internal')

#---------- Angle Maximises Radial Component of Directional Derivative ---------

# Make a scatter plot of the angle that maximises the amplitude of the radial 
# component of the directional derivative against the angle of the gradient of
# the polarisation intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenGradAng',\
 'RadDirecAng_internal', x_title = 'Angle of Gradient Polarisation Intensity', y_title =\
 'Ang Max Rad Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'RadDirecAng_vs_PolarIntenGradAng_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenGradAng',\
 'RadDirecAng_internal', x_title = 'Angle of Gradient Polarisation Intensity', y_title =\
 'Ang Max Rad Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'RadDirecAng_vs_PolarIntenGradAng_long',
 multiplier = None, emis_mech_mult = 'internal')

#-------- Angle Maximises Tangential Component of Directional Derivative -------

# Make a scatter plot of the angle that maximises the amplitude of the 
# tangential component of the directional derivative against the angle of the 
# gradient of the polarisation angle, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarAngleGradAng_internal',\
 'TangDirecAng_internal', x_title = 'Angle of Grad Polar Angle', y_title =\
 'Ang Max Tang Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'TangDirecAng_vs_PolarAngGradAng_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarAngleGradAng_internal',\
 'TangDirecAng_internal', x_title = 'Angle of Grad Polar Angle', y_title =\
 'Ang Max Tang Comp Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'TangDirecAng_vs_PolarAngGradAng_long',
 multiplier = None, emis_mech_mult = 'internal')

#--------------------- Amplitude of Wavelength Gradient ------------------------

# Make a scatter plot of the wavelength gradient against the Faraday depth, for
# internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepth',\
 'WavGrad_internal', x_title = 'Faraday Depth', y_title =\
 'Wavelength Gradient', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'WavGrad_vs_FaradayDepth_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepth',\
 'WavGrad_internal', x_title = 'Faraday Depth', y_title =\
 'Wavelength Gradient', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'WavGrad_vs_FaradayDepth_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the wavelength gradient against the polarisation 
# intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'WavGrad_internal', x_title = 'Polarisation Intensity', y_title =\
 'Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'WavGrad_vs_PolarInten_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'WavGrad_internal', x_title = 'Polarisation Intensity', y_title =\
 'Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'WavGrad_vs_PolarInten_long',
 multiplier = None, emis_mech_mult = 'internal')

#----------------- Radial Component of Wavelength Gradient ---------------------

# Make a scatter plot of the radial component of the wavelength gradient against
# the wavelength derivative of polarisation intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenWavGrad',\
 'RadWavGrad_internal', x_title = 'Polar Inten Wav Grad', y_title =\
 'Rad Comp Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.03, save_name = 'RadWavGrad_vs_PolarIntenWavGrad_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenWavGrad',\
 'RadWavGrad_internal', x_title = 'Polar Inten Wav Grad', y_title =\
 'Rad Comp Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.03, save_name = 'RadWavGrad_vs_PolarIntenWavGrad_long',
 multiplier = None, emis_mech_mult = 'internal')

#--------------- Tangential Component of Wavelength Gradient -------------------

# Make a scatter plot of the tangential component of the wavelength gradient 
# against the rotation measure, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'RotMeas_internal',\
 'TangWavGrad_internal', x_title = 'Rotation Measure', y_title =\
 'Tang Comp Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.04, save_name = 'TangWavGrad_vs_RotMeas_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'RotMeas_internal',\
 'TangWavGrad_internal', x_title = 'Rotation Measure', y_title =\
 'Tang Comp Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.04, save_name = 'TangWavGrad_vs_RotMeas_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the tangential component of the wavelength gradient 
# against the Faraday depth, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepth',\
 'TangWavGrad_internal', x_title = 'Faraday Depth', y_title =\
 'Tang Comp Wavelength Gradient', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'TangWavGrad_vs_FaradayDepth_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepth',\
 'TangWavGrad_internal', x_title = 'Faraday Depth', y_title =\
 'Tang Comp Wavelength Gradient', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'TangWavGrad_vs_FaradayDepth_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the tangential component of the wavelength gradient 
# against the polarisation intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'TangWavGrad_internal', x_title = 'Polarisation Intensity', y_title =\
 'Tang Comp Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'TangWavGrad_vs_PolarInten_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'TangWavGrad_internal', x_title = 'Polarisation Intensity', y_title =\
 'Tang Comp Wavelength Gradient', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'TangWavGrad_vs_PolarInten_long',
 multiplier = None, emis_mech_mult = 'internal')

#-------- Curvature in Direction that Maximises Directional Derivative ---------

# Make a scatter plot of the curvature in the direction that maximises the
# directional derivative against the polarisation intensity, for internal 
# emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'Curv4MaxDirecDeriv_internal', x_title = 'Polarisation Intensity', y_title =\
 'Curvature for Max Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'Curv4MaxDirecDeriv_vs_PolarInten_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarInten_internal',\
 'Curv4MaxDirecDeriv_internal', x_title = 'Polarisation Intensity', y_title =\
 'Curvature for Max Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'Curv4MaxDirecDeriv_vs_PolarInten_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the curvature in the direction that maximises the
# directional derivative against the perpendicular component of the magnetic
# field, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_projected',\
 'Curv4MaxDirecDeriv_internal', x_title = 'Perp B Field', y_title =\
 'Curvature for Max Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'Curv4MaxDirecDeriv_vs_BPerp_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_projected',\
 'Curv4MaxDirecDeriv_internal', x_title = 'Perp B Field', y_title =\
 'Curvature for Max Direc Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'Curv4MaxDirecDeriv_vs_BPerp_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the curvature in the direction that maximises the
# directional derivative against the maximum amplitude of the directional
# derivative, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'DirecAmpMax_internal',\
 'Curv4MaxDirecDeriv_internal', x_title = 'Max Amp Direc Deriv', y_title =\
 'Curvature for Max Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 0, cutoff = 0.02, save_name = 'Curv4MaxDirecDeriv_vs_DirecAmpMax_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'DirecAmpMax_internal',\
 'Curv4MaxDirecDeriv_internal', x_title = 'Max Amp Direc Deriv', y_title =\
 'Curvature for Max Direc Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 49, cutoff = 0.02, save_name = 'Curv4MaxDirecDeriv_vs_DirecAmpMax_long',
 multiplier = None, emis_mech_mult = 'internal')

#-------------------------- Wavelength Curvature -------------------------------

# Make a scatter plot of the wavelength curvature against the wavelength
# derivative of polarisation intensity, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenWavGrad',\
 'WavCurv_internal', x_title = 'Polar Inten Wav Grad', y_title =\
 'Wavelength Curvature', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'WavCurv_vs_PolarIntenWavGrad_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'PolarIntenWavGrad',\
 'WavCurv_internal', x_title = 'Polar Inten Wav Grad', y_title =\
 'Wavelength Curvature', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'WavCurv_vs_PolarIntenWavGrad_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the wavelength curvature against the wavelength
# derivative of polarisation angle, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'RotMeas_internal',\
 'WavCurv_internal', x_title = 'Rotation Measure', y_title =\
 'Wavelength Curvature', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.03, save_name = 'WavCurv_vs_RotMeas_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'RotMeas_internal',\
 'WavCurv_internal', x_title = 'Rotation Measure', y_title =\
 'Wavelength Curvature', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.03, save_name = 'WavCurv_vs_RotMeas_long',
 multiplier = None, emis_mech_mult = 'internal')

#------------------------- Maximum Mixed Derivative ----------------------------

# Make a scatter plot of the maximum amplitude of the mixed derivative against
# the product of the maximum amplitude of the directional derivative with
# the rotation measure, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'DirecAmpMax_internal',\
 'MixDerivMax_internal', x_title = 'Max Direc Deriv x RM', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.03, save_name = 'MixDerivMax_vs_MaxDirecRotMeas_short',
 multiplier = 'RotMeas_internal', emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'DirecAmpMax_internal',\
 'MixDerivMax_internal', x_title = 'Max Direc Deriv x RM', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'internal', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.03, save_name = 'MixDerivMax_vs_MaxDirecRotMeas_long',
 multiplier = 'RotMeas_internal', emis_mech_mult = 'internal')

# Make a scatter plot of the maximum amplitude of the mixed derivative against
# the Faraday depth, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepth',\
 'MixDerivMax_internal', x_title = 'Faraday Depth', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'MixDerivMax_vs_FaradayDepth_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepth',\
 'MixDerivMax_internal', x_title = 'Faraday Depth', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'MixDerivMax_vs_FaradayDepth_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the maximum amplitude of the mixed derivative against
# the density, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'dens_projected',\
 'MixDerivMax_internal', x_title = 'Projected Density', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'MixDerivMax_vs_Density_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'dens_projected',\
 'MixDerivMax_internal', x_title = 'Projected Density', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'MixDerivMax_vs_Density_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the maximum amplitude of the mixed derivative against
# the standard deviation of the Faraday depth, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'StDevFaradayDepth',\
 'MixDerivMax_internal', x_title = 'StDev Faraday Depth', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'MixDerivMax_vs_StDevFaradayDepth_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'StDevFaradayDepth',\
 'MixDerivMax_internal', x_title = 'StDev Faraday Depth', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'MixDerivMax_vs_StDevFaradayDepth_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the maximum amplitude of the mixed derivative against
# the parallel component of the magnetic field, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_para_projected',\
 'MixDerivMax_internal', x_title = 'Para B Field', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'MixDerivMax_vs_BPara_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_para_projected',\
 'MixDerivMax_internal', x_title = 'Para B Field', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'MixDerivMax_vs_BPara_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the maximum amplitude of the mixed derivative against
# the perpendicular component of the magnetic field, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_projected',\
 'MixDerivMax_internal', x_title = 'Perp B Field', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'MixDerivMax_vs_BPerp_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_projected',\
 'MixDerivMax_internal', x_title = 'Perp B Field', y_title =\
 'Max Amp Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'MixDerivMax_vs_BPerp_long',
 multiplier = None, emis_mech_mult = 'internal')

#--------------------- Angle Maximises Mixed Derivative ------------------------

# Make a scatter plot of the angle that maximises the amplitude of the mixed 
# derivative against the angle of the gradient of the perpendicular 
# component of the magnetic field, for internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_grad_ang',\
 'MixMaxAng_internal', x_title = 'Angle of Grad of Perp B Field', y_title =\
 'Ang Max Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'MixMaxAng_vs_BPerpGradAng_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'B_perp_grad_ang',\
 'MixMaxAng_internal', x_title = 'Angle of Grad of Perp B Field', y_title =\
 'Ang Max Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'MixMaxAng_vs_BPerpGradAng_long',
 multiplier = None, emis_mech_mult = 'internal')

# Make a scatter plot of the angle that maximises the amplitude of the mixed 
# derivative against the angle of the gradient of the Faraday depth, for
# internal emission
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepthGradAng',\
 'MixMaxAng_internal', x_title = 'Angle of Grad of Faraday Depth', y_title =\
 'Ang Max Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 2, cutoff = 0.02, save_name = 'MixMaxAng_vs_FaradayDepthGradAng_short',
 multiplier = None, emis_mech_mult = 'internal')
correl_plot('b1p2', 'b.1p2', 'b1p.05', 'b.1p.05', 'FaradayDepthGradAng',\
 'MixMaxAng_internal', x_title = 'Angle of Grad of Faraday Depth', y_title =\
 'Ang Max Mixed Deriv', emis_mech_x = 'backlit', emis_mech_y = 'internal',\
 slice1 = 47, cutoff = 0.02, save_name = 'MixMaxAng_vs_FaradayDepthGradAng_long',
 multiplier = None, emis_mech_mult = 'internal')