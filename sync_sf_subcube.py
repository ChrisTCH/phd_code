#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated magnetic      #
# field strengths, and calculates the observed synchrotron emission maps for a #
# cube that is saturated with a uniform, isotropic distribution of cosmic rays #
# with power spectrum index gamma. These maps are generated for sub-cubes of   #
# full simulation cube, and then structure functions are calculated for each   #
# map. The slopes of these structure functions are plotted against the size of #
# the sub-cube, and for each simulation a plot of the structure function for   #
# different sub-cubes is produced.                                             #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 21/4/2015                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.ndimage to handle rotation of data cubes.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage

# Import the function that calculates the structure function and the function 
# that calculates the radially averaged structure function.
from sf_fft import sf_fft
from sfr import sfr

# Set a variable to hold the number of bins to use in calculating the 
# structure functions
num_bins = 20

# Create a string for the directory that contains the simulated synchrotron
# intensity maps to use. 
simul_loc = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

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

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can be 'x', 'y', or 'z'
line_o_sight = 'z'

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create a variable that stores the index corresponding to the value of gamma to
# use in the calculations
gam_index = 2

# Create a variable that just holds the value of gamma being used
gamma = gamma_arr[gam_index]

# Create a variable that controls how many sub-cube sizes are being used
free_num = 41

# Create an array that specifies the different sub-cube sizes to use
# NOTE: The values chosen here are based on the known sizes of the simulation
# cubes
iter_array = np.linspace(312, 512, free_num)

# Create an empty array, where each entry specifies the calculated slope of
# the structure function of the synchrotron intensity image minus 1, of the 
# corresponding simulation, for a particular value of the free parameter related
# to the observational effect being studied. Each row corresponds to a value of 
# the free parameter, and each column corresponds to a simulation.
m_arr = np.zeros((len(simul_arr),len(iter_array)))

# Create an empty array, where each entry specifies the calculated intercept of
# the structure function of the synchrotron intensity image, of the 
# corresponding simulation, for a particular value of the free parameter related
# to the observational effect being studied. Each row corresponds to a value of 
# the free parameter, and each column corresponds to a simulation.
intercept_arr = np.zeros((len(simul_arr),len(iter_array)))

# Create an empty array, where each entry specifies the residuals of the linear
# fit to the structure function of the synchrotron intensity image, of the 
# corresponding simulation, for a particular value of the free parameter related
# to the observational effect being studied. Each row corresponds to a value of 
# the free parameter, and each column corresponds to a simulation. 
residual_arr = np.zeros((len(simul_arr),len(iter_array)))

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/'

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for j in range(len(simul_arr)):
	# Create an empty array, where each entry specifies the calculated value
	# of the structure function of the synchrotron intensity image, 
	# of the corresponding simulation, for a particular value of the free 
	# parameter related to the observational effect being studied. Each row 
	# corresponds to a value of the free parameter, and each column 
	# corresponds to a radial value.
	sf_arr = np.zeros((len(iter_array),num_bins))

	# Create an empty array, where each entry specifies the radius values
	# used to calculate the structure function of the synchrotron intensity 
	# image, of the corresponding simulation, for a particular value of the 
	# free parameter related to the observational effect being studied. Each
	# row corresponds to a value of the free parameter, and each column 
	# corresponds to a radial value.
	rad_arr = np.zeros((len(iter_array),num_bins))

	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS file that contains the x-component of the simulated magnetic
	# field
	mag_x_fits = fits.open(data_loc + 'magx.fits')

	# Extract the data for the simulated x-component of the magnetic field
	mag_x_data = mag_x_fits[0].data

	# Open the FITS file that contains the y-component of the simulated magnetic 
	# field
	mag_y_fits = fits.open(data_loc + 'magy.fits')

	# Extract the data for the simulated y-component of the magnetic field
	mag_y_data = mag_y_fits[0].data

	# Open the FITS file that contains the z-component of the simulated magnetic 
	# field
	mag_z_fits = fits.open(data_loc + 'magz.fits')

	# Extract the data for the simulated z-component of the magnetic field
	mag_z_data = mag_z_fits[0].data

	# Depending on the line of sight, the strength of the magnetic field 
	# perpendicular to the line of sight is calculated in different ways
	if line_o_sight == 'z':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and y component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_y_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the z axis, we need to integrate along axis 0. (Numpy convention is 
		# that axes are ordered as (z, y, x))
		int_axis = 0
	elif line_o_sight == 'y':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the x and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_x_data, 2.0) + np.power(mag_z_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the y axis, we need to integrate along axis 1.
		int_axis = 1
	elif line_o_sight == 'x':
		# Calculate the magnitude of the magnetic field perpendicular to the line of
		# sight, which is just the square root of the sum of the y and z component
		# magnitudes squared.
		mag_perp = np.sqrt( np.power(mag_y_data, 2.0) + np.power(mag_z_data, 2.0) )

		# Construct a variable which tells the script which axis we need to 
		# integrate along to calculate the synchrotron maps. Since the line of sight
		# is the x axis, we need to integrate along axis 2.
		int_axis = 2

	# Calculate the result of raising the perpendicular magnetic field strength
	# to the power of gamma, for these slices
	mag_perp_gamma = np.power(mag_perp, gamma)

	# Loop over the sub-cube sizes being studied, so that we can calculate the
	# synchrotron map for each sub-cube size
	for i in range(len(iter_array)):
		# Calculate the minimum index to include when extracting the sub-cube
		ind_min = int(256 - iter_array[i] / 2.0)

		# Calculate the maximum index to exclude when extracting the sub-cube
		ind_max = int(256 + iter_array[i] / 2.0)

		# Extract a sub-cube of the required size from the full cube
		sub_mag_perp_gamma = mag_perp_gamma[ind_min:ind_max,ind_min:ind_max,ind_min:ind_max]

		# Integrate the perpendicular magnetic field strength raised to the power
		# of gamma along the required axis, to calculate the observed synchrotron 
		# map for these slices. This integration is performed by the trapezoidal 
		# rule. To normalise the calculated synchrotron map, divide by the number 
		# of pixels along the integration axis. Note the array is ordered by(z,y,x)!
		# NOTE: Set dx to whatever the pixel spacing is
		sync_arr = np.trapz(sub_mag_perp_gamma, dx = 1.0, axis = int_axis) /\
		 np.shape(sub_mag_perp_gamma)[int_axis]

		# Calculate the structure function (two-dimensional) of the synchrotron
		# intensity map. Note that no_fluct = True is set, because we are not 
		# subtracting the mean from the synchrotron map before calculating the 
		# structure function.
		strfn = sf_fft(sync_arr, no_fluct = True)

		# Radially average the calculated 2D structure function, using the 
		# specified number of bins.
		rad_sf = sfr(strfn, num_bins, verbose = False)

		# Extract the calculated radially averaged structure function
		sf = rad_sf[1]

		# Extract the radius values used to calculate this structure function
		sf_rad_arr = rad_sf[0]

		# Store the values for the radially averaged structure function in the 
		# corresponding array
		sf_arr[i] = sf

		# Store the radius values used to calculate the structure function in
		# the corresponding array
		rad_arr[i] = sf_rad_arr

		# Calculate the spectral index of the structure function calculated for
		# this value of gamma. Note that only the first third of the structure
		# function is used in the calculation, as this is the part that is 
		# close to a straight line. 
		spec_ind_data = np.polyfit(np.log10(\
			sf_rad_arr[11:16]),\
			np.log10(sf[11:16]), 1, full = True)

		# Extract the returned coefficients from the polynomial fit
		coeff = spec_ind_data[0]

		# Extract the sum of the residuals from the polynomial fit
		residual_arr[j,i] = spec_ind_data[1]

		# Enter the value of m, the slope of the structure function minus 1,
		# into the corresponding array
		m_arr[j,i] = coeff[0]-1.0

		# Enter the value of the intercept of the linear fit into an array
		intercept_arr[j,i] = coeff[1]

	# When the code reaches this point, all of the structure functions have been
	# calculated for the different sub-cubes, and their slopes have been 
	# recorded, for this simulation.

	# Plot the structure functions for this simulation, for different sub-cube
	# sizes

	# Create a figure to display a plot of the structure functions for different
	# values of the sub-cube size
	fig1 = plt.figure()

	# Create an axis for this figure
	ax1 = fig1.add_subplot(111)

	# Plot the structure function for various values of the sub-cube size
	plt.plot(rad_arr[0], sf_arr[0],'b-o',label = 'Size ='\
		+'{}'.format(iter_array[0]))
	plt.plot(rad_arr[10], sf_arr[10],'r-o',\
		label= 'Size =' +'{0:.2f}'.format(iter_array[10]))
	plt.plot(rad_arr[20], sf_arr[20],'g-o',\
		label= 'Size =' +'{0:.2f}'.format(iter_array[20]))
	plt.plot(rad_arr[30], sf_arr[30],'k-o',\
		label= 'Size =' +'{0:.2f}'.format(iter_array[30]))
	plt.plot(rad_arr[40], sf_arr[40],'m-o',\
		label= 'Size =' +'{0:.2f}'.format(iter_array[40]))

	# Plot the line of best fit for each structure function, taking into 
	# account that the line of best fit is on a log-log graph
	plt.plot(rad_arr[0], np.power(10.0,intercept_arr[j,0]) * \
		np.power(rad_arr[0], m_arr[j,0] + 1) ,'b--')
	plt.plot(rad_arr[10], np.power(10.0,intercept_arr[j,10]) * \
		np.power(rad_arr[10], m_arr[j,10] + 1),'r--')
	plt.plot(rad_arr[20], np.power(10.0,intercept_arr[j,20]) * \
		np.power(rad_arr[20], m_arr[j,20] + 1),'g--')
	plt.plot(rad_arr[30], np.power(10.0,intercept_arr[j,30]) * \
		np.power(rad_arr[30], m_arr[j,30] + 1),'k--')
	plt.plot(rad_arr[40], np.power(10.0,intercept_arr[j,40]) * \
		np.power(rad_arr[40], m_arr[j,40] + 1),'m--')

	# Set the x-axis of the plot to be logarithmically scaled
	plt.xscale('log')

	# Add a label to the x-axis
	plt.xlabel('Radial Separation [pixels]', fontsize = 20)

	# Set the y-axis of the plot to be logarithmically scaled
	plt.yscale('log')

	# Add a label to the y-axis
	plt.ylabel('Structure Function', fontsize = 20)

	# Add a title to the plot
	plt.title('SF {} Gam{}'.format(short_simul[j], gamma), fontsize = 20)

	# Force the legend to appear on the plot
	plt.legend(loc = 4)

	# Save the figure using the given filename and format
	plt.savefig(save_loc + 'SF_{}_subcube_gam{}.png'.format(short_simul[j]\
		,gamma), format = 'png')

	# Close the figure, now that it has been saved.
	plt.close()

	# Close all of the FITS files, to save memory
	mag_x_fits.close()
	mag_y_fits.close()
	mag_z_fits.close()

# When the code reaches this point, structure functions have been calculated
# for all simulations, and for all sub-cube sizes

# Plot structure function slopes vs sub-cube size for low B

# Create a figure to display a plot of the SF slope - 1 as a function of the
# sub-cube size, for simulations with b = 0.1 
fig2 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the SF slope - 1 as a function of the sub-cube size for simulations
# with b = 0.1
plt.plot(iter_array, m_arr[0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, m_arr[1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, m_arr[2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, m_arr[3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, m_arr[4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, m_arr[5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, m_arr[6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, m_arr[7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Sub-cube Size [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Sub-cube Size b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])

# Force the legend to appear on the plot
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_subcube_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the sub-cube size has been saved
print 'Plot of the SF slope - 1 as a function of sub-cube size saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# SF slope - 1 vs Sub-Cube Size - High Magnetic Field

# Create a figure to display a plot of the SF slope - 1 as a function of the
# sub-cube size, for simulations with b = 1 
fig3 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the SF slope - 1 as a function of the sub-cube size for simulations
# with b = 1
plt.plot(iter_array, m_arr[8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, m_arr[9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, m_arr[10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, m_arr[11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, m_arr[12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, m_arr[13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, m_arr[14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, m_arr[15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Sub-cube Size [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('SF slope - 1', fontsize = 20)

# Add a title to the plot
plt.title('SF Slope vs Sub-cube Size b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box3 = ax3.get_position()
ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])

# Force the legend to appear on the plot
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'm_subcube_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the SF slope - 1 as a 
# function of the sub-cube size has been saved
print 'Plot of the SF slope - 1 as a function of sub-cube saved b=1'

# Close the figure, now that it has been saved.
plt.close()

#---------------------------- Residuals ----------------------------------------

# Plot residuals vs sub-cube size for low B

# Create a figure to display a plot of the residuals as a function of the
# sub-cube size, for simulations with b = 0.1 
fig4 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the residuals as a function of the sub-cube size for simulations
# with b = 0.1
plt.plot(iter_array, residual_arr[0],'b-o',label = '{}'.format(short_simul[0]))
plt.plot(iter_array, residual_arr[1],'b--o',label= '{}'.format(short_simul[1]))
plt.plot(iter_array, residual_arr[2],'r-o',label = '{}'.format(short_simul[2]))
plt.plot(iter_array, residual_arr[3],'r--o',label= '{}'.format(short_simul[3]))
plt.plot(iter_array, residual_arr[4],'g-o',label = '{}'.format(short_simul[4]))
plt.plot(iter_array, residual_arr[5],'g--o',label= '{}'.format(short_simul[5]))
plt.plot(iter_array, residual_arr[6],'c-o',label = '{}'.format(short_simul[6]))
plt.plot(iter_array, residual_arr[7],'c--o',label= '{}'.format(short_simul[7]))

# Add a label to the x-axis
plt.xlabel('Sub-cube Size [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Sub-cube Size b.1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box4 = ax4.get_position()
ax4.set_position([box4.x0, box4.y0, box4.width * 0.8, box4.height])

# Force the legend to appear on the plot
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'resid_subcube_b.1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sub-cube size has been saved
print 'Plot of the residuals as a function of sub-cube size saved b=0.1'

# Close the figure, now that it has been saved.
plt.close()

# Residuals vs Sub-Cube Size - High Magnetic Field

# Create a figure to display a plot of the residuals as a function of the
# sub-cube size, for simulations with b = 1 
fig5 = plt.figure(figsize = (10,6))

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot the residuals as a function of the sub-cube size for simulations
# with b = 1
plt.plot(iter_array, residual_arr[8],'b-o',label = '{}'.format(short_simul[8]))
plt.plot(iter_array, residual_arr[9],'b--o',label= '{}'.format(short_simul[9]))
plt.plot(iter_array, residual_arr[10],'r-o',label = '{}'.format(short_simul[10]))
plt.plot(iter_array, residual_arr[11],'r--o',label= '{}'.format(short_simul[11]))
plt.plot(iter_array, residual_arr[12],'g-o',label = '{}'.format(short_simul[12]))
plt.plot(iter_array, residual_arr[13],'g--o',label= '{}'.format(short_simul[13]))
plt.plot(iter_array, residual_arr[14],'c-o',label = '{}'.format(short_simul[14]))
plt.plot(iter_array, residual_arr[15],'c--o',label= '{}'.format(short_simul[15]))

# Add a label to the x-axis
plt.xlabel('Sub-cube Size [pixels]', fontsize = 20)

# Add a label to the y-axis
plt.ylabel('Residuals', fontsize = 20)

# Add a title to the plot
plt.title('Residuals vs Sub-cube Size b1 Gam{}'.format(gamma), fontsize = 20)

# Shrink the width of the plot axes
box5 = ax5.get_position()
ax5.set_position([box5.x0, box5.y0, box5.width * 0.8, box5.height])

# Force the legend to appear on the plot
ax5.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the figure using the given filename and format
plt.savefig(save_loc + 'resid_subcube_b1_gam{}.png'.format(gamma), format = 'png')

# Print a message to the screen to show that the plot of the residuals as a 
# function of the sub-cube size has been saved
print 'Plot of the residuals as a function of sub-cube saved b=1'

# Close the figure, now that it has been saved.
plt.close()