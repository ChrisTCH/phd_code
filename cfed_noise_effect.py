#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the mean and standard deviation of synchrotron   #
# intensity maps that are influenced by finite angular resolution and noise.   #
# Each of these quantities is plotted against zeta, to see which quantities    #
# are sensitive tracers of zeta. Quantities are also plotted against the noise #
# level to directly show how observations affect measured statistics.          #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 6/5/2016                                                         #
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

# Set the dpi at which to save the image
save_dpi = 300

# Set a variable that holds the number of timesteps we have for the simulations
num_timestep = 5

# Create a variable that controls whether the moments of the log normalised PDFs
# are calculated
log = True

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create a string for the specific simulated data set to use in calculations.
# The directories end in:
# 512sM5Bs5886_20 (Solenoidal turbulence, timestep 20)
# 512sM5Bs5886_25 (Solenoidal turbulence, timestep 25)
# 512sM5Bs5886_30 (Solenoidal turbulence, timestep 30)
# 512sM5Bs5886_35 (Solenoidal turbulence, timestep 35)
# 512sM5Bs5886_40 (Solenoidal turbulence, timestep 40)
# 512m075M5Bs5887_20 (zeta = 0.75, timestep 20)
# 512m075M5Bs5887_25 (zeta = 0.75, timestep 25)
# 512m075M5Bs5887_30 (zeta = 0.75, timestep 30)
# 512m075M5Bs5887_35 (zeta = 0.75, timestep 35)
# 512m075M5Bs5887_40 (zeta = 0.75, timestep 40)
# 512mM5Bs5887_20 (zeta = 0.5, timestep 20)
# 512mM5Bs5887_25 (zeta = 0.5, timestep 25)
# 512mM5Bs5887_30 (zeta = 0.5, timestep 30)
# 512mM5Bs5887_35 (zeta = 0.5, timestep 35)
# 512mM5Bs5887_40 (zeta = 0.5, timestep 40)
# 512m025M5Bs5887_20 (zeta = 0.25, timestep 20)
# 512m025M5Bs5887_25 (zeta = 0.25, timestep 25)
# 512m025M5Bs5887_30 (zeta = 0.25, timestep 30)
# 512m025M5Bs5887_35 (zeta = 0.25, timestep 35)
# 512m025M5Bs5887_40 (zeta = 0.25, timestep 40)
# 512cM5Bs5886_20 (Compressive turbulence, timestep 20)
# 512cM5Bs5886_25 (Compressive turbulence, timestep 25)
# 512cM5Bs5886_30 (Compressive turbulence, timestep 30)
# 512cM5Bs5886_35 (Compressive turbulence, timestep 35)
# 512cM5Bs5886_40 (Compressive turbulence, timestep 40)
simul_arr = ['512sM5Bs5886_20/', '512sM5Bs5886_25/', '512sM5Bs5886_30/',\
'512sM5Bs5886_35/', '512sM5Bs5886_40/', '512m075M5Bs5887_20/',\
'512m075M5Bs5887_25/', '512m075M5Bs5887_30/', '512m075M5Bs5887_35/',\
'512m075M5Bs5887_40/', '512mM5Bs5887_20/', '512mM5Bs5887_25/',\
'512mM5Bs5887_30/', '512mM5Bs5887_35/', '512mM5Bs5887_40/',\
'512m025M5Bs5887_20/', '512m025M5Bs5887_25/', '512m025M5Bs5887_30/',\
'512m025M5Bs5887_35/', '512m025M5Bs5887_40/', '512cM5Bs5886_20/',\
'512cM5Bs5886_25/', '512cM5Bs5886_30/', '512cM5Bs5886_35/', '512cM5Bs5886_40/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation
sim_labels = [r'$\zeta = 1.0$', r'$\zeta = 0.75$', r'$\zeta = 0.5$',\
 r'$\zeta = 0.25$', r'$\zeta = 0$']

# Create an array that gives the value of zeta for each simulation
zeta = np.array([1.0,0.75,0.5,0.25,0.0])

# Create a variable that holds the number of simulations being used
num_sims = len(sim_labels)

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Create a variable that controls how many data points are being used for the
# free parameter
free_num = 50

# Create a variable that just shows we are looking at the effects of noise
# and angular resolution
obs_effect = 'noise_res'

# Create an array of values that will be used to determine the standard
# deviation of the Gaussian distribution from which noise values are 
# generated. The standard deviation will be calculated by multiplying the
# median synchrotron intensity by the values in this array.
iter_array = np.linspace(0.01, 1.00, free_num)

# Create a label for the x-axis of plots that are made against noise
# standard deviation
xlabel = 'Final Noise StandDev [perc median inten]'

# Create a string to be used in the titles of any plots that are made 
# against noise standard deviation
title_string = 'Noise StandDev'

# Create a string to be used in legends involving spectral channel width
leg_string = 'Noise = ' 

# Create a string that specifies the folder in which all output should
# be saved for this set of tests
save_direc = 'NoiseRes_Study/'

# Create a variable that represents the standard deviation of the 
# Gaussian used to smooth the synchrotron maps. Value is in pixels.
smooth_stdev = 1.3

# Create a variable representing the final angular resolution of
# the image after smoothing. The final resolution is calculated by 
# quadrature from the initial resolution (1 pixel) and the standard 
# deviation of the convolving Gaussian.
final_res = np.sqrt(1.0 + np.power(smooth_stdev,2.0))

# Create a string to be used in legends involving angular resolution
leg_string_ang = 'AngRes = ' 

# Create an empty array, where each entry specifies the calculated mean of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of the free parameter related to the observational effect 
# being studied. The first index iterates over the value of the free parameter,
# the second over the line of sight, as x,y,z, and the third over the simulation
mean_arr = np.zeros((len(iter_array),3,len(simul_arr)))

# Create an empty array, where each entry specifies the calculated standard
# deviation of the synchrotron intensity image of the corresponding simulation 
# for a particular value of the free parameter related to the observational 
# effect being studied. The first index iterates over the value of the free 
# parameter, the second over the line of sight, as x,y,z, and the third over the
# simulation
stdev_arr = np.zeros((len(iter_array),3,len(simul_arr)))

# Create an array that will hold the values for the noise level of the final
# synchrotron maps produced, in the same units as the generated noise. 
# The first index iterates over the value of the free 
# parameter, the second over the line of sight, as x,y,z, and the third over the
# simulation
final_noise = np.zeros((len(iter_array),3,len(simul_arr)))

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + save_direc

# Loop over the simulations, as we need to calculate the statistics for each
# simulation
for k in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[k]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[k])

	# Loop over the lines of sight, to calculate the mean and standard deviation
	# for each line of sight
	for j in range(3):
		# Open the FITS file that contains the synchrotron intensity maps for this
		# simulation
		sync_fits = fits.open(data_loc + 'synint_{}_gam{}.fits'.format(line_o_sight[j],gamma))

		# Extract the data for the simulated synchrotron intensities
		sync_map = sync_fits[0].data

		# Print a message to the screen to show that the data has been loaded
		print 'Synchrotron intensity loaded successfully'

		# Loop over the various values of the free parameter related to the 
		# observational effect being studied, to calculate the various statistics
		# for the synchrotron map observed for each value of the free parameter
		for i in range(len(iter_array)):
			# Calculate the standard deviation of the Gaussian noise that will 
			# affect the synchrotron maps. This needs to be done individually 
			# for lines of sight along each of the axes, because of the lines of
			# sight have different intensity maps.
			noise_stdev = iter_array[i] * np.median(sync_map)

			# Create an array of values that are randomly drawn from a Gaussian
			# distribution with the specified standard deviation. This 
			# represents the noise at each pixel of the image. 
			noise_matrix = np.random.normal(scale = noise_stdev,\
			 size = np.shape(sync_map))

			# Add the noise maps onto the synchrotron intensity maps, to produce
			# the mock 'observed' maps
			sync_map_free_param = sync_map + noise_matrix

			# Create a Gaussian kernel to use to smooth the synchrotron map,
			# using the given standard deviation
			gauss_kernel = Gaussian2DKernel(smooth_stdev)

			# Smooth the synchrotron maps to the required resolution by 
			# convolution with the above Gaussian kernel.
			sync_map_free_param = convolve_fft(sync_map_free_param,\
			 gauss_kernel, boundary = 'wrap')

			# To plot against the final noise level, we need to perform some 
			# additional calculations
			
			# Start by smoothing the initial synchrotron intensity map to
			# the required resolution. (No noise added)
			sync_map_no_noise = convolve_fft(sync_map, gauss_kernel,\
			 boundary = 'wrap')

			# Subtract this smoothed synchrotron map (with no noise) from the
			# full map (noise added, then smoothed)
			noise_map = sync_map_free_param - sync_map_no_noise

			# Calculate the standard deviation of the noise (in same units as
			# the intensity)
			stdev_final_noise = np.std(noise_map)

			# Express the calculated standard deviation as a fraction of the 
			# median synchrotron intensity of the map, and store the value in
			# the corresponding matrix
			final_noise[i,j,k] = stdev_final_noise / np.median(sync_map)

			# Find the indices in the map where the synchrotron intensity
			# is negative, and ignore them
			sync_map_free_param = sync_map_free_param[sync_map_free_param > 0]

			# Flatten the synchrotron intensity maps for this value of gamma, for
			# lines of sight along each of the axes
			flat_sync = sync_map_free_param.flatten()

			# If we are calculating the moments of the log PDFs, then calculate the
			# logarithm of the synchrotron intensity values
			if log == True:
				# In this case we are calculating the moments of the log normalised
				# PDFs, so calculate the log of the synchrotron intensities
				flat_sync = np.log10(flat_sync / np.mean(flat_sync, dtype = np.float64))

			# Calculate the mean of the synchrotron intensity map, and store the
			# result in the corresponding array
			mean_arr[i,j,k] = np.mean(flat_sync, dtype=np.float64)

			# Calculate the standard deviation of the synchrotron intensity map, and
			# store the result in the corresponding array
			stdev_arr[i,j,k] = np.std(flat_sync, dtype=np.float64)

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			simul_arr[k], line_o_sight[j])

		# At this point, all of the statistics that need to be calculated for
		# every line of sight have been calculated.

		# Close the fits files, to save memory
		sync_fits.close()

# Create average arrays for each of the statistics.
mean_avg_arr = np.zeros((len(iter_array),3,num_sims))
stdev_avg_arr = np.zeros((len(iter_array),3,num_sims))
final_noise_avg = np.zeros((len(iter_array),3,num_sims))

# Create error arrays for each of the statistics. These errors are calculated 
# by the standard error of the mean of the statistics calculated for different
# snapshots of the synchrotron maps.
mean_err_arr = np.zeros((len(iter_array),3,num_sims))
stdev_err_arr = np.zeros((len(iter_array),3,num_sims))

# Average the calculated statistics over the time snapshots, to obtain the 
# final value of each statistic for each simulation. Calculate the standard 
# error of the mean between snapshots while doing this.
for i in range(num_sims):
	# Calculate the average mean and standard deviation of synchrotron intensity
	# for each simulation, by averaging over the timesteps
	mean_avg_arr[:,:,i] = np.mean(mean_arr[:,:,i*num_timestep:(i+1)*num_timestep\
		],axis=2,dtype=np.float64)
	stdev_avg_arr[:,:,i] = np.mean(stdev_arr[:,:,i*num_timestep:\
		(i+1)*num_timestep],axis=2,dtype=np.float64)
	final_noise_avg[:,:,i] = np.mean(final_noise[:,:,i*num_timestep:\
		(i+1)*num_timestep],axis=2,dtype=np.float64)

	# Calculate the standard error of the mean of the statistics of synchrotron
	# intensity, over the different timesteps
	mean_err_arr[:,:,i] = np.std(mean_arr[:,:,i*num_timestep:(i+1)*num_timestep\
		],axis=2,dtype=np.float64)/np.sqrt(num_timestep)
	stdev_err_arr[:,:,i] = np.std(stdev_arr[:,:,i*num_timestep:\
		(i+1)*num_timestep],axis=2,dtype=np.float64)/np.sqrt(num_timestep)

# When the code reaches this point, the statistics have been calculated for
# every simulation, so it is time to start plotting

#------------------------------- Mean vs Zeta ----------------------------------

# Mean vs zeta (x-LOS)

# Create a figure to display a plot of the mean as a function of zeta for a 
# line of sight along the x axis, with multiple plots for different values of 
# the free parameter related to the observational effect being studied.
fig1 = plt.figure()

# Create an axis for this figure
ax1 = fig1.add_subplot(111)

# Plot the mean as a function of zeta for various values of the
# free parameter related to the observational effect being studied.
plt.errorbar(zeta, mean_avg_arr[0,0,:],fmt='b-o',label = leg_string\
	+'{0:.2f}'.format(iter_array[0]),yerr=mean_err_arr[0,0,:])
plt.errorbar(zeta, mean_avg_arr[9,0,:],fmt='g-^',label = leg_string\
	+'{0:.2f}'.format(iter_array[9]),yerr=mean_err_arr[9,0,:])
plt.errorbar(zeta, mean_avg_arr[19,0,:],fmt='r-s',label = leg_string\
	+'{0:.2f}'.format(iter_array[19]),yerr=mean_err_arr[19,0,:])
plt.errorbar(zeta, mean_avg_arr[29,0,:],fmt='k-*',label = leg_string\
	+'{0:.2f}'.format(iter_array[29]),yerr=mean_err_arr[29,0,:])
plt.errorbar(zeta, mean_avg_arr[39,0,:],fmt='m-D',label = leg_string\
	+'{0:.2f}'.format(iter_array[39]),yerr=mean_err_arr[39,0,:])
plt.errorbar(zeta, mean_avg_arr[49,0,:],fmt='c-p',label = leg_string\
	+'{0:.2f}'.format(iter_array[49]),yerr=mean_err_arr[49,0,:])

# Add a label to the x-axis
plt.xlabel(r'Turbulent Driving Parameter $\zeta$', fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Mean', fontsize = 18)

# Add a title to the plot
plt.title(r'Mean vs $\zeta$ {0}LOS Ang{1:.2f}'.format(line_o_sight[0],final_res))

# Force the legend to appear on the plot
plt.legend(loc = 4)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mean_zeta_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[0]), format = 'png')

# Print a message to the screen to show that the plot of the mean as a 
# function of zeta has been saved for this line of sight
print 'Plot of the mean as a function of zeta saved, {}LOS'.format(line_o_sight[0])

# Close the figure, now that it has been saved.
plt.close()

# Mean vs zeta (y-LOS)

# Create a figure to display a plot of the mean as a function of zeta for a 
# line of sight along the y axis, with multiple plots for different values of 
# the free parameter related to the observational effect being studied.
fig2 = plt.figure()

# Create an axis for this figure
ax2 = fig2.add_subplot(111)

# Plot the mean as a function of zeta for various values of the
# free parameter related to the observational effect being studied.
plt.errorbar(zeta, mean_avg_arr[0,1,:],fmt='b-o',label = leg_string\
	+'{0:.2f}'.format(iter_array[0]),yerr=mean_err_arr[0,1,:])
plt.errorbar(zeta, mean_avg_arr[9,1,:],fmt='g-^',label = leg_string\
	+'{0:.2f}'.format(iter_array[9]),yerr=mean_err_arr[9,1,:])
plt.errorbar(zeta, mean_avg_arr[19,1,:],fmt='r-s',label = leg_string\
	+'{0:.2f}'.format(iter_array[19]),yerr=mean_err_arr[19,1,:])
plt.errorbar(zeta, mean_avg_arr[29,1,:],fmt='k-*',label = leg_string\
	+'{0:.2f}'.format(iter_array[29]),yerr=mean_err_arr[29,1,:])
plt.errorbar(zeta, mean_avg_arr[39,1,:],fmt='m-D',label = leg_string\
	+'{0:.2f}'.format(iter_array[39]),yerr=mean_err_arr[39,1,:])
plt.errorbar(zeta, mean_avg_arr[49,1,:],fmt='c-p',label = leg_string\
	+'{0:.2f}'.format(iter_array[49]),yerr=mean_err_arr[49,1,:])

# Add a label to the x-axis
plt.xlabel(r'Turbulent Driving Parameter $\zeta$', fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Mean', fontsize = 18)

# Add a title to the plot
plt.title(r'Mean vs $\zeta$ {0}LOS Ang{1:.2f}'.format(line_o_sight[1],final_res))

# Force the legend to appear on the plot
plt.legend(loc = 4)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mean_zeta_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[1]), format = 'png')

# Print a message to the screen to show that the plot of the mean as a 
# function of zeta has been saved for this line of sight
print 'Plot of the mean as a function of zeta saved, {}LOS'.format(line_o_sight[1])

# Close the figure, now that it has been saved.
plt.close()

# Mean vs zeta (z-LOS)

# Create a figure to display a plot of the mean as a function of zeta for a 
# line of sight along the z axis, with multiple plots for different values of 
# the free parameter related to the observational effect being studied.
fig3 = plt.figure()

# Create an axis for this figure
ax3 = fig3.add_subplot(111)

# Plot the mean as a function of zeta for various values of the
# free parameter related to the observational effect being studied.
plt.errorbar(zeta, mean_avg_arr[0,2,:],fmt='b-o',label = leg_string\
	+'{0:.2f}'.format(iter_array[0]),yerr=mean_err_arr[0,2,:])
plt.errorbar(zeta, mean_avg_arr[9,2,:],fmt='g-^',label = leg_string\
	+'{0:.2f}'.format(iter_array[9]),yerr=mean_err_arr[9,2,:])
plt.errorbar(zeta, mean_avg_arr[19,2,:],fmt='r-s',label = leg_string\
	+'{0:.2f}'.format(iter_array[19]),yerr=mean_err_arr[19,2,:])
plt.errorbar(zeta, mean_avg_arr[29,2,:],fmt='k-*',label = leg_string\
	+'{0:.2f}'.format(iter_array[29]),yerr=mean_err_arr[29,2,:])
plt.errorbar(zeta, mean_avg_arr[39,2,:],fmt='m-D',label = leg_string\
	+'{0:.2f}'.format(iter_array[39]),yerr=mean_err_arr[39,2,:])
plt.errorbar(zeta, mean_avg_arr[49,2,:],fmt='c-p',label = leg_string\
	+'{0:.2f}'.format(iter_array[49]),yerr=mean_err_arr[49,2,:])

# Add a label to the x-axis
plt.xlabel(r'Turbulent Driving Parameter $\zeta$', fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Mean', fontsize = 18)

# Add a title to the plot
plt.title(r'Mean vs $\zeta$ {0}LOS Ang{1:.2f}'.format(line_o_sight[2],final_res))

# Force the legend to appear on the plot
plt.legend(loc = 4)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mean_zeta_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[2]), format = 'png')

# Print a message to the screen to show that the plot of the mean as a 
# function of zeta has been saved for this line of sight
print 'Plot of the mean as a function of zeta saved, {}LOS'.format(line_o_sight[2])

# Close the figure, now that it has been saved.
plt.close()

#-------------------------- Mean vs Observational Effect -----------------------

# Mean vs Observational Effect (x-LOS)

# Create a figure to display a plot of the mean as a function of the
# observational effect, for all simulations
fig4 = plt.figure()

# Create an axis for this figure
ax4 = fig4.add_subplot(111)

# Plot the mean as a function of the observational effect for simulations
plt.errorbar(final_noise_avg[:,0,0], mean_avg_arr[:,0,0],fmt='b-o',label= '{}'.format(\
	sim_labels[0]),yerr=mean_err_arr[:,0,0])
plt.errorbar(final_noise_avg[:,0,1], mean_avg_arr[:,0,1],fmt='r-^',label= '{}'.format(\
	sim_labels[1]),yerr=mean_err_arr[:,0,1])
plt.errorbar(final_noise_avg[:,0,2], mean_avg_arr[:,0,2],fmt='g-s',label= '{}'.format(\
	sim_labels[2]),yerr=mean_err_arr[:,0,2])
plt.errorbar(final_noise_avg[:,0,3], mean_avg_arr[:,0,3],fmt='k-*',label= '{}'.format(\
	sim_labels[3]),yerr=mean_err_arr[:,0,3])
plt.errorbar(final_noise_avg[:,0,4], mean_avg_arr[:,0,4],fmt='m-D',label= '{}'.format(\
	sim_labels[4]),yerr=mean_err_arr[:,0,4])

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Mean', fontsize = 18)

# Add a title to the plot
plt.title('Mean vs ' + title_string + ' {0}LOS Ang{1:.2f}'.format(line_o_sight[0],\
 final_res), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc=1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mean_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[0]), format = 'png')

# Print a message to the screen to show that the plot of the mean as a 
# function of the observational effect has been saved
print 'Plot of the mean as a function of observational effect saved, {}LOS'.\
format(line_o_sight[0])

# Close the figure, now that it has been saved.
plt.close()

# Mean vs Observational Effect (y-LOS)

# Create a figure to display a plot of the mean as a function of the
# observational effect, for all simulations
fig5 = plt.figure()

# Create an axis for this figure
ax5 = fig5.add_subplot(111)

# Plot the mean as a function of the observational effect for simulations
plt.errorbar(final_noise_avg[:,1,0], mean_avg_arr[:,1,0],fmt='b-o',label= '{}'.format(\
	sim_labels[0]),yerr=mean_err_arr[:,1,0])
plt.errorbar(final_noise_avg[:,1,1], mean_avg_arr[:,1,1],fmt='r-^',label= '{}'.format(\
	sim_labels[1]),yerr=mean_err_arr[:,1,1])
plt.errorbar(final_noise_avg[:,1,2], mean_avg_arr[:,1,2],fmt='g-s',label= '{}'.format(\
	sim_labels[2]),yerr=mean_err_arr[:,1,2])
plt.errorbar(final_noise_avg[:,1,3], mean_avg_arr[:,1,3],fmt='k-*',label= '{}'.format(\
	sim_labels[3]),yerr=mean_err_arr[:,1,3])
plt.errorbar(final_noise_avg[:,1,4], mean_avg_arr[:,1,4],fmt='m-D',label= '{}'.format(\
	sim_labels[4]),yerr=mean_err_arr[:,1,4])

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Mean', fontsize = 18)

# Add a title to the plot
plt.title('Mean vs ' + title_string + ' {0}LOS Ang{1:.2f}'.format(line_o_sight[1],\
 final_res), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc=1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mean_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[1]), format = 'png')

# Print a message to the screen to show that the plot of the mean as a 
# function of the observational effect has been saved
print 'Plot of the mean as a function of observational effect saved, {}LOS'.\
format(line_o_sight[1])

# Close the figure, now that it has been saved.
plt.close()

# Mean vs Observational Effect (z-LOS)

# Create a figure to display a plot of the mean as a function of the
# observational effect, for all simulations
fig6 = plt.figure()

# Create an axis for this figure
ax6 = fig6.add_subplot(111)

# Plot the mean as a function of the observational effect for simulations
plt.errorbar(final_noise_avg[:,2,0], mean_avg_arr[:,2,0],fmt='b-o',label= '{}'.format(\
	sim_labels[0]),yerr=mean_err_arr[:,2,0])
plt.errorbar(final_noise_avg[:,2,1], mean_avg_arr[:,2,1],fmt='r-^',label= '{}'.format(\
	sim_labels[1]),yerr=mean_err_arr[:,2,1])
plt.errorbar(final_noise_avg[:,2,2], mean_avg_arr[:,2,2],fmt='g-s',label= '{}'.format(\
	sim_labels[2]),yerr=mean_err_arr[:,2,2])
plt.errorbar(final_noise_avg[:,2,3], mean_avg_arr[:,2,3],fmt='k-*',label= '{}'.format(\
	sim_labels[3]),yerr=mean_err_arr[:,2,3])
plt.errorbar(final_noise_avg[:,2,4], mean_avg_arr[:,2,4],fmt='m-D',label= '{}'.format(\
	sim_labels[4]),yerr=mean_err_arr[:,2,4])

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Mean', fontsize = 18)

# Add a title to the plot
plt.title('Mean vs ' + title_string + ' {0}LOS Ang{1:.2f}'.format(line_o_sight[2], final_res), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc=1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'Mean_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[2]), format = 'png')

# Print a message to the screen to show that the plot of the mean as a 
# function of the observational effect has been saved
print 'Plot of the mean as a function of observational effect saved, {}LOS'.\
format(line_o_sight[2])

# Close the figure, now that it has been saved.
plt.close()

#------------------------- Standard Deviation vs Zeta --------------------------

# Standard Deviation vs zeta (x-LOS)

# Create a figure to display a plot of the standard deviation as a function of 
# zeta for a line of sight along the x axis, with multiple plots for different 
# values of the free parameter related to the observational effect being studied
fig7 = plt.figure()

# Create an axis for this figure
ax7 = fig7.add_subplot(111)

# Plot the standard deviation as a function of zeta for various values of the
# free parameter related to the observational effect being studied.
plt.errorbar(zeta, stdev_avg_arr[0,0,:],fmt='b-o',label = leg_string\
	+'{0:.2f}'.format(iter_array[0]),yerr=stdev_err_arr[0,0,:])
plt.errorbar(zeta, stdev_avg_arr[9,0,:],fmt='g-^',label = leg_string\
	+'{0:.2f}'.format(iter_array[9]),yerr=stdev_err_arr[9,0,:])
plt.errorbar(zeta, stdev_avg_arr[19,0,:],fmt='r-s',label = leg_string\
	+'{0:.2f}'.format(iter_array[19]),yerr=stdev_err_arr[19,0,:])
plt.errorbar(zeta, stdev_avg_arr[29,0,:],fmt='k-*',label = leg_string\
	+'{0:.2f}'.format(iter_array[29]),yerr=stdev_err_arr[29,0,:])
plt.errorbar(zeta, stdev_avg_arr[39,0,:],fmt='m-D',label = leg_string\
	+'{0:.2f}'.format(iter_array[39]),yerr=stdev_err_arr[39,0,:])
plt.errorbar(zeta, stdev_avg_arr[49,0,:],fmt='c-p',label = leg_string\
	+'{0:.2f}'.format(iter_array[49]),yerr=stdev_err_arr[49,0,:])

# Add a label to the x-axis
plt.xlabel(r'Turbulent Driving Parameter $\zeta$', fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Standard Deviation', fontsize = 18)

# Add a title to the plot
plt.title(r'StDev vs $\zeta$ {0}LOS Ang{1:.2f}'.format(line_o_sight[0],final_res))

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'StDev_zeta_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[0]), format = 'png')

# Print a message to the screen to show that the plot of the standard deviation
# as a function of zeta has been saved for this line of sight
print 'Plot of the stdev as a function of zeta saved, {}LOS'.format(line_o_sight[0])

# Close the figure, now that it has been saved.
plt.close()

# Standard Deviation vs zeta (y-LOS)

# Create a figure to display a plot of the standard deviation as a function of 
# zeta for a line of sight along the y axis, with multiple plots for different 
# values of the free parameter related to the observational effect being studied
fig8 = plt.figure()

# Create an axis for this figure
ax8 = fig8.add_subplot(111)

# Plot the standard deviation as a function of zeta for various values of the
# free parameter related to the observational effect being studied.
plt.errorbar(zeta, stdev_avg_arr[0,1,:],fmt='b-o',label = leg_string\
	+'{0:.2f}'.format(iter_array[0]),yerr=stdev_err_arr[0,1,:])
plt.errorbar(zeta, stdev_avg_arr[9,1,:],fmt='g-^',label = leg_string\
	+'{0:.2f}'.format(iter_array[9]),yerr=stdev_err_arr[9,1,:])
plt.errorbar(zeta, stdev_avg_arr[19,1,:],fmt='r-s',label = leg_string\
	+'{0:.2f}'.format(iter_array[19]),yerr=stdev_err_arr[19,1,:])
plt.errorbar(zeta, stdev_avg_arr[29,1,:],fmt='k-*',label = leg_string\
	+'{0:.2f}'.format(iter_array[29]),yerr=stdev_err_arr[29,1,:])
plt.errorbar(zeta, stdev_avg_arr[39,1,:],fmt='m-D',label = leg_string\
	+'{0:.2f}'.format(iter_array[39]),yerr=stdev_err_arr[39,1,:])
plt.errorbar(zeta, stdev_avg_arr[49,1,:],fmt='c-p',label = leg_string\
	+'{0:.2f}'.format(iter_array[49]),yerr=stdev_err_arr[49,1,:])

# Add a label to the x-axis
plt.xlabel(r'Turbulent Driving Parameter $\zeta$', fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Standard Deviation', fontsize = 18)

# Add a title to the plot
plt.title(r'StDev vs $\zeta$ {0}LOS Ang{1:.2f}'.format(line_o_sight[1],final_res))

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'StDev_zeta_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[1]), format = 'png')

# Print a message to the screen to show that the plot of the standard deviation
# as a function of zeta has been saved for this line of sight
print 'Plot of the stdev as a function of zeta saved, {}LOS'.format(line_o_sight[1])

# Close the figure, now that it has been saved.
plt.close()

# Standard Deviation vs zeta (z-LOS)

# Create a figure to display a plot of the standard deviation as a function of 
# zeta for a line of sight along the z axis, with multiple plots for different 
# values of the free parameter related to the observational effect being studied
fig9 = plt.figure()

# Create an axis for this figure
ax9 = fig9.add_subplot(111)

# Plot the standard deviation as a function of zeta for various values of the
# free parameter related to the observational effect being studied.
plt.errorbar(zeta, stdev_avg_arr[0,2,:],fmt='b-o',label = leg_string\
	+'{0:.2f}'.format(iter_array[0]),yerr=stdev_err_arr[0,2,:])
plt.errorbar(zeta, stdev_avg_arr[9,2,:],fmt='g-^',label = leg_string\
	+'{0:.2f}'.format(iter_array[9]),yerr=stdev_err_arr[9,2,:])
plt.errorbar(zeta, stdev_avg_arr[19,2,:],fmt='r-s',label = leg_string\
	+'{0:.2f}'.format(iter_array[19]),yerr=stdev_err_arr[19,2,:])
plt.errorbar(zeta, stdev_avg_arr[29,2,:],fmt='k-*',label = leg_string\
	+'{0:.2f}'.format(iter_array[29]),yerr=stdev_err_arr[29,2,:])
plt.errorbar(zeta, stdev_avg_arr[39,2,:],fmt='m-D',label = leg_string\
	+'{0:.2f}'.format(iter_array[39]),yerr=stdev_err_arr[39,2,:])
plt.errorbar(zeta, stdev_avg_arr[49,2,:],fmt='c-p',label = leg_string\
	+'{0:.2f}'.format(iter_array[49]),yerr=stdev_err_arr[49,2,:])

# Add a label to the x-axis
plt.xlabel(r'Turbulent Driving Parameter $\zeta$', fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Standard Deviation', fontsize = 18)

# Add a title to the plot
plt.title(r'StDev vs $\zeta$ {0}LOS Ang{1:.2f}'.format(line_o_sight[2],final_res))

# Force the legend to appear on the plot
plt.legend(loc = 1)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'StDev_zeta_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[2]), format = 'png')

# Print a message to the screen to show that the plot of the standard deviation
# as a function of zeta has been saved for this line of sight
print 'Plot of the stdev as a function of zeta saved, {}LOS'.format(line_o_sight[2])

# Close the figure, now that it has been saved.
plt.close()

#------------------- Standard Deviation vs Observational Effect ----------------

# Standard Deviation vs Observational Effect (x-LOS)

# Create a figure to display a plot of the standard deviation as a function of 
# the observational effect, for all simulations
fig10 = plt.figure()

# Create an axis for this figure
ax10 = fig10.add_subplot(111)

# Plot the standard deviation as a function of the observational effect for simulations
plt.errorbar(final_noise_avg[:,0,0], stdev_avg_arr[:,0,0],fmt='b-o',label= '{}'.format(\
	sim_labels[0]),yerr=stdev_err_arr[:,0,0])
plt.errorbar(final_noise_avg[:,0,1], stdev_avg_arr[:,0,1],fmt='r-^',label= '{}'.format(\
	sim_labels[1]),yerr=stdev_err_arr[:,0,1])
plt.errorbar(final_noise_avg[:,0,2], stdev_avg_arr[:,0,2],fmt='g-s',label= '{}'.format(\
	sim_labels[2]),yerr=stdev_err_arr[:,0,2])
plt.errorbar(final_noise_avg[:,0,3], stdev_avg_arr[:,0,3],fmt='k-*',label= '{}'.format(\
	sim_labels[3]),yerr=stdev_err_arr[:,0,3])
plt.errorbar(final_noise_avg[:,0,4], stdev_avg_arr[:,0,4],fmt='m-D',label= '{}'.format(\
	sim_labels[4]),yerr=stdev_err_arr[:,0,4])

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Standard Deviation', fontsize = 18)

# Add a title to the plot
plt.title('StDev vs ' + title_string + ' {0}LOS Ang{1:.2f}'.format(line_o_sight[0],\
	final_res), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc=4)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'StDev_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[0]), format = 'png')

# Print a message to the screen to show that the plot of the standard deviation
# as a function of the observational effect has been saved
print 'Plot of the stdev as a function of observational effect saved, {}LOS'.\
format(line_o_sight[0])

# Close the figure, now that it has been saved.
plt.close()

# Standard Deviation vs Observational Effect (y-LOS)

# Create a figure to display a plot of the standard deviation as a function of 
# the observational effect, for all simulations
fig11 = plt.figure()

# Create an axis for this figure
ax11 = fig11.add_subplot(111)

# Plot the standard deviation as a function of the observational effect for 
# simulations
plt.errorbar(final_noise_avg[:,1,0], stdev_avg_arr[:,1,0],fmt='b-o',label= '{}'.format(\
	sim_labels[0]),yerr=stdev_err_arr[:,1,0])
plt.errorbar(final_noise_avg[:,1,1], stdev_avg_arr[:,1,1],fmt='r-^',label= '{}'.format(\
	sim_labels[1]),yerr=stdev_err_arr[:,1,1])
plt.errorbar(final_noise_avg[:,1,2], stdev_avg_arr[:,1,2],fmt='g-s',label= '{}'.format(\
	sim_labels[2]),yerr=stdev_err_arr[:,1,2])
plt.errorbar(final_noise_avg[:,1,3], stdev_avg_arr[:,1,3],fmt='k-*',label= '{}'.format(\
	sim_labels[3]),yerr=stdev_err_arr[:,1,3])
plt.errorbar(final_noise_avg[:,1,4], stdev_avg_arr[:,1,4],fmt='m-D',label= '{}'.format(\
	sim_labels[4]),yerr=stdev_err_arr[:,1,4])

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Standard Deviation', fontsize = 18)

# Add a title to the plot
plt.title('StDev vs ' + title_string + ' {0}LOS Ang{1:.2f}'.format(line_o_sight[1],\
	final_res), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc=4)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'StDev_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[1]), format = 'png')

# Print a message to the screen to show that the plot of the standard deviation
# as a function of the observational effect has been saved
print 'Plot of the stdev as a function of observational effect saved, {}LOS'.\
format(line_o_sight[1])

# Close the figure, now that it has been saved.
plt.close()

# Standard Deviation vs Observational Effect (z-LOS)

# Create a figure to display a plot of the standard deviation as a function of 
# the observational effect, for all simulations
fig12 = plt.figure()

# Create an axis for this figure
ax12 = fig12.add_subplot(111)

# Plot the mean as a function of the observational effect for simulations
plt.errorbar(final_noise_avg[:,2,0], stdev_avg_arr[:,2,0],fmt='b-o',label= '{}'.format(\
	sim_labels[0]),yerr=stdev_err_arr[:,2,0])
plt.errorbar(final_noise_avg[:,2,1], stdev_avg_arr[:,2,1],fmt='r-^',label= '{}'.format(\
	sim_labels[1]),yerr=stdev_err_arr[:,2,1])
plt.errorbar(final_noise_avg[:,2,2], stdev_avg_arr[:,2,2],fmt='g-s',label= '{}'.format(\
	sim_labels[2]),yerr=stdev_err_arr[:,2,2])
plt.errorbar(final_noise_avg[:,2,3], stdev_avg_arr[:,2,3],fmt='k-*',label= '{}'.format(\
	sim_labels[3]),yerr=stdev_err_arr[:,2,3])
plt.errorbar(final_noise_avg[:,2,4], stdev_avg_arr[:,2,4],fmt='m-D',label= '{}'.format(\
	sim_labels[4]),yerr=stdev_err_arr[:,2,4])

# Add a label to the x-axis
plt.xlabel(xlabel, fontsize = 18)

# Add a label to the y-axis
plt.ylabel('Standard Deviation', fontsize = 18)

# Add a title to the plot
plt.title('StDev vs ' + title_string + ' {0}LOS Ang{1:.2f}'.format(line_o_sight[2],\
	final_res), fontsize = 20)

# Force the legend to appear on the plot
plt.legend(loc=4)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'StDev_{}_gam{}_{}LOS.png'.format(obs_effect,gamma,\
	line_o_sight[2]), format = 'png')

# Print a message to the screen to show that the plot of the standard deviation
# as a function of the observational effect has been saved
print 'Plot of the stdev as a function of observational effect saved, {}LOS'.\
format(line_o_sight[2])

# Close the figure, now that it has been saved.
plt.close()