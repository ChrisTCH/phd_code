#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of simulated synchrotron   #
# intensities, and calculates the skewness of the simulated distribution by    #
# three different methods. This includes calculating the sample skewness, and  #
# performing least squares fits to log-normal and gamma distributions, to      #
# calculate the skewness of the simulated synchrotron intensity distribution   #
# according to these best fits. This is performed for each simulation, and     #
# then plots comparing the three different methods of calculating skewness are #
# created. For each simulation, the best fit parameters are recorded, as well  #
# as a measure of how good the fit is.                                         #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 8/5/2015                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.optimize to perform least squares optimisation,
# scipy.special to use the gamma function, scipy.stats to calculate sample
# skewness.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import leastsq
import scipy.special as scp
from scipy import stats

# Define a function that will calculate the residuals (difference
# between the data and the model) for a log-normal distribution.
def resid_lognorm(param, data, x_arr):
	'''
	Description
		This function calculates the difference between the given data, and the
		log-normal distribution for the given parameter values. This is to be 
		used in determining the parameter values that minimise the squared
		differences.

	Required Input
		param - A one-dimensional Numpy array with two entries. The first entry
				should be the location parameter M, and the second entry should
				be the scale parameter s. The location parameter can be any
				real number, but the scale parameter must be a positive number.
				If the scale parameter is not positive, then it is changed to a
				small, positive number.
		data - A one-dimensional Numpy array giving the value of the data at 
			   each value of the independent variable x.
		x_arr - A one-dimensional Numpy array giving the values of the 
				independent variable x that are used to obtain the data. These
				will be used to evaluate the log-normal distribution for the 
				given parameter values.

	Output
		resid - A one-dimensional Numpy array specifying the difference between 
				the data and log-normal distribution for the given parameter 
				values. 
	'''

	# First extract the given parameter values from the parameter array
	# M is the location parameter, and S is the scale parameter for a lognormal
	# distribution.
	M, S = param

	# Check to see if the scale parameter is a positive number
	if S <= 0:
		# In this case the value of the scale parameter is not positive, and so
		# it is invalid. Change it to a small positive number
		S = 1.0 * np.power(10.0,-6.0)

	# Calculate the difference between the given data and the value of the
	# log-normal distribution for the given parameter values
	resid = data - np.exp(-np.power(np.log(x_arr) - M, 2.0) /\
	 (2.0 * np.power(S,2.0))) / (S * x_arr * np.sqrt(2.0 * np.pi))

	# Return the calculated residuals to the calling function
	return resid

# Define a function that will calculate the residuals (difference
# between the data and the model) for a gamma distribution.
def resid_gamma(param, data, x_arr):
	'''
	Description
		This function calculates the difference between the given data, and the
		gamma distribution for the given parameter values. This is to be 
		used in determining the parameter values that minimise the squared
		differences.

	Required Input
		param - A one-dimensional Numpy array with two entries. The first entry
				should be the shape parameter alpha, and the second entry should
				be the scale parameter theta. Both the shape and scale 
				parameters must be greater than zero.
		data - A one-dimensional Numpy array giving the value of the data at 
			   each value of the independent variable x.
		x_arr - A one-dimensional Numpy array giving the values of the 
				independent variable x that are used to obtain the data. These
				will be used to evaluate the gamma distribution for the 
				given parameter values.

	Output
		resid - A one-dimensional Numpy array specifying the difference between 
				the data and gamma distribution for the given parameter 
				values. 
	'''

	# First extract the given parameter values from the parameter array
	# alpha is the shape parameter, and theta is the scale parameter for a gamma
	# distribution.
	alpha, theta = param

	# Check to see if the shape parameter is a positive number
	if alpha <= 0:
		# In this case the shape parameter is not a positive number, so change 
		# it to be a small positive number
		alpha = 1.0 * np.power(10.0,-2.0)

	# Check to see if the scale parameter is a positive number
	if theta <= 0:
		# In this case the scale parameter is not a positive number, so change 
		# it to be a small positive number
		theta = 1.0 * np.power(10.0,-2.0)

	# Calculate the difference between the given data and the value of the
	# gamma distribution for the given parameter values
	resid = data - np.power(x_arr, alpha - 1) * np.exp(-x_arr/theta) /\
	 (scp.gamma(alpha) * np.power(theta, alpha))

	# Return the calculated residuals to the calling function
	return resid

#-------------------------------------------------------------------------------

# Set a variable to hold the number of bins to use in calculating the 
# histograms of the simulated synchrotron intensity 
num_bins = 100

# Create a string for the directory that contains the simulated synchrotron
# intensity maps to use. 
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

# Create an array that specifies the value of gamma used to produce each 
# synchrotron intensity map
gamma_arr = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0])

# Create empty arrays, for the best-fit location and scale parameters of the
# log-normal distribution. Each entry specifies the calculated best-fit 
# parameter to the synchrotron intensity distribution of the corresponding 
# simulation for a particular value of gamma. Each row corresponds to a 
# simulation, and each column corresponds to a value of gamma. 
M_arr = np.zeros((len(simul_arr),len(gamma_arr)))
S_arr = np.zeros((len(simul_arr),len(gamma_arr)))

# Set an initial guess to use when trying to determine the best fit values of
# M and S for a log-normal distribution. This is a Numpy array with two 
# entries, with the first entry being the initial guess for M, and the second
# entry being the initial guess for S.
init_ln = np.array([0.0,1.0])

# Create empty arrays, for the best-fit shape and scale parameters of the
# gamma distribution. Each entry specifies the calculated best-fit 
# parameter to the synchrotron intensity distribution of the corresponding 
# simulation for a particular value of gamma. Each row corresponds to a 
# simulation, and each column corresponds to a value of gamma. 
alpha_arr = np.zeros((len(simul_arr),len(gamma_arr)))
theta_arr = np.zeros((len(simul_arr),len(gamma_arr)))

# Set an initial guess to use when trying to determine the best fit values of
# alpha and theta for a gamma distribution. This is a Numpy array with two 
# entries, with the first entry being the initial guess for alpha, and the 
# second entry being the initial guess for theta.
init_gam = np.array([2.0,2.0])

# Create empty arrays, for the skewness of the synchrotron intensity 
# distribution as calculated from the best-fit parameters to the log-normal and
# gamma distributions. Each entry specifies the calculated skewness for the
# synchrotron intensity distribution of the corresponding simulation for a
# particular value of gamma. Each row corresponds to a simulation, and each
# column corresponds to a value of gamma. 
skew_lognorm_arr = np.zeros((len(simul_arr),len(gamma_arr)))
skew_gamma_arr = np.zeros((len(simul_arr),len(gamma_arr)))

# Create empty arrays, for the sum of the squared residuals of the best-fit
# log-normal and gamma distributions to the synchrotron intensity distribution.
# Each entry specifies the calculated sum of the squared residuals for the
# synchrotron intensity distribution of the corresponding simulation for a
# particular value of gamma. Each row corresponds to a simulation, and each
# column corresponds to a value of gamma.
resid_final_lognorm_arr = np.zeros((len(simul_arr),len(gamma_arr)))
resid_final_gamma_arr = np.zeros((len(simul_arr),len(gamma_arr)))

# Create an empty array, where each entry specifies the calculated skewness of
# the synchrotron intensity image of the corresponding simulation for a 
# particular value of gamma. Each row corresponds to a simulation, and each 
# column corresponds to a value of gamma. 
# NOTE: We will calculate the biased skewness of the entire sample of 
# synchrotron intensities
skew_sample_arr = np.zeros((len(simul_arr),len(gamma_arr)))

# Create a new string representing the directory in which all plots should
# be saved
save_loc = simul_loc + 'Ultimate_Output/'

# Loop over the simulations, as we need to calculate the skewness for each
# simulation
for j in range(len(simul_arr)):
	# Create a string that gives the full directory of the simulation currently
	# being used
	data_loc = simul_loc + simul_arr[j]

	# Print a message to the screen to show what simulation is being used
	print 'Starting calculation for simulation {}'.format(simul_arr[j])

	# Open the FITS file that contains the simulated synchrotron intensity maps
	sync_fits = fits.open(data_loc + 'synint_p1-4.fits')

	# Extract the data for the simulated synchrotron intensities
	# This is a 3D data cube, where the slices along the third axis are the
	# synchrotron intensities observed for different values of gamma, the power
	# law index of the cosmic ray electrons emitting the synchrotron emission.
	sync_data = sync_fits[0].data

	# Print a message to the screen to show that the synchrotron data has been 
	# loaded successfully
	print 'Simulated synchrotron data loaded'

	# Loop over the various values of gamma, to calculate the skewness
	# for the synchrotron map observed for each value of gamma
	for i in range(len(gamma_arr)):
		# Extract the synchrotron intensity map for this value of gamma
		sync_map = sync_data[i]

		# Flatten the synchrotron intensity maps for this value of gamma
		flat_sync = sync_map.flatten()

		# Calculate the biased skewness of the synchrotron intensity maps, and
		# store the results in the corresponding array.
		skew_sample_arr[j,i] = stats.skew(flat_sync)

		# Create a histogram of the flattened synchrotron intensity values for
		# this map. The output from this histogram will be used to perform the
		# least squares fit
		sync_hist, bin_edges = np.histogram(flat_sync, bins = num_bins,\
			density = True)

		# We want the centres of the bins, not the edges. Create a new array 
		# that specifies the bin centres, from the bin edges
		bin_centres = ((bin_edges + np.roll(bin_edges,-1)) / 2.0)[0:-1]

		# Perform a least squares fit between the synchrotron intensity map and
		# a log-normal distribution. The returned variables are the best-fit
		# parameters for the distribution, the covariance matrix, a dictionary
		# with information about the fitting, a string giving information
		# if there is a failure, and an integer flag if a solution was found
		param_ln, cov_ln, info_ln, mesg_ln, ier_ln = leastsq(resid_lognorm,\
			init_ln, args = (sync_hist, bin_centres), full_output = True)

		# Extract the best-fit parameters for the log-normal distribution, and
		# store them in the corresponding array
		M_arr[j,i], S_arr[j,i] = param_ln

		# Calculate the skewness of the best-fit log-normal distribution from 
		# the best-fit parameters for this distribution
		skew_lognorm_arr[j,i] = np.sqrt(np.exp(S_arr[j,i] * S_arr[j,i]) - 1) *\
		(2 + np.exp(S_arr[j,i] * S_arr[j,i]))

		# Extract the sum of the squared residuals for the best-fit log-normal
		# distribution,  and store it in the corresponding array
		resid_final_lognorm_arr[j,i] = np.sum(np.power(info_ln["fvec"],2.0))

		# Perform a least squares fit between the synchrotron intensity map and
		# a gamma distribution. The returned variables are the best-fit
		# parameters for the distribution, the covariance matrix, a dictionary
		# with information about the fitting, a string giving information
		# if there is a failure, and an integer flag if a solution was found
		param_gam, cov_gam, info_gam, mesg_gam, ier_gam = leastsq(resid_gamma,\
			init_gam, args = (sync_hist, bin_centres), full_output = True)

		# Extract the best-fit parameters for the gamma distribution, and
		# store them in the corresponding array
		alpha_arr[j,i], theta_arr[j,i] = param_gam

		# Calculate the skewness of the best-fit gamma distribution from 
		# the best-fit parameters for this distribution
		skew_gamma_arr[j,i] = 2.0 / np.sqrt(alpha_arr[j,i])

		# Extract the sum of the squared residuals for the best-fit gamma
		# distribution,  and store it in the corresponding array
		resid_final_gamma_arr[j,i] = np.sum(np.power(info_gam["fvec"],2.0))

		if i == 2:
			# Produce a plot of the best-fit log-normal and gamma distributions,
			# on top of the actual histogram of synchrotron intensity values
			fig3 = plt.figure()

			# Create an axis for this figure
			ax3 = fig3.add_subplot(111)

			# Calculate the values of the log-normal distribution at each 
			# of the bin centres
			ln_dist = np.exp(-np.power(np.log(bin_centres) - M_arr[j,i], 2.0) /\
	 		(2.0 * np.power(S_arr[j,i],2.0))) / (S_arr[j,i] * bin_centres\
	 		 * np.sqrt(2.0 * np.pi))

	 		# Calculate the values of the gamma distribution at each of the 
	 		# bin centres
	 		gam_dist = np.power(bin_centres, alpha_arr[j,i] - 1) *\
	 		 np.exp(-bin_centres/theta_arr[j,i]) / (scp.gamma(alpha_arr[j,i])\
	 		  * np.power(theta_arr[j,i], alpha_arr[j,i]))

			# Plot the histogram of the synchrotron intensity values, as well
			# as the best-fit log-normal and gamma distributions
			plt.plot(bin_centres, sync_hist, 'bo', label ='True hist')
			plt.plot(bin_centres, ln_dist, 'ro', label ='Lognorm Dist')
			plt.plot(bin_centres, gam_dist, 'go', label ='Gamma Dist')

			# Add a label to the x-axis
			plt.xlabel('Sync Inten [arb units]', fontsize = 20)

			# Add a label to the y-axis
			plt.ylabel('Normalised PDF', fontsize = 20)

			# Add a title to the plot
			plt.title('Hist and Best Fits Gam 2 z LOS {}'.format(short_simul[j]), fontsize = 20)

			# Force the legend to appear on the plot
			plt.legend(loc = 1)

			# Save the figure using the given filename and format
			plt.savefig(save_loc + 'Hists_gamma_2_{}.png'.format(short_simul[j]), format = 'png')

			# Print a message to the screen to show that the histograms have
			# been saved
			print 'Histograms saved for {}'.format(short_simul[j])

			# Close the figure, now that it has been saved.
			plt.close()

	# When the code reaches this point, the skewness values have been calculated
	# for all values of gamma, and all the distributions we are interested in

	# Produce a plot of skewness against gamma, where all the methods used to 
	# calculate the skewness are shown on the same plot
	fig = plt.figure()

	# Create an axis for this figure
	ax = fig.add_subplot(111)

	# Plot the skewness as a function of gamma for each method of calculating
	# the skewness
	plt.plot(gamma_arr, skew_sample_arr[j,:], 'bo', label ='Sample')
	plt.plot(gamma_arr, skew_lognorm_arr[j,:], 'ro', label ='Lognorm Dist')
	plt.plot(gamma_arr, skew_gamma_arr[j,:], 'go', label ='Gamma Dist')

	# Add a label to the x-axis
	plt.xlabel('Gamma', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Skewness', fontsize = 20)

	# Add a title to the plot
	plt.title('Skew vs Gamma z LOS {}'.format(short_simul[j]), fontsize = 20)

	# Force the legend to appear on the plot
	plt.legend(loc = 2)

	# Save the figure using the given filename and format
	plt.savefig(save_loc + 'Skew_gamma_z_{}.png'.format(short_simul[j]), format = 'png')

	# Print a message to the screen to show that the plot of the skewness as a 
	# function of gamma has been saved
	print 'Plot of the skewness as a function of gamma saved z for {}'.format(short_simul[j])

	# Close the figure, now that it has been saved.
	plt.close()

	# Produce a plot of the sum of the squared residuals against gamma, where 
	# the methods used to calculate the skewness are shown on the same plot
	fig2 = plt.figure()

	# Create an axis for this figure
	ax2 = fig2.add_subplot(111)

	# Plot the sum of the squared residuals as a function of gamma for each
	# method of calculating the skewness
	plt.plot(gamma_arr, resid_final_lognorm_arr[j,:], 'ro', label ='Lognorm Dist')
	plt.plot(gamma_arr, resid_final_gamma_arr[j,:], 'go', label ='Gamma Dist')

	# Add a label to the x-axis
	plt.xlabel('Gamma', fontsize = 20)

	# Add a label to the y-axis
	plt.ylabel('Squared Residuals', fontsize = 20)

	# Add a title to the plot
	plt.title('Squared Residuals vs Gamma z LOS {}'.format(short_simul[j]), fontsize = 20)

	# Force the legend to appear on the plot
	plt.legend(loc = 2)

	# Save the figure using the given filename and format
	plt.savefig(save_loc + 'Resid_gamma_z_{}.png'.format(short_simul[j]), format = 'png')

	# Print a message to the screen to show that the plot of the residuals as a 
	# function of gamma has been saved
	print 'Plot of the residuals as a function of gamma saved z for {}'.format(short_simul[j])

	# Close the figure, now that it has been saved.
	plt.close()

# When the code reaches this point, the skewness has been calculated for all
# simulations, and all methods of calculating the skewness. Plots of skewness
# against gamma have been produced for every simulation.

# Print a message to show that all plots were produced successfully
print 'All plots produced successfully'
		