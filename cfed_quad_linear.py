#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# calculates the quadrupole ratios of the synchrotron intensity maps, for      #
# different lines of sight. Linear fits to the quadrupole ratios are then      #
# performed on certain radial scales, and the residuals of these fits plotted  #
# against the Alfvenic Mach number. This code is intended to be used with      #
# simulations produced by Christoph Federrath.                                 #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 2/2/2016                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

# Import the functions that calculate the structure and correlation functions
# using FFT, as well as the function that calculates the radially averaged 
# structure or correlation functions. Also import the function that calculates
# multipoles of the 2D structure functions, and the function that calculates the
# magnitude and argument of the quadrupole ratio
from sf_fft import sf_fft
from cf_fft import cf_fft
from sfr import sfr
from calc_multipole_2D import calc_multipole_2D
from calc_quad_ratio import calc_quad_ratio

# Set a variable to hold the number of bins to use in calculating the 
# correlation functions
num_bins = 25

# Set a variable to hold the final index to be used to calculate the standard
# deviation of the first derivative of the quadrupole ratio
end_index = 17

# Set a variable to hold the first index to be used to calculate the standard
# deviation of the first derivative of the quadrupole ratio
first_index = 8

# Set a variable for how many data points should be used to calculate the
# standard deviation of the first derivative of the quadrupole ratio
num_eval_points = end_index - first_index

# Create a string to hold the location to which plots should be saved
save_loc = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create a string to hold the location to which plots of the quadrupole ratio
# moduli should be saved
save_quad = '/Users/chrisherron/Documents/PhD/CFed_2016/quad_linear_fits/'

#------------------------------------------------------------------------------

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use, for Christoph's simulations
simul_loc_cfed = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create a string for the specific simulated data sets to use in calculations.
# This is just for Christoph's simulations
# The directories end in:
# 512sM5Bs5886_20 (Solenoidal turbulence, timestep 20)
# 512sM5Bs5886_25 (Solenoidal turbulence, timestep 25)
# 512sM5Bs5886_30 (Solenoidal turbulence, timestep 30)
# 512sM5Bs5886_35 (Solenoidal turbulence, timestep 35)
# 512sM5Bs5886_40 (Solenoidal turbulence, timestep 40)
# 512cM5Bs5886_20 (Compressive turbulence, timestep 20)
# 512cM5Bs5886_25 (Compressive turbulence, timestep 25)
# 512cM5Bs5886_30 (Compressive turbulence, timestep 30)
# 512cM5Bs5886_35 (Compressive turbulence, timestep 35)
# 512cM5Bs5886_40 (Compressive turbulence, timestep 40)
spec_locs_cfed = ['512sM5Bs5886_20/', '512sM5Bs5886_25/', '512sM5Bs5886_30/',\
'512sM5Bs5886_35/', '512sM5Bs5886_40/', '512cM5Bs5886_20/', '512cM5Bs5886_25/',\
'512cM5Bs5886_30/', '512cM5Bs5886_35/', '512cM5Bs5886_40/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation, for Christoph's simulations
sim_labels_cfed = ['Sol20', 'Sol25', 'Sol30', 'Sol35', 'Sol40', 'Comp 20',\
 'Comp 25', 'Comp 30', 'Comp 35', 'Comp 40',]

# Create an array, where each entry specifies the calculated Alfvenic Mach 
# number for each simulation, for Christoph's simulations
alf_mach_arr_cfed = np.array([2.0,2.1,2.1,2.1,2.1, 2.3,2.0,2.0,2.0,2.0])

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube. This can include 'x', 'y', or 'z'. 
# Only use lines of sight perpendicular to the mean magnetic field.
# For Christoph's simulations
line_o_sight_cfed = ['x', 'y']

#------------------------------------------------------------------------------

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. For Blakesley's simulations.
simul_loc_bbur = '/Volumes/CAH_ExtHD/Madison_2014/Simul_Data/'

# Create a string for the specific simulated data set to use in calculations.
# For Blakesley's simulations
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

# Create a list, where each entry is a string describing the directory 
# containing the current simulation to study, for Blakesley's simulations
spec_locs_bbur = ['c512b.1p.0049/', 'c512b.1p.0077/', 'b.1p.01_Oct_Burk/', \
'c512b.1p.025/', 'c512b.1p.05/', 'b.1p.1_Oct_Burk/', 'c512b.1p.7/', \
'b.1p2_Aug_Burk/', 'c512b1p.0049/', 'c512b1p.0077/', 'b1p.01_Oct_Burk/',\
'c512b1p.025/', 'c512b1p.05/', 'b1p.1_Oct_Burk/', 'c512b1p.7/', \
'b1p2_Aug_Burk/', 'c512b3p.01/', 'c512b5p.01/']

# Create a list, where each entry is a string describing the initial magnetic
# field and pressure used to run each simulation, for Blakesley's simulations
short_labels_bbur = ['b.1p.0049', 'b.1p.0077', 'b.1p.01', 'b.1p.025', 'b.1p.05',\
'b.1p.1', 'b.1p.7', 'b.1p2', 'b1p.0049', 'b1p.0077', 'b1p.01', 'b1p.025',\
'b1p.05', 'b1p.1', 'b1p.7', 'b1p2', 'b3p.01', 'b5p.01']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# low magnetic field simulations used to produce plots
low_B_short_M = ['Ms7.02Ma1.76', 'Ms2.38Ma1.86', 'Ms0.83Ma1.74', 'Ms0.45Ma1.72']

# Create strings giving the simulation codes in terms of Mach numbers, for the
# high magnetic field simulations used to produce plots
high_B_short_M = ['Ms6.78Ma0.52', 'Ms2.41Ma0.67', 'Ms0.87Ma0.7', 'Ms0.48Ma0.65']

# Create an array, where each entry specifies the calculated Alfvenic Mach 
# number for each simulation, for Blakesley's simulations
alf_mach_arr_bbur = np.array([1.41278383, 1.77294593, 1.75575508, 1.50830194,\
 1.69455875, 1.85993991, 1.74231524, 1.71939152, 0.49665052, 0.50288954,\
 0.51665006, 0.54928564, 0.57584022, 0.67145057, 0.70015313, 0.65195539,\
 0.21894299, 0.14357068])

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube. This can include 'x', 'y', or 'z'. 
# Only use lines of sight perpendicular to the mean magnetic field.
# For Blakesley's simulations
line_o_sight_bbur = ['z', 'y']

#------------------------------------------------------------------------------

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Set the index of the gamma array, that is needed to obtain the synchrotron 
# intensity map produced for this value of gamma in Blakesley's simulations
gam_index = 2

# Create an empty array, where each entry specifies the calculated residuals
# for the linear fit to the quadrupole ratio modulus for the corresponding 
# simulation, for a particular value of gamma. The first index gives the 
# simulation, and the second index gives the line of sight.
quad_resid_arr_cfed = np.zeros((len(spec_locs_cfed),2))
quad_resid_arr_bbur = np.zeros((len(spec_locs_bbur),2))

# Loop over Christoph's simulations that we are using to make the plot
for i in range(len(spec_locs_cfed)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc_cfed + spec_locs_cfed[i]

	# Loop over the lines of sight, to calculate the quadrupole ratio for each 
	# line of sight
	for j in range(2):
		# Open the FITS file that contains the synchrotron intensity maps for this
		# simulation
		sync_fits = fits.open(data_loc + 'synint_{}_gam{}.fits'.format(\
			line_o_sight_cfed[j],gamma))

		# Extract the data for the simulated synchrotron intensities
		sync_data = sync_fits[0].data

		# Print a message to the screen to show that the data has been loaded
		print 'Synchrotron intensity loaded successfully'

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn = sf_fft(sync_data, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image.
		norm_strfn = np.fft.fftshift(norm_strfn)

		# Calculate the magnitude and argument of the quadrupole ratio
		quad_mod, quad_arg, quad_rad = calc_quad_ratio(norm_strfn, num_bins)

		# Perform a linear fit to the quadrupole ratio on the radial scales
		# that have been specified
		spec_ind_data = np.polyfit(np.log10(quad_rad[first_index:end_index]),\
			quad_mod[first_index:end_index], 1, full = True)

		# Enter the value of the residuals into the corresponding array
		quad_resid_arr_cfed[i,j] = spec_ind_data[1]

		# Extract the returned coefficients from the polynomial fit
		coeff = spec_ind_data[0]

		#--------------------- Fitting Line of Best Fit ------------------------

		# Extract the slope of the structure function
		m_val = coeff[0]

		# Extract the intercept of the linear fit
		intercept = coeff[1]

		# Calculate the y values of the line of best fit, to use in plotting
		fit_line = m_val * np.log10(quad_rad) + intercept

		# Create a figure, to plot the quadrupole ratio modulus with line of 
		# best fit
		fig = plt.figure()

		# Create an axis for the first subplot to be produced, which is for the
		# x line of sight
		ax = fig.add_subplot(111)

		# Plot the quadrupole ratio modulus for this simulation, for this 
		# line of sight
		plt.plot(quad_rad, quad_mod, '-o', label = '{}'.format(sim_labels_cfed[i]))

		# Plot the line of best fit to the quadrupole ratio modulus
		plt.plot(quad_rad, fit_line, '--', label = 'Best Fit')

		# Plot a rectangular box around the region used to produce the fit
		plt.axvspan(quad_rad[first_index], quad_rad[end_index - 1], facecolor = 'r', alpha = 0.5)

		# Make the x axis of the plot logarithmic
		ax.set_xscale('log')

		# Force the legend to appear on the plot
		plt.legend(fontsize = 8, numpoints=1)

		# Add a label to the x-axis
		plt.xlabel('Radial Separation [pixels]', fontsize = 20)

		# Add a label to the y-axis
		plt.ylabel('Quadrupole Ratio', fontsize = 20)

		# Add a title to the figure
		plt.title('Quad Ratio {} {} Alf: {}'.format(sim_labels_cfed[i],\
			line_o_sight_cfed[j], alf_mach_arr_cfed[i]))

		# Save the figure using the given filename and format
		plt.savefig(save_quad + 'quad_ratio_{}_{}LOS_gam{}.eps'.format(\
			sim_labels_cfed[i], line_o_sight_cfed[j], gamma), format = 'eps')

		# Close the figure so that it does not stay in memory
		plt.close()

		#-----------------------------------------------------------------------

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			spec_locs_cfed[i], line_o_sight_cfed[j])

# Loop over Blakesley's simulations that we are using to make the plot
for i in range(len(spec_locs_bbur)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc_bbur + spec_locs_bbur[i]

	# Loop over the lines of sight, to calculate the quadrupole ratio for each 
	# line of sight
	for j in range(2):
		# Open the FITS file that contains the synchrotron intensity maps for this
		# simulation
		if line_o_sight_bbur[j] == 'z':
			# Open the FITS file for a line of sight along the z axis
			sync_fits = fits.open(data_loc + 'synint_p1-4.fits'.format(\
				line_o_sight_bbur[j]))
		elif line_o_sight_bbur[j] == 'y':
			# Open the FITS file for a line of sight along the y axis
			sync_fits = fits.open(data_loc + 'synint_p1-4y.fits'.format(\
				line_o_sight_bbur[j]))

		# Extract the data for the simulated synchrotron intensities
		sync_data = sync_fits[0].data

		# Extract the slice for the value of gamma
		sync_data = sync_data[gam_index]

		# Print a message to the screen to show that the data has been loaded
		print 'Synchrotron intensity loaded successfully'

		# Calculate the 2D structure function for this slice of the synchrotron
		# intensity data cube. Note that no_fluct = True is set, because we are
		# not subtracting the mean from the synchrotron maps before calculating
		# the structure function. We are also calculating the normalised 
		# structure function, which only takes values between 0 and 2.
		norm_strfn = sf_fft(sync_data, no_fluct = True, normalise = True)

		# Shift the 2D structure function so that the zero radial separation
		# entry is in the centre of the image.
		norm_strfn = np.fft.fftshift(norm_strfn)

		# Calculate the magnitude and argument of the quadrupole ratio
		quad_mod, quad_arg, quad_rad = calc_quad_ratio(norm_strfn, num_bins)

		# Perform a linear fit to the quadrupole ratio on the radial scales
		# that have been specified
		spec_ind_data = np.polyfit(np.log10(quad_rad[first_index:end_index]),\
			quad_mod[first_index:end_index], 1, full = True)

		# Enter the value of the residuals into the corresponding array
		quad_resid_arr_bbur[i,j] = spec_ind_data[1]

		# Extract the returned coefficients from the polynomial fit
		coeff = spec_ind_data[0]

		#--------------------- Fitting Line of Best Fit ------------------------

		# Extract the slope of the structure function
		m_val = coeff[0]

		# Extract the intercept of the linear fit
		intercept = coeff[1]

		# Calculate the y values of the line of best fit, to use in plotting
		fit_line = m_val * np.log10(quad_rad) + intercept

		# Create a figure, to plot the quadrupole ratio modulus with line of 
		# best fit
		fig = plt.figure()

		# Create an axis for the first subplot to be produced, which is for the
		# x line of sight
		ax = fig.add_subplot(111)

		# Plot the quadrupole ratio modulus for this simulation, for this 
		# line of sight
		plt.plot(quad_rad, quad_mod, '-o', label = '{}'.format(short_labels_bbur[i]))

		# Plot the line of best fit to the quadrupole ratio modulus
		plt.plot(quad_rad, fit_line, '--', label = 'Best Fit')

		# Plot a rectangular box around the region used to produce the fit
		plt.axvspan(quad_rad[first_index], quad_rad[end_index - 1], facecolor = 'r', alpha = 0.5)

		# Make the x axis of the plot logarithmic
		ax.set_xscale('log')

		# Force the legend to appear on the plot
		plt.legend(fontsize = 8, numpoints=1)

		# Add a label to the x-axis
		plt.xlabel('Radial Separation [pixels]', fontsize = 20)

		# Add a label to the y-axis
		plt.ylabel('Quadrupole Ratio', fontsize = 20)

		# Add a title to the figure
		plt.title('Quad Ratio {} {} Alf: {}'.format(short_labels_bbur[i],\
			line_o_sight_bbur[j], alf_mach_arr_bbur[i]))

		# Save the figure using the given filename and format
		plt.savefig(save_quad + 'quad_ratio_{}_{}LOS_gam{}.eps'.format(\
			short_labels_bbur[i], line_o_sight_bbur[j], gamma), format = 'eps')

		# Close the figure so that it does not stay in memory
		plt.close()

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'All statistics calculated for simulation {} LOS {}'.format(\
			spec_locs_bbur[i], line_o_sight_bbur[j])

# When the code reaches this point, the quadrupole ratio has been saved for every 
# simulation, and every line of sight, so start making the final plots.

#-------------------------------------------------------------------------------

# Create a figure to plot the residuals of the linear fit to the
# quadrupole ratio modulus as a function of the Alfvenic Mach number
fig3 = plt.figure(3, figsize=(7,5), dpi = 300)

# Create an axis for the plot
ax9 = fig3.add_subplot(111)

# Plot the values of the residuals of the linear fit for 
# Christoph and Blakesley's simulations
plt.plot(alf_mach_arr_bbur, quad_resid_arr_bbur[:,0], 'o', label = 'BB zLOS')
plt.plot(alf_mach_arr_bbur, quad_resid_arr_bbur[:,1], 'o', label = 'BB yLOS')
plt.plot(alf_mach_arr_cfed, quad_resid_arr_cfed[:,0], '^', label = 'CF xLOS')
plt.plot(alf_mach_arr_cfed, quad_resid_arr_cfed[:,1], '^', label = 'CF yLOS')

# Force the legend to appear on the plot
plt.legend(loc = 2, fontsize = 8, numpoints=1)

# Add labels to the axes
plt.xlabel('Alfvenic Mach Number', fontsize = 18)
plt.ylabel('Residuals Fit Quad Ratio', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(save_loc + 'quad_resid_alf_gam{}_{}_{}.eps'.format(gamma,\
	first_index, end_index), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()

#-------------------------------------------------------------------------------

# Now that all of the statistics have been calculated, print them out to the 
# screen. Loop over all of the lines of sight, and the different simulations,
# and print out results for the simulations
for j in range(2):
	# For this line of sight, loop over Christoph's simulations
	for i in range(len(spec_locs_cfed)):
		# Print out the value of the mean for this line of sight
		print "{} {} LOS Quad Resid: {}".format(sim_labels_cfed[i],\
		 line_o_sight_cfed[j], quad_resid_arr_cfed[i,j])

# Loop over Blakesley's simulations
for j in range(2):
	# For this line of sight, loop over the simulations
	for i in range(len(spec_locs_bbur)):
		# Print out the value of the mean for this line of sight
		print "{} {} LOS Quad Resid: {}".format(short_labels_bbur[i],\
		 line_o_sight_bbur[j], quad_resid_arr_bbur[i,j])