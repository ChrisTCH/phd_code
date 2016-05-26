#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# produces histograms of the synchrotron intensity values for different        #
# lines of sight. Plots are then produced of the histograms. Values of the     #
# mean and standard deviation of the histograms are also given. This code is   #
# intended to be used with simulations produced by Christoph Federrath.        #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 21/1/2016                                                        #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

# Set a variable to hold the number of bins to use in calculating the 
# histograms
num_bins = 30

# Create a variable that controls whether the PDFs are produced on a log scale
log = True

# Create a variable that controls whether the y-axis of the plots is on a 
# log scale
log_counts = True

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create a string for the specific simulated data sets to use in calculations.
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
spec_locs = ['512sM5Bs5886_20/', '512sM5Bs5886_25/', '512cM5Bs5886_20/',\
'512cM5Bs5886_25/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation
sim_labels = ['Sol 20', 'Sol 25', 'Comp 20', 'Comp 25']

# Create a variable that controls whether the line of sight is assumed to be
# along the x, y or z axis of the data cube when constructing the synchrotron
# maps. This can include 'x', 'y', or 'z'. Synchrotron maps are produced for 
# each line of sight included in the array
line_o_sight = ['x', 'y', 'z']

# Create a variable that specifies the gamma values that will be used to produce
# these synchrotron emission maps
gamma = 2.0

# Create a three dimensional array that will hold all of the information
# for the histograms of synchrotron intensity. The first index gives the 
# simulation the second gives the line of sight, and the third axis goes along 
# radius. The x, y and z axes are numbered with indices 0, 1 and 2 respectively
hist_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will just hold the radius values used
# to make all of the histograms. The first axis represents the simulation used, 
# the second represents the line of sight, and the third axis goes over radius.
# The x, y and z axes are numbered with indices 0, 1 and 2 respectively
hist_bin_edges_arr = np.zeros((len(spec_locs), 3, num_bins + 1))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]

	# Loop over the lines of sight, to calculate the correlation function, 
	# structure function and quadrupole ratio for each line of sight
	for j in range(3):
		# Open the FITS file that contains the synchrotron intensity maps for this
		# simulation
		sync_fits = fits.open(data_loc + 'synint_{}_gam{}.fits'.format(line_o_sight[j],gamma))

		# Extract the data for the simulated synchrotron intensities
		sync_data = sync_fits[0].data

		# Print a message to the screen to show that the data has been loaded
		print 'Synchrotron intensity loaded successfully'

		# Flatten the synchrotron intensity map
		flat_sync = sync_data.flatten()

		# If we are producing PDFs on a log scale, then normalise the PDF by
		# dividing by the mean synchrotron intensity, and then calculate the
		# logarithm of the result
		if log == True:
			# In this case we are calculating the PDFs on a log scale, so 
			# calculate the log of the normalised PDFs
			flat_sync = np.log10( flat_sync / np.mean(flat_sync, dtype = np.float64) )

		# Calculate the histogram for the flattened synchrotron intensity map
		hist_arr[i,j], hist_bin_edges_arr[i,j] = np.histogram(flat_sync, bins=num_bins)

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'Histogram calculated for simulation {} LOS {}'.format(\
			spec_locs[i], line_o_sight[j])

# Calculate the centres of all of the bins
hist_bin_cent_arr = (hist_bin_edges_arr[:,:,:-1] + hist_bin_edges_arr[:,:,1:]) / 2.0

# Calculate the widths for all of the bins to be plotted
hist_bin_width_arr = 0.7 * (hist_bin_edges_arr[:,:,1] - hist_bin_edges_arr[:,:,0])

# When the code reaches this point, the histograms have been made for every 
# simulation, so start making the final plots.

# -------------------- Histograms of Synchrotron Intensity ---------------------

# Here we want to produce one plot with six subplots. There should be two rows
# of subplots, with three subplots in each row. The top row will be the 
# histograms for the solenoidal simulation, and the bottom row will be for the
# compressive simulation.
# The left column will be for a line of sight along the x axis, the centre
# column for a line of sight along the y axis, and the right column will be for
# a line of sight along the z axis.

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(10,6), dpi = 300)

# Create an axis for the first subplot to be produced, which is for the
# solenoidal simulation and a line of sight along the x axis
ax1 = fig.add_subplot(231)

# Produce histograms for each solenoidal simulation
plt.bar(hist_bin_cent_arr[0,0], hist_arr[0,0], align='center',\
 width=hist_bin_width_arr[0,0], alpha = 0.6)
plt.bar(hist_bin_cent_arr[1,0], hist_arr[1,0], align='center',\
 width=hist_bin_width_arr[1,0], alpha = 0.6, edgecolor = 'g', facecolor = 'None')

# Make the y-axis of the plot logarithmic if required
if log_counts == True:
	ax1.set_yscale('log')

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for the
# solenoidal simulation and a line of sight along the y axis. Make the y
# axis limits the same as for the x axis plot
ax2 = fig.add_subplot(232, sharey = ax1)

# Produce histograms for each solenoidal simulation
plt.bar(hist_bin_cent_arr[0,1], hist_arr[0,1], align='center',\
 width=hist_bin_width_arr[0,1], alpha = 0.6)
plt.bar(hist_bin_cent_arr[1,1], hist_arr[1,1], align='center',\
 width=hist_bin_width_arr[1,1], alpha = 0.6, edgecolor = 'g', facecolor = 'None')

# Make the y-axis of the plot logarithmic if required
if log_counts == True:
	ax2.set_yscale('log')

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for the
# solenoidal simulation and a line of sight along the z axis. Make the y
# axis limits the same as for the x axis plot
ax3 = fig.add_subplot(233, sharey = ax1)

# Produce histograms for each solenoidal simulation
plt.bar(hist_bin_cent_arr[0,2], hist_arr[0,2], align='center',\
 width=hist_bin_width_arr[0,2], alpha = 0.6, label='{}'.format(sim_labels[0]))
plt.bar(hist_bin_cent_arr[1,2], hist_arr[1,2], align='center', edgecolor = 'g',\
 facecolor = 'None',\
 width=hist_bin_width_arr[1,2], alpha = 0.6, label='{}'.format(sim_labels[1]))

# Make the y-axis of the plot logarithmic if required
if log_counts == True:
	ax3.set_yscale('log')

# Make the x axis tick labels invisible
plt.setp( ax3.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc=1, fontsize = 9, numpoints=1)

# Create an axis for the fourth subplot to be produced, which is for the 
# compressive simulation and a line of sight along the x axis.
# Make the x axis limits the same as for the first plot
ax4 = fig.add_subplot(234, sharex = ax1, sharey = ax1)

# Produce histograms for each solenoidal simulation
plt.bar(hist_bin_cent_arr[2,0], hist_arr[2,0], align='center',\
 width=hist_bin_width_arr[2,0], alpha = 0.6)
plt.bar(hist_bin_cent_arr[3,0], hist_arr[3,0], align='center',\
 width=hist_bin_width_arr[3,0], alpha = 0.6, edgecolor = 'g', facecolor = 'None')

# Make the y-axis of the plot logarithmic if required
if log_counts == True:
	ax4.set_yscale('log')

# Create an axis for the fifth subplot to be produced, which is for the 
# compressive simulation and a line of sight along the y axis.
# Make the x axis limits the same as for the second plot, and the y axis limits
# the same as for the fourth plot
ax5 = fig.add_subplot(235, sharex = ax2, sharey = ax4)

# Produce histograms for each solenoidal simulation
plt.bar(hist_bin_cent_arr[2,1], hist_arr[2,1], align='center',\
 width=hist_bin_width_arr[2,1], alpha = 0.6)
plt.bar(hist_bin_cent_arr[3,1], hist_arr[3,1], align='center',\
 width=hist_bin_width_arr[3,1], alpha = 0.6, edgecolor = 'g', facecolor = 'None')

# Make the y-axis of the plot logarithmic if required
if log_counts == True:
	ax5.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax5.get_yticklabels(), visible=False)

# Create an axis for the sixth subplot to be produced, which is for the 
# compressive simulation and a line of sight along the z axis.
# Make the x axis limits the same as for the third plot, and the y axis limits
# the same as for the fourth plot
ax6 = fig.add_subplot(236, sharex = ax3, sharey = ax4)

# Produce histograms for each solenoidal simulation
plt.bar(hist_bin_cent_arr[2,2], hist_arr[2,2], align='center',\
 width=hist_bin_width_arr[2,2], alpha = 0.6, label='{}'.format(sim_labels[2]))
plt.bar(hist_bin_cent_arr[3,2], hist_arr[3,2], align='center', edgecolor = 'g',\
 facecolor = 'None',\
 width=hist_bin_width_arr[3,2], alpha = 0.6, label='{}'.format(sim_labels[3]))

# Make the y-axis of the plot logarithmic if required
if log_counts == True:
	ax6.set_yscale('log')

# Force the legend to appear on the plot
plt.legend(loc=1, fontsize = 9, numpoints=1)

# Make the y axis tick labels invisible
plt.setp( ax6.get_yticklabels(), visible=False)

# Add a label to the x-axis
plt.figtext(0.5, 0.0, 'Synchrotron Intensity [units]', ha = 'center', \
	va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.03, 0.5, 'Raw Count', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.15, 0.94, 'a) Sol x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure b
plt.figtext(0.42, 0.94, 'b) Sol y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.7, 0.94, 'c) Sol z-LOS', fontsize = 18)

# Add some text to the figure, to label the left plot as figure d
plt.figtext(0.15, 0.475, 'd) Comp x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure e
plt.figtext(0.42, 0.475, 'e) Comp y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure f
plt.figtext(0.7, 0.475, 'f) Comp z-LOS', fontsize = 18)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'sync_int_pdfs_all_sims_time_gam{}_log{}_y{}.eps'.\
	format(gamma,log, log_counts), format = 'eps')

# Close the figure so that it does not stay in memory
plt.close()