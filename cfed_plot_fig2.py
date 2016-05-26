#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of synchrotron intensity   #
# produced at different times in the evolution of the simulation, and          #
# produces time averaged histograms of the synchrotron intensity values for    #
# different lines of sight. Plots are then produced of the time averaged       #
# histograms, with error bars to indicate the standard deviation of the number #
# of counts in each bin. This code is intended to be used with simulations     #
# produced by Christoph Federrath.                                             #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 2/3/2016                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

# Set the dpi at which to save the image
save_dpi = 300

# Set a variable to hold the number of bins to use in calculating the 
# histograms
num_bins = 20

# Set a variable that holds the number of timesteps we have for the simulations
num_timestep = 5

# Create a variable that controls whether the PDFs are produced on a log scale
log = True

# Depending on whether the log PDFs are being plotted, determine the bin edges
if log == True:
	# Specify the bin edges to use when calculating the histograms of the simulations
	bin_edges = np.linspace(-1.0, 1.0, num = num_bins+1)
else:
	# Specify the bin edges to use when calculating the histograms of the simulations
	bin_edges = np.linspace(1e-10, 5e-9, num = num_bins+1)

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
spec_locs = ['512sM5Bs5886_20/', '512sM5Bs5886_25/', '512sM5Bs5886_30/',\
'512sM5Bs5886_35/', '512sM5Bs5886_40/', '512cM5Bs5886_20/',\
'512cM5Bs5886_25/', '512cM5Bs5886_30/', '512cM5Bs5886_35/', '512cM5Bs5886_40/']

# Create an array of strings, where each string gives the legend label for 
# a corresponding simulation
sim_labels = [r'Sol: $\zeta = 1.0$', r'Comp: $\zeta = 0$']

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

# Create a three dimensional array that will hold all of the information
# for the histograms of synchrotron intensity. The first index gives the 
# simulation, the second gives the line of sight, and the third axis goes along 
# the intensity bins. The x, y and z axes are numbered with indices 0, 1 and 2
# respectively
hist_arr = np.zeros((len(spec_locs), 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the time averaged histograms of synchrotron intensity. The first index 
# gives the simulation, the second gives the line of sight, and the third axis 
# goes along the intensity bins. The x, y and z axes are numbered with indices 
# 0, 1 and 2 respectively
hist_timeavg_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))

# Create a three dimensional array that will hold all of the information
# for the standard deviation of the number of counts in each bin of the 
# histograms of synchrotron intensity. The first index gives the simulation, the
# second gives the line of sight, and the third axis goes along the intensity 
# bins. The x, y and z axes are numbered with indices 0, 1 and 2 respectively
hist_stdev_arr = np.zeros((len(spec_locs)/num_timestep, 3, num_bins))

# Loop over the different simulations that we are using to make the plot
for i in range(len(spec_locs)):
	# Create a string for the full directory path to use in this calculation
	data_loc = simul_loc + spec_locs[i]

	# Loop over the lines of sight, to calculate the histogram for each line
	# of sight
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
		hist_arr[i,j], edges = np.histogram(flat_sync, bins = bin_edges)

		# Close the fits files, to save memory
		sync_fits.close()

		# Print a message to show that the calculation has finished successfully
		# for this simulation
		print 'Histogram calculated for simulation {} LOS {}'.format(\
			spec_locs[i], line_o_sight[j])

# Loop over the simulations, so that we can calculate the time averaged 
# histograms, and the standard error of the mean in each bin
for i in range(num_sims):
	# Calculate the time averaged histograms for the simulations
	hist_timeavg_arr[i] = np.mean(hist_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype = np.float64)

	# Calculate the standard error of the mean number of counts in each bin
	hist_stdev_arr[i] = np.std(hist_arr[i*num_timestep:(i+1)*num_timestep],\
	 axis = 0, dtype=np.float64) / np.sqrt(num_timestep)

# Calculate the centres of all of the bins
hist_bin_cent_arr = (bin_edges[:-1] + bin_edges[1:]) / 2.0

# Calculate the widths for all of the bins to be plotted
hist_bin_width_arr = 1.0 * (bin_edges[1] - bin_edges[0])

# When the code reaches this point, the histograms have been made for every 
# simulation, so start making the final plots.

# -------------------- Histograms of Synchrotron Intensity ---------------------

# Here we want to produce one plot with three subplots. There should be one row
# of subplots, with three subplots in the row.
# The left column will be for a line of sight along the x axis, the centre
# column for a line of sight along the y axis, and the right column will be for
# a line of sight along the z axis.

# Create an array, that specifies the edge colour for each simulation
edge_col_arr = ['k', 'k']

# Create an array, that specifies the face colour for each simulation
face_col_arr = ['b', 'lightsalmon']

# Create an array, that specifies the colour of the error bars for each simulation
ecol_arr = ['b', 'k']

# Create a figure to hold all of the subplots
fig = plt.figure(1, figsize=(11,7), dpi = 300)

# Create an axis for the first subplot to be produced, which is for
# a line of sight along the x axis
ax1 = fig.add_subplot(231)

# Produce histograms for each simulation
for i in range(num_sims):
	plt.bar(hist_bin_cent_arr, hist_timeavg_arr[i,0], align='center',\
	 width=hist_bin_width_arr, alpha = 0.6, yerr = hist_stdev_arr[i,0],\
	 edgecolor = edge_col_arr[i], fc = face_col_arr[i],\
	 ecolor = ecol_arr[i], capsize = 3, error_kw = {'elinewidth':1.5, 'markeredgewidth':1.5})

# Make the x axis tick labels invisible
plt.setp( ax1.get_xticklabels(), visible=False)

# Create an axis for the second subplot to be produced, which is for
# a line of sight along the y axis. Make the y axis limits the same as for the
# x axis plot
ax2 = fig.add_subplot(232, sharey = ax1)

# Produce histograms for each simulation
for i in range(num_sims):
	plt.bar(hist_bin_cent_arr, hist_timeavg_arr[i,1], align='center',\
	 width=hist_bin_width_arr, alpha = 0.6, yerr = hist_stdev_arr[i,1],\
	 edgecolor = edge_col_arr[i], facecolor = face_col_arr[i],\
	 ecolor = ecol_arr[i], capsize = 3, error_kw = {'elinewidth':1.5, 'markeredgewidth':1.5})

# Make the x axis tick labels invisible
plt.setp( ax2.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax2.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for
# a line of sight along the z axis. Make the y axis limits the same as for the 
# x axis plot
ax3 = fig.add_subplot(233, sharey = ax1)

# Produce histograms for each simulation
for i in range(num_sims):
	plt.bar(hist_bin_cent_arr, hist_timeavg_arr[i,2], align='center',\
	 width=hist_bin_width_arr, alpha = 0.6, yerr = hist_stdev_arr[i,2],\
	 edgecolor = edge_col_arr[i], facecolor = face_col_arr[i],\
	 ecolor = ecol_arr[i], capsize = 3, error_kw = {'elinewidth':1.5, 'markeredgewidth':1.5},\
	  label='{}'.format(sim_labels[i]))

# Make the x axis tick labels invisible
plt.setp( ax3.get_xticklabels(), visible=False)

# Make the y axis tick labels invisible
plt.setp( ax3.get_yticklabels(), visible=False)

# Force the legend to appear on the plot
plt.legend(loc=1, fontsize = 13, numpoints=1)

# Add a label to the y-axis
plt.figtext(0.03, 0.7, 'Counts', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add some text to the figure, to label the left plot as figure a
plt.figtext(0.19, 0.94, 'a) x-LOS', fontsize = 18)

# Add some text to the figure, to label the centre plot as figure b
plt.figtext(0.46, 0.94, 'b) y-LOS', fontsize = 18)

# Add some text to the figure, to label the right plot as figure c
plt.figtext(0.74, 0.94, 'c) z-LOS', fontsize = 18)

#------------------------------- Log Counts ---------------------------------

# Create an axis for the first subplot to be produced, which is for
# a line of sight along the x axis
ax4 = fig.add_subplot(234, sharex = ax1)

# Produce histograms for each simulation
for i in range(num_sims):
	plt.bar(hist_bin_cent_arr, hist_timeavg_arr[i,0], align='center',\
	 width=hist_bin_width_arr, alpha = 0.6, yerr = hist_stdev_arr[i,0],\
	 edgecolor = edge_col_arr[i], fc = face_col_arr[i],\
	 ecolor = ecol_arr[i], capsize = 3, error_kw = {'elinewidth':1.5, 'markeredgewidth':1.5})

# Make the y-axis of the plot logarithmic
ax4.set_yscale('log')

# Create an axis for the second subplot to be produced, which is for
# a line of sight along the y axis. Make the y axis limits the same as for the
# x axis plot
ax5 = fig.add_subplot(235, sharex = ax2, sharey = ax4)

# Produce histograms for each simulation
for i in range(num_sims):
	plt.bar(hist_bin_cent_arr, hist_timeavg_arr[i,1], align='center',\
	 width=hist_bin_width_arr, alpha = 0.6, yerr = hist_stdev_arr[i,1],\
	 edgecolor = edge_col_arr[i], facecolor = face_col_arr[i],\
	 ecolor = ecol_arr[i], capsize = 3, error_kw = {'elinewidth':1.5, 'markeredgewidth':1.5})

# Make the y-axis of the plot logarithmic
ax5.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax5.get_yticklabels(), visible=False)

# Create an axis for the third subplot to be produced, which is for
# a line of sight along the z axis. Make the y axis limits the same as for the 
# x axis plot
ax6 = fig.add_subplot(236, sharex = ax3, sharey = ax4)

# Produce histograms for each simulation
for i in range(num_sims):
	plt.bar(hist_bin_cent_arr, hist_timeavg_arr[i,2], align='center',\
	 width=hist_bin_width_arr, alpha = 0.6, yerr = hist_stdev_arr[i,2],\
	 edgecolor = edge_col_arr[i], facecolor = face_col_arr[i],\
	 ecolor = ecol_arr[i], capsize = 3, error_kw = {'elinewidth':1.5, 'markeredgewidth':1.5},\
	  label='{}'.format(sim_labels[i]))

# Make the y-axis of the plot logarithmic
ax6.set_yscale('log')

# Make the y axis tick labels invisible
plt.setp( ax6.get_yticklabels(), visible=False)

# Add a label to the y-axis
plt.figtext(0.03, 0.3, 'Log Counts', ha = 'left', \
	va = 'center', fontsize = 20, rotation = 'vertical')

# Add a label to the x-axis
plt.figtext(0.5, 0.0, r'$\mathcal{I} = \log_{10} \left( \frac{I}{<I>} \right)$',\
 ha = 'center',	va = 'bottom', fontsize = 20)

# Save the figure using the given filename and format
plt.savefig(simul_loc + 'Publication_Plots/fig2.png', dpi = save_dpi, format = 'png')

# Close the figure so that it does not stay in memory
plt.close()