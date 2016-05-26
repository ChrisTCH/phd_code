#------------------------------------------------------------------------------#
#                                                                              #
# This code is a Python script that reads in arrays of the x, y and z          #
# components of the magnetic field, and saves images of the magnetic field     #
# vectors for slices of the full cube.                                         #
#                                                                              #
# Author: Chris Herron                                                         #
# Start Date: 1/3/2016                                                         #
#                                                                              #
#------------------------------------------------------------------------------#

# First import numpy for array handling, matplotlib for plotting, astropy.io
# for fits manipulation, scipy.stats for calculating statistical quantities
import numpy as np
import matplotlib.pyplot as plt
import h5py
from mayavi import mlab

# Create a string for the directory that contains the simulated magnetic fields
# and synchrotron intensity maps to use. 
simul_loc = '/Users/chrisherron/Documents/PhD/CFed_2016/'

# Create an variable that gives the timestep of the simulation to use
timestep = 20

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
spec_loc = '512cM5Bs5886_{}/'.format(timestep)

# Create a string for the full directory path to use in this calculation
data_loc = simul_loc + spec_loc

# Load the data files for the x, y and z components of the magnetic field
mag_x_hdf = h5py.File(data_loc+'OUT_hdf5_plt_cnt_00{}_magx'.format(timestep),'r')
mag_y_hdf = h5py.File(data_loc+'OUT_hdf5_plt_cnt_00{}_magy'.format(timestep),'r')
mag_z_hdf = h5py.File(data_loc+'OUT_hdf5_plt_cnt_00{}_magz'.format(timestep),'r')

# Extract the magnetic field arrays from the data files
mag_x = mag_x_hdf['magx']
mag_y = mag_y_hdf['magy']
mag_z = mag_z_hdf['magz']

# Calculate the amplitude of the magnetic field
B_mag = np.sqrt(np.power(mag_x,2) + np.power(mag_y,2) + np.power(mag_z,2))

# Create arrays of the x, y and z co-ordinates of each element of the array 
X, Y, Z = np.mgrid[0:512,0:512,0:512]

# Create an array of values that specify the indices along the x axis of the 
# array, at which images of the vector field for that slice will be saved
slice_arr = np.linspace(0,500,num=26, endpoint = True)

# Set the maximum and minimum values to go on the colourbar
vmax = 6e-5
vmin = 0

# Loop over the values of the slice array, to produce an image of the vector 
# field for each slice
for i in range(len(slice_arr)):
	# Create a new figure 
	fig = mlab.figure(size = (800,700))

	# Create an image of the vector field for this slice of the array
	mlab.quiver3d(X[slice_arr[i],::8,::8], Y[slice_arr[i],::8,::8],\
	 Z[slice_arr[i],::8,::8], mag_x[slice_arr[i],::8,::8],\
	 mag_y[slice_arr[i],::8,::8],mag_z[slice_arr[i],::8,::8],scalars =\
	  B_mag[slice_arr[i],::8,::8], vmax=vmax,vmin=vmin)

	# Make the co-ordinate axes appear on the plot 
	mlab.xlabel('x-axis')
	mlab.ylabel('y-axis')
	mlab.zlabel('z-axis')

	# Rotate the camera so that it is face on
	mlab.view(azimuth=0, elevation = 90)

	# Add a colourbar to the figure
	mlab.colorbar(orientation='vertical')

	# Save the figure
	mlab.savefig(data_loc + 'B_vectors_slice_{}.jpg'.format(slice_arr[i]))

	# Close the figure
	mlab.close()

	# Print a line to the screen to show that the figure has been saved
	print 'Figure saved for slice number {}'.format(i)