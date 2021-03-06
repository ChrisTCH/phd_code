#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code is a script that produces Figure 4 of the paper on the   #
# polarisation gradients of the CGPS. The figure has 2 panels, showing the    #
# polarisation gradients of the high latitude extension of the CGPS at 150    #
# arc second resolution, compared to Stokes I.                                #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 19/2/2016                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, astropy.io.fits, matplotlib and aplpy
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import aplpy

# Create a string object which stores the directory of the CGPS data
data_loc = '/Volumes/CAH_ExtHD/CGPS_2015/'

# Create a string that will be used to save the figure
filename = '/Users/chrisherron/Documents/PhD/My_Papers/CGPS_Polar_Grad/fig9.png'

# Set the dpi at which to save the image
save_dpi = 100

# Set the convention for cartesian co-ordinates used for the CGPS
convention = 'wells'

# Set the colour scale to use with the polarisation gradient
colour_p = 'viridis'

# Set the intensity scaling to use with the polarisation gradient
stretch_p = 'linear'

# Set the colour scale to use with the total intensity
colour_i = 'magma'

# Set the intensity scaling to use with the total intensity
stretch_i = 'linear'

# Set the minimum intensity value to be included in the colour scale, for the
# polarisation gradient
vmin_p = 0

# Set the maximum intensity value to be included in the colour scale, for the
# polarisation gradient
vmax_p = 5.0

# Set the minimum intensity value to be included in the colour scale, for the
# total intensity
vmin_i = 3.5

# Set the maximum intensity value to be included in the colour scale, for the
# total intensity
vmax_i = 9.0

# Open the FITS file that contains the total intensity image for the
# Galactic plane
cgps_I_fits = fits.open(data_loc + 'Sto_I_high_lat.fits')

# Open the FITS file that contains the polarisation gradient image for the
# high latitude extension at 150 arcsecond resolution
cgps_gradP_150_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_smoothed/Polar_Grad_high_lat_smooth2_150.fits')

# Create a figure that will be used to hold all of the subplots
fig = plt.figure(figsize=(9,9), dpi=save_dpi)

#------------------------------ 150 Resolution ---------------------------------

# Add an image of the polarisation gradient at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig, subplot=[0.55,0.1,0.4,0.8],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig1.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=vmin_p, vmax=vmax_p, cmap=colour_p, stretch=stretch_p)

# Add a co-ordinate grid to the image
fig1.add_grid()
# Change the grid lines to be more transparent
fig1.grid.set_alpha(1)
# Set the grid lines to be dashed
fig1.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig1.grid.set_linewidth(0.5)

# Add a colour bar to the image
fig1.add_colorbar()
# Set the colour bar to the right of the panel
fig1.colorbar.set_location('right')
# Make the colour bar be placed right next to the panel
fig1.colorbar.set_pad(0.0)
# Lower the size of the text on the colourbar
fig1.colorbar.set_font(size='small')

# Lower the size of the y-axis ticks, and hide the y axis label
fig1.hide_yaxis_label()
fig1.hide_ytick_labels()
fig1.tick_labels.set_font(size='small')
fig1.ticks.set_yspacing(2.5)

# Lower the size of the x-axis ticks, and hide the x axis label
fig1.hide_xaxis_label()
fig1.tick_labels.set_xformat('ddd.d')
fig1.ticks.set_xspacing(2.5)

#------------------------------ Total Intensity --------------------------------

# Add an image of the total intensity to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_I_fits, figure=fig, subplot=[0.1,0.1,0.4,0.8],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig2.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin=vmin_i, vmax=vmax_i, cmap=colour_i, stretch=stretch_i)

# Add a co-ordinate grid to the image
fig2.add_grid()
# Change the grid lines to be more transparent
fig2.grid.set_alpha(1)
# Set the grid lines to be dashed
fig2.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig2.grid.set_linewidth(0.5)

# Add a colour bar to the image
fig2.add_colorbar()
# Set the colour bar to the right of the panel
fig2.colorbar.set_location('right')
# Make the colour bar be placed right next to the panel
fig2.colorbar.set_pad(0.0)
# Lower the size of the text on the colourbar
fig2.colorbar.set_font(size='small')

# Hide the y axis label and ticks for this figure
fig2.hide_yaxis_label()
fig2.tick_labels.set_yformat('ddd.d')
fig2.tick_labels.set_font(size='small')
fig2.ticks.set_yspacing(2.5)

# Lower the size of the x-axis ticks, and hide the x axis label
fig2.hide_xaxis_label()
fig2.tick_labels.set_xformat('ddd.d')
fig2.ticks.set_xspacing(2.5)

#-------------------------------------------------------------------------------

# Add a label to the x-axis
plt.figtext(0.53, 0.25, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.5, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

#-------------------------------------------------------------------------------

# Save the image using the given filename
fig.savefig(filename, dpi = save_dpi, format = 'png', bbox_inches='tight')

# Close all of the figures
plt.close(fig)
fig1.close()
fig2.close()

# Close all of the FITS files that were opened
cgps_I_fits.close()
cgps_gradP_150_fits.close()