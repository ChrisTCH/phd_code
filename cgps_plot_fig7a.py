#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code is a script that produces Figure 7a of the paper on the  #
# polarisation gradients of the CGPS. The figure has 3 panels, which will show#
# the skewness of the polarisation gradient for different box sizes.          #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 22/8/2016                                                       #
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
data_loc = '/Users/chrisherron/Documents/PhD/CGPS_2015/'

# Create a string that will be used to save the figure
filename = '/Users/chrisherron/Documents/PhD/My_Papers/CGPS_Polar_Grad/'

# Set the dpi at which to save the image
save_dpi = 100

# Set the convention for cartesian co-ordinates used for the CGPS
convention = 'wells'

# Set the colour scale to use with the skewness
colour_s = 'cubehelix'

# Set the intensity scaling to use with the skewness
stretch_s = 'linear'

# Set the minimum intensity value to be included in the colour scale, for the
# skewness
vmin_s = 0

# Set the maximum intensity value to be included in the colour scale, for the
# skewness
vmax_s = 1.2

# Open the FITS file that contains the skewness of the polarisation gradient
# smoothed to an angular resolution of 150 arcseconds, with 1% truncation,
# and 20 beams across half the box, for the entire Galactic plane.
cgps_skew_150_20_fits = fits.open(data_loc +\
 'Polar_Grad_plane_all_mask_trunc1_beams20_skew_sparse/'+\
 'Polar_Grad_plane_all_mask_smooth2_150_beams20_skewness_sparse.fits')

# Open the FITS file that contains the skewness of the polarisation gradient
# smoothed to an angular resolution of 150 arcseconds, with 1% truncation,
# and 40 beams across half the box, for the entire Galactic plane.
cgps_skew_150_40_fits = fits.open(data_loc +\
 'Polar_Grad_plane_all_mask_trunc1_beams40_skew_sparse/'+\
 'Polar_Grad_plane_all_mask_smooth2_150_beams40_skewness_sparse.fits')

# Open the FITS file that contains the skewness of the polarisation gradient
# smoothed to an angular resolution of 150 arcseconds, with 1% truncation,
# and 60 beams across half the box, for the entire Galactic plane.
cgps_skew_150_60_fits = fits.open(data_loc +\
 'Polar_Grad_plane_all_mask_trunc1_beams60_skew_sparse/'+\
 'Polar_Grad_plane_all_mask_smooth2_150_beams60_skewness_sparse.fits')

#------------------------------ Figure a ---------------------------------------

# Create a figure that will be used to hold all of the subplots
fig_a = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 141 < l < 169 ----------------------------------

# Add an image of the skewness measured when the box has 20 beams to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_skew_150_20_fits, figure=fig_a, subplot=[0.1,0.691,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig1.recenter(155.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=vmin_s, vmax=vmax_s, cmap = colour_s, stretch = stretch_s)

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
fig1.tick_labels.set_yformat('ddd.d')
fig1.tick_labels.set_font(size='small')

# Lower the size of the x-axis ticks, and hide the x axis label
fig1.hide_xaxis_label()
fig1.hide_xtick_labels()
fig1.ticks.set_xspacing(5)

# Add an image of the skewness measured when the box has 40 beams to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_skew_150_40_fits, figure=fig_a, subplot=[0.1,0.535,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig2.recenter(155.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin_s,vmax=vmax_s,cmap=colour_s, stretch=stretch_s)

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

# Lower the size of the y-axis ticks, and hide the y axis label
fig2.hide_yaxis_label()
fig2.tick_labels.set_yformat('ddd.d')
fig2.tick_labels.set_font(size='small')

# Hide the x axis label and ticks for this figure
fig2.hide_xaxis_label()
fig2.hide_xtick_labels()
fig2.ticks.set_xspacing(5)

# Add an image of the skewness measured when the box has 60 beams to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_60_fits, figure=fig_a, subplot=[0.1,0.38,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig3.recenter(155.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig3.show_colorscale(vmin=vmin_s, vmax=vmax_s, cmap = colour_s, stretch = stretch_s)

# Add a co-ordinate grid to the image
fig3.add_grid()
# Change the grid lines to be more transparent
fig3.grid.set_alpha(1)
# Set the grid lines to be dashed
fig3.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig3.grid.set_linewidth(0.5)

# Add a colour bar to the image
fig3.add_colorbar()
# Set the colour bar to the right of the panel
fig3.colorbar.set_location('right')
# Make the colour bar be placed right next to the panel
fig3.colorbar.set_pad(0.0)
# Lower the size of the text on the colourbar
fig3.colorbar.set_font(size='small')

# Lower the size of the y-axis ticks, and hide the y axis label
fig3.hide_yaxis_label()
fig3.tick_labels.set_yformat('ddd.d')
fig3.tick_labels.set_font(size='small')

# Lower the size of the x-axis ticks, and hide the x axis label
fig3.hide_xaxis_label()
fig3.tick_labels.set_xformat('ddd.d')
fig3.ticks.set_xspacing(5)

# Add a label to the x-axis
plt.figtext(0.53, 0.38, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.65, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

# Save the image using the given filename
fig_a.savefig(filename + 'fig13.png', dpi = save_dpi, format = 'png', bbox_inches='tight')

# Close all of the figures
plt.close(fig_a)
fig1.close()
fig2.close()
fig3.close()

#-------------------------------------------------------------------------------

# Close all of the FITS files that were opened
cgps_skew_150_20_fits.close()
cgps_skew_150_40_fits.close()
cgps_skew_150_60_fits.close()