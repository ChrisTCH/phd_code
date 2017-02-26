#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code is a script that produces Figure 8 of the paper on the   #
# polarisation gradients of the CGPS. The figure has 6 panels, showing the    #
# skewness of the polarisation gradients of the high latitude extension of    #
# the CGPS at 6 different resolutions.                                        #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 18/2/2016                                                       #
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
filename = '/Users/chrisherron/Documents/PhD/My_Papers/CGPS_Polar_Grad/fig12.png'

# Set the dpi at which to save the image
save_dpi = 100

# Set the convention for cartesian co-ordinates used for the CGPS
convention = 'wells'

# Set the colour scale to use with the images
colour = 'cubehelix'

# Set the intensity scaling to use with the images
stretch = 'linear'

# Set the minimum intensity value to be included in the colour scale
vmin = 0

# Set the maximum intensity value to be included in the colour scale
vmax = 0.9

# Open the FITS file that contains the skewness image for the
# high latitude extension at 75 arcsecond resolution
cgps_skew_75_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_all_mask_trunc1_skew_sparse/'+\
 'Polar_Grad_high_lat_all_mask_smooth2_75_skewness_sparse.fits')

# Open the FITS file that contains the skewness image for the
# high latitude extension at 105 arcsecond resolution
cgps_skew_105_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_all_mask_trunc1_skew_sparse/'+\
 'Polar_Grad_high_lat_all_mask_smooth2_105_skewness_sparse.fits')

# Open the FITS file that contains the skewness image for the
# high latitude extension at 150 arcsecond resolution
cgps_skew_150_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_all_mask_trunc1_skew_sparse/'+\
 'Polar_Grad_high_lat_all_mask_smooth2_150_skewness_sparse.fits')

# Open the FITS file that contains the skewness image for the
# high latitude extension at 240 arcsecond resolution
cgps_skew_240_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_all_mask_trunc1_skew_sparse/'+\
 'Polar_Grad_high_lat_all_mask_smooth2_240_skewness_sparse.fits')

# Open the FITS file that contains the skewness image for the
# high latitude extension at 480 arcsecond resolution
cgps_skew_480_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_all_mask_trunc1_skew_sparse/'+\
 'Polar_Grad_high_lat_all_mask_smooth2_480_skewness_sparse.fits')

# Open the FITS file that contains the skewness image for the
# high latitude extension at 1200 arcsecond resolution
cgps_skew_1200_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_all_mask_trunc1_skew_sparse/'+\
 'Polar_Grad_high_lat_all_mask_smooth2_1200_skewness_sparse.fits')

# Create a figure that will be used to hold all of the subplots
fig = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 75 Resolution ------------------------------

# Add an image of the skewness at 75 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_skew_75_fits, figure=fig, subplot=[0.1,0.67,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig1.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin = vmin, vmax=vmax, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig1.add_grid()
# Change the grid lines to be more transparent
fig1.grid.set_alpha(1)
# Set the grid lines to be dashed
fig1.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig1.grid.set_linewidth(0.5)

# Add circles representing the evaluation areas
fig1.show_circles([104.1,109.7],[6.7,6.8],[0.6,0.7], edgecolor = 'r')

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
fig1.ticks.set_yspacing(2.5)

# Hide the x axis label and ticks for this figure
fig1.hide_xaxis_label()
fig1.hide_xtick_labels()
fig1.ticks.set_xspacing(2.5)

# Add a label to show the resolution of the panel
fig1.add_label(0.5, 1.05, '(a) 75\"', relative=True)

#------------------------------ 105 Resolution ---------------------------------

# Add an image of the skewness at 105 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_skew_105_fits, figure=fig, subplot=[0.535,0.67,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig2.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin, vmax=vmax, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig2.add_grid()
# Change the grid lines to be more transparent
fig2.grid.set_alpha(1)
# Set the grid lines to be dashed
fig2.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig2.grid.set_linewidth(0.5)

# Add circles representing the evaluation areas
fig2.show_circles([104.1,109.7],[6.7,6.8],[0.6,0.7], edgecolor = 'r')

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
fig2.hide_ytick_labels()
fig2.ticks.set_yspacing(2.5)

# Hide the x axis label and ticks for this figure
fig2.hide_xaxis_label()
fig2.hide_xtick_labels()
fig2.ticks.set_xspacing(2.5)

# Add a label to show the resolution of the panel
fig2.add_label(0.5, 1.05, '(b) 105\" ', relative=True)

#------------------------------ 150 Resolution ---------------------------------

# Add an image of the skewness at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_fits, figure=fig, subplot=[0.1,0.43,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig3.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig3.show_colorscale(vmin = vmin, vmax=vmax, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig3.add_grid()
# Change the grid lines to be more transparent
fig3.grid.set_alpha(1)
# Set the grid lines to be dashed
fig3.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig3.grid.set_linewidth(0.5)

# Add circles representing the evaluation areas
fig3.show_circles([104.1,109.7],[6.7,6.8],[0.6,0.7], edgecolor = 'r')

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
fig3.ticks.set_yspacing(2.5)

# Hide the x axis label and ticks for this figure
fig3.hide_xaxis_label()
fig3.hide_xtick_labels()
fig3.ticks.set_xspacing(2.5)

# Add a label to show the resolution of the panel
fig3.add_label(0.5, 1.05, '(c) 150\" ', relative=True)

#------------------------------ 240 Resolution ---------------------------------

# Add an image of the skewness at 240 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig4 = aplpy.FITSFigure(cgps_skew_240_fits, figure=fig, subplot=[0.535,0.43,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig4.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig4.show_colorscale(vmin = vmin, vmax=vmax, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig4.add_grid()
# Change the grid lines to be more transparent
fig4.grid.set_alpha(1)
# Set the grid lines to be dashed
fig4.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig4.grid.set_linewidth(0.5)

# Add circles representing the evaluation areas
fig4.show_circles([104.1,109.7],[6.7,6.8],[0.6,0.7], edgecolor = 'r')

# Add a colour bar to the image
fig4.add_colorbar()
# Set the colour bar to the right of the panel
fig4.colorbar.set_location('right')
# Make the colour bar be placed right next to the panel
fig4.colorbar.set_pad(0.0)
# Lower the size of the text on the colourbar
fig4.colorbar.set_font(size='small')

# Hide the y axis label and ticks for this figure
fig4.hide_yaxis_label()
fig4.hide_ytick_labels()
fig4.ticks.set_yspacing(2.5)

# Hide the x axis label and ticks for this figure
fig4.hide_xaxis_label()
fig4.hide_xtick_labels()
fig4.ticks.set_xspacing(2.5)

# Add a label to show the resolution of the panel
fig4.add_label(0.5, 1.05, '(d) 240\" ', relative=True)

#------------------------------ 480 Resolution ---------------------------------

# Add an image of the skewness at 480 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig5 = aplpy.FITSFigure(cgps_skew_480_fits, figure=fig, subplot=[0.1,0.19,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig5.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig5.show_colorscale(vmin = vmin, vmax=vmax, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig5.add_grid()
# Change the grid lines to be more transparent
fig5.grid.set_alpha(1)
# Set the grid lines to be dashed
fig5.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig5.grid.set_linewidth(0.5)

# Add circles representing the evaluation areas
fig5.show_circles([104.1,109.7],[6.7,6.8],[0.6,0.7], edgecolor = 'r')

# Add a colour bar to the image
fig5.add_colorbar()
# Set the colour bar to the right of the panel
fig5.colorbar.set_location('right')
# Make the colour bar be placed right next to the panel
fig5.colorbar.set_pad(0.0)
# Lower the size of the text on the colourbar
fig5.colorbar.set_font(size='small')

# Lower the size of the y-axis ticks, and hide the y axis label
fig5.hide_yaxis_label()
fig5.tick_labels.set_yformat('ddd.d')
fig5.tick_labels.set_font(size='small')
fig5.ticks.set_yspacing(2.5)

# Lower the size of the x-axis ticks, and hide the x axis label
fig5.hide_xaxis_label()
fig5.tick_labels.set_xformat('ddd.d')
fig5.ticks.set_xspacing(2.5)

# Add a label to show the resolution of the panel
fig5.add_label(0.5, 1.05, '(e) 480\" ', relative=True)

#------------------------------ 1200 Resolution --------------------------------

# Add an image of the skewness at 1200 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig6 = aplpy.FITSFigure(cgps_skew_1200_fits, figure=fig, subplot=[0.535,0.19,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig6.recenter(108.2,6.0,width=14.5,height=14.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig6.show_colorscale(vmin = vmin, vmax=vmax, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig6.add_grid()
# Change the grid lines to be more transparent
fig6.grid.set_alpha(1)
# Set the grid lines to be dashed
fig6.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig6.grid.set_linewidth(0.5)

# Add circles representing the evaluation areas
fig6.show_circles([104.1,109.7],[6.7,6.8],[0.6,0.7], edgecolor = 'r')

# Add a colour bar to the image
fig6.add_colorbar()
# Set the colour bar to the right of the panel
fig6.colorbar.set_location('right')
# Make the colour bar be placed right next to the panel
fig6.colorbar.set_pad(0.0)
# Lower the size of the text on the colourbar
fig6.colorbar.set_font(size='small')

# Hide the y axis label and ticks for this figure
fig6.hide_yaxis_label()
fig6.hide_ytick_labels()
fig6.ticks.set_yspacing(2.5)

# Lower the size of the x-axis ticks, and hide the x axis label
fig6.hide_xaxis_label()
fig6.tick_labels.set_xformat('ddd.d')
fig6.tick_labels.set_font(size='small')
fig6.ticks.set_xspacing(2.5)

# Add a label to show the resolution of the panel
fig6.add_label(0.5, 1.05, '(f) 1200\" ', relative=True)

#-------------------------------------------------------------------------------

# Add a label to the x-axis
plt.figtext(0.53, 0.175, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.55, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

#-------------------------------------------------------------------------------

# Save the image using the given filename
fig.savefig(filename, dpi = save_dpi, format = 'png', bbox_inches='tight')

# Close all of the figures
plt.close(fig)
fig1.close()
fig2.close()
fig3.close()
fig4.close()
fig5.close()
fig6.close()

# Close all of the FITS files that were opened
cgps_skew_75_fits.close()
cgps_skew_105_fits.close()
cgps_skew_150_fits.close()
cgps_skew_240_fits.close()
cgps_skew_480_fits.close()
cgps_skew_1200_fits.close()