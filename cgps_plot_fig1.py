#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code is a script that produces Figure 1 of the paper on the   #
# polarisation gradients of the CGPS. The figure has 6 panels, showing the    #
# polarisation gradients of the high latitude extension of the CGPS at 6      #
# different resolutions.                                                      #
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
data_loc = '/Volumes/CAH_ExtHD/CGPS_2015/'

# Create a string that will be used to save the figure
filename = '/Users/chrisherron/Documents/PhD/My_Papers/CGPS_Polar_Grad/fig1.png'

# Set the dpi at which to save the image
save_dpi = 100

# Set the convention for cartesian co-ordinates used for the CGPS
convention = 'wells'

# Set the colour scale to use with the images
colour = 'viridis'

# Set the intensity scaling to use with the images
stretch = 'linear'

# Set the minimum intensity value to be included in the colour scale
vmin = 0

# Open the FITS file that contains the polarisation gradient image for the
# high latitude extension at 75 arcsecond resolution
cgps_gradP_75_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_smoothed/Polar_Grad_high_lat_smooth2_75.fits')

# Open the FITS file that contains the polarisation gradient image for the
# high latitude extension at 105 arcsecond resolution
cgps_gradP_105_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_smoothed/Polar_Grad_high_lat_smooth2_105.fits')

# Open the FITS file that contains the polarisation gradient image for the
# high latitude extension at 150 arcsecond resolution
cgps_gradP_150_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_smoothed/Polar_Grad_high_lat_smooth2_150.fits')

# Open the FITS file that contains the polarisation gradient image for the
# high latitude extension at 240 arcsecond resolution
cgps_gradP_240_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_smoothed/Polar_Grad_high_lat_smooth2_240.fits')

# Open the FITS file that contains the polarisation gradient image for the
# high latitude extension at 480 arcsecond resolution
cgps_gradP_480_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_smoothed/Polar_Grad_high_lat_smooth2_480.fits')

# Open the FITS file that contains the polarisation gradient image for the
# high latitude extension at 1200 arcsecond resolution
cgps_gradP_1200_fits = fits.open(data_loc +\
 'Polar_Grad_high_lat_smoothed/Polar_Grad_high_lat_smooth2_1200.fits')

# Create a figure that will be used to hold all of the subplots
fig = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 75 Resolution ----------------------------------

# Add an image of the polarisation gradient at 75 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_gradP_75_fits, figure=fig, subplot=[0.1,0.691,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig1.recenter(111.0,7.5,width=5.0,height=5.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin = vmin, vmax = 12.0, cmap = colour, stretch = stretch)

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

# Hide the x axis label and ticks for this figure
fig1.hide_xaxis_label()
fig1.hide_xtick_labels()

# Add a label to show the resolution of the panel
fig1.add_label(0.5, 1.05, '(a) 75\"', relative=True)

#------------------------------ 105 Resolution ---------------------------------

# Add an image of the polarisation gradient at 105 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_gradP_105_fits, figure=fig, subplot=[0.545,0.691,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig2.recenter(111.0,7.5,width=5.0,height=5.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin, vmax = 7.5, cmap = colour, stretch = stretch)

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
fig2.hide_ytick_labels()

# Hide the x axis label and ticks for this figure
fig2.hide_xaxis_label()
fig2.hide_xtick_labels()

# Add a label to show the resolution of the panel
fig2.add_label(0.5, 1.05, '(b) 105\" ', relative=True)

#------------------------------ 150 Resolution ---------------------------------

# Add an image of the polarisation gradient at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig, subplot=[0.1,0.443,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig3.recenter(111.0,7.5,width=5.0,height=5.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig3.show_colorscale(vmin = vmin, vmax = 5.4, cmap = colour, stretch = stretch)

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

# Hide the x axis label and ticks for this figure
fig3.hide_xaxis_label()
fig3.hide_xtick_labels()

# Add a label to show the resolution of the panel
fig3.add_label(0.5, 1.05, '(c) 150\" ', relative=True)

#------------------------------ 240 Resolution ---------------------------------

# Add an image of the polarisation gradient at 240 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig4 = aplpy.FITSFigure(cgps_gradP_240_fits, figure=fig, subplot=[0.545,0.443,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig4.recenter(111.0,7.5,width=5.0,height=5.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig4.show_colorscale(vmin = vmin, vmax = 3.5, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig4.add_grid()
# Change the grid lines to be more transparent
fig4.grid.set_alpha(1)
# Set the grid lines to be dashed
fig4.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig4.grid.set_linewidth(0.5)

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

# Hide the x axis label and ticks for this figure
fig4.hide_xaxis_label()
fig4.hide_xtick_labels()

# Add a label to show the resolution of the panel
fig4.add_label(0.5, 1.05, '(d) 240\" ', relative=True)

#------------------------------ 480 Resolution ---------------------------------

# Add an image of the polarisation gradient at 480 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig5 = aplpy.FITSFigure(cgps_gradP_480_fits, figure=fig, subplot=[0.1,0.195,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig5.recenter(111.0,7.5,width=5.0,height=5.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig5.show_colorscale(vmin = vmin, vmax = 1.5, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig5.add_grid()
# Change the grid lines to be more transparent
fig5.grid.set_alpha(1)
# Set the grid lines to be dashed
fig5.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig5.grid.set_linewidth(0.5)

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

# Lower the size of the x-axis ticks, and hide the x axis label
fig5.hide_xaxis_label()
fig5.tick_labels.set_xformat('ddd.d')

# Add a label to show the resolution of the panel
fig5.add_label(0.5, 1.05, '(e) 480\" ', relative=True)

#------------------------------ 1200 Resolution --------------------------------

# Add an image of the polarisation gradient at 1200 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig6 = aplpy.FITSFigure(cgps_gradP_1200_fits, figure=fig, subplot=[0.545,0.195,0.4,0.25],\
 convention = convention)

# Change the figure so that the edges of the high latitude extension are 
# ignored
fig6.recenter(111.0,7.5,width=5.0,height=5.0)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig6.show_colorscale(vmin = vmin, vmax = 0.8, cmap = colour, stretch = stretch)

# Add a co-ordinate grid to the image
fig6.add_grid()
# Change the grid lines to be more transparent
fig6.grid.set_alpha(1)
# Set the grid lines to be dashed
fig6.grid.set_linestyle('dashed')
# Set the grid lines to be thin
fig6.grid.set_linewidth(0.5)

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

# Lower the size of the x-axis ticks, and hide the x axis label
fig6.hide_xaxis_label()
fig6.tick_labels.set_xformat('ddd.d')
fig6.tick_labels.set_font(size='small')

# Add a label to show the resolution of the panel
fig6.add_label(0.5, 1.05, '(f) 1200\" ', relative=True)

#-------------------------------------------------------------------------------

# Add a label to the x-axis
plt.figtext(0.53, 0.175, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.00, 0.55, 'Galactic Latitude', ha = 'left', \
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
cgps_gradP_75_fits.close()
cgps_gradP_105_fits.close()
cgps_gradP_150_fits.close()
cgps_gradP_240_fits.close()
cgps_gradP_480_fits.close()
cgps_gradP_1200_fits.close()