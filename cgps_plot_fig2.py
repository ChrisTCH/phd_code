#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code is a script that produces Figure 2 of the paper on the   #
# polarisation gradients of the CGPS. The figure has 6 sections, which will   #
# be spread across three pages, showing the polarisation gradients of the     #
# Galactic plane mosaic of the CGPS, along with Stokes I.                     #
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
filename = '/Users/chrisherron/Documents/PhD/My_Papers/CGPS_Polar_Grad/'

# Set the dpi at which to save the image
save_dpi = 300

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

# Set the colour scale to use with the skewness
colour_s = 'cubehelix'

# Set the intensity scaling to use with the skewness
stretch_s = 'linear'

# Set the minimum intensity value to be included in the colour scale, for the
# polarisation gradient
vmin_p = 0

# Set the maximum intensity value to be included in the colour scale, for the
# polarisation gradient
vmax_p = 5.0

# Set the minimum intensity value to be included in the colour scale, for the
# skewness
vmin_s = 0

# Set the maximum intensity value to be included in the colour scale, for the
# skewness
vmax_s = 1.2

# Open the FITS file that contains the polarisation gradient image for the
# Galactic plane at 150 arcsecond resolution
cgps_gradP_150_fits = fits.open(data_loc +\
 'Polar_Grad_plane_smoothed/Polar_Grad_plane_smooth2_150.fits')

# Open the FITS file that contains the skewness of the polarisation gradient
# smoothed to an angular resolution of 150 arcseconds, with 1% truncation,
# for the entire Galactic plane.
cgps_skew_150_fits = fits.open(data_loc +\
 'Polar_Grad_plane_all_mask_trunc1_skew_sparse/'+\
 'Polar_Grad_plane_all_mask_smooth2_150_skewness_sparse.fits')

# Open the FITS file that contains the total intensity image for the
# Galactic plane
cgps_I_fits = fits.open(data_loc + 'Sto_I_plane.fits')

#------------------------------ Figure a ---------------------------------------

# Create a figure that will be used to hold all of the subplots
fig_a = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 164 < l < 192 ----------------------------------

# Add an image of the total intensity to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_I_fits, figure=fig_a, subplot=[0.1,0.691,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig1.recenter(178.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=4.0, vmax=6.0, cmap = colour_i, stretch = stretch_i)

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

# Add an image of the polarisation at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig_a, subplot=[0.1,0.535,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig2.recenter(178.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin_p,vmax=vmax_p,cmap=colour_p, stretch=stretch_p)

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

# Add an image of the skewness to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_fits, figure=fig_a, subplot=[0.1,0.38,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig3.recenter(178.0,1.0,width=28.0,height=8.5)

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

# Add a label to the x-axis
plt.figtext(0.53, 0.38, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.65, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

# Save the image using the given filename
fig_a.savefig(filename + 'fig2a.eps', dpi = save_dpi, format = 'eps', bbox_inches='tight')

# Close all of the figures
plt.close(fig_a)
fig1.close()
fig2.close()
fig3.close()

#------------------------------ Figure b ---------------------------------------

# Create a figure that will be used to hold all of the subplots
fig_b = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 141 < l < 169 ----------------------------------

# Add an image of the total intensity to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_I_fits, figure=fig_b, subplot=[0.1,0.691,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig1.recenter(155.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=4.0, vmax=6.0, cmap = colour_i, stretch = stretch_i)

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

# Add an image of the polarisation at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig_b, subplot=[0.1,0.535,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig2.recenter(155.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin_p,vmax=vmax_p,cmap=colour_p, stretch=stretch_p)

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

# Add an image of the skewness to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_fits, figure=fig_b, subplot=[0.1,0.38,0.8,0.22],\
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

# Add a label to the x-axis
plt.figtext(0.53, 0.38, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.65, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

# Save the image using the given filename
fig_b.savefig(filename + 'fig2b.eps', dpi = save_dpi, format = 'eps', bbox_inches='tight')

# Close all of the figures
plt.close(fig_b)
fig1.close()
fig2.close()
fig3.close()

#------------------------------ Figure c ---------------------------------------

# Create a figure that will be used to hold all of the subplots
fig_c = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 118 < l < 146 ----------------------------------

# Add an image of the total intensity to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_I_fits, figure=fig_c, subplot=[0.1,0.691,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig1.recenter(132.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=4.0, vmax=7.0, cmap = colour_i, stretch = stretch_i)

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

# Add an image of the polarisation at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig_c, subplot=[0.1,0.535,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig2.recenter(132.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin_p,vmax=vmax_p,cmap=colour_p, stretch=stretch_p)

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

# Add an image of the skewness to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_fits, figure=fig_c, subplot=[0.1,0.38,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig3.recenter(132.0,1.0,width=28.0,height=8.5)

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

# Add a label to the x-axis
plt.figtext(0.53, 0.38, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.65, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

# Save the image using the given filename
fig_c.savefig(filename + 'fig2c.eps', dpi = save_dpi, format = 'eps', bbox_inches='tight')

# Close all of the figures
plt.close(fig_c)
fig1.close()
fig2.close()
fig3.close()

#------------------------------ Figure d ---------------------------------------

# Create a figure that will be used to hold all of the subplots
fig_d = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 95 < l < 123 ----------------------------------

# Add an image of the total intensity to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_I_fits, figure=fig_d, subplot=[0.1,0.691,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig1.recenter(109.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=4.0, vmax=9.0, cmap = colour_i, stretch = stretch_i)

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

# Add an image of the polarisation at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig_d, subplot=[0.1,0.535,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig2.recenter(109.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin_p,vmax=vmax_p,cmap=colour_p, stretch=stretch_p)

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

# Add an image of the skewness to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_fits, figure=fig_d, subplot=[0.1,0.38,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig3.recenter(109.0,1.0,width=28.0,height=8.5)

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

# Add a label to the x-axis
plt.figtext(0.53, 0.38, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.65, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

# Save the image using the given filename
fig_d.savefig(filename + 'fig2d.eps', dpi = save_dpi, format = 'eps', bbox_inches='tight')

# Close all of the figures
plt.close(fig_d)
fig1.close()
fig2.close()
fig3.close()

#------------------------------ Figure e ---------------------------------------

# Create a figure that will be used to hold all of the subplots
fig_e = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 72 < l < 100 ----------------------------------

# Add an image of the total intensity to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_I_fits, figure=fig_e, subplot=[0.1,0.691,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig1.recenter(86.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=4.0, vmax=20.0, cmap = colour_i, stretch = 'sqrt')

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

# Add an image of the polarisation at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig_e, subplot=[0.1,0.535,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig2.recenter(86.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin_p,vmax=vmax_p,cmap=colour_p, stretch=stretch_p)

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

# Add an image of the skewness to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_fits, figure=fig_e, subplot=[0.1,0.38,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig3.recenter(86.0,1.0,width=28.0,height=8.5)

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

# Add a label to the x-axis
plt.figtext(0.53, 0.38, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.65, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

# Save the image using the given filename
fig_e.savefig(filename + 'fig2e.eps', dpi = save_dpi, format = 'eps', bbox_inches='tight')

# Close all of the figures
plt.close(fig_e)
fig1.close()
fig2.close()
fig3.close()

#------------------------------ Figure f ---------------------------------------

# Create a figure that will be used to hold all of the subplots
fig_f = plt.figure(figsize=(9,15), dpi=save_dpi)

#------------------------------ 49 < l < 77 ----------------------------------

# Add an image of the total intensity to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cgps_I_fits, figure=fig_f, subplot=[0.1,0.691,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig1.recenter(63.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin=4.0, vmax=11.0, cmap = colour_i, stretch = stretch_i)

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

# Add an image of the polarisation at 150 arcsecond resolution to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cgps_gradP_150_fits, figure=fig_f, subplot=[0.1,0.535,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig2.recenter(63.0,1.0,width=28.0,height=8.5)

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin_p,vmax=vmax_p,cmap=colour_p, stretch=stretch_p)

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

# Add an image of the skewness to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cgps_skew_150_fits, figure=fig_f, subplot=[0.1,0.38,0.8,0.22],\
 convention = convention)

# Change the figure so that only the desired region is plotted
fig3.recenter(63.0,1.0,width=28.0,height=8.5)

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

# Add a label to the x-axis
plt.figtext(0.53, 0.38, 'Galactic Longitude', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.65, 'Galactic Latitude', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

# Save the image using the given filename
fig_f.savefig(filename + 'fig2f.eps', dpi = save_dpi, format = 'eps', bbox_inches='tight')

# Close all of the figures
plt.close(fig_f)
fig1.close()
fig2.close()
fig3.close()

#-------------------------------------------------------------------------------

# Close all of the FITS files that were opened
cgps_I_fits.close()
cgps_gradP_150_fits.close()
cgps_skew_150_fits.close()