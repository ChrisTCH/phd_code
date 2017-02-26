#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code is a script that produces Figure 1 of the paper on the   #
# synchrotron statistics for Christoph's simulations. The figure has 6 panels,#
# showing the synchrotron intensity images along different lines of sight for #
# the solenoidal and compressive simulations, at a chosen timestep.           #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 2/3/2016                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, astropy.io.fits, matplotlib and aplpy
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import aplpy

# Create a string object which stores the directory of the synchrotron data
data_loc = '/Volumes/CAH_ExtHD/CFed_2016/'

# Create a string that will be used to save the figure
filename = '/Users/chrisherron/Documents/PhD/CFed_2016/Publication_Plots/fig1.eps'

# Set the timestep to use when producing the plot
timestep = 20

# Set the value of gamma used to produce the synchrotron intensity images
gamma = 2.0

# Set the dpi at which to save the image
save_dpi = 150

# Set the colour scale to use with the images
colour = 'viridis'

# Set the intensity scaling to use with the images
stretch = 'log'
stretch_c = 'log'

# Set the minimum intensity value to be included in the colour scale
vmin = 7e-11

# Set the maximum intensity value to be included in the colour scale
vmax = 1e-09

# Open the FITS file that contains the synchrotron intensity map for the 
# compressive simulation for a line of sight along the x axis
cfed_comp_x_fits = fits.open(data_loc +\
 '512cM5Bs5886_{}/synint_x_gam{}.fits'.format(timestep, gamma))

# Open the FITS file that contains the synchrotron intensity map for the 
# compressive simulation for a line of sight along the y axis
cfed_comp_y_fits = fits.open(data_loc +\
 '512cM5Bs5886_{}/synint_y_gam{}.fits'.format(timestep, gamma))

# Open the FITS file that contains the synchrotron intensity map for the 
# compressive simulation for a line of sight along the x axis
cfed_comp_z_fits = fits.open(data_loc +\
 '512cM5Bs5886_{}/synint_z_gam{}.fits'.format(timestep, gamma))

# Open the FITS file that contains the synchrotron intensity map for the 
# solenoidal simulation for a line of sight along the x axis
cfed_sol_x_fits = fits.open(data_loc +\
 '512sM5Bs5886_{}/synint_x_gam{}.fits'.format(timestep, gamma))

# Open the FITS file that contains the synchrotron intensity map for the 
# solenoidal simulation for a line of sight along the y axis
cfed_sol_y_fits = fits.open(data_loc +\
 '512sM5Bs5886_{}/synint_y_gam{}.fits'.format(timestep, gamma))

# Open the FITS file that contains the synchrotron intensity map for the 
# solenoidal simulation for a line of sight along the z axis
cfed_sol_z_fits = fits.open(data_loc +\
 '512sM5Bs5886_{}/synint_z_gam{}.fits'.format(timestep, gamma))

# Create a figure that will be used to hold all of the subplots
fig = plt.figure(figsize=(9,15), dpi=save_dpi)

#---------------------------------- Sol x -------------------------------------

# Add an image of the synchrotron intensity for a solenoidal simulation, line
# of sight along the x axis to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig1 = aplpy.FITSFigure(cfed_sol_x_fits, figure=fig, subplot=[0.08,0.69,0.4,0.25])

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig1.show_colorscale(vmin = vmin, vmax = vmax, cmap = colour, stretch = stretch)

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
fig1.tick_labels.set_font(size='small')

# Hide the x axis label and ticks for this figure
fig1.hide_xaxis_label()
fig1.hide_xtick_labels()

# Add a label to this panel
fig1.add_label(0.5, 1.05, '(a) Sol x-LOS', relative=True)

#---------------------------------- Comp x ------------------------------------

# Add an image of the synchrotron intensity for the compressive simulation, line
# of sight along the x axis to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig2 = aplpy.FITSFigure(cfed_comp_x_fits, figure=fig, subplot=[0.535,0.69,0.4,0.25])

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig2.show_colorscale(vmin = vmin, vmax = vmax, cmap = colour, stretch = stretch_c)

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
fig2.add_label(0.5, 1.05, '(b) Comp x-LOS', relative=True)

#---------------------------------- Sol y -------------------------------------

# Add an image of the synchrotron intensity for a solenoidal simulation, line
# of sight along the y axis to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig3 = aplpy.FITSFigure(cfed_sol_y_fits, figure=fig, subplot=[0.08,0.44,0.4,0.25])

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig3.show_colorscale(vmin = vmin, vmax = vmax, cmap = colour, stretch = stretch)

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
fig3.tick_labels.set_font(size='small')

# Hide the x axis label and ticks for this figure
fig3.hide_xaxis_label()
fig3.hide_xtick_labels()

# Add a label to show the resolution of the panel
fig3.add_label(0.5, 1.05, '(c) Sol y-LOS ', relative=True)

#--------------------------------- Comp y --------------------------------------

# Add an image of the synchrotron intensity for a compressive simulation, line
# of sight along the y axis to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig4 = aplpy.FITSFigure(cfed_comp_y_fits, figure=fig, subplot=[0.535,0.44,0.4,0.25])

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig4.show_colorscale(vmin = vmin, vmax = vmax, cmap = colour, stretch = stretch_c)

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
fig4.add_label(0.5, 1.05, '(d) Comp y-LOS ', relative=True)

#--------------------------------- Sol z --------------------------------------

# Add an image of the synchrotron intensity for a solenoidal simulation, line
# of sight along the z axis to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig5 = aplpy.FITSFigure(cfed_sol_z_fits, figure=fig, subplot=[0.08,0.19,0.4,0.25])

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig5.show_colorscale(vmin = vmin, vmax = vmax, cmap = colour, stretch = stretch)

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
fig5.tick_labels.set_font(size='small')

# Lower the size of the x-axis ticks, and hide the x axis label
fig5.hide_xaxis_label()

# Add a label to show the resolution of the panel
fig5.add_label(0.5, 1.05, '(e) Sol z-LOS ', relative=True)

#--------------------------------- Comp z --------------------------------------

# Add an image of the synchrotron intensity for a compressive simulation, line
# of sight along the z axis to the figure
# The subplot argument gives (xmin,ymin,dx,dy), for placement of the subplot
# Need a 10 percent margin on each side of the figure
fig6 = aplpy.FITSFigure(cfed_comp_z_fits, figure=fig, subplot=[0.535,0.19,0.4,0.25])

# In this case the colour scale range is set manually using the 
# specified values for vmin and vmax
# Make a colour scale version of the image appear on the frame, 
# using the colour map specified by the user
fig6.show_colorscale(vmin = vmin, vmax = vmax, cmap = colour, stretch = stretch_c)

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
fig6.tick_labels.set_font(size='small')

# Add a label to show the resolution of the panel
fig6.add_label(0.5, 1.05, '(f) Comp z-LOS ', relative=True)

#-------------------------------------------------------------------------------

# Add a label to the x-axis
plt.figtext(0.53, 0.165, '[pixels]', ha = 'center', \
    va = 'bottom', fontsize = 20)

# Add a label to the y-axis
plt.figtext(0.005, 0.55, '[pixels]', ha = 'left', \
    va = 'center', fontsize = 20, rotation = 'vertical')

#-------------------------------------------------------------------------------

# Save the image using the given filename
fig.savefig(filename, dpi = save_dpi, format = 'eps', bbox_inches='tight')

# Close all of the figures
plt.close(fig)
fig1.close()
fig2.close()
fig3.close()
fig4.close()
fig5.close()
fig6.close()

# Close all of the FITS files that were opened
cfed_comp_x_fits.close()
cfed_comp_y_fits.close()
cfed_comp_z_fits.close()
cfed_sol_x_fits.close()
cfed_sol_y_fits.close()
cfed_sol_z_fits.close()