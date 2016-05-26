#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, as well as Stokes Q and U, and calculates the angle in the image      #
# plane for which the radial component of the directional derivative of the   #
# complex polarisation vector is maximised. This produces an image of the     #
# angle for which the radial component is maximised. The formula is given on  #
# page 31 of PhD Logbook 5.                                                   #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 20/1/2016                                                       #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Rad_Direc_Ang, which will calculate the angle in the
# image plane for which the radial component of the directional derivative of
# the complex  polarisation vector is maximised. 
def calc_Rad_Direc_Ang(Q = None, U = None, dQ_dy = None, dQ_dx = None,\
dU_dy = None, dU_dx = None):
    '''
    Description
        This function calculates the angle for which the radial component of 
        the directional derivative of the complex polarisation vector is 
        maximised, when given the first order partial derivatives of the Stokes 
        Q and U values with respect to the x and y axes for the image, as well 
        as Stokes Q and U themselves. The formula is given on page 31 of PhD
        Logbook 5.
        
    Required Input
        Q, U - Stokes Q and Stokes U respectively. Must be Numpy arrays of the
               same size, where each entry of the array is a float.
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                       vertical and horizontal axes of the image respectively.
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays must have the same size as the arrays
                       containing the values of the partial derivatives of
                       Stokes Q.
                   
    Output
        rad_comp_ang - A Numpy array containing the value of the angle for which 
                   the radial component of the directional derivative of the
                   complex polarisation vector is maximised, at each pixel in 
                   the image. Has the same shape as the provided Stokes Q and U 
                   arrays.
    '''
    
    # Calculate the numerator needed to calculate the angle that maximises the
    # radial component of the directional derivative
    numer = Q*dQ_dy + U*dU_dy

    # Calculate the denominator needed to calculate the angle that maximises the
    # radial component of the directional derivative
    denom = Q*dQ_dx + U*dU_dx

    # Calculate the angle for which the radial component of the directional
    # derivative is maximised, at each pixel
    rad_comp_ang = np.rad2deg(np.arctan2(numer,denom))
    
    # Return the angle that maximises the radial component of the directional 
    # derivative to the caller
    return rad_comp_ang