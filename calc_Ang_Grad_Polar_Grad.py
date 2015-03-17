#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# and second order partial derivatives of Stokes Q and U for an image, and    #
# produces an array of the angle of the gradient of the polarisation          #
# gradient magnitude value at each pixel. The formula for the calculation of  #
# the gradient of the polarisation gradient magnitude is on page 108 of PhD   #
# Logbook 1. The angle of the gradient of the polarisation gradient array     #
# is returned to the caller.                                                  #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 2/7/2014                                                        #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Ang_Grad_Polar_Grad, which will calculate the
# angle of the gradient of the polarisation gradient magnitude from the first
# and second order partial derivatives of the Stokes Q and U values with respect
# to the x and y axes.
def calc_Ang_Grad_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx, d2Q_dy2 = None,\
d2Q_dydx = None, d2Q_dx2 = None, d2U_dy2 = None, d2U_dydx = None,\
d2U_dx2 = None):
    '''
    Description
        This function calculates the angle of the gradient of the 
        polarisation gradient magnitude at each point of an image, when given
        the first and second order partial derivatives of Stokes Q and U values
        with respect to the x and y axes for the image. The formula used for the
        calculation is given on page 108 of PhD Logbook 1. 
        
    Required Input
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                       vertical and horizontal axes of the image respectively.
                       Arrays must have the same size.
                       
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays must have the same size as the arrays
                       containing the values of the partial derivatives of
                       Stokes Q.
        
        d2Q_dy2, d2Q_dx2, d2Q_dydx - Second order partial derivatives of Stokes
                       Q with respect to the vertical, horizontal, and both axes
                       of the image respectively. These arrays must have the 
                       same size as the arrays containing the values of the 
                       partial derivatives of Stokes Q.
        
        d2U_dy2, d2U_dx2, d2U_dydx - Second order partial derivatives of Stokes
                       U with respect to the vertical, horizontal, and both axes
                       of the image respectively. These arrays must have the 
                       same size as the arrays containing the values of the 
                       partial derivatives of Stokes Q.
                   
    Output
        ang_grad_polar_grad - A Numpy array containing the value of the angle of
                     the gradient of the polarisation gradient magnitude at each
                     point. This array has the same size as the partial 
                     derivative arrays.
    '''
    
    # Calculate the x component of the gradient of the polarisation gradient
    # magnitude
    x_comp = dQ_dx * d2Q_dx2 + dQ_dy * d2Q_dydx + dU_dx * d2U_dx2 +\
    dU_dy * d2U_dydx
    
    # Calculate the y component of the gradient of the polarisation gradient 
    # magnitude
    y_comp = dQ_dx * d2Q_dydx + dQ_dy * d2Q_dy2 + dU_dx * d2U_dydx +\
    dU_dy * d2U_dy2
    
    # Calculate the angle of the gradient of the polarisation gradient
    # magnitude
    ang_grad_polar_grad = np.rad2deg(np.arctan2(y_comp, x_comp))
    
    # Return the angle of the gradient of the polarisation gradient 
    # magnitude to the calling function
    return ang_grad_polar_grad