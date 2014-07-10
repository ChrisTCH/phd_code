#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# and second order partial derivatives of Stokes Q and U for an image, and    #
# produces an array of the angle between the polarisation gradient and the    #
# gradient of the polarisation gradient magnitude. The formula for the        #
# calculation of the angle between the polarisation gradient and the gradient #
# of the polarisation gradient magnitude is on page 114 of PhD Logbook 1. The #
# angle between the polarisation gradient and the gradient of the             #
# polarisation gradient magnitude array is returned to the caller. Note that  #
# the angle calculated is technically the Hermitian angle between the         #
# polarisation gradient and the gradient of the polarisation gradient         #
# magnitude.                                                                  #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 2/7/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Ang_Betw_DP_DModDP, which will calculate the angle
# between the polarisation gradient and the gradient of the polarisation 
# gradient magnitude from the first and second order partial derivatives of the
# Stokes Q and U values with respect to the x and y axes.
def calc_Ang_Betw_DP_DModDP(dQ_dy, dQ_dx, dU_dy, dU_dx, d2Q_dy2 = None,\
d2Q_dydx = None, d2Q_dx2 = None, d2U_dy2 = None, d2U_dydx = None,\
d2U_dx2 = None):
    '''
    Description
        This function calculates the angle between the polarisation gradient and
        the gradient of the polarisation gradient magnitude at each point of an
        image, when given the first and second order partial derivatives of 
        Stokes Q and U values with respect to the x and y axes for the image. 
        The formula used for the calculation is given on page 114 of PhD
        Logbook 1. 
        
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
        ang_betw_DP_DModDP - A Numpy array containing the value of the angle
                     between the polarisation gradient and the gradient of the
                     polarisation gradient magnitude at each point. This array 
                     has the same size as the partial derivative arrays.
    '''
    
    # Calculate the magnitude of the polarisation gradient
    polar_grad = np.sqrt(np.power(dQ_dy, 2.0) + np.power(dQ_dx, 2.0) +\
    np.power(dU_dy, 2.0) + np.power(dU_dx, 2.0) )
    
    # Calculate the x component of the gradient of the polarisation gradient
    # magnitude
    x_comp = dQ_dx * d2Q_dx2 + dQ_dy * d2Q_dydx + dU_dx * d2U_dx2 +\
    dU_dy * d2U_dydx
    
    # Calculate the y component of the gradient of the polarisation gradient 
    # magnitude
    y_comp = dQ_dx * d2Q_dydx + dQ_dy * d2Q_dy2 + dU_dx * d2U_dydx +\
    dU_dy * d2U_dy2
    
    # Calculate the magnitude of the gradient of the polarisation gradient
    # magnitude
    grad_polar_grad = np.sqrt( np.power(x_comp,2.0) + np.power(y_comp,2.0))
    
    # Calculate the sum of the squared partial derivatives of Stokes Q and U
    # in the x direction
    sum_x = np.power(dQ_dx,2.0) + np.power(dU_dx,2.0)
    
    # Calculate the sum of the squared partial derivatives of Stokes Q and U
    # in the y direction
    sum_y = np.power(dQ_dy,2.0) + np.power(dU_dy,2.0)
    
    # Calculate the sum of the multiplied partial derivatives of Stokes Q and U
    # in the x and y directions
    sum_xy = dQ_dx * dQ_dy + dU_dx * dU_dy
    
    # Calculate the numerator of the term required to calculate the angle
    # between the polarisation gradient and the gradient of the polarisation
    # gradient magnitude
    numer = np.sqrt(np.power(x_comp,2.0) * sum_x + np.power(y_comp,2.0) * sum_y\
    + 2.0 * x_comp * y_comp * sum_xy)
    
    # Calculate the denominator of the term required to calculate the angle
    # between the polarisation gradient and the gradient of the polarisation
    # gradient magnitude
    denom = polar_grad * grad_polar_grad
    
    # Calculate the angle between the polarisation gradient and the gradient
    # of the polarisation gradient magnitude
    ang_betw_DP_DModDP = np.rad2deg(np.arccos(numer / denom))
    
    # Return the angle between the polarisation gradient and the gradient of
    # the polarisation gradient magnitude to the calling function
    return ang_betw_DP_DModDP