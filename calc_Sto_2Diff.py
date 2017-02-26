#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and Stokes U values for an image, and #
# calculates all of the second order partial derivatives of Stokes Q and U    #
# along both the horizontal and the vertical axes of the image. These partial #
# derivatives are then returned to the caller.                                #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 12/6/2014                                                       #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Sto_2Diff, which will calculate the second order 
# partial derivatives of the Stokes Q and U values with respect to the x and y
# axes of the image.
def calc_Sto_2Diff(dQ_dy, dQ_dx, dU_dy, dU_dx, pix_sep = 1.0):
    '''
    Description
        This function calculates the second order partial derivatives of Stokes
        Q and U along the horizontal (x) and vertical (y) axes of an image, and 
        returns them to the calling function. 
        
    Required Input
        dQ_dy, dQ_dx - Numpy arrays containing the value of the first order 
                   partial derivatives of Stokes Q along the y and x axes
                   respectively. The array must conform to the 
                   convention that the second last dimension represents the 
                   y-axis, and the last dimension represents the x-axis, so that
                   each entry of the array represents a fixed y-value, and 
                   contains an array giving the Stokes Q value at each x-value.
                   For a 2D array, the dimensions should be ordered as (y,x),
                   and for a 3D array (-,y,x).
        dU_dy, dU_dx - Numpy arrays containing the value of the first order
                   partial derivatives of Stokes U along the y and x axes
                   respectively. Must have the same size as the arrays for 
                   partial derivatives of Stokes Q. Must satisfy the same
                   conventions as the Stokes Q array.
        pix_sep - A float denoting the separation between two adjacent points
                  in the provided arrays of Stokes Q and U, in radians. This
                  is required for the calculated derivatives to have the 
                  correct units. 
                   
    Output
        The order of the output quantities is d2Q_dy2, d2Q_dydx, d2Q_dx2, 
        d2U_dy2, d2U_dydx, d2U_dx2.
        d2Q_dy2, d2Q_dydx, d2Q_dx2 - Second order partial derivatives of Stokes
                       Q with respect to the vertical axis only, both the 
                       vertical and horizontal axes, and horizontal axis only
                       respectively. These arrays have the same size as the 
                       arrays for partial derivatives of Stokes Q.
        d2U_dy2, d2U_dydx, d2U_dx2 - Second order partial derivatives of Stokes
                       U with respect to the vertical axis only, both the 
                       vertical and horizontal axes, and horizontal axis only
                       respectively. These arrays have the same size as the 
                       arrays for partial derivatives of Stokes U.
    '''
    
    # Calculate the number of dimensions in the given arrays
    num_dim = len(np.shape(dQ_dy))

    # Depending on the number of dimensions, set parameters that control the
    # axes that we differentiate across. These values represent the index
    # for the correct axes
    axis_y = num_dim - 2
    axis_x = num_dim - 1

    # Using the gradient function of numpy, calculate two of the second
    # derivatives of Stokes Q. Note that this line calculates the derivative
    # of dQ_dy with respect to y and x, thus calculating d2Q_dy2 and d2Q_dxdy.
    # pix_sep is used to make sure that the returned matrices have the correct
    # units.
    d2Q_dy2 = np.gradient(dQ_dy, pix_sep, axis = axis_y)
    d2Q_dxdy = np.gradient(dQ_dy, pix_sep, axis = axis_x)
    
    # Calculate the derivative of dQ_dx with respect to y and x, thus 
    # calculating d2Q_dydx and d2Q_dx2
    d2Q_dydx = np.gradient(dQ_dx, pix_sep, axis = axis_y)
    d2Q_dx2 = np.gradient(dQ_dx, pix_sep, axis = axis_x)
    
    # Calculate the derivative of dU_dy with respect to y and x, thus
    # calculating d2U_dy2 and d2U_dxdy
    d2U_dy2 = np.gradient(dU_dy, pix_sep, axis = axis_y)
    d2U_dxdy = np.gradient(dU_dy, pix_sep, axis = axis_x)
    
    # Calculate the derivative of dU_dx with respect to y and x, thus
    # calculating d2U_dydx and d2U_dx2
    d2U_dydx = np.gradient(dU_dx, pix_sep, axis = axis_y)
    d2U_dx2 = np.gradient(dU_dx, pix_sep, axis = axis_x)
    
    # Return all of the calculated second derivatives of Stokes Q and U
    return d2Q_dy2, d2Q_dydx, d2Q_dx2, d2U_dy2, d2U_dydx, d2U_dx2