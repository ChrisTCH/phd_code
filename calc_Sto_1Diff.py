#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of Stokes Q   #
# and Stokes U values for an image, and calculates the partial derivatives of #
# Stokes Q and U along both the horizontal and the vertical axes of the       #
# image. These partial derivatives are then returned to the caller.           #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 12/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Sto_1Diff, which will calculate the first partial
# derivatives of the Stokes Q and U values.
def calc_Sto_1Diff(Stokes_Q, Stokes_U):
    '''
    Description
        This function calculates the partial derivatives of Stokes Q and U 
        along the horizontal (x) and vertical (y) axes of an image, and 
        returns them to the calling function. These are the first order 
        partial derivatives.
        
    Required Input
        Stokes_Q - A Numpy array containing the value of Stokes Q at each
                   pixel of the image. The array must conform to the 
                   convention that the first dimension represents the y-axis,
                   and the second dimension represents the x-axis, so that
                   each entry of the array represents a fixed y-value, and 
                   contains an array giving the Stokes Q value at each x-value.
        Stokes_U - A Numpy array containing the value of Stokes U at each
                   pixel of the image. Must have the same size as the Stokes
                   Q array. Must satisfy the same conventions as the Stokes Q
                   array.
                   
    Output
        The order of the output quantities is dQ_dy, dQ_dx, dU_dy, dU_dx.
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays have the same size as the Stokes Q array.
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays have the same size as the Stokes U array.
    '''
    
    # To calculate the derivatives of Stokes Q and U along the x and y axes,
    # the gradient function of numpy is used.
    # Numpy performs the calculation by using central differences in the 
    # interior of the provided arrays, and using first differences at the
    # boundaries of the array. This function returns an array with two entries.
    # The first entry is the derivative along the y axis, and the second
    # entry is the derivative along the x axis.
    
    # Calculate the partial derivatives of Stokes Q along the y and x axes
    dQ_dy, dQ_dx = np.gradient(Stokes_Q)
    
    # Calculate the partial derivatives of Stokes U along the y and x axes
    dU_dy, dU_dx = np.gradient(Stokes_U)
    
    # Return the calculated partial derivatives to the caller.
    return dQ_dy, dQ_dx, dU_dy, dU_dx 