#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the        #
# partial derivatives of Stokes Q and U with respect to the x and y axes, and #
# produces an array of the angle of the polarisation gradient with respect to #
# the x axis at each pixel.                                                   #
# The formula for the calculation of angle of the polarisation gradient is    #
# the same as that in Gaensler et al 2011, Nature 478.                        #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 10/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Ang_Grad, which will calculate the angle of the 
# polarisation gradient with respect to the x axis of the image from the
# first derivatives of the Stokes Q and U values.
def calc_Ang_Grad(dQ_dy = None, dQ_dx = None, dU_dy = None, dU_dx = None):
    '''
    Description
        This function calculates the angle of the polarisation gradient with
        respect to the x axis at each point of an image, when given the first
        derivatives of the Stokes Q or U values with respect to the x and y 
        axes for the image. The formula used for the calculation is given in
        Gaensler et al 2011, Nature 478.
        
    Required Input
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                       vertical and horizontal axes of the image respectively.
                       
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays must have the same size as the arrays
                       containing the values of the partial derivatives of
                       Stokes Q.
                   
    Output
        grad_ang - A Numpy array containing the value of the angle of the
                   polarisation gradient at each point with respect to the x
                   axis. This array has the same size as the input arrays. The
                   returned angles are between +/- 90 degrees.
    '''
    
    # Calculate the term whose inverse tan gives the angle of the polarisation
    # gradient with respect to the x axis of the image.
    term = np.sign(dQ_dx * dQ_dy + dU_dx * dU_dy) * \
    np.sqrt( np.power(dQ_dy, 2.0) + np.power(dU_dy, 2.0) )\
    / np.sqrt( np.power(dQ_dx, 2.0) + np.power(dU_dx, 2.0))
    
    # Calculate the angle of the polarisation gradient with respect to the 
    # x axis.
    grad_ang = np.arctan(term)
    
    # Convert the calculated angles from radians to degrees
    grad_ang = np.rad2deg(grad_ang)
    
    # Return the array containing the angle of the polarisation gradient with
    # respect to the x axis at each pixel
    return grad_ang