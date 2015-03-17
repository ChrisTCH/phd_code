#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U for an image, and produces an   #
# array of the polarisation gradient magnitude value at each pixel. The       #
# formula for the calculation of the polarisation gradient is the same as     #
# that in Gaensler et al 2011, Nature 478. The polarisation gradient array is #
# returned to the caller.                                                     #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 6/6/2014                                                        #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Polar_Grad, which will calculate the polarisation
# gradient magnitude from the first order partial derivatives of the Stokes Q
# and U values with respect to the x and y axes.
def calc_Polar_Grad(dQ_dy, dQ_dx, dU_dy, dU_dx):
    '''
    Description
        This function calculates the magnitude of the polarisation gradient
        at each point of an image, when given the first order partial 
        derivatives of Stokes Q and U values with respect to the x and y axes
        for the image. The formula used for the calculation is given in 
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
        polar_grad - A Numpy array containing the value of the magnitude of the
                     polarisation gradient at each point. This array has the 
                     same size as the partial derivative arrays.
    '''
    
    # Calculate the magnitude of the polarisation gradient
    polar_grad = np.sqrt(np.power(dQ_dy, 2.0) + np.power(dQ_dx, 2.0) +\
    np.power(dU_dy, 2.0) + np.power(dU_dx, 2.0) )
    
    # Return the polarisation gradient to the calling function
    return polar_grad