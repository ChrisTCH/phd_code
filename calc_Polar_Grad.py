#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of Stokes Q   #
# and Stokes U values for an image, and produces an array of the polarisation #
# gradient magnitude values at each pixel. The formula for the calculation of #
# the polarisation gradient is the same as that in Gaensler et al 2011,       #
# Nature 478. The polarisation gradient array is returned to the caller.      #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 6/6/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Polar_Grad, which will calculate the polarisation
# gradient magnitude from the Stokes Q and U values.
def calc_Polar_Grad(Stokes_Q, Stokes_U):
    '''
    Description
        This function calculates the magnitude of the polarisation gradient
        at each point of an image, when given the Stokes Q and U values for
        the image. The formula used for the calculation is given in 
        Gaensler et al 2011, Nature 478.
        
    Required Input
        Stokes_Q - A Numpy array containing the value of Stokes Q at each
                   pixel of the image.
        Stokes_U - A Numpy array containing the value of Stokes U at each
                   pixel of the image. Must have the same size as the Stokes
                   Q image.
                   
    Output
        polar_grad - A Numpy array containing the value of the magnitude of the
                     polarisation gradient at each point. This array has the 
                     same size as the Stokes Q and U arrays.
    '''
    # First, it is necessary to calculate the derivatives of Stokes Q and U
    # along the x and y axes. This is achieved with the gradient function of 
    # numpy.
    # Numpy performs the calculation by using central differences in the 
    # interior of the provided arrays, and using first differences at the
    # boundaries of the array. This function returns an array with two entries.
    # The first entry is the derivative along the y axis, and the second
    # entry is the derivative along the x axis.
    
    # Calculate the partial derivatives of Stokes Q along the y and x axes
    dQ_dy, dQ_dx = np.gradient(Stokes_Q)
    
    # Calculate the partial derivatives of Stokes U along the y and x axes
    dU_dy, dU_dx = np.gradient(Stokes_U)
    
    # Calculate the magnitude of the polarisation gradient
    polar_grad = np.sqrt(np.power(dQ_dy, 2.0) + np.power(dQ_dx, 2.0) +\
    np.power(dU_dy, 2.0) + np.power(dU_dx, 2.0) )
    
    # Return the polarisation gradient to the calling function
    return polar_grad