#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the        #
# second order partial derivatives of Stokes Q and U with respect to the x    #
# and y axes, and calculates the modulus of the divergence of the             #
# polarisation gradient at each pixel. The formula is given on page 105 of    #
# PhD Logbook 1, and page 52 of PhD Logbook 2. This quantity is rotationally  #
# and translationally invariant in the Q-U plane.                             #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 14/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Mod_Div_Grad, which will calculate the modulus of
# the divergence of the polarisation gradient from the second order partial
# derivatives of the Stokes Q and U values.
def calc_Mod_Div_Grad(d2Q_dy2 = None, d2Q_dx2 = None, d2U_dy2 = None,\
d2U_dx2 = None):
    '''
    Description
        This function calculates the modulus of the divergence of the
        polarisation gradient, when given the second order partial
        derivatives of the Stokes Q and U values with respect to the x and y 
        axes for the image. The formula for this quantity is given on page 105
        of PhD Logbook 1, and page 52 of PhD Logbook 2.
        
    Required Input
        d2Q_dy2, d2Q_dx2 - Second order partial derivatives of Stokes Q with
                       respect to the vertical and horizontal axes of the image
                       respectively.
                       
        d2U_dy2, d2U_dx2 - Second order partial derivatives of Stokes U with
                       respect to the vertical and horizontal axes of the image
                       respectively.These arrays must have the same size as the
                       arrays containing the values of the partial derivatives
                       of Stokes Q.
                   
    Output
        mod_div_grad - A Numpy array containing the value of the modulus of the
                   divergnce of the polarisation gradient at each point. This
                   array has the same size as the input arrays.
    '''
    
    # Calculate the modulus of the divergence of the polarisation gradient
    # using the provided arrays of the second order partial derivatives.
    mod_div_grad = np.sqrt( np.power(d2Q_dy2 + d2Q_dx2, 2.0) + np.power(d2U_dy2 + d2U_dx2 ,2.0))
    
    # Return the array of modulus of the divergence of the polarisation 
    # gradient values to the caller
    return mod_div_grad
    