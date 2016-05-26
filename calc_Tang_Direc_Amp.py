#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, as well as Stokes Q and U, and calculates the amplitude of the        #
# tangential component of the directional derivative of the complex           #
# polarisation vector. This amplitude is the maximum absolute value of the    #
# tangential component over a range of angles theta with respect to the       #
# horizontal axis of the image. This produces an image of the amplitude of    #
# the tangential component. The formula is given on page 36 of PhD Logbook 5. #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 18/6/2014                                                       #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Tang_Direc_Amp, which will calculate the amplitude of
# the tangential component of the directional derivative of the complex 
# polarisation vector. This amplitude is the maximum absolute value of the    
# tangential component over a range of angles theta with respect to the       
# horizontal axis of the image.
def calc_Tang_Direc_Amp(Q = None, U = None, dQ_dy = None, dQ_dx = None,\
dU_dy = None, dU_dx = None):
    '''
    Description
        This function calculates the amplitude of the tangential component of 
        the directional derivative of the complex polarisation vector when given
        the first order partial derivatives of the Stokes Q and U values with
        respect to the x and y axes for the image, as well as Stokes Q and U
        themselves. This amplitude is the maximum absolute value of the 
        tangential component over a range of angles theta with respect to the
        horizontal axis of the image. The formula is given on page 36 of PhD
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
        tang_comp_amp - A Numpy array containing the value of the amplitude of 
                   the tangential component of the directional derivative of the
                   complex polarisation vector at each pixel in the image. Has
                   the same shape as the provided Stokes Q and U arrays.
    '''
    
    # Calculate the numerator needed to calculate the tangential component of 
    # the directional derivative
    numer = np.power(Q*dU_dx - U*dQ_dx,2.0) + np.power(Q*dU_dy - U*dQ_dy,2.0)

    # Calculate the denominator needed to calculate the tangential component of
    # the directional derivative
    denom = np.power(Q,2.0) + np.power(U,2.0)
    
    # Calculate the amplitude of the tangential component of the directional
    # derivative at each pixel
    tang_comp_amp = np.sqrt(numer/denom)
    
    # Return the amplitude of the tangential component of the directional 
    # derivative to the caller
    return tang_comp_amp