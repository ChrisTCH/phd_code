#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, as well as Stokes Q and U, and calculates the amplitude of the        #
# directional derivative of the complex polarisation vector. This amplitude   #
# is the maximum absolute value of the tangential component over a range of   #
# angles theta with respect to the horizontal axis of the image. This         #
# produces an image of the amplitude of the directional derivative. The       #
# formula is given on page 45 of PhD Logbook 5.                               #
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

# Define the function calc_Direc_Amp_Max, which will calculate the amplitude of
# the directional derivative of the complex polarisation vector. This amplitude
# is the maximum absolute value of the directional derivative over a range of
# angles theta with respect to the horizontal axis of the image.
def calc_Direc_Amp_Max(Q = None, U = None, dQ_dy = None, dQ_dx = None,\
dU_dy = None, dU_dx = None):
    '''
    Description
        This function calculates the amplitude of the directional derivative of
        the complex polarisation vector when given the first order partial 
        derivatives of the Stokes Q and U values with respect to the x and y 
        axes for the image, as well as Stokes Q and U themselves. This amplitude
        is the maximum absolute value of the directional derivative over a range
        of angles theta with respect to the horizontal axis of the image. The 
        formula is given on page 45 of PhD Logbook 5.
        
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
        direc_amp_max - A Numpy array containing the value of the amplitude of 
                   the directional derivative of the complex polarisation vector
                   at each pixel in the image. Has the same shape as the 
                   provided Stokes Q and U arrays.
    '''
    
    # Calculate the polarisation gradient squared, as this will simplify the
    # calculation of the maximum amplitude of the directional derivative
    polar_grad_sq = np.power(dQ_dx,2.0) + np.power(dU_dx,2.0) +\
     np.power(dQ_dy,2.0) + np.power(dU_dy,2.0)
    
    # Calculate the maximum amplitude of the directional derivative at each
    # pixel, squared
    direc_amp_max_sq = 0.5*(polar_grad_sq + np.sqrt(np.power(polar_grad_sq,2.0)\
     - 4.0*np.power(dQ_dx*dU_dy - dQ_dy*dU_dx,2.0) ) )
    
    # Calculate the maximum amplitude of the directional derivative
    direc_amp_max = np.sqrt(direc_amp_max_sq)

    # Return the amplitude of the directional derivative to the caller
    return direc_amp_max