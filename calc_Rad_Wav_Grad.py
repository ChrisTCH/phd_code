#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to wavelength      #
# squared, as well as Stokes Q and U, and calculates the radial component of  #
# the polarisation wavelength gradient. This produces an image of the radial  #
# component. The formula is given on page 61 of PhD Logbook 5.                #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 26/10/2016                                                      #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Rad_Wav_Grad, which will calculate the radial 
# component of the polarisation wavelength gradient.
def calc_Rad_Wav_Grad(Q = None, U = None, dQ_dl = None, dU_dl = None):
    '''
    Description
        This function calculates the radial component of the polarisation 
        wavelength gradient when given the first order partial derivatives of 
        the Stokes Q and U values with respect to the wavelength squared, as 
        well as Stokes Q and U themselves. The formula is given on page 61 of 
        PhD Logbook 5.
        
    Required Input
        Q, U - Stokes Q and Stokes U respectively. Must be Numpy arrays of the
               same size, where each entry of the array is a float.
        dQ_dl - Partial derivative of Stokes Q with respect to the wavelength
                squared. Must have the same size as the Q and U arrays.
        dU_dl - Partial derivative of Stokes U with respect to the wavelength
                squared. Must have the same size as the Q and U arrays.
                   
    Output
        rad_wav_grad - A Numpy array containing the value of the radial 
                component of the polarisation wavelength gradient at each pixel 
                in the image. Has the same shape as the provided Stokes Q and U 
                arrays.
    '''

    # Calculate the radial component of the polarisation wavelength gradient at 
    # each pixel
    rad_wav_grad = np.cos(np.arctan2(U,Q)) * dQ_dl +\
                     np.sin(np.arctan2(U,Q)) * dU_dl
    
    # Return the radial component of the polarisation wavelength gradient to the
    # caller
    return rad_wav_grad