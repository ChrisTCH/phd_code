#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U for an array, and produces an   #
# array of the polarisation wavelength gradient magnitude value at each pixel.#
# The polarisation wavelength gradient array is returned to the caller.       #
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

# Define the function calc_Wav_Grad, which will calculate the polarisation
# wavelength gradient magnitude from the first order partial derivatives of the
# Stokes Q and U values with respect to the wavelength squared.
def calc_Wav_Grad(dQ_dl, dU_dl):
    '''
    Description
        This function calculates the magnitude of the polarisation wavelength 
        gradient at each point of an array, when given the first order partial 
        derivatives of Stokes Q and U values with respect to the wavelength 
        squared for the array. 
        
    Required Input
        dQ_dl - Partial derivative of Stokes Q with respect to the wavelength
                squared.
                       
        dU_dl - Partial derivative of Stokes U with respect to the wavelength 
                squared. This array must have the same size as the array
                containing the values of the partial derivative of Stokes Q.
                   
    Output
        wav_grad - A Numpy array containing the value of the magnitude of the
                   polarisation wavelength gradient at each point. This array 
                   has the same size as the partial derivative arrays.
    '''
    
    # Calculate the magnitude of the polarisation wavelength gradient
    wav_grad = np.sqrt(np.power(dQ_dl, 2.0) + np.power(dU_dl, 2.0) )
    
    # Return the polarisation wavelength gradient to the calling function
    return wav_grad