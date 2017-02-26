#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of Stokes Q   #
# and Stokes U values for an image, and calculates the partial derivatives of #
# Stokes Q and U along the third axis of the array, which represents the      #
# square of the wavelength. These partial derivatives are then returned to    #
# the caller.                                                                 #
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

# Define the function calc_Sto_Wav_1Diff, which will calculate the first partial
# derivatives of the Stokes Q and U values with respect to the square of the 
# wavelength
def calc_Sto_Wav_1Diff(Stokes_Q, Stokes_U, wave_sq_space = 1.0):
    '''
    Description
        This function calculates the partial derivatives of Stokes Q and U 
        along the third axis of the array, representing the square of the 
        wavelength, and returns them to the calling function. These are the 
        first order partial derivatives.
        
    Required Input
        Stokes_Q - A Numpy array containing the value of Stokes Q at each
                   pixel of the image, for various wavelengths. The array must
                   conform to the convention that the first dimension represents
                   the square of the wavelength, the second dimension represent 
                   the y-axis, and the third dimension represents the x-axis.
        Stokes_U - A Numpy array containing the value of Stokes U at each
                   pixel of the image. Must have the same size as the Stokes
                   Q array. Must satisfy the same conventions as the Stokes Q
                   array.
        wave_sq_space - The wavelength squared spacing between adjacent slices 
                   of the Stokes Q and U arrays, in m^2.  
                   
    Output
        The order of the output quantities is dQ_dl, dU_dl, where l is the 
        wavelength squared.
        dQ_dl - Partial derivative of Stokes Q with respect to the wavelength 
                squared. This array has the same size as the Stokes Q array.
        dU_dl - Partial derivative of Stokes Q with respect to the wavelength 
                squared. This array has the same size as the Stokes Q array.
    '''
    
    # To calculate the derivatives of Stokes Q and U along the wavelength 
    # squared axis, the gradient function of numpy is used.
    # Numpy performs the calculation by using central differences in the 
    # interior of the provided arrays, and using first differences at the
    # boundaries of the array. This function returns an array specifying the
    # derivative of the given array along the specified axis.
    
    # Calculate the partial derivative of Stokes Q with respect to the 
    # wavelength squared. wave_sq_space is used to make sure that the returned 
    # matrix has the correct units.
    dQ_dl = np.gradient(Stokes_Q, wave_sq_space, axis = 0)
    
    # Calculate the partial derivative of Stokes U with respect to the 
    # wavelength squared.
    dU_dl = np.gradient(Stokes_U, wave_sq_space, axis = 0)
    
    # Return the calculated partial derivatives to the caller.
    return dQ_dl, dU_dl