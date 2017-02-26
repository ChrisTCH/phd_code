#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of both the   #
# first and second order partial derivatives of Stokes Q and U with respect   #
# to wavelength squared, and calculates the wavelength curvature of the       #
# complex polarisation vector. The wavelength curvature is the curvature of   #
# the path in the Q-U plane traced out when we vary the wavelength at a       #
# pixel. The formula is given on page 63 of PhD Logbook 5.                    #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 27/10/2016                                                      #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Wav_Curv, which will calculate the wavelength 
# curvature of the complex polarisation vector from the first and second order 
# partial derivatives of the Stokes Q and U values.
def calc_Wav_Curv(dQ_dl = None, dU_dl = None, d2Q_dl2 = None, d2U_dl2 = None):
    '''
    Description
        This function calculates the wavelength curvature of the complex
        polarisation vector, when given the first and second order partial 
        derivatives of the Stokes Q and U values with respect to the wavelength
        squared. The wavelength curvature is the curvature of the path in the 
        Q-U plane traced out as the observing wavelength changes. The formula 
        for this quantity is given on page 63 of PhD Logbook 5.
        
    Required Input
        dQ_dl - A Numpy array specifying the partial derivative of Stokes Q with
                respect to the wavelength squared.
        dU_dl - A Numpy array specifying the partial derivative of Stokes U with
                respect to the wavelength squared. This array must have the same
                size as the array containing the values of the partial 
                derivative of Stokes Q.
        d2Q_dl2 - A Numpy array specifying the second order partial derivative 
                of Stokes Q with respect to the wavelength squared. This array 
                must have the same size as the array containing the values of 
                the partial derivative of Stokes Q.
        d2U_dl2 - A Numpy array specifying the second order partial derivative 
                of Stokes U with respect to the wavelength squared. This array 
                must have the same size as the array containing the values of 
                the partial derivative of Stokes Q.
                   
    Output
        wav_curv - A Numpy array containing the value of the wavelength
                curvature of the complex polarisation vector at each pixel in 
                the image. This array has the same size as the array containing 
                the values of the partial derivative of Stokes Q.
    '''

    # Calculate the denominator of the wavelength curvature
    denom = np.power( np.power(dQ_dl,2.0) + np.power(dU_dl,2.0),1.5)
    
    # Calculate the numerator of the wavelength curvature
    numer = dQ_dl * d2U_dl2 - dU_dl * d2Q_dl2
    
    # Calculate the wavelength curvature image for this value of theta
    wav_curv = numer / denom
    
    # Calculate the numerator only to avoid a noisy data cube
    # wav_curv = numer
    
    # Return the wavelength curvature to the caller
    return wav_curv