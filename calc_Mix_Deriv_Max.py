#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, and calculates the maximum amplitude of the mixed derivative of the   #
# polarisation. This amplitude is the maximum rate of change of the           #
# directional derivative with respect to wavelength. This produces an image   #
# of the maximum amplitude of the mixed derivative. The formula is given on   #
# page 66 of PhD Logbook 5.                                                   #
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

# Define the function calc_Mix_Deriv_Max, which will calculate the maximum 
# amplitude of the mixed derivative of the complex polarisation vector. This 
# amplitude is the maximum rate of change of the directional derivative with 
# respect to wavelength.
def calc_Mix_Deriv_Max(dQ_dy = None, dQ_dx = None, dU_dy = None, dU_dx = None,\
  wave_sq_space = 1.0):
    '''
    Description
        This function calculates the maximum amplitude of the mixed derivative 
        of the complex polarisation vector when given the first order partial 
        derivatives of the Stokes Q and U values with respect to the x and y 
        axes for the image. This amplitude is the maximum rate of change of the 
        directional derivative with respect to wavelength. The formula is given 
        on page 66 of PhD Logbook 5.
        
    Required Input
        dQ_dy, dQ_dx - Numpy arrays containing the partial derivatives of Stokes
                Q with respect to the vertical and horizontal axes of the image 
                respectively. The arrays must conform to the convention that the
                first dimension represents the square of the wavelength, the 
                second dimension represent the y-axis, and the third dimension 
                represents the x-axis. The arrays must have the same size.
        dU_dy, dU_dx - Numpy arrays containing the partial derivatives of Stokes
                U with respect to the vertical and horizontal axes of the image 
                respectively. The arrays must conform to the convention that the
                first dimension represents the square of the wavelength, the 
                second dimension represent the y-axis, and the third dimension 
                represents the x-axis. The arrays must have the same size.
        wave_sq_space - The wavelength squared spacing between adjacent slices 
                of the Stokes Q and U arrays, in m^2. 
                   
    Output
        mix_deriv_max - A Numpy array containing the value of the maximum 
                amplitude of the mixed derivative of the complex polarisation 
                vector at each pixel in the image. Has the same shape as the 
                provided partial derivative arrays.
    '''
    
    # Calculate the derivative of the partial derivatives of Stokes Q, with 
    # respect to the wavelength squared
    d2Q_dldy = np.gradient(dQ_dy, wave_sq_space, axis = 0)
    d2Q_dldx = np.gradient(dQ_dx, wave_sq_space, axis = 0)

    # Calculate the derivative of the partial derivatives of Stokes U, with 
    # respect to the wavelength squared
    d2U_dldy = np.gradient(dU_dy, wave_sq_space, axis = 0)
    d2U_dldx = np.gradient(dU_dx, wave_sq_space, axis = 0)

    # Calculate the polarisation gradient squared, with the derivatives replaced
    # by mixed derivatives, as this will simplify the calculation of the maximum
    # amplitude of the mixed derivative
    polar_grad_sq = np.power(d2Q_dldx,2.0) + np.power(d2U_dldx,2.0) +\
     np.power(d2Q_dldy,2.0) + np.power(d2U_dldy,2.0)
    
    # Calculate the maximum amplitude of the mixed derivative at each
    # pixel, squared
    mix_max_sq = 0.5*(polar_grad_sq + np.sqrt(np.power(polar_grad_sq,2.0)\
     - 4.0*np.power(d2Q_dldx*d2U_dldy - d2Q_dldy*d2U_dldx,2.0) ) )
    
    # Calculate the maximum amplitude of the mixed derivative
    mix_deriv_max = np.sqrt(mix_max_sq)

    # Return the amplitude of the mixed derivative to the caller
    return mix_deriv_max