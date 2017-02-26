#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of both the   #
# first and second order partial derivatives of Stokes Q and U with respect   #
# to the x and y axes, and calculates the directional curvature of the        #
# complex polarisation vector in the direction of the image plane that        #
# maximises the amplitude of the directional derivative. The directional      #
# curvature is the curvature of the path in the Q-U plane traced out when     #
# moving at an angle theta with respect to the horizontal axis.               #
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

# Import the function that calculates the angle that maximises the amplitude
# of the directional derivative
from calc_Direc_Max_Ang import calc_Direc_Max_Ang 

# Define the function calc_Curv4MaxDirecDeriv, which will calculate the 
# directional curvature of the complex polarisation vector in the direction
# that maximises the amplitude of the directional derivative, from the first and
# second order partial derivatives of the Stokes Q and U values.
def calc_Curv4MaxDirecDeriv(dQ_dy = None, dQ_dx = None, dU_dy = None,\
dU_dx = None, d2Q_dy2 = None, d2Q_dx2 = None, d2U_dy2 = None, d2U_dx2 = None,\
d2Q_dydx = None, d2U_dydx = None):
    '''
    Description
        This function calculates the directional curvature of the complex
        polarisation vector in the direction that maximises the amplitude of the
        directional derivative, when given Stokes Q and U, and the first and 
        second order partial derivatives of Stokes Q and U with respect to the x
        and y axes for the image. The directional curvature is the curvature of
        the path in the Q-U plane traced out when moving at an angle theta with
        respect to the horizontal axis.
        
    Required Input
        Q, U - Stokes Q and Stokes U respectively. Must be Numpy arrays of the
               same size, where each entry of the array is a float.
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                vertical and horizontal axes of the image respectively. Must 
                have the same size as the Stokes Q and U arrays.
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                vertical and horizontal axes of the image respectively. Must 
                have the same size as the Stokes Q and U arrays.
        d2Q_dy2, d2Q_dx2 - Second order partial derivatives of Stokes Q with
                respect to the vertical and horizontal axes of the image
                respectively. Must have the same size as the Stokes Q and U 
                arrays.
        d2U_dy2, d2U_dx2 - Second order partial derivatives of Stokes U with
                respect to the vertical and horizontal axes of the image
                respectively. Must have the same size as the Stokes Q and U 
                arrays.
        d2Q_dydx, d2U_dydx - Second order partial derivatives of Stokes Q and U
                with respect to the vertical and horizontal axes of the 
                image. These arrays must have the same size as the
                arrays containing the values of the partial derivatives
                of Stokes Q.
                   
    Output
        curv_max_direc - A Numpy array containing the value of the directional
                curvature of the complex polarisation vector at each pixel in 
                the image, in the direction that maximises the directional 
                derivative.
    '''
    
    # Calculate the angle that maximises the directional derivative
    ang_max_deg = calc_Direc_Max_Ang(dQ_dy, dQ_dx, dU_dy, dU_dx)
    
    # Convert the angle that maximises the directional derivative to radians
    ang_max = np.deg2rad(ang_max_deg)
    
    # Calculate the sum of the squared partial derivatives of Stokes Q and U
    # in the x direction
    sum_x = np.power(dQ_dx,2.0) + np.power(dU_dx,2.0)
    
    # Calculate the sum of the squared partial derivatives of Stokes Q and U
    # in the y direction
    sum_y = np.power(dQ_dy,2.0) + np.power(dU_dy,2.0)
    
    # Calculate the sum of the multiplied partial derivatives of Stokes Q and U
    # in the x and y directions
    sum_xy = dQ_dx * dQ_dy + dU_dx * dU_dy
    
    # Calculate the cosine squared of the angle values
    cos2theta = np.power(np.cos(ang_max), 2.0)
    
    # Calculate the sine squared of the angle values
    sin2theta = np.power(np.sin(ang_max), 2.0)
    
    # Calculate the product of the cosine and sine of the angle values
    cos_sin_theta = np.cos(ang_max) * np.sin(ang_max)
    
    # Calculate the cosine cubed of the angle values
    cos3theta = np.power(np.cos(ang_max), 3.0)
    
    # Calculate the sine cubed of the angle values
    sin3theta = np.power(np.sin(ang_max), 3.0)
    
    # Calculate the product of cos squared with sin
    cos2_sin_theta = cos2theta * np.sin(ang_max)
    
    # Calculate the product of sin squared with cos
    cos_sin2_theta = np.cos(ang_max) * sin2theta
    
    # Calculate the denominator of the directional curvature
    denom = np.power(cos2theta * sum_x + sin2theta * sum_y +\
    2 * cos_sin_theta * sum_xy, 1.5)
    
    # Calculate the numerator of the directional curvature
    numer = cos3theta * (dQ_dx * d2U_dx2 - dU_dx * d2Q_dx2) +\
    2 * cos2_sin_theta * (dQ_dx * d2U_dydx - dU_dx * d2Q_dydx) +\
    cos2_sin_theta * (dQ_dy * d2U_dx2 - dU_dy * d2Q_dx2) +\
    2 * cos_sin2_theta * (dQ_dy * d2U_dydx - dU_dy * d2Q_dydx) +\
    cos_sin2_theta * (dQ_dx * d2U_dy2 - dU_dx * d2Q_dy2) +\
    sin3theta * (dQ_dy * d2U_dy2 - dU_dy * d2Q_dy2)
    
    # Calculate the directional curvature image for the angles
    curv_max_direc = np.absolute(numer / denom)
    
    # Calculate the numerator only to avoid a noisy data cube
    # curv_max_direc = np.absolute(numer)
    
    # Return the directional curvature data cube to the caller
    return curv_max_direc