#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the second #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, and calculates the directional acceleration of the complex            #
# polarisation vector over a range of angles theta with respect to the        #
# horizontal axis of the image. The directional acceleration is the magnitude #
# of the acceleration vector that results from moving along the path in the   #
# Q-U plane traced out when moving at an angle theta with respect to the      #
# horizontal axis. This produces a data cube where the dimensions are x, y    #
# and theta, so that each slice of the image cube shows the directional       #
# acceleration for a certain value of theta. The formula is given on page 76  #
# of PhD Logbook 2.                                                           #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 1/7/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Direc_Accel, which will calculate the directional 
# acceleration of the complex polarisation vector over a range of angles theta 
# from second order partial derivatives of the Stokes Q and U values.
def calc_Direc_Accel(d2Q_dy2 = None, d2Q_dx2 = None, d2U_dy2 = None,\
d2U_dx2 = None, num_theta = 2):
    '''
    Description
        This function calculates the directional acceleration of the complex
        polarisation vector over a range of angles theta with respect to the
        horizontal axis of the image, when given the second order partial
        derivatives of the Stokes Q and U values with respect to the x and y
        axes for the image. The directional acceleration is the magnitude of the
        acceleration vector that results from moving along the path in the Q-U
        plane traced out when moving at an angle theta with respect to the 
        horizontal axis. The formula for this quantity is given on page 76 of
        PhD Logbook 2.
        
    Required Input
        d2Q_dy2, d2Q_dx2 - Second order partial derivatives of Stokes Q with
                       respect to the vertical and horizontal axes of the image
                       respectively. These arrays must have the same size as 
                       each other.
        d2U_dy2, d2U_dx2 - Second order partial derivatives of Stokes U with
                       respect to the vertical and horizontal axes of the image
                       respectively. These arrays must have the same size as the
                       second order partial derivative arrays for Stokes Q.
        num_theta - The number of theta values for which to evaluate the 
                    directional curvature of the polarisation vector. These
                    values are equally distributed between -180 and 180 degrees.
                    Must be an integer. If a float is provided then it will be
                    rounded down to the nearest integer. Be very careful with
                    this parameter, as large values will greatly increase the
                    data size of the produced matrix.
                   
    Output
        direc_accel - A Numpy array containing the value of the directional
                    acceleration of the complex polarisation vector at each
                    pixel in the image, for each value of theta used. This is
                    a three dimensional array, whose first dimension is the 
                    x-axis of the image, the second dimension is the y-axis
                    of the image, and the third dimension is the theta axis. 
                    Each slice taken across the third axis produces an image of
                    the directional acceleration for a specific value of theta.
        theta - The Numpy array of theta values for which the directional 
                acceleration of the complex polarisation vector has been 
                calculated. Values are in degrees.
    '''
    
    # Create an array of theta values for which the directional acceleration
    # will be calculated
    theta = np.linspace(-180.0, 180.0, num = np.floor(num_theta),\
    endpoint = False)
    
    # Convert the angles theta to radians
    theta_rad = np.deg2rad(theta)
    
    # Calculate the sum of the squared second order partial derivatives of
    # Stokes Q and U in the x direction
    sum_xx = np.power(d2Q_dx2,2.0) + np.power(d2U_dx2,2.0)
    
    # Calculate the sum of the squared second order partial derivatives of
    # Stokes Q and U in the y direction
    sum_yy = np.power(d2Q_dy2,2.0) + np.power(d2U_dy2,2.0)
    
    # Calculate the sum of the multiplied second order partial derivatives of
    # Stokes Q and U in the x and y directions
    sum_xy = d2Q_dx2 * d2Q_dy2 + d2U_dx2 * d2U_dy2
    
    # Calculate the cosine of the theta values to the power of 4
    cos4theta = np.power(np.cos(theta_rad), 4.0)
    
    # Calculate the sine of the theta values to the power of 4
    sin4theta = np.power(np.sin(theta_rad), 4.0)
    
    # Calculate the product of the cosine and sine of the theta values squared
    cos_sin_theta_2 = np.power(np.cos(theta_rad) * np.sin(theta_rad), 2.0)
    
    # Unpack the tuple representing the size of the provided partial derivative
    # arrays, so that an array which will hold all of the directional 
    # acceleration values can be produced.
    y_length, x_length = np.shape(d2Q_dy2)
    
    # Create an empty array that will store the produced directional 
    # acceleration images for each theta value. Each entry in this array will
    # correspond to a particular value of theta.
    direc_accel = np.zeros((len(theta), y_length, x_length))
    
    # Iterate through the different values of theta for which the directional 
    # acceleration is to be calculated.
    for i in range(len(theta)):
        # Calculate the directional acceleration for this value of theta
        direc_accel[i] = np.sqrt(cos4theta[i] * sum_xx + 2 * cos_sin_theta_2[i]\
        * sum_xy + sin4theta[i] * sum_yy)

    # Return the directional acceleration data cube to the caller
    return direc_accel, theta