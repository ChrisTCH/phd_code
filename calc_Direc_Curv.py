#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of both the   #
# first and second order partial derivatives of Stokes Q and U with respect   #
# to the x and y axes, and calculates the directional curvature of the        #
# complex polarisation vector over a range of angles theta with respect to    #
# the horizontal axis of the image. The directional curvature is the          #
# curvature of the path in the Q-U plane traced out when moving at an angle   #
# theta with respect to the horizontal axis. This produces a data cube where  #
# the dimensions are x, y and theta, so that each slice of the image cube     #
# shows the directional curvature for a certain value of theta. The formula   #
# is given on page 22 of PhD Logbook 2.                                       #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 16/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Direc_Curv, which will calculate the directional 
# curvature of the complex polarisation vector over a range of angles theta 
# from the first and second order partial derivatives of the Stokes Q and U 
# values.
def calc_Direc_Curv(dQ_dy = None, dQ_dx = None, dU_dy = None, dU_dx = None,\
d2Q_dy2 = None, d2Q_dx2 = None, d2U_dy2 = None, d2U_dx2 = None, num_theta = 2):
    '''
    Description
        This function calculates the directional curvature of the complex
        polarisation vector over a range of angles theta with respect to the
        horizontal axis of the image, when given the first and second order
        partial derivatives of the Stokes Q and U values with respect to the x
        and y axes for the image. The directional curvature is the curvature of
        the path in the Q-U plane traced out when moving at an angle theta with
        respect to the horizontal axis.The formula for this quantity is given
        on page 22 of PhD Logbook 2.
        
    Required Input
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                       vertical and horizontal axes of the image respectively.
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays must have the same size as the arrays
                       containing the values of the partial derivatives of
                       Stokes Q.
        d2Q_dy2, d2Q_dx2 - Second order partial derivatives of Stokes Q with
                       respect to the vertical and horizontal axes of the image
                       respectively. These arrays must have the same size as the
                       arrays containing the values of the partial derivatives
                       of Stokes Q.
        d2U_dy2, d2U_dx2 - Second order partial derivatives of Stokes U with
                       respect to the vertical and horizontal axes of the image
                       respectively. These arrays must have the same size as the
                       arrays containing the values of the partial derivatives
                       of Stokes Q.
        num_theta - The number of theta values for which to evaluate the 
                    directional curvature of the polarisation vector. These
                    values are equally distributed between -180 and 180 degrees.
                    Must be an integer. If a float is provided then it will be
                    rounded down to the nearest integer. Be very careful with
                    this parameter, as large values will greatly increase the
                    data size of the produced matrix.
                   
    Output
        direc_curv - A Numpy array containing the value of the directional
                    curvature of the complex polarisation vector at each
                    pixel in the image, for each value of theta used. This is
                    a three dimensional array, whose first dimension is the 
                    x-axis of the image, the second dimension is the y-axis
                    of the image, and the third dimension is the theta axis. 
                    Each slice taken across the third axis produces an image of
                    the directional curvature for a specific value of theta.
        theta - The Numpy array of theta values for which the directional 
                derivative of the complex polarisation vector has been 
                calculated. Values are in degrees.
    '''
    
    # Create an array of theta values for which the directional curvature
    # will be calculated
    theta = np.linspace(-180.0, 180.0, num = np.floor(num_theta),\
    endpoint = False)
    
    # Convert the angles theta to radians
    theta_rad = np.deg2rad(theta)
    
    # Calculate the sum of the squared partial derivatives of Stokes Q and U
    # in the x direction
    sum_x = np.power(dQ_dx,2.0) + np.power(dU_dx,2.0)
    
    # Calculate the sum of the squared partial derivatives of Stokes Q and U
    # in the y direction
    sum_y = np.power(dQ_dy,2.0) + np.power(dU_dy,2.0)
    
    # Calculate the sum of the multiplied partial derivatives of Stokes Q and U
    # in the x and y directions
    sum_xy = dQ_dx * dQ_dy + dU_dx * dU_dy
    
    # Calculate the cosine squared of the theta values
    cos2theta = np.power(np.cos(theta_rad), 2.0)
    
    # Calculate the sine squared of the theta values
    sin2theta = np.power(np.sin(theta_rad), 2.0)
    
    # Calculate the product of the cosine and sine of the theta values
    cos_sin_theta = np.cos(theta_rad) * np.sin(theta_rad)
    
    # Calculate the cosine cubed of the theta values
    cos3theta = np.power(np.cos(theta_rad), 3.0)
    
    # Calculate the sine cubed of the theta values
    sin3theta = np.power(np.sin(theta_rad), 3.0)
    
    # Calculate the product of cos squared with sin
    cos2_sin_theta = cos2theta * np.sin(theta_rad)
    
    # Calculate the product of sin squared with cos
    cos_sin2_theta = np.cos(theta_rad) * sin2theta
    
    # Unpack the tuple representing the size of the provided partial derivative
    # arrays, so that an array which will hold all of the directional curvature
    # values can be produced.
    y_length, x_length = np.shape(dQ_dy)
    
    # Create an empty array that will store the produced directional curvature
    # images for each theta value. Each entry in this array will correspond to
    # a particular value of theta.
    direc_curv = np.zeros((len(theta), y_length, x_length))
    
    # Iterate through the different values of theta for which the directional 
    # curvature is to be calculated.
    for i in range(len(theta)):
        # Calculate the denominator of the directional curvature
        denom = np.power(cos2theta[i] * sum_x + sin2theta[i] * sum_y +\
        2 * cos_sin_theta[i] * sum_xy, 1.5)
        
        # Calculate the numerator of the directional curvature
        numer = cos3theta[i] * (dQ_dx * d2U_dx2 - dU_dx * d2Q_dx2) +\
        cos2_sin_theta[i] * (dQ_dy * d2U_dx2 - dU_dy * d2Q_dx2) +\
        cos_sin2_theta[i] * (dQ_dx * d2U_dy2 - dU_dx * d2Q_dy2) +\
        sin3theta[i] * (dQ_dy * d2U_dy2 - dU_dy * d2Q_dy2)
        
        # Calculate the directional curvature image for this value of theta
        direc_curv[i] = numer / denom
        
        # Calculate the numerator only to avoid a noisy data cube
        #direc_curv[i] = numer
        
        # Calculate the radius of curvature
        #direc_curv[i] = denom / numer
    
    # Return the directional curvature data cube to the caller
    return direc_curv, theta