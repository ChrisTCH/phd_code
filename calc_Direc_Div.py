#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, and calculates the modulus of the directional derivative of the       #
# complex polarisation vector over a range of angles theta with respect to    #
# the horizontal axis of the image. This produces a data cube where the       #
# dimensions are x, y and theta, so that each slice of the image cube shows   #
# the modulus of the directional derivative for a certain value of theta. The #
# formula is given on page 60 of PhD Logbook 1, and page 53 of PhD Logbook 2. #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 16/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Direc_Div, which will calculate the directional 
# derivative of the complex polarisation vector over a range of angles theta 
# from the first order partial derivatives of the Stokes Q and U values.
def calc_Direc_Div(dQ_dy = None, dQ_dx = None, dU_dy = None, dU_dx = None,\
num_theta = 2):
    '''
    Description
        This function calculates the modulus of the directional derivative of 
        the complex polarisation vector over a range of angles theta with
        respect to the horizontal axis of the image, when given the first order
        partial derivatives of the Stokes Q and U values with respect to the x
        and y axes for the image. The formula for this quantity is given on page
        60 of PhD Logbook 1, and page 53 of PhD Logbook 2.
        
    Required Input
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                       vertical and horizontal axes of the image respectively.
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays must have the same size as the arrays
                       containing the values of the partial derivatives of
                       Stokes Q.
        num_theta - The number of theta values for which to evaluate the 
                    modulus of the directional derivative of the polarisation
                    vector. These values are equally distributed between -180 
                    and 180 degrees. Must be an integer. If a float is provided
                    then it will be rounded down to the nearest integer. Be
                    very careful with this parameter, as large values will
                    greatly increase the data size of the produced matrix.
                   
    Output
        direc_div - A Numpy array containing the value of the directional
                    derivative of the complex polarisation vector at each
                    pixel in the image, for each value of theta used. This is
                    a three dimensional array, whose first dimension is the 
                    x-axis of the image, the second dimension is the y-axis
                    of the image, and the third dimension is the theta axis. 
                    Each slice taken across the third axis produces an image of
                    the directional derivative for a specific value of theta.
        theta - The Numpy array of theta values for which the directional 
                derivative of the complex polarisation vector has been 
                calculated. Values are in degrees.
    '''
    
    # Create an array of theta values for which the directional derivative
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
    
    # Unpack the tuple representing the size of the provided partial derivative
    # arrays, so that an array which will hold all of the directional derivative
    # values can be produced.
    y_length, x_length = np.shape(dQ_dy)
    
    # Create an empty array that will store the produced directional derivative
    # images for each theta value. Each entry in this array will correspond to
    # a particular value of theta.
    direc_div = np.zeros((len(theta), y_length, x_length))
    
    # Iterate through the different values of theta for which the directional 
    # derivative is to be calculated.
    for i in range(len(theta)):
        # Calculate the directional derivative image for this value of theta
        direc_div[i] = np.sqrt(cos2theta[i] * sum_x + sin2theta[i] * sum_y +\
        2 * cos_sin_theta[i] * sum_xy)
    
    # Return the directional derivative data cube to the caller
    return direc_div, theta