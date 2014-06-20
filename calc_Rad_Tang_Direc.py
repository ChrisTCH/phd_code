#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, as well as Stokes Q and U, and calculates the radial and tangential   #
# components of the directional derivative of the complex polarisation        #
# vector over a range of angles theta with respect to the horizontal axis of  #
# the image. This produces two data cubes where the dimensions are x, y and   #
# theta, so that each slice of an image cube shows the radial or tangential   #
# component of the directional derivative for a certain value of theta. The   #
# formulae are given on page 57 of PhD Logbook 2.                             #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 17/6/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Rad_Tang_Direc, which will calculate the radial and
# tangential components of the directional derivative of the complex
# polarisation vector over a range of angles theta from the first order partial
# derivatives of the Stokes Q and U values, as well as Stokes Q and U 
# themselves.
def calc_Rad_Tang_Direc(Q = None, U = None, dQ_dy = None, dQ_dx = None,\
dU_dy = None, dU_dx = None, num_theta = 2):
    '''
    Description
        This function calculates the radial and tangential components of the 
        directional derivative of the complex polarisation vector over a range
        of angles theta with respect to the horizontal axis of the image, when
        given the first order partial derivatives of the Stokes Q and U values
        with respect to the x and y axes for the image, as well as Stokes Q and
        U themselves. The formulae for these quantities are given
        on page 57 of PhD Logbook 2.
        
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
        num_theta - The number of theta values for which to evaluate the 
                    radial and tangential components of the directional 
                    derivative of the polarisation vector. These values are 
                    equally distributed between -180 and 180 degrees. Must be
                    an integer. If a float is provided then it will be rounded
                    down to the nearest integer. Be very careful with this
                    parameter, as large values will greatly increase the data
                    size of the produced matrix.
                   
    Output
        rad_comp - A Numpy array containing the value of the radial component
                   of the directional derivative of the complex polarisation
                   vector at each pixel in the image, for each value of theta
                   used. This is a three dimensional array, whose first 
                   dimension is the x-axis of the image, the second dimension
                   is the y-axis of the image, and the third dimension is the
                   theta axis. Each slice taken across the third axis produces
                   an image of the radial component of the directional 
                   derivative for a specific value of theta.
        tang_comp - A Numpy array containing the value of the tangential
                   component of the directional derivative of the complex
                   polarisation vector at each pixel in the image, for each
                   value of theta used. This is a three dimensional array, 
                   whose first dimension is the x-axis of the image, the second
                   dimension is the y-axis of the image, and the third dimension
                   is the theta axis. Each slice taken across the third axis
                   produces an image of the tangential component of the
                   directional derivative for a specific value of theta.
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
    
    # Calculate the cosine of theta
    cos_theta = np.cos(theta_rad)
    
    # Calculate the sine of theta
    sin_theta = np.sin(theta_rad)
    
    # Calculate the cosine of the argument of a point in the Q-U plane
    cos_arg = np.cos(np.arctan2(U,Q))
    
    # Calculate the sine of the argument of a point in the Q-U plane
    sin_arg = np.sin(np.arctan2(U,Q))
    
    # Unpack the tuple representing the size of the provided partial derivative
    # arrays, so that an array which will hold all of the directional derivative
    # values can be produced.
    y_length, x_length = np.shape(dQ_dy)
    
    # Create an empty array that will store the produced radial component
    # images for each theta value. Each entry in this array will correspond to
    # a particular value of theta.
    rad_comp = np.zeros((len(theta), y_length, x_length))
    
    # Create an empty array that will store the produced tangential component
    # images for each theta value. Each entry in this array will correspond to
    # a particular value of theta.
    tang_comp = np.zeros((len(theta), y_length, x_length))
    
    # Iterate through the different values of theta for which the radial and 
    # tangential components are to be calculated.
    for i in range(len(theta)):
        # Calculate the image of the radial component of the directional 
        # derivative for this value of theta
        rad_comp[i] = cos_arg * (dQ_dx * cos_theta[i] + dQ_dy * sin_theta[i])\
        + sin_arg * (dU_dx * cos_theta[i] + dU_dy * sin_theta[i])
        
        # Calculate the image of the tangential component of the directional
        # derivative for this value of theta
        tang_comp[i] = -sin_arg * (dQ_dx * cos_theta[i] + dQ_dy * sin_theta[i])\
        + cos_arg * (dU_dx * cos_theta[i] + dU_dy * sin_theta[i])
    
    # Return the radial and tangential components of the directional derivative
    # to the caller, along with the theta values
    return rad_comp, tang_comp, theta