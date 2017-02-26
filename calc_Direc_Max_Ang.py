#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# order partial derivatives of Stokes Q and U with respect to the x and y     #
# axes, and calculates the angle in the image plane for which the directional #
# derivative of the complex polarisation vector is maximised. This produces   #
# an image of the angle for which the directional derivative is maximised.    #
# The formula is given on page 70 of PhD Logbook 6.                           #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 20/1/2016                                                       #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Direc_Max_Ang, which will calculate the angle in the
# image plane for which the directional derivative of the complex polarisation
# vector is maximised. 
def calc_Direc_Max_Ang(dQ_dy = None, dQ_dx = None,dU_dy = None, dU_dx = None):
    '''
    Description
        This function calculates the angle for which the directional derivative
        of the complex polarisation vector is maximised, when given the first 
        order partial derivatives of the Stokes Q and U values with respect to
        the x and y axes for the image, as well as Stokes Q and U themselves. 
        The formula is given on page 70 of PhD Logbook 6.
        
    Required Input
        dQ_dy, dQ_dx - Partial derivatives of Stokes Q with respect to the
                       vertical and horizontal axes of the image respectively.
        dU_dy, dU_dx - Partial derivatives of Stokes U with respect to the
                       vertical and horizontal axes of the image respectively.
                       These arrays must have the same size as the arrays
                       containing the values of the partial derivatives of
                       Stokes Q.
                   
    Output
        direc_max_ang - A Numpy array containing the value of the angle for which 
                   the directional derivative of the complex polarisation vector
                   is maximised, at each pixel in the image. Has the same shape 
                   as the provided Stokes Q and U arrays. In degrees, between
                   plus or minus 90 degrees.
    '''
    
    # Calculate the numerator needed to calculate the angle that maximises the
    # directional derivative using the sin formula
    numer_sin = 2.0*(dQ_dx*dQ_dy + dU_dx*dU_dy)

    # Calculate the numerator needed to calculate the angle that maximises the
    # directional derivative using the cos formula
    numer_cos = -1.0*(np.power(dQ_dy,2.0) - np.power(dQ_dx,2.0) +\
     np.power(dU_dy,2.0) - np.power(dU_dx,2.0) )

    # Calculate the polarisation gradient squared, as this will simplify the
    # calculation
    polar_grad_sq = np.power(dQ_dx,2.0) + np.power(dU_dx,2.0) +\
     np.power(dQ_dy,2.0) + np.power(dU_dy,2.0)

    # Calculate the denominator needed to calculate the angle that maximises the
    # directional derivative
    denom = np.sqrt(np.power(polar_grad_sq,2.0) - \
        4.0 * np.power(dQ_dx*dU_dy - dQ_dy*dU_dx,2.0))

    # Calculate the quantities that we will calculate the inverse sin and cos
    # of, so that we can check all of the values are valid
    inv_sin = numer_sin/denom
    inv_cos = numer_cos/denom

    # Find entries where the numerator for the sin calculation is larger than
    # the denominator
    fix_sin_one = inv_sin > 1.0

    # Convert the improper values, so that they won't break the inverse sin
    # function. 
    inv_sin[fix_sin_one] = 1.0

    # Now find entries where the numerator for the sin calculation has an 
    # absolute value larger than the denominator, and is negative
    fix_sin_neg_one = inv_sin < -1.0

    # Convert the improper values, so they won't break the inverse sin function
    inv_sin[fix_sin_neg_one] = -1.0

    # Find entries where the numerator for the cos calculation is larger than
    # the denominator
    fix_cos_one = inv_cos > 1.0

    # Convert the improper values, so that they won't break the inverse cos
    # function. 
    inv_cos[fix_cos_one] = 1.0

    # Now find entries where the numerator for the cos calculation has an 
    # absolute value larger than the denominator, and is negative
    fix_cos_neg_one = inv_cos < -1.0

    # Convert the improper values, so they won't break the inverse cos function
    inv_cos[fix_cos_neg_one] = -1.0

    # Calculate double the angle for which the directional derivative is 
    # maximised, at each pixel, using inverse sin. In radians, between +/- pi/2.
    double_theta_sin = np.arcsin(inv_sin)

    # Calculate double the angle for which the directional derivative is 
    # maximised, at each pixel, using inverse cos. In radians, between 0 and pi.
    double_theta_cos = np.arccos(inv_cos)

    # Find the entries in the array where the angle returned by inverse cos
    # is more than pi/2, as in this case the value for double the angle lies
    # in the second or third quadrant, so we need to adjust the angle that
    # is measured by inverse sin
    theta_cos_entries = double_theta_cos > np.pi/2.0

    # Find the entries of the array where the measured angle was in the first
    # quadrant, but it is supposed to be in the second quadrant
    second_quad = np.logical_and(theta_cos_entries, double_theta_sin >= 0)

    # Find the entries of the array where the measured angle was in the fourth
    # quadrant, but it is supposed to be in the third quadrant
    third_quad = np.logical_and(theta_cos_entries, double_theta_sin < 0)

    # For entries that are supposed to be in the second quadrant, adjust the
    # value of the measured angle
    double_theta_sin[second_quad] = np.pi - double_theta_sin[second_quad]

    # For entries that are supposed to be in the third quadrant, adjust the
    # value of the measured angle
    double_theta_sin[third_quad] = -1.0 * np.pi - double_theta_sin[third_quad]
    
    # Now calculate the angle that maximises the directional derivative in
    # degrees, between +/- 90 degrees
    direc_max_ang = np.rad2deg(0.5 * double_theta_sin)

    # Return the angle that maximises the directional derivative to the caller
    return direc_max_ang