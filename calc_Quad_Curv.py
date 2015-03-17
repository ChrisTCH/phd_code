#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the first  #
# and second order partial derivatives of Stokes Q and U with respect to the  #
# x and y axes, and calculates the quadrature of the curvatures in the x and  #
# y directions at each pixel. The formula is given on page 96 of PhD Logbook  #
# 1, and page 23 of PhD Logbook 2.                                            #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 14/6/2014                                                       #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Quad_Curv, which will calculate the quadrature of
# the curvatures from both the first and second order partial derivatives of
# the Stokes Q and U values.
def calc_Quad_Curv(dQ_dy = None, dQ_dx = None, dU_dy = None, dU_dx = None,\
d2Q_dy2 = None, d2Q_dx2 = None, d2U_dy2 = None, d2U_dx2 = None):
    '''
    Description
        This function calculates the quadrature of the curvatures in the x and
        y directions of the image, when given both the first and second order
        partial derivatives of the Stokes Q and U values with respect to the x
        and y axes for the image. The formula for this quantity is given on page
        96 of PhD Logbook 1, and page 23 of PhD Logbook 2.
        
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
                   
    Output
        quad_curv - A Numpy array containing the value of the quadrature of the
                    curvatures at each point. This array has the same size as
                    the input arrays.
    '''
    
    # The formula for the quadrature of the curvatures involves the square root
    # of two terms, each of which involves a numerator and a denominator. The 
    # first term is the curvature in the x direction squared, and the second 
    # term is the curvature in the y direction squared. I will calculate the
    # numerator and denominator of each term separately, then calculate each
    # term, before calculating the final quadrature.
    
    # Calculate the numerator of the first term.
    num_1 = dQ_dx * d2U_dx2 - dU_dx * d2Q_dx2
    
    # Calculate the denominator of the first term
    denom_1 = np.power(np.power(dQ_dx,2.0) + np.power(dU_dx,2.0) ,1.5)
    #print 'Smallest value in denom_1 is: {}'.format(np.min(denom_1))
    #print 'Average value in denom_1 is {}'.format(np.mean(denom_1))
    #print 'Median value in denom_1 is {}'.format(np.median(denom_1))
    
    # Check the denominator for very small values
    
    # Calculate the first term
    term_1 = np.power(num_1 / denom_1, 2.0)
    
    # Calculate the numerator of the second term.
    num_2 = dQ_dy * d2U_dy2 - dU_dy * d2Q_dy2
    
    # Calculate the denominator of the second term
    denom_2 = np.power(np.power(dQ_dy,2.0) + np.power(dU_dy,2.0) ,1.5)
    #print 'Smallest value in denom_2 is: {}'.format(np.min(denom_2))
    #print 'Average value in denom_2 is {}'.format(np.mean(denom_2))
    #print 'Median value in denom_2 is {}'.format(np.median(denom_2))
    
    # Calculate the second term
    term_2 = np.power(num_2 / denom_2, 2.0)
    
    # Calculate the quadrature of the curvatures in the x and y directions
    quad_curv = np.sqrt(term_1 + term_2)
    #print 'Smallest value in quad_curv is: {}'.format(np.min(quad_curv))
    #print 'Average value in quad_curv is {}'.format(np.mean(quad_curv))
    #print 'Median value in quad_curv is {}'.format(np.median(quad_curv))
    
    # Return the quadrature of the curvatures to the caller
    return quad_curv
    
    