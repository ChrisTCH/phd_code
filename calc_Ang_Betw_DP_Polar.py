#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of the angle  #
# that the polarisation gradient forms with the x axis, and the polarisation  #
# angle that is observed, and calculates the difference in these angles. This #
# means that the acute angle subtended by the polarisation gradient and the   #
# plane of polarisation is calculated. This calculation requires arrays of    #
# the angle that the polarisation gradient makes with the x-axis, and the     #
# polarisation angle. A map of the acute angle between the polarisation       #
# gradient and the polarisation angle is returned to the caller.              #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 17/3/2015                                                       #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Ang_Betw_DP_Polar, which will calculate the angle
# between the polarisation gradient and the polarisation angle, when given these
# quantities
def calc_Ang_Betw_DP_Polar(ang_polar_grad = None, polar_angle = None):
    '''
    Description
        This function calculates the angle between the polarisation gradient and
        the polarisation angle at each point of an image, when given these 
        quantities. Both quantities must be in degrees.
        
    Required Input
        ang_polar_grad - A Numpy array containing the value of the angle that
                         the polarisation gradient makes with the x-axis at each
                         point. This angle is measured counter-clockwise. This 
                         must be in degrees.

        polar_angle - A Numpy array containing the value of the observed 
                      polarisation angle at each point. This angle is measured
                      from the vertical, counter-clockwise. This array must have
                      the same shape as the ang_polar_grad array, and be in 
                      degrees.
                   
    Output
        ang_betw_DP_Polar - A Numpy array containing the value of the acute 
                     angle between the polarisation gradient and the 
                     polarisation angle at each point. This array 
                     has the same size as the given angle arrays.
    '''
    
    # Add 90 degrees to all of the polarisation angle values, so that
    # the angles are measured from the x-axis, in a counter-clockwise direction
    polar_angle = polar_angle + 90.0

    # Calculate the angle between the observed polarisation angle and the 
    # polarisation gradient, by subtracting one from the other, and taking the 
    # absolute value.
    ang_betw_DP_polar = np.abs(polar_angle - ang_polar_grad)
    
    # There are some situations where the angle calculated above will be over 90
    # degrees, but I am defining the angle between the polarisation gradient and
    # the observed polarisation angle to be the acute angle between them. Thus, 
    # for pixels where the angular separation is above 90 degrees, we need to 
    # calculate the acute angle from the obtuse angle.
    
    # First find the pixel locations where the angular separation is above 90 
    # degrees
    ang_above_90 = ang_betw_DP_polar > 90.0
    
    # For the pixels that have angular separation above 90 degrees, replace this 
    # value by the acute angle.
    ang_betw_DP_polar[ang_above_90] = 180.0 - ang_betw_DP_polar[ang_above_90]
    
    # Return the angle between the polarisation gradient and the polarisation
    # angle to the calling function
    return ang_betw_DP_Polar