#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of Stokes Q   #
# and U for an image, and produces an array of the observed polarisation      #
# angle at each pixel. The polarisation angle array is returned to the        #
# caller. Note that under the IAU convention, the polarisation angle is       #
# measured counter-clockwise from the vertical direction.                     #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 6/6/2014                                                        #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Polar_Angle, which will calculate the observed 
# polarisation angle from the Stokes Q and U values at each pixel.
def calc_Polar_Angle(Sto_Q, Sto_U):
    '''
    Description
        This function calculates the observed polarisation angle at each point
        of an image, when given the Stokes Q and U values at each pixel
        for the image. Note that under the IAU convention, the polarisation 
        angle is measured counter-clockwise from the vertical direction.
        
    Required Input
        Sto_Q - A Numpy array containing the Stokes Q value at each point of 
                the image. Each entry of the array must be a float.
        Sto_U - A Numpy array containing the Stokes U value at each point of
                the image. Each entry of the array must be a float. Array must
                have the same size as the Stokes Q array.
                   
    Output
        polar_angle - A Numpy array containing the value of the observed
                polarisation angle at each point. This array has the same size 
                as the Stokes arrays. The polarisation is measured 
                counter-clockwise from the vertical direction, in degrees.
    '''
    
    # Calculate the observed polarisation angle (in radians)
    polar_angle = 0.5 * np.arctan2(Sto_U, Sto_Q)
    
    # Convert the calculated angles from radians to degrees
    polar_angle = np.rad2deg(polar_angle)
    
    # Return the polarisation angle to the calling function
    return polar_angle