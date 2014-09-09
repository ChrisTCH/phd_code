#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of Stokes Q   #
# and U for an image, and produces an array of the observed polarisation      #
# intensity at each pixel. The polarisation intensity array is returned to    #
# the caller.                                                                 #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 19/8/2014                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_Polar_Inten, which will calculate the observed 
# polarisation intensity from the Stokes Q and U values at each pixel.
def calc_Polar_Inten(Sto_Q, Sto_U):
    '''
    Description
        This function calculates the observed polarisation intensity at each
        point of an image, when given the Stokes Q and U values at each pixel
        for the image. 
        
    Required Input
        Sto_Q - A Numpy array containing the Stokes Q value at each point of 
                the image. Each entry of the array must be a float.
        Sto_U - A Numpy array containing the Stokes U value at each point of
                the image. Each entry of the array must be a float. Array must
                have the same size as the Stokes Q array.
                   
    Output
        polar_inten - A Numpy array containing the value of the observed
                     polarisation intensity at each point. This array has the 
                     same size as the Stokes arrays.
    '''
    
    # Calculate the observed polarisation intensity
    polar_inten = np.sqrt(np.power(Sto_Q,2.0) + np.power(Sto_U,2.0))
    
    # Return the polarisation intensity to the calling function
    return polar_inten