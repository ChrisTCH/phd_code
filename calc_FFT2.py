#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives a two-dimensional    #
# Numpy array of float values (e.g. an image) and computes the two-           #
# dimensional Fast Fourier Transform (FFT) of the image. The sample spacing   #
# between adjacent pixels in this array can also be provided, to ensure that  #
# the calculated frequencies are correct. Four arrays are returned to the     #
# calling function, the first two of which represent the amplitude and phase  #
# of the frequency components composing the image. The other two arrays       #
# specify the frequency value of each pixel in the returned image arrays.     #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 3/7/2014                                                        #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_FFT2, which will calculate the Fast Fourier Transform
# of the provided two-dimensional Numpy array.
def calc_FFT2(array, spacing = 1.0):
    '''
    Description
        This function calculates the two-dimensional Fast Fourier Transform of
        the provided two-dimensional Numpy array. The amplitude and phase 
        spectra are each returned as two-dimensional arrays of the same shape
        as the input array. Two one-dimensional arrays are also returned, which
        specify the horizontal and vertical frequency of each pixel in the
        returned amplitude and phase arrays.
        
    Required Input
        array - A two-dimensional Numpy array, of any size, consisting of 
                complex numbers.
        
        spacing - The pixel separation in whatever units are appropriate. A 
                  spacing of 2 seconds returns frequencies in cycles per second,
                  where the fact that data points are separated by 2 seconds is
                  taken into account.
                   
    Output
        amp_spec - The amplitude spectrum of the provided array obtained by a 
                   two-dimensional Fast Fourier Transform. Zero frequency 
                   component placed in the centre of the array. Same shape as 
                   the provided array.
        
        phase_spec - The phase spectrum of the provided array obtained by a 
                     two-dimensional Fast Fourier Transform. Zero frequency 
                     component placed in the centre of the array. Phases are in 
                     degrees. Same shape as the provided array.
        
        axis0_freq - A 1D array specifying the frequency of each pixel along
                     axis 0 of the array. Length of this array is the same as 
                     the length of axis 0, e.g. the y-axis of the array.
        
        axis1_freq - A 1D array specifying the frequency of each pixel along
                     axis 1 of the array. Length of this array is the same as 
                     the length of axis 1, e.g. the x-axis of the array.
    '''
    
    # Calculate the lengths of each axis of the provided array
    axis0_length, axis1_length = np.shape(array)
    
    # Calculate the Fast Fourier Transform of the provided array. Each entry
    # of this array is a complex number.
    fft = np.fft.fft2(array)
    
    # Calculate the amplitude spectrum of the array, by calculating the modulus
    # of each pixel value in the fft array. Also shift the values in the array
    # so that zero frequency is in the centre.
    amp_spec = np.fft.fftshift(np.abs(fft))
    
    # Calculate the phase spectrum of the array, by calculating the argument
    # of each pixel value in the fft array. Also shift the values in the array
    # so that zero frequency is in the centre.
    phase_spec = np.fft.fftshift(np.angle(fft, deg = True))
    
    # Calculate the frequencies along axis 0 of the amplitude and phase 
    # spectra arrays, using the given spacing. Also shift the values in the 
    # array so that zero frequency is in the centre.
    axis0_freq = np.fft.fftshift(np.fft.fftfreq(axis0_length, d = spacing))
    
    # Calculate the frequencies along axis 1 of the amplitude and phase 
    # spectra arrays, using the given spacing. Also shift the values in the 
    # array so that zero frequency is in the centre.
    axis1_freq = np.fft.fftshift(np.fft.fftfreq(axis1_length, d = spacing))
    
    # Return the amplitude and phase spectra, followed by the frequencies for
    # axis 0 and axis 1 of these spectra.
    return amp_spec, phase_spec, axis0_freq, axis1_freq