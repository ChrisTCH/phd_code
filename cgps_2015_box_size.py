#-----------------------------------------------------------------------------#
#                                                                             #
# This is a script which is designed to generate numerous samples from a      #
# Gaussian distribution, where each sample has a certain size, and calculate  #
# the skewness of each sample. A histogram of these skewness values will be   #
# produced, which describes how well the samples produced approximate a       #
# Gaussian distribution. By running the script for different sample sizes, we #
# can determine how many data points are required to properly calculate the   #
# skewness.                                                                   #
#                                                                             #
# Author: Chris Herron                                                        #
# Start Date: 26/8/2015                                                       #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import the various packages which are required for the proper functioning of
# this script, scipy.stats for calculating statistical quantities
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Create a variable that will determine how many data points are in each sample
sample_size = 21*21

# Create a variable that controls how many samples will be drawn
num_samples = 100000

# Randomly generate samples from a standard Gaussian distribution. Each row
# of this 2D array represents a separate sample
gauss_sample = np.random.normal(size = (num_samples, sample_size))

# Create an array that will hold the values of the skewness calculated for each
# sample
skew_arr = np.zeros(num_samples)

# Loop over the samples, and calculate the skewness for each one
for i in range(num_samples):
	skew_arr[i] = stats.skew(gauss_sample[i])

# Now that all of the skewness values have been calculated, print out the 
# mean skewness value, and the standard deviation in the skewness values
print 'The mean of the skewness values is {}'.format(np.mean(skew_arr))
print 'The standard deviation of the skewness values is {}'.format(np.std(skew_arr))

# Plot a histogram of the skewness values
plt.hist(skew_arr, bins = 20)
# Add the specified x-axis label to the plot
plt.xlabel('Skewness')
# Add a y-axis label to the plot
plt.ylabel('Counts')
# Add the specified title to the plot
plt.title('Histogram skewness: Sample size {}: Num samples {}'.format(sample_size, num_samples))
# Make the histogram appear
plt.show()