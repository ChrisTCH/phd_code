# Author: Eli Bressert
import numpy as np

def separation(array1, array2):
    """
    Description
        This is a vectorized form of the angular separation function 
        that Astropy uses. Eventually, Astropy will adopt some form of 
        similar code, but for now this should do. 

    Required input
        array1: Numpy array of N rows by 2 columns. This should be
                an array of points in units of radians. Latitude values
				in first column, longitude in second column.

        array2: Numpy array of N rows by 2 columns. This should be
                an array of points in units of radians. Latitude values
				in first column, longitude in second column.
	
	Output
		One dimensional Numpy array of length N, i-th element contains
		the angular separation in radians between the i-th elements
		of array1 and array2.
    """

    lat1, lat2 = array1[:, 0], array2[:, 0]
    lon1, lon2 = array1[:, 1], array2[:, 1]
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon
    return np.arctan2((num1 ** 2 + num2 ** 2) ** 0.5, denominator)
	
# Creating fake data for testing
points = 40000
x1 = np.random.uniform(low=0, high=np.deg2rad(359.9999), size=points)
x2 = np.random.uniform(low=0, high=np.deg2rad(359.9999), size=points)
y1 = np.random.uniform(low=np.deg2rad(-90), high=np.deg2rad(90), size=points)
y2 = np.random.uniform(low=np.deg2rad(-90), high=np.deg2rad(90), size=points)

a1 = np.column_stack([x1, y1]).astype(np.float32)
a2 = np.column_stack([x2, y2]).astype(np.float32)

# Mapping distances from a1[1] to a2[i]
ang_sep = separation(a1, a2)
