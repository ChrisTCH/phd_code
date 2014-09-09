#!/bin/bash

# This is a bash script that uses MIRIAD functions to smooth the Stokes Q and U
# mosaics of the CGPS data, and then converts the smoothed images into FITS
# files.
#
# Author: Chris Herron
#
# Start Date: 26/8/2014

# Create a variable that stores the directory location of the CGPS data
data_loc=~/Documents/PhD/CGPS_Data/

# Create a variable that stores the input Stokes Q MIRIAD directory
Q_dir=cgps_Sto_Q_mosaic.mir

# Create a variable that specifies the full directory of the input Stokes Q
Q_in=$data_loc$Q_dir

# Create a variable that stores the input Stokes U MIRIAD directory
U_dir=cgps_Sto_U_mosaic.mir

# Create a variable that specifies the full directory of the input Stokes U
U_in=$data_loc$U_dir

# Create a variable that specifies the FWHM of the Gaussian to be used to
# smooth the data, in arcseconds
smth=390

# Run MIRIAD's convol task on the Stokes Q map to smooth it
convol map=$Q_in fwhm=$smth out=${data_loc}cgps_Sto_Q_smooth2_${smth}.mir options=final

# Run MIRIAD's convol task on the Stokes U map to smooth it
convol map=$U_in fwhm=$smth out=${data_loc}cgps_Sto_U_smooth2_${smth}.mir options=final

# Run MIRIAD's fits task to convert the MIRIAD files to FITS files. For Stokes Q:
fits in=${data_loc}cgps_Sto_Q_smooth2_${smth}.mir op=xyout out=${data_loc}cgps_Sto_Q_smooth2_${smth}.fits

# Run MIRIAD's fits task to convert the MIRIAD files to FITS files. For Stokes U:
fits in=${data_loc}cgps_Sto_U_smooth2_${smth}.mir op=xyout out=${data_loc}cgps_Sto_U_smooth2_${smth}.fits

# Print a message to the screen to show that everything completed successfully
echo "Smoothed images produced successfully"