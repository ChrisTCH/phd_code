#!/bin/bash

# This is a bash script that uses MIRIAD functions to smooth the Stokes Q and U
# mosaics of the CGPS data, and then converts the smoothed images into FITS
# files. This is performed for a range of smoothing scales.
#
# Author: Chris Herron
#
# Start Date: 19/8/2015

# Create a variable that stores the directory location of the CGPS data
data_loc=~/Documents/PhD/CGPS_2015/
#data_loc=/import/shiva1/herron/cgps_2015/

# Create a string that will be used to control what Q and U FITS files are used
# to perform calculations, and that will be appended into the filename of 
# anything produced in this script. This is either 'high_lat' or 'plane'
save_append=plane

# Create a variable that stores the input Stokes Q MIRIAD directory
Q_dir=Sto_Q_${save_append}_final_mask.mir

# Create a variable that specifies the full directory of the input Stokes Q
Q_in=$data_loc$Q_dir

# Create a variable that stores the input Stokes U MIRIAD directory
U_dir=Sto_U_${save_append}_final_mask.mir

# Create a variable that specifies the full directory of the input Stokes U
U_in=$data_loc$U_dir

# Create a list that specifies all of the final resolutions that we are to smooth the
# Stokes Q and U data to
final_res_list="75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300
315 330 345 360 375 390 405 420 450 480 510 540 570 600 630 660 720 780
840 900 960 1020 1080 1140 1200"

# Loop over the final resolutions that the Q and U data is to be smoothed to
for res in `echo $final_res_list`
do
	# Run MIRIAD's convol task on the Stokes Q map to smooth it
	convol map=$Q_in fwhm=$res out=${data_loc}Sto_Q_${save_append}_mask_smooth2_${res}.mir options=final

	# Run MIRIAD's convol task on the Stokes U map to smooth it
	convol map=$U_in fwhm=$res out=${data_loc}Sto_U_${save_append}_mask_smooth2_${res}.mir options=final

	# Run MIRIAD's fits task to convert the MIRIAD files to FITS files. For Stokes Q:
	fits in=${data_loc}Sto_Q_${save_append}_mask_smooth2_${res}.mir op=xyout out=${data_loc}Sto_Q_${save_append}_mask_smooth2_${res}.fits

	# Run MIRIAD's fits task to convert the MIRIAD files to FITS files. For Stokes U:
	fits in=${data_loc}Sto_U_${save_append}_mask_smooth2_${res}.mir op=xyout out=${data_loc}Sto_U_${save_append}_mask_smooth2_${res}.fits

	# Remove the MIRIAD directories that were produced by the smoothing
	rm -rdfv ${data_loc}Sto_Q_${save_append}_mask_smooth2_${res}.mir
	rm -rdfv ${data_loc}Sto_U_${save_append}_mask_smooth2_${res}.mir

	# Print something to show that the smoothing at this resolution has been done
	echo "Smoothed to $res"

# This is everything that needs be done in the for loop
done

# Print a message to the screen to show that everything completed successfully
echo "Smoothed images produced successfully"