#!/bin/bash

# This is a bash script that uses Montage functions to convert the Stokes Q and U
# mosaics of the CGPS data from 64 bit to 32 bit. This is performed for a range
# of smoothing scales.
#
# Author: Chris Herron
#
# Start Date: 10/11/2015

# Create a variable that stores the directory location of the CGPS data
data_loc=~/Documents/PhD/CGPS_2015/
#data_loc=/import/shiva1/herron/cgps_2015/

# Create a string that will be used to control what Q and U FITS files are being
# converted, and that will be appended into the filename of 
# anything produced in this script. This is either 'high_lat' or 'plane'
save_append=plane_all_mask

# Create a variable that stores the input Stokes Q directory
Q_dir=Sto_Q_${save_append}_smoothed/

# Create a variable that specifies the full directory of the input Stokes Q
Q_in=$data_loc$Q_dir

# Create a variable that stores the input Stokes U directory
U_dir=Sto_U_${save_append}_smoothed/

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
	# Run Montage's mConvert task on the Stokes Q map to convert it
	mConvert -b -32 ${Q_in}Sto_Q_${save_append}_smooth2_${res}.fits ${Q_in}Sto_Q_${save_append}_smooth2_${res}_32.fits

	# Run Montage's mConvert task on the Stokes U map to convert it
	mConvert -b -32 ${U_in}Sto_U_${save_append}_smooth2_${res}.fits ${U_in}Sto_U_${save_append}_smooth2_${res}_32.fits

	# Remove the 64 bit smoothed Stokes Q and U maps
	rm -rdfv ${Q_in}Sto_Q_${save_append}_smooth2_${res}.fits
	rm -rdfv ${U_in}Sto_U_${save_append}_smooth2_${res}.fits

	# Rename the 32 bit smoothed Stokes Q and U maps
	mv ${Q_in}Sto_Q_${save_append}_smooth2_${res}_32.fits ${Q_in}Sto_Q_${save_append}_smooth2_${res}.fits
	mv ${U_in}Sto_U_${save_append}_smooth2_${res}_32.fits ${U_in}Sto_U_${save_append}_smooth2_${res}.fits

	# Print something to show that the smoothing at this resolution has been done
	echo "Converted $res"

# This is everything that needs be done in the for loop
done

# Print a message to the screen to show that everything completed successfully
echo "Smoothed images produced successfully"