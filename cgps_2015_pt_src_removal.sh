#!/bin/bash

# This is a bash script that uses the Aegean and BANE programs to remove point
# sources from the CGPS images. It produces images of the background and rms
# of the CGPS images, tables that identify and fit all of the point sources
# in the images, and a map of the CGPS data that has had all of the point 
# sources subtracted off.
#
# Author: Chris Herron
#
# Start Date: 6/10/2015

# Create a variable that stores the directory location of the Aegean and BANE
# programs
code_loc=~/Documents/CH_Apps/Aegean-dev/

# Create a variable that stores the directory location of the CGPS data
data_loc=~/Documents/CH_Apps/Aegean-dev/

# Create a variable that stores the input image name
img_name=Sto_I_high_lat_sub

# Create a variable that specifies the full directory of the input image
img_loc=$data_loc${img_name}.fits

# Run the BANE program on the input image, to characterise the background and 
# rms of the image. FITS files of the background and rms are saved in the same
# folder as the input image.
python ${code_loc}BANE.py $img_loc --out=$data_loc$img_name 

# Run the Aegean program on the input image, using the produced background and 
# rms images
python ${code_loc}aegean.py --seedclip=2 --floodclip=2 \
--background=${data_loc}${img_name}_bkg.fits \
--noise=${data_loc}${img_name}_rms.fits \
--table=${data_loc}${img_name}_src_table.vot $img_loc

# Run the residual.py program, which takes the original input image, and 
# subtracts off all of the point sources
python ${code_loc}residual.py $img_loc ${data_loc}${img_name}_src_table_comp.vot \
${data_loc}${img_name}_pt_src_removed.fits

# Print a message to the screen to show that everything completed successfully
echo "Point sources removed successfully"