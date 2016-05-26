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
data_loc=~/Documents/PhD/CGPS_2015/

# Create a variable that stores the input image name
img_name=Sto_U_plane

# Create a variable that specifies the full directory of the input image
img_loc=$data_loc${img_name}.fits

# # Create a variable that stores the input Stokes Q image name
# Q_name=Sto_Q_plane

# # Create a variable that specifies the full directory of the Stokes Q image
# Q_loc=$data_loc${Q_name}.fits

# # Create a variable that stores the input Stokes U image name
# U_name=Sto_U_plane

# # Create a variable that specifies the full directory of the Stokes U image
# U_loc=$data_loc${U_name}.fits

# Run the BANE program on the input image, to characterise the background and 
# rms of the image. FITS files of the background and rms are saved in the same
# folder as the input image.
python ${code_loc}BANE.py $img_loc --out=$data_loc$img_name --grid=2 2 --box=18 18 --cores=4

# Run the Aegean program on the input image, using the produced background and 
# rms images
python ${code_loc}aegean.py --cores=4 --negative --seedclip=4 --floodclip=4 --maxsummits=5 \
--background=${data_loc}${img_name}_bkg.fits \
--noise=${data_loc}${img_name}_rms.fits \
--table=${data_loc}${img_name}_src_table.vot $img_loc

# Run the residual.py program, which takes the original input polarised intensity
# image, and uses the fitted sources to mask each source in polarised intensity,
# and then mask the corresponding locations in Stokes Q and U.
python ${code_loc}residual_src_mask.py $img_loc ${data_loc}${img_name}_src_table_comp.vot \
${data_loc}${img_name}_pt_src_mask.fits 0.05

# python ${code_loc}residual_polar_mask.py $img_loc ${data_loc}${img_name}_src_table_comp.vot \
# ${data_loc}${img_name}_pt_src_mask.fits $Q_loc ${data_loc}${Q_name}_pt_src_mask.fits \
# $U_loc ${data_loc}${U_name}_pt_src_mask.fits

# Print a message to the screen to show that everything completed successfully
echo "Point sources removed successfully"

#-------------------------------------------------------------------------------

# Use this code to run tests on the sub image of the CGPS

# # Create a variable that stores the directory location of the CGPS data
# data_loc=~/Documents/CH_Apps/Aegean-dev/

# # Create a variable that stores the input image name
# img_name=Sto_U_high_lat_sub

# # Create a variable that specifies the full directory of the input image
# img_loc=$data_loc${img_name}.fits

# # Run the BANE program on the input image, to characterise the background and 
# # rms of the image. FITS files of the background and rms are saved in the same
# # folder as the input image.
# python ${code_loc}BANE.py $img_loc --out=$data_loc$img_name --grid=2 2 --box=18 18

# # Run the Aegean program on the input image, using the produced background and 
# # rms images
# python ${code_loc}aegean.py --negative --seedclip=4 --floodclip=4 --maxsummits=5 \
# --background=${data_loc}${img_name}_bkg.fits \
# --noise=${data_loc}${img_name}_rms.fits \
# --table=${data_loc}${img_name}_src_table.vot $img_loc

# # Run the residual_src_mask.py program, which takes the list of found sources
# # and then masks them in the original image.
# python ${code_loc}residual_src_mask.py $img_loc ${data_loc}${img_name}_src_table_comp.vot \
# ${data_loc}${img_name}_pt_src_mask.fits

# # Print a message to the screen to show that everything completed successfully
# echo "Point sources removed successfully"
