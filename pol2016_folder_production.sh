#!/bin/bash

# This is a bash script that automatically produces all of the folders that
# will be required for the qualiative study of polarisation diagnostics. The 
# folders are produced on an external hard drive.
#
# Author: Chris Herron
#
# Start Date: 20/10/2016

# Create a variable that stores the base directory that I'll produce folders in 
data_loc=/Volumes/CAH_ExtHD/Pol_2016/

# Create a list that specifies all of the simulations that we need to produce
# folders for
declare -a sim_list=("b.1p.0049" "b.1p.0077" "b.1p.01" "b.1p.025" "b.1p.05"
"b.1p.1" "b.1p.7" "b.1p2" "b.5p.0049" "b.5p.0077" "b.5p.01" "b.5p.025" "b.5p.05"
"b.5p.1" "b.5p.7" "b.5p2" "b1p.0049" "b1p.0077" "b1p.01" "b1p.025" "b1p.05" 
"b1p.1" "b1p.7" "b1p2")

# Create a list that specifies the lones of sight that we need to produce
# folders for, for each simulation
declare -a los_list=("x_los" "y_los" "z_los")

# Loop over the simulations, and produce a folder for each simulation 
for sim in "${sim_list[@]}"
do 
	# Create a folder for this simulation
	mkdir "$data_loc${sim}"

	# Loop over the lines of sight, to produce a folder for each line of sight
	for los in "${los_list[@]}"
	do 
		# Create a folder for this line of sight, and this simulation
		mkdir "$data_loc${sim}/${los}"
	done
done