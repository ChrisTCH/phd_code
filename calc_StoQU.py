#-----------------------------------------------------------------------------#
#                                                                             #
# This piece of code describes a function which receives arrays of density,   #
# and the magnetic field, and produces images of Stokes Q and U for the       #
# simulation. These images can be generated for different lines of sight into #
# the cube, different frequencies, different spectral indices, and for the    #
# cases where emission comes from within the cube, or the cube is backlit by  #
# polarised emission.                                                         #
#                                                                             #
# Author: Chris Herron                                                        #
# Email: cher7851@uni.sydney.edu.au                                           #
# Start Date: 20/10/2016                                                      #
# To be published as Herron et al (in prep), please contact before using in   #
# any publications, and do not distribute without permission.                 #
#                                                                             #
#-----------------------------------------------------------------------------#

# Import numpy, which is the only package required
import numpy as np

# Define the function calc_StoQU, which will calculate the observed Stokes Q and
# U images for the simulation, based on the line of sight, propagation 
# mechanism, frequency, and spectral index
def calc_StoQU(dens, magx, magy, magz, freq, dl, los, emis_mech, spec_ind = -1.0):
    '''
    Description
        This function calculates the Stokes Q and U images that would be 
        observed from a given turbulent cube at a given frequency, for a chosen
        line of sight. It is possible to control the spectral index of the 
        emission, as well as the emission mechanism, which is either emission
        being generated within the cube, or the cube being backlit by polarised
        emission. 
        
    Required Input
        dens - A 3D Numpy array, where each entry specifies the density of the
               ionised material at that pixel. The cube must already have been
               scaled to units of cm^-3.
        magx, magy, magz - 3D Numpy arrays, specifying the component of the 
               magnetic field in the x, y and z directions respectively at each
               pixel. These cubes must have already been scaled to units of 
               micro Gauss. All must have the same size as the dens array.
        freq - A decimal specifying the frequency at which the Stokes Q and U
               maps should be calculated. This should be specified in Hz.
        dl - A decimal specifying the size of each pixel of the simulation cube 
               in parsecs.
        los - The line of sight along which to integrate when calculating the
               observed Stokes Q and U images. This should be 0, 1, or 2, to 
               specify the axis along which we integrate. These correspond to 
               the z, y and x axes respectively.
        emis_mech - This is a string, which specifies whether the synchrotron 
               emission is generated within the cube, or whether the cube is 
               backlit by polaried emission. It can either by 'internal' or
               'backlit'. If emission is generated internally, then the
               produced Stokes Q and U maps are normalised so that the mean 
               polarised intensity in the absence of depolarisation is 1. If 
               the cube is backlit by polarised emission, then the uniform wall
               of emission that passes through the cube has unit polarised 
               intensity, such that Q = 1 everywhere across the wall.
        spec_ind - This is a decimal specifying the spectral index of the 
               synchrotron emission. Physically realistic values are between 
               0 and -3.0.
                   
    Output
        StoQ, StoU - 2D Numpy arrays giving the Stokes Q and Stokes U that 
               would be observed for the simulation cube, and the given
               parameters.
    '''
    
    # Define a variable that holds the value of the speed of light, in m s^-1
    c = 3.0 * np.power(10.0,8.0)

    # Calculate the wavelength of the emission, in m
    wavelength = c / freq

    # Define a variable that holds the value of the constant of proportionality
    # used to calculate the RM. This constant has units of rad m^-2 cm^3 micro
    # Gauss^-1 pc^-1
    K = 0.81

    # Check which line of sight is being used, so that we can calculate
    # the magnetic field parallel to the line of sight
    if los == 0:
        # Define the component of the magnetic field parallel to the line
        # of sight
        mag_para = magz
 
    elif los == 1:
        # Define the component of the magnetic field parallel to the line
        # of sight
        mag_para = magy

    elif los == 2:
        # Define the component of the magnetic field parallel to the line
        # of sight
        mag_para = magx

    # Check to see whether emission is generated internally, or whether the cube
    # is backlit by polarised emission
    if emis_mech == 'backlit':
        # In the case where the cube is backlit by polarised emission, we 
        # assume that the wall of polarisation has Q = 1 everywhere, and U = 0,
        # so that the initial polarisation angle is 0 degrees. The final 
        # polarised intensity is 1 everywhere, so we only need to find out
        # the final polarisation angle. We calculate the final polarisation 
        # angle by calculating the RM, and multiplying this by the square of the
        # wavelength.

        # Calculate the rotation measure in rad m^-2, using the density, 
        # magnetic field parallel to the line of sight, and distance between 
        # pixels
        # Integrate the product of the electron density and the magnetic field
        # strength along the line of sight to calculate the rotation measure
        # map. This integration is performed by the trapezoidal rule. Note the
        # array is ordered by (z,y,x)!
        RM = K * np.trapz(dens * mag_para, dx = dl, axis = los)

        # Calculate the observed polarisation angle of the emission in rad
        obs_pol_angle = RM * np.power(wavelength, 2.0)

        # Calculate the observed Stokes Q, based on the observed polarisation
        # angle
        StoQ = np.cos(2.0 * obs_pol_angle)

        # Calculate the observed Stokes U, based on the observed polarisation
        # angle
        StoU = np.sin(2.0 * obs_pol_angle)

    elif emis_mech == 'internal':
        # Check which line of sight is being used, so that we can calculate
        # the magnetic field perpendicular to the line of sight, and the
        # intrinsic polarisation angle
        if los == 0:
            # Calculate the magnitude of the magnetic field perpendicular to the 
            # line of sight, which is just the square root of the sum of the x and y
            # component magnitudes squared.
            mag_perp = np.sqrt( np.power(magx, 2.0) + np.power(magy, 2.0) )

            # Calculate the intrinsic polarisation angle at each pixel, based
            # on the line of sight and the magnetic field direction. The 
            # plane of polarisation is perpendicular to the direction of the 
            # magnetic field perpendicular to the line of sight.
            # The angle is calculated in radians, between +/- pi/2
            intrins_pol_angle = np.arctan(magy/magx)
     
        elif los == 1:
            # Calculate the magnitude of the magnetic field perpendicular to the 
            # line of sight, which is just the square root of the sum of the x and z
            # component magnitudes squared.
            mag_perp = np.sqrt( np.power(magx, 2.0) + np.power(magz, 2.0) )

            # Calculate the intrinsic polarisation angle at each pixel, based
            # on the line of sight and the magnetic field direction. The 
            # plane of polarisation is perpendicular to the direction of the 
            # magnetic field perpendicular to the line of sight.
            # The angle is calculated in radians, between +/- pi/2
            intrins_pol_angle = -1.0 * np.arctan(magz/magx)

        elif los == 2:
            # Calculate the magnitude of the magnetic field perpendicular to the 
            # line of sight, which is just the square root of the sum of the y and z
            # component magnitudes squared.
            mag_perp = np.sqrt( np.power(magy, 2.0) + np.power(magz, 2.0) )

            # Calculate the intrinsic polarisation angle at each pixel, based
            # on the line of sight and the magnetic field direction. The 
            # plane of polarisation is perpendicular to the direction of the 
            # magnetic field perpendicular to the line of sight.
            # The angle is calculated in radians, betweeen +/- pi/2
            intrins_pol_angle = -1.0 * np.arctan(magy/magz)

        # Calculate the synchrotron emissivity throughout the cube
        sync_emis = np.power(mag_perp, 1.0 - spec_ind) * np.power(freq,spec_ind)

        # Calculate the intrisic fractional polarisation of the synchrotron
        # emission
        p_frac_intrins = (3.0 - 3.0 * spec_ind)/(5.0 - 3.0 * spec_ind)

        # Calculate the polarisation emissivity at each location
        polar_emis = p_frac_intrins * sync_emis

        # Calculate the integral of the polarised emissivities along the line
        # of sight, ignoring depolarisation effects
        polar_emis_integral = np.trapz(polar_emis, dx = dl, axis = los)

        # Average the integral of the polarised emissivities over the obtained
        # image, to obtain a number that will be used to normalise the 
        # observed Stokes Q and U values
        pol_norm = np.mean(polar_emis_integral, dtype = np.float64)

        # Calculate the length of the cube along the line of sight
        num_pix = dens.shape[los]

        # Define the cumulative Stokes Q and U, which will be used to calculate
        # the observed Stokes Q and U values
        cumul_Q = np.zeros((num_pix,num_pix))
        cumul_U = np.zeros((num_pix,num_pix))

        # Define the cumulative Faraday depth, which will be used to calculate
        # how much the emission from a certain pixel is rotated as it moves 
        # through the cube
        cumul_Fara_depth = np.zeros((num_pix,num_pix))

        # Starting from the front of the cube, calculate the contribution to 
        # Stokes Q and U from each pixel along the line of sight, and add this
        # to the cumulative total, taking into account Faraday rotation. We 
        # need to do this for every pixel in the final observed image.
        for i in range(num_pix):
            # Check which line of sight we are integrating along, so that
            # we cycle through the correct slices of the cube.
            if los == 0:
                # Calculate the polarisation angle of the emission, after it has 
                # propagated to the front of the cube, and been rotated by the
                # material in front of it
                polar_angle = intrins_pol_angle[i,:,:] + cumul_Fara_depth *\
                             np.power(wavelength,2.0)

                # Calculate the contribution to Stokes Q and U of this slice, 
                # based on the intrinsic polarisation angle and the polarisation
                # emissivity
                StoQ_slice = polar_emis[i,:,:] * np.cos(2.0 * polar_angle) * dl
                StoU_slice = polar_emis[i,:,:] * np.sin(2.0 * polar_angle) * dl

                # Calculate the contribution to the Faraday depth of this slice
                # This is in rad m^-2.
                Fara_depth_slice = K * dens[i,:,:] * mag_para[i,:,:] * dl

            elif los == 1:
                # Calculate the polarisation angle of the emission, after it has 
                # propagated to the front of the cube, and been rotated by the
                # material in front of it
                polar_angle = intrins_pol_angle[:,i,:] + cumul_Fara_depth *\
                             np.power(wavelength,2.0)

                # Calculate the contribution to Stokes Q and U of this slice, 
                # based on the intrinsic polarisation angle and the polarisation
                # emissivity
                StoQ_slice = polar_emis[:,i,:] * np.cos(2.0 * polar_angle) * dl
                StoU_slice = polar_emis[:,i,:] * np.sin(2.0 * polar_angle) * dl

                # Calculate the contribution to the Faraday depth of this slice
                # This is in rad m^-2.
                Fara_depth_slice = K * dens[:,i,:] * mag_para[:,i,:] * dl

            elif los == 2:
                # Calculate the polarisation angle of the emission, after it has 
                # propagated to the front of the cube, and been rotated by the
                # material in front of it
                polar_angle = intrins_pol_angle[:,:,i] + cumul_Fara_depth *\
                             np.power(wavelength,2.0)

                # Calculate the contribution to Stokes Q and U of this slice, 
                # based on the intrinsic polarisation angle and the polarisation
                # emissivity
                StoQ_slice = polar_emis[:,:,i] * np.cos(2.0 * polar_angle) * dl
                StoU_slice = polar_emis[:,:,i] * np.sin(2.0 * polar_angle) * dl

                # Calculate the contribution to the Faraday depth of this slice
                # This is in rad m^-2.
                Fara_depth_slice = K * dens[:,:,i] * mag_para[:,:,i] * dl

            # Add the contribution to Stokes Q and U from this slice to the 
            # cumulative total (this is where depolarisation occurs, due to the
            # addition of emission from different depths)
            cumul_Q = cumul_Q + StoQ_slice
            cumul_U = cumul_U + StoU_slice

            # Add the Faraday depth from this slice to the cumulative total
            cumul_Fara_depth = cumul_Fara_depth + Fara_depth_slice

        # Now that the loop has finished, we have calculated the final Stokes
        # Q and U values that are observed, taking into account depolarisation

        # Normalise the observed Stokes Q and U values
        StoQ = cumul_Q / pol_norm
        StoU = cumul_U / pol_norm

        # Cast the Stokes Q and U matrices as 32-bit floats, to halve the data
        # that is required by these matrices
        StoQ = StoQ.astype(np.float32)
        StoU = StoU.astype(np.float32)

    else:
        # In this case an invalid value was entered, so print an error message 
        # to the screen
        print 'You must enter a valid emission mechanism, either backlit or '\
        + 'internal.'
    
    # Return Stokes Q and Stokes U to the calling function
    return StoQ, StoU