from cherab.core.model import SingleRayAttenuator, BeamCXLine, BeamEmissionLine
from cherab.core import Beam, Line
from cherab.core.atomic import elements, deuterium
from cherab.openadas import OpenADAS
from cherab.core.model.beam.beam_emission import SIGMA_TO_PI, SIGMA1_TO_SIGMA0, \
    PI2_TO_PI3, PI4_TO_PI3
from raysect.optical import Point3D, Vector3D, translate, rotate, rotate_basis, ConstantSF
from raysect.optical.material.emitter import UnityVolumeEmitter, UniformVolumeEmitter
from raysect.primitive import Cylinder

import numpy as np
import matplotlib.pyplot as plt


class test_beam:
    # Creates an unity volume emmiter in the place of the NBI, for testing purposes, see raysect doc. for more details on UnityVolumeEmitter
    south_pos = Point3D(0.188819939, -6.88824321, 0.0)  # Position of PINI grid center
    duct_pos = Point3D(0.539, -1.926, 0.00)  # position of beam duct
    beam_axis = south_pos.vector_to(duct_pos).normalise()

    up = Vector3D(0, 0, 1)
    beam_rotation = rotate_basis(beam_axis, up)
    beam_position = translate(south_pos.x, south_pos.y, south_pos.z)

    beam_width = 0.1

    def __init__(self, world):
        test_cylinder = Cylinder(self.beam_width,10, parent=world, transform=self.beam_position * self.beam_rotation, material = UnityVolumeEmitter())


"""
THe beam Class for the simulation
"""
class n_beam:

    south_pos = Point3D(0.188819939, -6.88824321, 0.0)  # Position of PINI grid center
    duct_pos = Point3D(0.539, -1.926, 0.00)  # position of beam duct
    beam_axis = south_pos.vector_to(duct_pos).normalise()

    up = Vector3D(0, 0, 1)
    beam_rotation = rotate_basis(beam_axis, up)

    beam_position = translate(south_pos.x, south_pos.y, south_pos.z)
    beam_energy = 65000
    beams = []
    P_ratios = [0.88,0.09,0.03]
    #P_ratios = [0, 0, 1]
    adas = OpenADAS(permit_extrapolation=True, missing_rates_return_null=True)  # create atomic data source
    integration_step = 0.0025

    def __init__(self,world,plasma):
        for i in range(1,4):

            self.beams.append(Beam(parent=world, transform=self.beam_position * self.beam_rotation))
            self.beams[i-1].plasma = plasma
            self.beams[i-1].atomic_data = self.adas
            self.beams[i-1].energy = 65000/i
            self.beams[i-1].power = 1.5e7*self.P_ratios[i-1]
            self.beams[i - 1].temperature = 20
            self.beams[i-1].element = elements.deuterium
            self.beams[i-1].sigma = 0.025 # def 0.025
            self.beams[i-1].divergence_x = 0  # 0.5
            self.beams[i-1].divergence_y = 0  # 0.5
            self.beams[i-1].length = 10.0
            self.beams[i-1].attenuator = SingleRayAttenuator(clamp_to_zero=True)
            self.beams[i-1].models = [
                BeamEmissionLine(Line(elements.deuterium, 0, (3, 2)),
                                   sigma_to_pi=SIGMA_TO_PI, sigma1_to_sigma0=SIGMA1_TO_SIGMA0,
                                   pi2_to_pi3=PI2_TO_PI3, pi4_to_pi3=PI4_TO_PI3),
                BeamCXLine(Line(elements.carbon, 5, (8, 7))),
                #BeamCXLine(Line(elements.deuterium, 0, (3,2))),
                #BeamCXLine(Line(elements.deuterium, 0, (4,2)))#,
            ]
            self.beams[i-1].integrator.step = self.integration_step
            self.beams[i-1].integrator.min_samples = 10
            #print(dir(self.beams[i-1].models.BeamCXLine(Line(elements.carbon, 5, (8, 7)))))

    def save_density(self,filename):
        # Fiber number to R mapping
        Fiber_to_R = np.load('Data/Inputs/RtoFnumber.npy')
        # Fiber number to Beam position mapping
        Fiber_to_beam_Z = np.load('Data/Inputs/FnumberToBeamPos.npy')
        #plt.style.use('seaborn-darkgrid')

        z = range(np.size(Fiber_to_beam_Z))
        beam_full_densities = np.array([self.beams[0].density(0, 0, Fiber_to_beam_Z[zz] ) for zz in z ])
        beam_half_densities = np.array([self.beams[1].density(0, 0, Fiber_to_beam_Z[zz] ) for zz in z])
        beam_third_densities = np.array([self.beams[2].density(0, 0, Fiber_to_beam_Z[zz])for zz in z])

        r = np.linspace(-0.2,0.2,100)
        beam_full_densities_radial = np.array([self.beams[0].density(0, rr, 1 ) for rr in r])
        beam_half_densities_radial = np.array([self.beams[1].density(0, rr, 1 ) for rr in r])
        beam_third_densities_radial = np.array([self.beams[2].density(0, rr, 1 )for rr in r])

        beam_densities = np.stack((beam_full_densities,beam_half_densities,beam_third_densities),1)
        np.save('Data/Inputs/BeamDensity%s.npy' %filename,beam_densities )

        plt.rc('font', size=16)
        plt.rc('axes', titlesize=16)
        plt.rc('axes', labelsize=14)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.rc('legend', fontsize=14)
        plt.rc('figure', titlesize=20)

        total_density = beam_full_densities + beam_half_densities + beam_third_densities
        total_density_radial = beam_full_densities_radial + beam_half_densities_radial + beam_third_densities_radial

        # plot the density profiles for the beam
        '''
        fig, [ax1,ax2] = plt.subplots(2,1,tight_layout = True,figsize = (10,6))
        ax1.plot(r, np.array(beam_full_densities_radial), label="full energy",linewidth = 2)
        ax1.plot(r, np.array(beam_half_densities_radial), label="half energy",linewidth = 2)
        ax1.plot(r, np.array(beam_third_densities_radial), label="third energy",linewidth = 2)
        ax1.set(xlabel = ('Beam radius [m]'),ylabel = ('Radial beam density [$m^{-3}$]'),title = ("Beam radial density at beam origin"))
        ax1.minorticks_on()
        ax1.grid(which = 'minor')

        ax1.legend()
        ax2.plot(Fiber_to_R, np.array(beam_full_densities) , label="full energy",linewidth = 2)
        ax2.plot(Fiber_to_R, np.array(beam_half_densities), label="half energy",linewidth = 2)
        ax2.plot(Fiber_to_R, np.array(beam_third_densities), label="third energy",linewidth = 2)
        ax2.set(xlabel=('Plasma major radius (R [m])'), ylabel=('Central beam density [$m^{-3}$]'),
                title=("Beam attenuation alsong the beam axis"))
        ax2.minorticks_on()
        ax2.grid(which='minor')
        ax2.legend()
        '''