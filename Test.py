# Core and external imports
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import numpy as np
import pyuda
import os
import time

# Cherab and raysect imports
from raysect.primitive import Box, Cylinder, import_obj
from raysect.optical import World, Ray, translate, Point3D, Vector3D, rotate, Spectrum
from raysect.optical.observer import VectorCamera
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.material import Lambert, UniformVolumeEmitter, UniformSurfaceEmitter
from raysect.optical.library.spectra.colours import green
from raysect.optical.spectralfunction import ConstantSF
from raysect.optical.observer import RGBPipeline2D, SpectralPowerPipeline0D, PowerPipeline2D, FibreOptic, RGBAdaptiveSampler2D
from raysect.optical.observer import PinholeCamera
from raysect.core.math import rotate_basis
from raysect.core.math.function.vector3d.function2d.autowrap import *

from cherab.core import Maxwellian, Species, Plasma
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.core.math import Constant3D, Slice3D, samplevector3d_grid, Interpolate2DLinear, ConstantVector3D, sample2d_grid, sample3d_points, AxisymmetricMapper, VectorAxisymmetricMapper, AxisymmetricMapper
from cherab.tools.observers import load_calcam_calibration
#from cherab.solps import load_solps_from_mdsplus
from cherab.openadas import OpenADAS
from cherab.core.utility.conversion import PerCm3ToPerM3

from cherab.tools.equilibrium.plot import plot_equilibrium
# Scipy imports
from scipy.constants import electron_mass, atomic_mass, Boltzmann, speed_of_light

# CHERAB MAST import
from cherab.mastu.equilibrium import MASTUEquilibrium
from cherab.mastu.machine import import_mastu_mesh
from cherab.mastu.machine.mast_m9_cad_files import *


#my imports
from TRANSP import TRANSP_data
from ContourPlotter import  plot_contourSCP, save_Zeff,plot_contour, plot_velocity,  plot_mag_field,  save_parameters
from GetFibre import fiber_observers
from Beam import n_beam, test_beam
from ProfileFunction import *

print('import completed')


#########WORLD SETUP###################
SimName = 'file_9209_BE'   # CX at the end of a name defines Carbon 529 transition, BE defines Balmer Alpha
if SimName[-2:] == 'CX':
    Sline = 'CV'
elif SimName[-2:] == 'BE':
    Sline = 'Dalfa-Beam'

# Shot number for pyuda import at time and Transp idex for TRANSP import at that time
PULSE = 30422
TIME, Trans_index = [0.24,35]
# pyuda import
equilibrium = MASTUEquilibrium(PULSE)
equilibrium_slice = equilibrium.time(TIME)
#plot_equilibrium(equilibrium_slice,detail=True)
# world creation
world = World()
plasma = Plasma(parent = world)
plasma.atomic_data = OpenADAS(data_path='/home/ddoller/PycharmProjects/CHwork/venv4/AData',permit_extrapolation=True)
print('World Setup')
#######################################

# choose one mesh import
#########MASTU MESH####################
#import_mastu_mesh(world, override_material=Lambert())
#########MAST MESH#####################
#for part in MAST_FULL_MESH:
#    f_path, material = part
#    import_obj(f_path, material = material, parent = world, scaling=0.001)
#print('Mesh setup')
#######################################


#########TRANSP IMPORT#################
#Extracting information from the TRANSP file
TRANSP_file_path = '/common/transp_shared/Data/result/MAST/29880/Z01/29880Z01.CDF'
#TRANSP_file_path = "/home/cmichael/tstore/29976/TRANSP/O20/29976O20.CDF"

# defines which data field to import, the names are TRANSP defined TI - ion temperature, NI -  ion density, NIMP - impurity density
data_fileds = ['TI','ND','TE','NE','NI','NIMP']
TRANSP = TRANSP_data(TRANSP_file_path,Trans_index,data_fileds) # the number is the timeslice of the shot
TRANSP.velocity_map() # maps the velocity onto flux surfaces and can be accesed as TRANSP.vel_el/ion/imp
print('TRANPS import completed')
#######################################

#########PLASMA GEOMETRY###############
#defining the plasma cylinder
plasma_radius = TRANSP.plasma_boundary[0]
plasma_height = TRANSP.plasma_boundary[1]-TRANSP.plasma_boundary[2]
plasma.geometry = Cylinder(plasma_radius,plasma_height )
plasma.geometry_transform = translate(0,0,-plasma_height/2)
#######################################

#########PLASMA COMPOSITION############

# custome defined profiles
velocity_profile = PythonFunction2D(vectorfunction2d)
vel_test = VectorAxisymmetricMapper(velocity_profile )
#Temp_profile = PythonFunction2D(function2d)
#temp_test = AxisymmetricMapper(Temp_profile)
'''
MAG_profile = PythonVectorFunction2D(vectorfunction2dMAG)
MAG_test = VectorAxisymmetricMapper(MAG_profile )

dens_profile = PythonFunction2D(function2ddensity)
dens_test = AxisymmetricMapper(dens_profile)
'''

#defining the distribution for each plasma species:
# Deuterium Ion
#D_dens = PerCm3ToPerM3.to(TRANSP.mapped_data['ND'])#
D_dens =  PerCm3ToPerM3.to(TRANSP.mapped_data['ND'])
D_dens1 =  TRANSP.mapped_data['user_ND'] # user_ND/TI.. are user defined profiels mapped on flux surfaces in the TRANSP file
d1_distribution = Maxwellian(D_dens, TRANSP.mapped_data['TI'] ,vel_test,
                             deuterium.atomic_weight * atomic_mass)
d1_species = Species(deuterium, 1, d1_distribution)

# Carbon
# Carbon density from totoal ion density - deuterium density .......... or user defined density
c6_species_density_1 = PerCm3ToPerM3.to(TRANSP.mapped_data['NI']-TRANSP.mapped_data['ND'])
c6_species_density_2 = TRANSP.mapped_data['user_NI']
c6_species_density = c6_species_density_1

c6_distribution = Maxwellian(c6_species_density,TRANSP.mapped_data['TI'],vel_test,carbon.atomic_weight * atomic_mass )
c6_species = Species(carbon,6, c6_distribution)

# Electrons
E_dens = c6_species_density*6 + D_dens # TRANSP.mapped_data['NE']# # #
e_distribution = Maxwellian(E_dens, TRANSP.mapped_data['TE'],TRANSP.vel_el, electron_mass)
plasma.electron_distribution = e_distribution

# Magnetic field:
plasma.b_field = VectorAxisymmetricMapper(equilibrium_slice.b_field)
# Assigning species to the plasma object
plasma.composition = [c6_species,d1_species]
print('plasma species completed')
#########################################

##########SOLPS PLASMA###################
"""
# Load plasma from SOLPS model
mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
ref_number = 69636  # 69637
sim = load_solps_from_mdsplus(mds_server, ref_number)
plasma = sim.create_plasma(parent=world)
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
print(sim.species_list)
mesh = sim.mesh
vessel = mesh.vessel
print('solps plasma completed')
"""
##########################################

##########PLASMA COMPOSITION DEBUGING#####
#plot_contourSCP(plasma.composition[deuterium,1].distribution.density)
#plot_velocity(plasma.composition[carbon,6].distribution.bulk_velocity)
#plot_mag_field(equilibrium_slice.b_field)
#save_velocity(plasma.composition[carbon, 6].distribution.bulk_velocity)
save_parameters(plasma,SimName)  # this function is required for the analysis code as it saves input parameter file
#TRANSP.plot_quantity('TI','ion temperature')
#plt.show()
#print(plasma.electron_distribution.effective_temperature(0.5,0.5,-0.3))
##########################################

##########EMISSION MODELS#################

#ciii_465 = Line(carbon, 2, ('2s1 3p1 3P4.0', '2s1 3s1 3S1.0'))
cvi_529 = Line(carbon, 5, (8,7))
d_alpha = Line(deuterium, 0, (3, 2))
d_beta = Line(deuterium, 0, (4, 2))
Brem = Bremsstrahlung()
plasma.models = [#RecombinationLine(ciii_465),
                 #ExcitationLine(d_alpha),
                 RecombinationLine(d_alpha),
                 #ExcitationLine(d_beta),
                 RecombinationLine(d_beta),
                 #RecombinationLine(cvi_529),
                 Brem
                ]

NBI = n_beam(world,plasma)  # NBI is created
#NBI_test = test_beam(world)
NBI.save_density(SimName)
#plt.show()
print('Emission models steup')

##########################################

##########CAMERA##########################
# Select from available Cameras
camera_path = "/home/cwade/diagnostics/rgb/development/mast_pinhole_cameras/20190813/realistic.nc"
camera_config = load_calcam_calibration(camera_path)
pixels_shape, pixel_origins, pixel_directions = camera_config


# Get the power and raw spectral data for scientific use.
RGB_unfiltered = RGBPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered Power (W)")
RGB_unfiltered.display_progress = False

RGB_fiber = RGBPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered Power (W)")
RGB_fiber.display_progress = False

#Setup for optical fiber observer 64 fibers spanning the NBI at midplane
fibersCX = fiber_observers('/home/ddoller/XDRtoSAV/radii_SS_290711.sav',SimName,world,quality='low',line=Sline)#Line which is going to be observed is defined by the SimName
#fibersCX.plot_fibers(plot3d = True)
fibersCX.fiber_observe()
fibersCX.fiber_save()

# THE CXRS setup position and angles for use in camera
A1, A2, A3, = fibersCX.fiber_all[32,:] #(0,1,0)
P1, P2, P3 = fibersCX.fiber_coords #(0,-1.4,0)
rotation = rotate_basis(Vector3D(A1,A2,A3),Vector3D(0,0,1))
'''
#camera = VectorCamera(pixel_origins, pixel_directions, pipelines=[RGB_unfiltered], parent=world)
#camera = VectorCamera(pixel_origins, pixel_directions, pipelines=[RGB_unfiltered], parent=world, transform = translate(*fibers.fiber_coords)*rotate(*fibers.rotation_angles))
camera = PinholeCamera((150*10,100*10), fov=60, parent = world, pipelines=[RGB_fiber], transform = translate(P1+0.05,P2+0.05,0.07-0.07)*rotation)
camera.spectral_bins = 10
camera.pixel_samples = 70
camera.spectral_rays = 1
camera.ray_extinction_prob = 0.8
camera.ray_max_depth = 5
camera.ray_extinction_min_depth = 3
print('Camera setup')
###########################################
'''

############ CAMERA SETUP FOR pretty pictures
rgb = RGBPipeline2D(display_update_time=50, display_unsaturated_fraction=0.995)
sampler = RGBAdaptiveSampler2D(rgb, min_samples=1, fraction=0.1, cutoff=0.01)
camera = PinholeCamera((150*10, 100*10), fov = 60, parent=world, transform=translate(P1+0.05,P2+0.05,0.07-0.07)*rotation, pipelines=[rgb], frame_sampler=sampler)
camera.spectral_bins = 12
camera.spectral_rays = 1
camera.pixel_samples = 5
camera.ray_max_depth = 5
camera.ray_extinction_min_depth = 3
camera.ray_extinction_prob = 0.8

'''
# start ray tracing
ion()
name = 'MAST'
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
render_pass = 1
while True:

    print("Rendering pass {}...".format(render_pass))
    camera.observe()
    if render_pass % 3 == 0 :
        rgb.save("SCP/{}_{}_pass_{}.png".format(name, timestamp, render_pass))
    print()

    render_pass += 1
rgb.save("SCP/{}_{}_pass_{}.png".format(name, timestamp, render_pass))
ioff()
rgb.display()
##############################
'''


###########RUN THE OBSERVATION#############
#camera.observe()
#RGB_unfiltered.save('RGB_1.jpg')
#RGB_fiber.save('Shots/RGB_4.jpg')
#RGB_fiber.display()
plt.show()
###########################################