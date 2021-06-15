
from scipy.io import readsav
import numpy as np
from raysect.optical.observer import SightLine,RGBPipeline2D, SpectralPowerPipeline0D, PowerPipeline2D, FibreOptic, PowerPipeline0D, SpectralRadiancePipeline0D
from raysect.optical import translate, rotate, Vector3D, rotate_basis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
#filename = '/home/ddoller/XDRtoSAV/radii_SS_290711.sav'
import shapely
from shapely.geometry import LineString, Point
from lmfit.models import GaussianModel

"""
I use 'fiber' which is the "US" spelling instead of 'fibre'
I try to stay consistent, but there might be some mistakes.
"""




class fiber_positions:
    """
    Attributes:
        property_list: lists the variable names stored int he sav file
        fiber_properties: it's a dictionary that stores the values for each property in property_list
        fiber_raw1, fiber_raw2, fiber_all: store the fiber unit vector directions for given set of fibers
        fiber_coords: stores the fiber coordinates, they are all located at one point, which is the 'PUPIL'
                      of the optical setup
    Methods:
        fiber_directions: creates fiber_raw1, fiber_raw2, fiber_all attributes.
        fiber_position: creates fiber_coords attribute
    """

    property_list = ['PUPIL','DIAPHRAGM','FOCALDIST','IMAGE','CDIR','RADII','COSFAC','Z']

    def __init__(self, filename):
        sorce = readsav(filename)
        sorce_rec = sorce["data"]

        self.fiber_properties = {}
        for pro in self.property_list:
            self.fiber_properties[pro] = sorce_rec[pro][0]
        self.fiber_directions()

    def fiber_directions(self):
        self.fiber_raw1 = self.fiber_properties['CDIR'][:,0,:]
        self.fiber_raw2 = self.fiber_properties['CDIR'][:,1,:]
        self.fiber_all = np.append(np.flip(self.fiber_raw2,0),self.fiber_raw1,axis=0)

    def fiber_position(self):
        self.fiber_coords = self.fiber_properties['PUPIL']/1000
        self.fiber_z_pos = (self.fiber_properties['Z'].flatten())/1000
        return

class fiber_observers(fiber_positions):

    """
    Attributes:
        spectral_power_fiber: Raysect-SpectralPowerPipeline0D() object
        fibers: is a list that contains all the FibreOptics Raysect objects, for each fiber one object
        + all the FibreOptics Attributes
    Methods:
        fiber_list: creates the fibers list.
        fiber_observe: applies the .observe() method to all fibers.
        fiber_save: saves the wavelengths and mean intensity values for each fiber in an 3D array
                    [wavelength/intensity,fiber,values] this is the native binary npy format, but the data will
                    also to saved to CSV file for possible processing elsewhere in format [wavelength+intensity, fibers]
    Note:
    """

   # specifies parameters for the FiberOptics Raysec observer
    acceptance_angle= 1
    radius=0.0001
   # min_wavelength=400     #C5 8->7 lambda (525 to 535)
   # max_wavelength=700       # wide spectrum lambda 400 to 700
   # spectral_bins=1000
   # spectral_rays=2
   # pixel_samples=100
    fibers = []
    sigthlines = []


    def __init__(self,filename, tosave, world,quality = 'low', line = 'noline'):
        super().__init__(filename)

        self.file = tosave  # file to save parameters and output data

        # different quality levels
        self.fiber_position()
        if quality == 'medium':
            self.spectral_bins = 1000
            self.spectral_rays = 1
            self.pixel_samples = 800
        elif quality == 'high':
            self.spectral_bins = 1000
            self.spectral_rays = 1
            self.pixel_samples = 2000
        elif quality == 'low':
            self.spectral_bins = 500
            self.spectral_rays = 1
            self.pixel_samples = 100
        else:
            self.spectral_bins = 1000
            self.spectral_rays = 1
            self.pixel_samples = 100

        # defines the part of the spectrum observed
        if line == 'Dalfa':
            self.min_wavelength = 650
            self.max_wavelength = 660
        elif line == 'Dbeta':
            self.min_wavelength = 480
            self.max_wavelength = 490
        elif line == 'Dalfa-Beam':
            self.min_wavelength = 655
            self.max_wavelength = 663
        elif line == 'CV':
            self.min_wavelength = 528
            self.max_wavelength = 530
        elif line == 'noline':
            self.min_wavelength = 500
            self.max_wavelength = 680
        else:
            self.min_wavelength = 400
            self.max_wavelength = 700
        self.fiber_list(world)




    def fiber_list(self,world):
        self.pipelines = []
        self.pipelinesP = []
        self.pipelinesPS = []

        x_pos = self.fiber_coords[0]
        y_pos = self.fiber_coords[1]
        z_pos = 0#self.fiber_coords[2]

        for i in range(np.size(self.fiber_all-1,0)):
            ###############
            self.pipelines.append(SpectralPowerPipeline0D(display_progress = False))
            self.pipelinesP.append(PowerPipeline0D())
            self.pipelinesPS.append(SpectralPowerPipeline0D(display_progress = False))
            ###############
            #self.fiber_all[i,2] = 0
            #rotation = rotate_basis(Vector3D(*self.fiber_all[i,:]),Vector3D(0,0,1))
            ################ Experimental cylinders for visualization of LOS##############
            #Cylinder(0.002,2, parent=world, transform = translate(x_pos,y_pos,z_pos) * rotation, material = Gold())
            ################

            self.fibers.append(FibreOptic([self.pipelines[i]],
                                          acceptance_angle = self.acceptance_angle,
                                          radius = self.radius,
                                          min_wavelength=self.min_wavelength,
                                          max_wavelength=self.max_wavelength,
                                          spectral_bins = self.spectral_bins,
                                          spectral_rays=self.spectral_rays,
                                          pixel_samples = self.pixel_samples,
                                          #transform = translate(x_pos,y_pos,z_pos) * rotation,
                                          parent = world))

            self.sigthlines.append(SightLine(1,[self.pipelinesP[i],self.pipelinesPS[i]],
                                          min_wavelength=self.min_wavelength,
                                          max_wavelength=self.max_wavelength,
                                          spectral_bins = self.spectral_bins,
                                          spectral_rays=self.spectral_rays,
                                          pixel_samples = self.pixel_samples,
                                          #transform = translate(x_pos,y_pos,z_pos) * rotation,
                                          parent = world))
        self.find_focal_point()
        #offset = 0.015   Defines offset at the poin ot the LOS and beam crossing
        #Zvector = offset/np.sqrt(offset**2 + self.d_to_beam**2)

        for i in range(np.size(self.fiber_all - 1, 0)):
            self.fiber_all[i, 2] = 0#  Zvector[i] # must be unncomented to apply offset
            rotation = rotate_basis(Vector3D(*self.fiber_all[i, :]), Vector3D(0, 0, 1))
            self.fibers[i].transform = translate(x_pos,y_pos,z_pos) * rotation
            self.sigthlines[i].transform = translate(x_pos, y_pos, z_pos) * rotation

    def fiber_observe(self, fixed_fib = []):
        #print(self.fibers[1].acceptance_angle,self.fibers[1].radius,self.fibers[1].transform)
        if self.fibers:
            if fixed_fib:
                for i in fixed_fib:
                    self.fibers[i].observe()
                    self.sigthlines[i].observe()
                    print("Fiber number %d" % i)
            else :
                for i in range(len(self.fibers)):
                    self.fibers[i].observe()
                    self.sigthlines[i].observe()
                    print("Fiber number %d" % i)
        else:
            warnings.warn("fibers are not defined, run fiber_list method first", UserWarning)

    def fiber_save(self, fixed_fib = []):
        if fixed_fib:
            wavelengths = self.fibers[fixed_fib[0]].pipelines[0].wavelengths.T
            intensities = self.fibers[fixed_fib[0]].pipelines[0].samples.mean.T
            intensitiesP = self.sigthlines[fixed_fib[0]].pipelines[0].value.mean
            intensitiesPS = self.sigthlines[fixed_fib[0]].pipelines[1].samples.mean.T
            intensities_variance = self.fibers[fixed_fib[0]].pipelines[0].samples.variance.T
            for i in fixed_fib[1:]:
                wavelengths = np.vstack((wavelengths,self.fibers[i].pipelines[0].wavelengths.T))
                intensities = np.vstack((intensities,self.fibers[i].pipelines[0].samples.mean.T))
                intensitiesP = np.vstack((intensitiesP,self.sigthlines[i].pipelines[0].value.mean))
                intensitiesPS = np.vstack((intensitiesPS, self.sigthlines[i].pipelines[1].samples.mean.T))
                intensities_variance = np.vstack((intensities_variance,self.fibers[i].pipelines[0].samples.variance.T))
            to_save = np.stack([wavelengths,intensities,intensities_variance,intensitiesPS])

        else:
            wavelengths = self.fibers[0].pipelines[0].wavelengths.T
            intensities = self.fibers[0].pipelines[0].samples.mean.T
            intensitiesP = self.sigthlines[0].pipelines[0].value.mean
            intensitiesPS = self.sigthlines[0].pipelines[1].samples.mean.T
            intensities_variance = self.fibers[0].pipelines[0].samples.variance.T
            for i in range(len(self.fibers)-1):
                wavelengths = np.vstack((wavelengths,self.fibers[i+1].pipelines[0].wavelengths.T))
                intensities = np.vstack((intensities,self.fibers[i+1].pipelines[0].samples.mean.T))
                intensitiesP = np.vstack((intensitiesP,self.sigthlines[i+1].pipelines[0].value.mean))
                intensitiesPS = np.vstack((intensitiesPS, self.sigthlines[i+1].pipelines[1].samples.mean.T))
                intensities_variance = np.vstack((intensities_variance,self.fibers[i+1].pipelines[0].samples.variance.T))
            to_save = np.stack([wavelengths,intensities,intensities_variance,intensitiesPS])


        np.save('Data/Outputs/%s.npy'% self.file, to_save)
        np.save('Data/Outputs/%sPower.npy'% self.file, intensitiesP)

    def fiber_test_one(self,n):
         test_pipe = SpectralPowerPipeline0D(display_progress = False)

        #self.fibers[n].observe()
        #wavelengths = self.fibers[n].pipelines[0].wavelengths.T
        #intensities = self.fibers[n].pipelines[0].samples.mean.T




    def plot_fibers(self, plot3d = False):
        fig, ax= plt.subplots(1,1, figsize = (7,7))
        ax.axis(ymin = -2,ymax = 2,xmin = -2,xmax = 2)
        #ax.set_aspect('equal','box')
        ax.grid()
        #axz.grid(which = 'both')

        ax.set(title=('Top view, observation lines'), ylabel=('[m]'), xlabel=('[m]'))
        #axz.set(title=('LOS crossing the beam path'), ylabel=('Z [m]'), xlabel=('Fiber number'))
        #axz.grid(which='both')

        plasma_center = plt.Circle((0,0),0.7, fill = False, lw = 2, alpha = 0.7, color = 'blue')
        plasma_edge = plt.Circle((0, 0), 1.5, fill=False, lw = 2, alpha = 0.7, color = 'blue')
        ax.add_artist(plasma_center)
        ax.add_artist(plasma_edge)

        xp,yp,zp = self.fiber_coords
        zp = 0

        focal_dist = self.fiber_properties['FOCALDIST']
        Z_beam = self.fiber_properties['Z']
        R_beam = self.fiber_properties['RADII']

        focal_dist = np.append(focal_dist[:,0],focal_dist[:,1])/1000
        Z_beam = np.append(np.flip(Z_beam[:,1]),Z_beam[:,0])/1000
        R_beam = np.append(np.flip(R_beam[:,1]),R_beam[:,0])/1000

        # save fiber to R mapping
        np.save('Data/Inputs/RtoFnumber.npy', R_beam)

        south_pos = np.array([0.188819939, -6.88824321, 0])  # Position of PINI grid center
        duct_pos = np.array([0.539, -1.926, 0])  # position of beam duct

        norm = np.linalg.norm(duct_pos-south_pos)
        beam = (duct_pos-south_pos)/norm
        beam_line = np.array([[south_pos[0],south_pos[0]+beam[0]*8],[south_pos[1],south_pos[1]+beam[1]*8]])

        self.cos_ver = []

        for i in [0,31,63] :#ange(np.size(self.fiber_all,0)):
            xf,yf,zf = self.fiber_all[i]*focal_dist[i]
            ax.arrow(xp,yp,xf*1.3,yf*1.3,head_width = 0.05,head_length = 0.1,width=0.01)
            if i == 0 :
                ax.annotate('%d' % i,(xp+xf+0.1,yp+yf))
            if i == 31:
                ax.annotate('%d' % i,(xp+xf+0.1,yp+yf))
            if i == 63 :
                ax.annotate('%d' % i,(xp+xf+0.1,yp+yf))


        ax.plot(beam_line[0],beam_line[1],linewidth = 10,color = 'green',alpha = 0.4)

        ax.annotate('R = 0.7m', xy = (0,0.7),size = 11)
        ax.annotate('R = 1.5m', xy=(1.5, 0),size = 11)
        ax.annotate('Beam line', xy = (south_pos[0]+beam[0]*8,south_pos[1]+beam[1]*8),size = 11)

        #axz.plot( Z_beam, marker='o', color='red')
        #axz.tick_params(axis='y', labelcolor = 'red')
        #axR = axz.twinx()
        #axR.plot(R_beam, marker='o', color='green')
        #axR.set_ylabel('R [m]')
        #axR.tick_params(axis='y', labelcolor = 'green')
        #axX = axz.twiny()
        tick_loc = np.linspace(0,63,10,dtype=int)
        #axX.set_xlim(axz.get_xlim())
        #axX.set_xticks(tick_loc)

        def tick_f(x):
            v = R_beam[x]
            return ['%.2f' % z for z in v]

       #axX.set_xticklabels(tick_f(tick_loc))
        #axX.set_xlabel("Major radius, LOS crossing the beam line [m]")
       # fig.tight_layout()
        plt.show()

        # calculate the point where LOS and beam cross in beam coordinate.
        point = np.zeros((np.size(self.fiber_all, 0),2))
        beam_z = np.zeros((np.size(self.fiber_all, 0)))
        for i in range(np.size(self.fiber_all, 0)):
            xf, yf, zf = self.fiber_all[i] * focal_dist[i]*1.5
            point[i] = line_intersect(south_pos[0],south_pos[1],south_pos[0]+beam[0]*8,south_pos[1]+beam[1]*8,xf+xp,yf+yp,xp,yp)
            beam_z[i] = np.linalg.norm([south_pos[0],south_pos[1]]-point[i])
        np.save('Data/Inputs/FnumberToBeamPos.npy', beam_z)

        if plot3d ==  True :
            fig3D = plt.figure(figsize = (10,10), dpi=100)
            ax3D = fig3D.add_subplot(111,projection='3d')
            #ax3D.set_aspect('equal')
            ax3D.set_xlim3d(-1.5,2)
            ax3D.set_ylim3d(-2.1,1)
            ax3D.set_zlim3d(-0.5,0.5)
            ax3D.scatter(xp,yp,zp, color = 'g')

        # draw cilinder
            l, h = np.mgrid[0:2*np.pi:20j, -0.5:0.5:10j]
            x = np.cos(l)*0.7
            y = np.sin(l)*0.7
            z = h
            ax3D.plot_wireframe(x,y,z, color = 'r')

        # draw the lines
            LOS = np.array([0,10,20,30,40,50,63])
            qui = self.fiber_all[LOS]
            print(qui.shape)
            ax3D.quiver(xp,yp,zp,qui[:,0],qui[:,1],qui[:,2], length = 3, color = 'black', arrow_length_ratio = 0.1)
            ax3D.plot(beam_line[0],beam_line[1], linewidth = 5)
            plt.show()

    def find_focal_point(self):

        # Calculates the radius of the observation cone when it crosses the beam
        # Calclates the Cos factor of velocity and LOS
        xp,yp,zp = self.fiber_coords


        focal_dist = self.fiber_properties['FOCALDIST']
        Z_beam = self.fiber_properties['Z']
        R_beam = self.fiber_properties['RADII']

        focal_dist = np.append(focal_dist[:,0],focal_dist[:,1])/1000
        Z_beam = np.append(np.flip(Z_beam[:,1]),Z_beam[:,0])/1000
        self.R_beam = np.append(np.flip(R_beam[:,1]),R_beam[:,0])/1000

        # save fiber to R mapping
        np.save('Data/Inputs/RtoFnumber.npy', self.R_beam)

        south_pos = np.array([0.188819939, -6.88824321, 0])  # Position of PINI grid center
        duct_pos = np.array([0.539, -1.926, 0])  # position of beam duct

        norm = np.linalg.norm(duct_pos-south_pos)
        beam = (duct_pos-south_pos)/norm
        beam_line = np.array([[south_pos[0],south_pos[0]+beam[0]*8],[south_pos[1],south_pos[1]+beam[1]*8]])

        self.cos_ver = []
        self.cos_hor = []
        self.angle_LB = []
        self.r_at_beam_save = []
        self.d_to_beam = []
        self.ac_angle = []
        self.Cos_fac_LOS_FLUX = []
        for i in range(np.size(self.fiber_all,0)):
            xf, yf, zf = self.fiber_all[i] * focal_dist[i]

            # intersection of the LOS and Beam
            A = (beam_line[0, 0], beam_line[1, 0])
            B = (beam_line[0, 1], beam_line[1, 1])

            C = (xp, yp)
            D = (xp+xf, yp+yf)

            line1 = LineString([A, B])
            line2 = LineString([C, D])
            int_pt = line1.intersection(line2)
            poi = int_pt.x, int_pt.y,  Z_beam[i] # LOS and BEAM intersection point

            # Calculation the vertical cosine factor
            dist = np.linalg.norm(poi-self.fiber_coords)
            self.d_to_beam = np.append(self.d_to_beam,dist)
            dist_plane = np.linalg.norm(poi[0:1]-self.fiber_coords[0:1])
            self.cos_ver = np.append(self.cos_ver,dist_plane/dist)

            # Cos-fac for LOS and fluxsurface Tanget
            P1 = np.array([0, 0])            # Origin
            P2 = np.array([poi[0], poi[1]])  # LOS and beam intercept
            P3 = np.array([xp ,yp])          # Pupil position
            P2P1 = P1 - P2
            P2P3 = P3 - P2
            ANGLE = np.abs(np.pi/2 - np.arccos(np.dot(P2P1,P2P3)/(np.linalg.norm(P2P1) * np.linalg.norm(P2P3))))
            self.Cos_fac_LOS_FLUX = np.append(self.Cos_fac_LOS_FLUX, np.cos(ANGLE) )


            # radius of LOS at beam
            f_point = self.fiber_coords + np.array([xf,yf,zf])
            r_LOS_at_f = 0.00515 #r
            r_LOS_at_pupil = self.fiber_properties['DIAPHRAGM']/1000/2
            #r_at_beam = self.fiber_properties['DIAPHRAGM']/1000*np.linalg.norm(poi-f_point)/focal_dist[i]+0.00515
            r_at_beam2 = r_LOS_at_f + ((r_LOS_at_pupil-r_LOS_at_f)/focal_dist[i])*np.linalg.norm(poi-f_point)
            self.r_at_beam_save = np.append(self.r_at_beam_save,r_at_beam2)
            #self.fibers[i].radius = r_at_beam2


            #calculate maximal acceptance angle
            self.fibers[i].acceptance_angle = np.rad2deg(np.arctan((r_at_beam2)/dist))
            #print(np.arctan((r_at_beam2)/dist))
            self.ac_angle = np.append(self.ac_angle,np.arctan((r_at_beam2)/dist))

            #angle between LOS and Beam
            No_Z_x , No_Z_y, _  =  self.fiber_all[i]
            self.angle_LB = np.append(self.angle_LB,angel_between_vectors(np.array([No_Z_x,No_Z_y,0]),beam))

            # calculation of the horizontal cosine factor
            v1 = np.array(poi)
            v2 = self.fiber_all[i]

            theta = np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            alpha = np.pi/2-theta
            self.cos_hor = np.append(self.cos_hor, np.cos(alpha))



        #np.save('Data/Fibers/VerCosFactor.npy', self.cos_ver)
        #np.save('Data/Fibers/HorCosFactor.npy', self.cos_hor)
        #np.save('Data/Fibers/RatBeam.npy', self.r_at_beam_save)
        #np.save('Data/Fibers/DtoBeam.npy', self.d_to_beam)
        #np.save('Data/Fibers/Angle_LB.npy', self.angle_LB)
        #np.save('Data/Fibers/Ac_angle.npy', self.ac_angle)
        np.savez('Data/Inputs/LOS%s' % self.file, cos_ver = self.cos_ver, cos_hor = self.cos_hor, r_at_beam_save = self.r_at_beam_save,
                 d_to_beam = self.d_to_beam, angle_LB=self.angle_LB, ac_angle = self.ac_angle, R_beam = self.R_beam, Cos_fac = self.Cos_fac_LOS_FLUX)



def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return x, y

def angel_between_vectors(vector_1,vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    return (angle)

"""  PUPIL: [-2122.8501, -272.25000, -449.53000],
    DIAPHRAGM: 40.000000,
    FOCALDIST: [1796.5394, 1787.5764, 1805.5323, 1778.9559, 1814.6257, 1770.3156, 1823.9600, 1761.8119,
1833.3988, 1753.4836, 1843.0883, 1745.2870, 1852.8868, 1737.2205, 1862.7842, 1729.3201, 1872.9569, 1721.5452,
1883.1857, 1713.7783, 1893.6382, 1706.2489, 1904.2653, 1698.7603, 1915.1254, 1691.3551, 1926.1683, 1684.1014,
1937.4541, 1676.9631, 1948.9895, 1669.8241, 1960.5573, 1662.9445, 1972.4880, 1656.1749, 1984.5201, 1649.4001,
1996.8246, 1642.7036, 2009.3438, 1636.1741, 2022.1478, 1629.7203, 2035.1770, 1623.3997, 2048.5044, 1617.1814,
2061.9050, 1610.9531, 2075.7810, 1604.9637, 2089.7432, 1598.9618, 2104.0334, 1593.0581, 2118.7458, 1587.2516,
2133.5635, 1581.5905, 2148.5728, 1575.9985, 2164.0229, 1570.5471],
    IMAGE: 10.000000,
    CDIR: [0.95291537, -0.23131017, 0.19608124, 0.88671803, -0.39819697, 0.23488355, 0.95129555, -0.23676389,
0.19743261, 0.88422036, -0.40310591, 0.23592350, 0.94966710, -0.24215360, 0.19873118, 0.88163501, -0.40808585,
0.23703536, 0.94799000, -0.24756183, 0.20007034, 0.87902689, -0.41305563, 0.23811080, 0.94626105, -0.25302166,
0.20142002, 0.87640756, -0.41799000, 0.23915283, 0.94452518, -0.25841486, 0.20271641, 0.87375361, -0.42290774,
0.24021570, 0.94276530, -0.26377025, 0.20400737, 0.87107813, -0.42781442, 0.24124204, 0.94095194, -0.26917937,
0.20530921, 0.86838627, -0.43268186, 0.24226362, 0.93913698, -0.27448729, 0.20658772, 0.86566716, -0.43753463,
0.24327718, 0.93726772, -0.27985033, 0.20787740, 0.86287671, -0.44244429, 0.24430878, 0.93538064, -0.28517815,
0.20913285, 0.86010939, -0.44727033, 0.24527761, 0.93345612, -0.29049471, 0.21041030, 0.85727441, -0.45212674,
0.24629651, 0.93151474, -0.29577541, 0.21165347, 0.85439962, -0.45699242, 0.24730386, 0.92954266, -0.30104578,
0.21288948, 0.85150874, -0.46181896, 0.24830647, 0.92755467, -0.30627978, 0.21409118, 0.84859681, -0.46663338,
0.24927229, 0.92550427, -0.31154394, 0.21536532, 0.84560031, -0.47150567, 0.25028500, 0.92346454, -0.31672907,
0.21655458, 0.84264535, -0.47626504, 0.25123778, 0.92136472, -0.32197699, 0.21775648, 0.83967048, -0.48101148,
0.25215369, 0.91923690, -0.32718533, 0.21898225, 0.83660215, -0.48581323, 0.25314495, 0.91709393, -0.33235824,
0.22017410, 0.83349991, -0.49062556, 0.25409544, 0.91492099, -0.33752018, 0.22135867, 0.83039463, -0.49537554,
0.25504500, 0.91273391, -0.34264615, 0.22250934, 0.82725596, -0.50013530, 0.25595364, 0.91051137, -0.34775817,
0.22368145, 0.82409632, -0.50485128, 0.25688595, 0.90827560, -0.35283375, 0.22481966, 0.82091659, -0.50955361,
0.25778115, 0.90598339, -0.35795224, 0.22597396, 0.81764239, -0.51431823, 0.25872314, 0.90368986, -0.36300349,
0.22709723, 0.81442845, -0.51896662, 0.25957641, 0.90134865, -0.36807263, 0.22823933, 0.81111848, -0.52367824,
0.26047650, 0.89896786, -0.37315625, 0.22937143, 0.80777454, -0.52836782, 0.26139596, 0.89657813, -0.37817565,
0.23050129, 0.80441624, -0.53304750, 0.26224980, 0.89415562, -0.38321114, 0.23159219, 0.80105060, -0.53766137,
0.26313153, 0.89172530, -0.38818234, 0.23268095, 0.79765213, -0.54228222, 0.26397160, 0.88925642, -0.39316607,
0.23375930, 0.79425383, -0.54684156, 0.26481143],
    RADII: [782.97284, 1118.6259, 792.07507, 1129.0112, 801.33490, 1139.5514, 810.87378, 1150.0551, 820.72974,
1160.4694, 830.66492, 1170.8444, 840.71185, 1181.1782, 851.02557, 1191.4196, 861.29309, 1201.6183, 871.80035,
1211.9268, 882.35461, 1222.0372, 892.99597, 1232.2107, 903.65643, 1242.3887, 914.37866, 1252.4742, 925.09656,
1262.5148, 935.95050, 1272.6733, 946.68842, 1282.5765, 957.60571, 1292.4332, 968.48865, 1302.4114, 979.33020,
1312.3904, 990.17902, 1322.2335, 1000.9739, 1332.0760, 1011.7646, 1341.8286, 1022.4916, 1351.5353, 1033.3241,
1361.3729, 1044.0217, 1370.9449, 1054.7675, 1380.6511, 1065.5463, 1390.3136, 1076.1937, 1399.9318, 1086.8676,
1409.4226, 1097.4058, 1418.9116, 1107.9647, 1428.2749],
    COSFAC: [1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000,
1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000,
1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000,
1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000,
1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000,
1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000,
1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000],
    Z: [-16.805481, -17.573029, -16.814301, -17.640961, -16.895081, -17.592896, -16.884399, -17.600372,
-16.868134, -17.646942, -16.922729, -17.638672, -16.958130, -17.684875, -16.987488, -17.715057, -17.003754,
-17.745178, -17.014313, -17.755005, -17.069550, -17.838409, -17.061798, -17.832581, -17.098328, -17.842285,
-17.134460, -17.836731, -17.213898, -17.883972, -17.147491, -17.858948, -17.208588, -17.890900, -17.263794,
-17.974609, -17.241150, -17.934906, -17.261322, -17.962158, -17.281311, -17.959625, -17.343109, -18.023254,
-17.345764, -18.020691, -17.389862, -18.069183, -17.411804, -18.047852, -17.455383, -18.132050, -17.460327,
-18.146759, -17.481873, -18.111359, -17.470398, -18.176239, -17.532074, -18.162262, -17.560791, -18.212494,
-17.605469, -18.233978]
<]"""
