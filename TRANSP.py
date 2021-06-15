import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.io import netcdf
from scipy.signal import sawtooth
from raysect.core.math import *
from raysect.core.math.function.float.function2d.interpolate import Interpolator2DMesh
from cherab.core.math.function import ScalarToVectorFunction2D
from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper, Constant2D



"""

Here the TRANSP class deal with opening and parsing of the TRANSP simulation results to get access to required fields
"""



class TRANSP_data:
    """
    Attributes:
        
    Methods:
        load_shot: pulls the requested data fields plus the flux surface shapes
        compute_flux_sufraces: caluculates the flux surface shape, with given mesh density
        mesh_creation: creates mesh from the flux surfaces
        data_mapping: combines the data from TRANSP files and the mesh, and creates CHERAB AxisimetricMapper
        object, 3D function that defines the value of given property
    compute_flux_surfaces:
        takes the Y and R moments that has been loaded plus the number of sample points around a circle, and computes
        positions of the flux surfaces in R,Y coordinates R-Major Radius Y-the vertical distance from mag-axis
    mesh_creation:
        creates a triangular mesh,  it does so such that one array defines coordinates vertices Nx2,
        and other array Mx3 stores the indices of the vertices that form mesh triangles
        Also, plasma_baundary variable stores the R-Max and Y-min,max of the mesh
    data_mapping:
        uses the  interpolator2DMesh fropm Raysect and AxisymmetricMapper from cherab, to create a CHERAB function
        object that returns a value for given quantity anywhere in the plasma, i.e. maps the profiles into the 3D space
    plot_quantity:
        quick way of plotting any quantity as a function of flux surfaces, input is the TRANSP variable name ('NI','TI'...)
        and a y-axis label
    """

    timeframe = 0
    #defines the density of created mesh, the number defines how much data pints lies on one flux surface
    mesh_density = 100
    mapped_data = {}



    def __init__(self, shotfile, timeframe, variables):
        self.variables = variables
        self.load_shot(shotfile, timeframe)
        self.user_profiles()
        self.compute_flux_surfaces()
        self.mesh_creation()
        self.data_mapping()

    def user_profiles(self):  # custom user profiles that will be mapped onto loaded flux surfaces
        n_flux = self.x[0,:].shape
        # density
        #imp_density = 4E17 + 2E17*np.cos(self.x[0,:]*16)
        imp_density = np.ones(n_flux)*1E13
        d_density = np.ones(n_flux) * 4E19


        temp = np.ones(n_flux) * 500
        #imp_density[20:] = 5E17*1.3
        #imp_density = 5E17 + 1E17*sawtooth(self.x[0,:]*15)
        el_density = d_density + 6 * imp_density
        self.quantities['user_NI'] = imp_density
        self.quantities['user_ND'] = d_density
        self.quantities['user_NE'] = el_density
        self.quantities['user_TI'] = temp
        self.quantities['user_TE'] = temp
       # print(imp_density)

    def load_shot(self,shotfile, timeframe):
        # location of the CDF TRANSP file
        self.file_path = shotfile
        self.shot_n = self.file_path[-12:-5]

        ## Open the NetCDF file,
        nc = netcdf.netcdf_file(self.file_path)

        # some of the key variable fields "NE", "TE", "NI", "TI", electron/ion temperature and density, 'X' is r/a
        # first command reads the defined variable, second copies it from file to memory.
        data = nc.variables['TIME']
        self.time = data[:].copy()
        print(self.time)
        data = nc.variables['X']
        self.x = data[:].copy()

        # pick a timeslice in the middle or the user defined time.
        if timeframe > len(self.time):
            self.halftime = len(self.time) // 2
            #warnings.warn('Default time used')
        else:
            self.halftime = timeframe
        self.timeslice = self.time[self.halftime]

        # initialize moment arrays, moments are in [cm]
        data = nc.variables['YMPA']
        self.YAXI = np.array(data[:].copy())
        data = nc.variables['RMJMP']
        self.RAXI = np.array(data[:].copy())
        data = nc.variables['YMC00']
        self.YMC = np.array(data[:].copy())
        data = nc.variables['RMC00']
        self.RMC = np.array(data[:].copy())
        self.YMC = self.YMC[..., np.newaxis]
        self.RMC = self.RMC[..., np.newaxis]
        data = nc.variables['YMS01']
        self.YMS = np.array(data[:].copy())
        data = nc.variables['RMS01']
        self.RMS = np.array(data[:].copy())
        self.YMS = self.YMS[..., np.newaxis]
        self.RMS = self.RMS[..., np.newaxis]

        # read all the moments from TRANSP file add append them into a 3D array (time,x,moment_n)
        # until there is no more moment variables
        data_available = True
        self.moment_n = 1
        while data_available:
            try:
                data = nc.variables['YMC%02d' % (self.moment_n)]
                self.YMC = np.dstack((self.YMC, np.array(data[:].copy())))
                data = nc.variables['RMC%02d' % (self.moment_n)]
                self.RMC = np.dstack((self.RMC, np.array(data[:].copy())))

                data = nc.variables['YMS%02d' % (self.moment_n + 1)]
                self.YMS = np.dstack((self.YMS, np.array(data[:].copy())))
                data = nc.variables['RMS%02d' % (self.moment_n + 1)]
                self.RMS = np.dstack((self.RMS, np.array(data[:].copy())))
                self.moment_n += 1
            except:
                print('end of data, number of moments %d' % self.moment_n)
                data_available = False

        # Creates a dictionary with all the requested datafields that are to be mapped onto a mesh
        self.quantities = {}
        for name in self.variables:
            data = nc.variables[name]
            buffer = data[:].copy()
            # add a check for the correct shape of the data field
            self.quantities[name] = buffer[self.halftime, :]

        # closes the file
        del data
        nc.close()

    def compute_flux_surfaces(self):

        # Calculating the points on the flux surfaces, This is the general formula :
        #      R(x,theta) = RMC00(x,t) + RMC01(x,t)*cos(theta) + RMS01(x,t)*sin(theta)
        # ;                    + RMC02(x,t)*cos(2*theta) + RMS02(x,t)*sin(2*theta)
        # ;                    + RMC03(x,t)*cos(3*theta) + RMS03(x,t)*sin(3*theta)
        # ;                    + ...
        #
        # ;    Y(x,theta) = YMC00(x,t) + YMC01(x,t)*cos(theta) + YMS01(x,t)*sin(theta)
        # ;                    + YMC02(x,t)*cos(2*theta) + YMS02(x,t)*sin(2*theta)
        # ;                    + YMC03(x,t)*cos(3*theta) + YMS03(x,t)*sin(3*theta)
        # ;                    + ...

        # First we create a vectors that will hold the moment order and a linearly spaced theta vector,
        # multiply to create a matrix of orders times theta value
        moment_order_sin = np.linspace(1, self.RMS.shape[2], self.RMS.shape[2])
        moment_order_cos = np.linspace(0, self.RMC.shape[2] - 1, self.RMC.shape[2])

        theta_steps = self.mesh_density  # number of steps in theta.
        self.theta_resolution = (2*pi)/theta_steps # resolution of theta in radians
        theta = np.linspace(0, 2 * pi, num=theta_steps, endpoint=False)
        self.theta_steps = theta

        cos_mx = np.cos(np.outer(moment_order_cos,theta))  # matrix of cos(0)..cos(theta)..cos(2*theta).....   one column for each theta
        sin_mx = np.sin(np.outer(moment_order_sin,theta))  # identical matrix for sin

        # preparing the coefficients for given time
        self.RMC = self.RMC[self.halftime, :, :]
        self.RMS = self.RMS[self.halftime, :, :]
        self.YMC = self.YMC[self.halftime, :, :]
        self.YMS = self.YMS[self.halftime, :, :]

        # multiply matrices of coefficients and sines/cosines, add them up to get R(x,theta), Y(x,theta)
        self.R = np.add(np.matmul(self.RMC, cos_mx), np.matmul(self.RMS, sin_mx))
        self.Y = np.add(np.matmul(self.YMC, cos_mx), np.matmul(self.YMS, sin_mx))

        self.m, self.n = self.R.shape

    def mesh_creation(self):
        # creates field for the mesh, vertex coordinates (N,2), triangles (M,3) 3 vertex indices, data (N,1)
        self.vertex_coords = np.empty((1 + self.n * self.m, 2))
        self.triangles = np.zeros((self.n * self.m * 2 + self.n, 3))

        # sets the middle point and the data there and converts to meters
        self.vertex_coords[0, 0] = self.RAXI[self.halftime, 0]/100
        self.vertex_coords[0, 1] = self.YAXI[self.halftime, 0]/100

        # seting up triangles in the inner most flux surface
        self.triangles[0:self.n, 0] = 0
        self.triangles[0:self.n, 1] = np.linspace(1, self.n, num=self.n)
        self.triangles[0:(self.n - 1), 2] = np.linspace(2, self.n, num=(self.n - 1))
        self.triangles[self.n - 1, 2] = 1

        # for loop that makes the vertices into triangles.
        index = 0
        tri_index = self.n - 2
        for i in range(self.m):
            for j in range(self.n):
                index += 1
                tri_index += 2
                self.vertex_coords[index, 0] = self.R[i, j]/100  # converting to meters
                self.vertex_coords[index, 1] = self.Y[i, j]/100

                self.triangles[tri_index, 0] = index
                self.triangles[tri_index + 1, 0] = index
                self.triangles[tri_index, 1] = index + self.n
                self.triangles[tri_index + 1, 1] = index + self.n + 1
                self.triangles[tri_index, 2] = index + self.n + 1
                self.triangles[tri_index + 1, 2] = index + 1

            self.triangles[tri_index + 1, 2] = index - self.n + 1
            self.triangles[tri_index, 2] = index + 1
            self.triangles[tri_index + 1, 1] = index + 1

        self.triangles = np.int_(np.delete(self.triangles, np.s_[(tri_index - self.n * 2)::], 0))

        # PlasmaBoundary returns the max or R, and max,min of Y mesh positions, so cylinder containing the plasma can be created
        self.plasma_boundary = [np.max(self.vertex_coords[:, 0]), np.max(self.vertex_coords[:, 1]), np.min(self.vertex_coords[:, 1])]

    def data_mapping(self):
        # for every entry in the dictionary we construct a datafield for all triangles and create the mesh
        for name, value in self.quantities.items():
            vertex_data = np.insert((np.ones((self.m, self.n)) * value[:, np.newaxis]).flatten(), 0, value[0])
            self.data_interpolated = Interpolator2DMesh(self.vertex_coords, vertex_data, self.triangles, limit=False)
            data_maped = AxisymmetricMapper(self.data_interpolated)
            self.mapped_data[name] = data_maped

    def plot_quantity(self,quantity,axis):
        fig, ax1 = plt.subplots()

        ax1.set(xlabel='r/a', ylabel=axis,
                title='TRANSP shot %s timeslice t =%1.3f' % (self.shot_n ,self.timeslice))
        ax1.grid()
        ax1.scatter(self.x[self.halftime, :], self.quantities[quantity])

    def velocity_map(self, plot = False):
        #open the file and load the variables
        nc = netcdf.netcdf_file(self.file_path)

        data = nc.variables['RMAJM']
        self.RMAJM = np.array(data[:].copy())

        data = nc.variables['VTORE_NC']
        self.VTORE_NC = np.array(data[:].copy())
        data = nc.variables['VTORX_NC']
        self.VTORX_NC = np.array(data[:].copy())
        data = nc.variables['VTORD_NC']
        self.VTORD_NC = np.array(data[:].copy())

        #data = nc.variables['VPOLE_NC']
        #self.VPORE_NC = np.array(data[:].copy())
        #data = nc.variables['VPOLX_NC']
        #self.VPORX_NC = np.array(data[:].copy())
        #data = nc.variables['VPOLD_NC']
        #self.VPORD_NC = np.array(data[:].copy())

        # closes the file
        del data
        nc.close()

        #pick time
        self.RMAJM = self.RMAJM[self.halftime, :]       / 100
        self.VTORE_NC = self.VTORE_NC[self.halftime, :] / 100
        self.VTORX_NC = self.VTORX_NC[self.halftime, :] / 100
        self.VTORD_NC = self.VTORD_NC[self.halftime, :] / 100
        #self.VPORE_NC = self.VPORE_NC[self.halftime, :] / 100
        #self.VPORX_NC = self.VPORX_NC[self.halftime, :] / 100
        #self.VPORD_NC = self.VPORD_NC[self.halftime, :] / 100

        np.save('Data/Inputs/VelInput.npy' , np.stack([self.VTORX_NC,self.RMAJM]))

        ########### Velocity, with full grid, and cos modulation
        number_fs = (len(self.RMAJM)-1)//2

        inboard_ve = self.VTORE_NC[0:number_fs]
        outboard_ve = self.VTORE_NC[(number_fs+1):(number_fs*2+1)]
        middle_ve = self.VTORE_NC[number_fs]

        inboard_vd = self.VTORD_NC[0:number_fs]
        outboard_vd = self.VTORD_NC[(number_fs + 1):(number_fs * 2 + 1)]
        middle_vd = self.VTORD_NC[number_fs]

        inboard_vx = self.VTORX_NC[0:number_fs]
        outboard_vx = self.VTORX_NC[(number_fs + 1):(number_fs * 2 + 1)]
        middle_vx = self.VTORX_NC[number_fs]

        el_vel = inboard_ve[:,np.newaxis] + np.outer((np.flip(outboard_ve)-inboard_ve),np.sin(self.theta_steps/2))
        x_vel = inboard_vx[:,np.newaxis] + np.outer((np.flip(outboard_vx)-inboard_vx),np.sin(self.theta_steps/2))
        d_vel = inboard_vd[:,np.newaxis] + np.outer((np.flip(outboard_vd)-inboard_vd),np.sin(self.theta_steps/2))

        data_el = np.insert(np.flip(el_vel).flatten(),0,middle_ve)
        self.VTORE = Interpolator2DMesh.instance(self.data_interpolated,data_el)
        data_x = np.insert(np.flip(x_vel).flatten(), 0, middle_vx)
        self.VTORX = Interpolator2DMesh.instance(self.data_interpolated, data_x)
        data_d = np.insert(np.flip(d_vel).flatten(), 0, middle_vd)
        self.VTORD = Interpolator2DMesh.instance(self.data_interpolated, data_d)
        '''
        # width on the mesh in meters this is in Z direction, centered around Z = 0
        mesh_width = 0.2/2
        x = np.repeat(self.RMAJM,2)
        y = np.tile([mesh_width,-mesh_width],len(self.RMAJM))
		
        # coordinates of the mesh verticies
        mesh_coords = np.vstack((x,y)).T
        
        # data       on the verticies
        VTORE_NC_data = np.repeat(self.VTORE_NC,2)
        VTORX_NC_data = np.repeat(self.VTORX_NC,2)
        VTORD_NC_data = np.repeat(self.VTORD_NC,2)
        #VPORE_NC_data = np.repeat(self.VPORE_NC,2)
        #VPORX_NC_data = np.repeat(self.VPORX_NC,2)
        #VPORD_NC_data = np.repeat(self.VPORD_NC,2)
        
        # mesh triagles indicies
        index1 = np.repeat(np.arange(0,len(x)-3,2),2)
        index2 = np.repeat(np.arange(3,len(x)-2,2),2)
        index2 = np.append(index2,len(index1)+1)
        index2 = np.insert(index2,0,1)
        index3_1 = np.arange(3,len(x),2)
        index3_2 = np.arange(2,len(x)-1,2)

        index3 = np.append(np.vstack(index3_1),np.vstack(index3_2),1).flatten()


        triangles2 = np.vstack((index1, index2, index3)).T
        triangles2 = np.ascontiguousarray(triangles2)
        mesh_coords = np.ascontiguousarray(mesh_coords)

        # interoplating and Mapping WARNING !!!!!!!!!
        #self.VTORE = Interpolator2DMesh(mesh_coords,VTORE_NC_data, triangles2, limit=False)
        #self.VTORD = Interpolator2DMesh(mesh_coords,VTORD_NC_data, triangles2, limit=False)
        #self.VTORX = Interpolator2DMesh(mesh_coords,VTORX_NC_data, triangles2, limit=False)
     
        #self.VPORE  = Interpolator2DMesh(mesh_coords,VPORE_NC_data, triangles2, limit=False)
        #self.VPORD  = Interpolator2DMesh(mesh_coords,VPORD_NC_data, triangles2, limit=False)
        #self.VPORX  = Interpolator2DMesh(mesh_coords,VPORX_NC_data, triangles2, limit=False)
        '''
        
        Zero_Constant = Constant2D(0)
        # mapping into 3D
        vel2D_el = ScalarToVectorFunction2D(Zero_Constant,self.VTORE,Zero_Constant)
        self.vel_el = VectorAxisymmetricMapper(vel2D_el)
        vel2D_de = ScalarToVectorFunction2D(Zero_Constant,self.VTORD,Zero_Constant)
        self.vel_de = VectorAxisymmetricMapper(vel2D_de)
        vel2D_x = ScalarToVectorFunction2D(Zero_Constant,self.VTORX,Zero_Constant)
        self.vel_x = VectorAxisymmetricMapper(vel2D_x)

        if plot == True :
            fig, [ax1,ax3,ax5] = plt.subplots(3,1,figsize = (15,10),constrained_layout = True)
            ax1.plot(self.RMAJM,self.VTORE_NC,marker = 'x')
            ax1.set(title=('Velocity Electrons, Toroidal'), ylabel=('[m/s]'),xlabel=('Major Radius [m]'))
            #ax2.plot(self.RMAJM,self.VPORE_NC)
            #ax2.set(title=('Velocity Electrons, Poloidal'), ylabel=('[m/s]'), xlabel=('Major Radius [m]'))
            ax3.plot(self.RMAJM, self.VTORD_NC)
            ax3.set(title=('Velocity Deuterium, Toroidal'), ylabel=('[m/s]'), xlabel=('Major Radius [m]'))
            #ax4.plot(self.RMAJM, self.VPORD_NC)
            #ax4.set(title=('Velocity Deuterium, Poloidal'), ylabel=('[m/s]'), xlabel=('Major Radius [m]'))
            ax5.plot(self.RMAJM, self.VTORX_NC)
            ax5.set(title=('Velocity Impurities, Toroidal'), ylabel=('[m/s]'), xlabel=('Major Radius [m]'))
            #ax6.plot(self.RMAJM, self.VPORX_NC)
            #ax6.set(title=('Velocity Impurities, Poloidal'), ylabel=('[m/s]'), xlabel=('Major Radius [m]'))
            fig.suptitle('Shot %s at time slice %i' % (self.shot_n,self.halftime))
            plt.show()