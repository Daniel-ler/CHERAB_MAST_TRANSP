import matplotlib
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from cherab.core.math import Slice3D, sample3d_points, samplevector2d_grid, samplevector3d_grid, samplevector3d_points, ConstantVector3D, sample2d_grid, sample3d_points, samplevector3d_grid
from cherab.core import Plasma
from cherab.core.atomic.elements import carbon, deuterium


def plot_contour(function3d):



    #Extracting the Distribution function strength frtom the plasam object
    X = [0]
    Y = np.linspace(0,1.75,100)
    Z = np.linspace(-1.5, 1.5 ,100)
    Dist = np.squeeze(sample2d_grid(Slice3D(function3d,'x',0),Y,Z))

    #plotting the Bfield
    fig, ax = plt.subplots()
    CS = ax.contour(Y, Z, np.transpose(Dist), levels = 20)
    fmt = ticker.ScalarFormatter()
    fmt.set_scientific(True)
    fmt.set_powerlimits((0,1))
    fmt.create_dummy_axis()
    ax.clabel(CS, inline = 1, fmt = fmt )
    ax.set_title('Density [per m^3]')
    ax.set(xlabel = '[m]', ylabel = '[m]')
    cbar = fig.colorbar(CS, cmap='plasma', extend='both')
    cbar.minorticks_on()
    plt.show()

    """
    #Extracting the Bfield strength frtom the plasam object
    X = [0]
    Y = np.linspace(0,2,100)
    Z = np.linspace(-2.2, 2.2 ,200)
    B_field3D = np.squeeze(samplevector3d_grid(plasma.b_field,X,Y,Z))
    B_field3DScalar = np.linalg.norm(B_field3D,axis = 2)
    """

def plot_contourSCP(function3d):

    plt.rc('font', size=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=18)
    plt.rc('figure', titlesize=20)

    #Extracting the Distribution function strength frtom the plasam object
    X = [0]
    Y = np.linspace(0,1.75,500)
    Z = np.linspace(-1.5, 1.5 ,500)
    Dist = np.squeeze(sample2d_grid(Slice3D(function3d,'x',0),Y,Z))

    Dist = np.ma.masked_where(Dist < 0.01,Dist)
    cmap = plt.cm.RdGy_r
    cmap.set_bad(color='red',alpha = 0)
    #plotting the Bfield
    fig, ax = plt.subplots(figsize = (6,7))
    ax.imshow(Dist.T,extent = [0,1.75,-1.5,1.5],cmap = cmap,origin = 'lower',alpha = 0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_color('snow')
    ax.spines['left'].set_color('snow')
    ax.xaxis.label.set_color('snow')
    ax.yaxis.label.set_color('snow')
    ax.tick_params(axis='both', colors='snow')
    ax.set_xticks([0,0.5,1,1.5])
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')
    # plt.title('Sampled Balmer Series Spectrum')
    fig.savefig('SCP/TokDensDeu.png', transparent=True)
    plt.show()

def plot_velocity(function3Dvector):

    X = [0]
    Y = np.linspace(0, 1.75, 100)
    X2 = np.linspace(0, 1.75, 100)
    Y2 = [0]
    Z = np.linspace(-1.5, 1.5, 100)
    Z2 = np.zeros(100)
    midline = np.array([Z2,Y,Z2]).T
    print(midline.shape)
    values = np.linalg.norm(np.squeeze(samplevector3d_grid(function3Dvector,X,Y,Z)),axis=2).T
    valuesRot = np.linalg.norm(np.squeeze(samplevector3d_grid(function3Dvector, X2, Y2, Z)), axis=2).T
    ValuesMid = np.linalg.norm(np.squeeze(samplevector3d_points(function3Dvector, midline)), axis=1).T

    fig, [ax,ax3] = plt.subplots(2,1)
    im = ax.imshow(values, vmin = np.min(values), vmax = np.max(values))
    #im2 = ax2.imshow(valuesRot, vmin=np.min(values), vmax=np.max(values))
    ax.set_aspect('equal')
    ax.set(title = 'particle velocity')
    cbar = fig.colorbar(im, cmap = 'plasma' ,extend = 'both',label = 'velocity [m/s]')
    cbar.minorticks_on()
    #ax2.set_aspect('equal')
    #ax2.set(title='particle velocity')
    ax3.plot(Y,ValuesMid,marker = 'o')

    plt.show()
def plot_mag_field(function3Dvector):
    X = [0]
    Y = np.linspace(0.1, 1.5, 10)
    Z = np.linspace(-2, 2, 10)

    values = np.linalg.norm(np.squeeze(samplevector2d_grid(function3Dvector,Y,Z)),axis=2).T

    fig, [ax, ax1] = plt.subplots(2,1)
    im = ax.imshow(values, vmin = np.min(values), vmax = np.max(values))

    ax.set_aspect('equal')
    ax.set(title = 'Magnetic Field')
    cbar = fig.colorbar(im, cmap = 'plasma' ,extend = 'both',label = 'B-Field [T]')
    cbar.minorticks_on()


    plt.show()


def save_parameters(plasma,filename):
    vel = save_velocity(plasma.composition[carbon, 6].distribution.bulk_velocity)
    temp =save_temperature(plasma.composition[carbon, 6].distribution.effective_temperature)
    den =save_density(plasma.composition[carbon, 6].distribution.density)
    dden = save_edensity(plasma.composition[deuterium, 1].distribution.density)
    eden = save_edensity(plasma.electron_distribution.density)
    np.savez('Data/Inputs/InputData%s' %filename, velocity = vel, temperature = temp, density = den, edensity = eden, ddensity = dden, zeff = save_Zeff(plasma), b_field = save_Bfield(plasma)) #, edensity = eden
    return



def save_velocity(function3Dvector):
    Y = np.load('Data/Inputs/RtoFnumber.npy')
    Z2 = np.zeros(np.size(Y))
    midline = np.array([Z2,Y,Z2]).T
    ValuesMid = np.linalg.norm(np.squeeze(samplevector3d_points(function3Dvector, midline)), axis=1).T
    return np.stack([ValuesMid, Y])

def save_temperature(function3D):
    Y = np.load('Data/Inputs/RtoFnumber.npy')
    X = np.zeros(np.size(Y))
    Z = np.zeros(np.size(Y))
    midline = np.array([X, Y, Z]).T
    ValuesMid = sample3d_points(function3D,midline)
    return np.stack([ValuesMid, Y])

def save_density(function3D):
    Y = np.load('Data/Inputs/RtoFnumber.npy')
    X = np.zeros(np.size(Y))
    Z = np.zeros(np.size(Y))
    midline = np.array([X, Y, Z]).T
    ValuesMid = sample3d_points(function3D,midline)
    return np.stack([ValuesMid, Y])

def save_edensity(function3D):
    Fiber_to_R = np.load('Data/Inputs/RtoFnumber.npy')
    X = np.zeros(np.size(Fiber_to_R))
    Y = Fiber_to_R
    Z = np.zeros(np.size(Fiber_to_R))
    midline = np.array([X, Y, Z]).T
    ValuesMid = sample3d_points(function3D,midline)
    return  ValuesMid

def save_Zeff(plasma):
    Y = np.load('Data/Inputs/RtoFnumber.npy')
    ValuesMid = np.zeros(np.size(Y))
    for i, y in enumerate(Y[2:]):
        ValuesMid[i+2] = plasma.z_effective(0,y,0)
    print(ValuesMid)
    return ValuesMid

def save_Bfield(plasma):
    Y = np.load('Data/Inputs/RtoFnumber.npy')
    print(Y.shape)
    ValuesMid = np.zeros(np.size(Y))
    for i, y in enumerate(Y):
        ValuesMid[i] = plasma.b_field(0,y,0).length
    return ValuesMid