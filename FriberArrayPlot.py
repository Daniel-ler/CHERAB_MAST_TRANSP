import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.constants import speed_of_light, atomic_mass, Boltzmann
from matplotlib.ticker import FormatStrFormatter
from lmfit.models import GaussianModel, VoigtModel, LinearModel
from scipy.integrate import simps, quad
from cherab.core.atomic import deuterium, carbon
from cherab.openadas import OpenADAS
from scipy.interpolate import interp1d

from ProcessingFunction import *

# This scripts opens and partialy analizes the output of the simulation

adas = OpenADAS(permit_extrapolation=True)
SimName = 'file_9209_CX'

###Simulation result
file = 'Data/Outputs/%s.npy' % SimName#'Data/Fibers/29880Z01TEST_Carbon-spectrum-density_test18_BE.npy' #_18
data = np.load(file)

wavelength = data[0,:,:]
signal = data[1,:,:]
signal_variance = data[2,:,:]

# power
filePower = 'Data/Outputs/%sPower.npy' %SimName
dataPower = np.squeeze(np.load(filePower))

###### Input parameters
inputfile = np.load('Data/Inputs/InputData%s.npz' %SimName)

#velocity input
indata = inputfile['velocity']
inR = indata[1,:]
inV = indata[0,:]

# Temperature
indata = inputfile['temperature']
inRT = indata[1,:]
inT = indata[0,:]

#carbon density input
density_input = inputfile['density']
E_density_input = inputfile['edensity'] # second axis is Fiber_to_R

inRden = density_input [1,density_input [1,:]>0.75]
inden = density_input [0,density_input [1,:]>0.75]

#Beam density
Beam_n = np.load('Data/Inputs/BeamDensity%s.npy' % SimName)
paramfile = np.load('Data/Inputs/LOS%s.npz' %SimName)
###
### Geometrical factors
#LOS radius and distance
LOS_r = paramfile['r_at_beam_save']

LOS_area = np.pi*LOS_r**2
D_to_Beam = paramfile['d_to_beam']
#print(D_to_Beam)

#LOS and Beam angle
AngleLB = paramfile['angle_LB']
# fiber acceptance angle
Ac_angle = paramfile['ac_angle']

# vertical cos factor
cosFV = paramfile['cos_ver']
cosFH = paramfile['cos_hor']
Cos_fac = np.array([0.99970883, 0.999648,   0.99958353, 0.99951601, 0.99944609, 0.9993744,
0.99930155, 0.99922818, 0.99915491, 0.99908235, 0.99901112, 0.99894184,
0.99887509, 0.9988115 , 0.99875162, 0.99869601, 0.99864523, 0.99859982,
0.99856028, 0.9985271 , 0.99850073, 0.99848162, 0.99847015, 0.99846669,
0.99847153, 0.99848493, 0.9985071 , 0.99853817, 0.99857818, 0.9986271,
0.99868479, 0.998751  , 0.99882536, 0.99890735, 0.99899626, 0.99909123,
0.99919117, 0.99929472, 0.99940029, 0.99950594, 0.9996094 , 0.99970801,
0.99979865, 0.99987769, 0.99994093, 0.99998355, 0.99999997, 0.99998384,
0.99992786, 0.99982373, 0.99966206, 0.99943212, 0.99912174, 0.99871733,
0.99820352, 0.99756306, 0.9967767 , 0.99582289, 0.99467773, 0.99331484,
0.99170508, 0.9898167 , 0.98761595, 0.98506536,])

#Fiber ,number to R, mapping
Fiber_to_R = paramfile['R_beam']
###

#style
plt.style.use('seaborn-darkgrid')

#color palette
palette = plt.get_cmap('plasma')

######multiple line plot
### fitting peaks
peak_wave_fit = []
peak_int_fit = []
FWHM = []

#integration
Integrated_int = []
integrated_BE_2 = np.empty((0,3),float)


Line = 'CX'
low_b,hig_b = Line_boundaries(Line)


fig1 = plt.figure()
spec = fig1.add_gridspec(ncols=1, nrows=2, height_ratios = [2,1])
ax1 = fig1.add_subplot(spec[0])
ax2 = fig1.add_subplot(spec[1])

fibers = np.arange(np.size(wavelength,0))
mask = np.ones(np.size(wavelength,0), dtype= bool)
mask[0:8] = False
fibers = fibers[mask]
print('active fibers',fibers)

for i in  fibers:#range(np.size(wavelength,0)):range(10,11):
    ax1.plot(wavelength[i,:],signal[i,:], marker = '', color = palette(i*4), linewidth = 1, alpha = 0.9, label =i+1)
    print(i)
    #GAUSSIAN FITTING
    peekH, peekC, peekW = gaussian_fit(wavelength[i, :],signal[i, :])
    peak_int_fit = np.append(peak_int_fit, peekH)
    peak_wave_fit = np.append(peak_wave_fit, peekC)
    FWHM = np.append(FWHM, peekW)

    #INTEGRATION
    Integrated_int = np.append(Integrated_int,integrate_signal(wavelength[i,:],signal[i,:],low_b,hig_b))

    if Line == 'BE':
        #PEAK FINDING, for BE
        peak_v, half_max, edges = find_BE_peak(wavelength[i,:],signal[i,:])
        integrated_BE = area_gaussian(peak_v,half_max)
        ax1.plot(edges[0,:],np.zeros(2),'x')
        ax1.plot(edges[1,:], np.zeros(2),'x')
        ax1.plot(edges[2,:], np.zeros(2),'x')
        integrated_BE_2 =  np.append(integrated_BE_2 ,integrate_BE_signal(wavelength[i,:],signal[i,:],edges),0)
        #integrated_allthree = integrate_signal(wavelength[i,:],signal[i,:],658,663)
        BE_signal_ratios = integrated_BE_2/np.sum(integrated_BE_2)

ax1.legend(loc = 2, ncol = 3, title = 'Fiber')
ax1.set(title = (r'Spectral power'),xlabel = 'Wavelength, [nm]',ylabel= 'Spectral power, [W/nm]' ) #, $C^{5+}(8 \rightarrow 7), \: \lambda = 529.05 nm$
ax2.set(title = (r'Spectral power Standard deviation, $C^{5+}(8 \rightarrow 7), \: \lambda = 529.05 nm$'),xlabel = 'Wavelength, [nm]',ylabel= 'Spectral power, [W/nm]' )
plt.show()

length = np.size(signal,0)

####################################

# Velocity calculation
lambda_0 = 529.07 # carbon
velocity = velocity_cal(peak_wave_fit, lambda_0, cosFH = Cos_fac[mask])

# Temperature calculation
C_mass = 12.0107
D_mass = 2.014
Temperature = Temp_cal(FWHM,C_mass,lambda_0)
if Line == 'BE':
    TemperatureInt = interp1d(inRT,inT)
    Temperature= TemperatureInt(Fiber_to_R)

plt.rc('font', size = 14)
plt.rc('axes', titlesize = 14)
plt.rc('axes', labelsize = 14)
plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
plt.rc('legend', fontsize = 12)


print(Fiber_to_R[1:]-Fiber_to_R[0:-1])

plt.plot(Fiber_to_R,LOS_r*2*1000,'o')
plt.title('Width of the collection area at the beam')
plt.xlabel('Major radius [m]')
plt.ylabel('width [mm]')
#rad.set(title = 'Width of the collection area at the beam', xlabel = 'Major radius [m]', ylabel = 'width [m]')


figT, [axv1, axv2, axv3] = plt.subplots(1,3,tight_layout = True,figsize = (15,3))
axv1.plot(Fiber_to_R[mask],velocity,'o',label = 'Measurement')
axv1.plot(Fiber_to_R[mask],inV[mask],'o',label = 'Input')
axv1.plot(np.NaN,np.NaN,color = 'none',label = '$\Delta R \doteq $1cm')
axv1.set(xlabel = 'Radius [m]',ylabel = 'Velocity [m/s]', title = 'Velocity')
axv2.plot(Fiber_to_R[mask],Temperature,'o',label = 'Measurement')
axv2.plot(Fiber_to_R[mask],inT[mask],'o',label = 'Input')
axv2.plot(np.NaN,np.NaN,color = 'none',label = '$\Delta R \doteq $1cm')
axv2.set(xlabel = 'Radius [m]',ylabel = 'Temperature [eV]', title = 'Ion Temperature')
axv2.ticklabel_format(style='plain',useOffset=False)
axv3.plot(Fiber_to_R[mask],inT[mask]/Temperature*100 - 100,'o',color = 'tab:red',label = 'Temperature')
axv3.plot(Fiber_to_R[mask],inV[mask]/velocity*100 -100,'o',color = 'tab:purple',label = 'Velocity')
axv3.plot(np.NaN,np.NaN,color = 'none',label = '$\Delta R \doteq $1cm')
axv3.set(xlabel = 'Radius [m]',ylabel = 'Deviation [%]', title = 'Relative error')
axv3.legend(frameon = True)
axv2.legend(frameon = True)
axv1.legend(frameon = True)


# Density calculation
Beam_radius = 0.025
Fiber_radius = 0.0001
cxr = adas.beam_cx_pec(deuterium, carbon, 6, (8, 7))
ber = adas.beam_emission_pec(deuterium,deuterium,1,(3,2))
bpr = adas.beam_population_rate(deuterium,2,deuterium,1)

Zeff = 6
Cden = 4E17
POPcoeff = np.array([[bpr(65000/1, E_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [bpr(65000/2, E_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [bpr(65000/3, E_density_input[f], Temperature[i]) for i,f in enumerate(fibers)]])*1000000
BEcoeff = np.array([[ber(65000/1, E_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [ber(65000/2, E_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [ber(65000/3, E_density_input[f], Temperature[i]) for i,f in enumerate(fibers)]])
CXcoeff = np.array([[cxr[0](65000/1, Temperature[i], Cden,   Zeff , 0.6) for i,f in enumerate(fibers)],
                   [cxr[0](65000/2, Temperature[i], Cden,  Zeff , 0.6) for i,f in enumerate(fibers)],
                   [cxr[0](65000/3, Temperature[i], Cden,   Zeff , 0.6) for i,f in enumerate(fibers)]])
CXcoeff2 = np.array([[cxr[1](65000/1, Temperature[i], Cden,   Zeff , 0.6) for i,f in enumerate(fibers)],
                   [cxr[1](65000/2, Temperature[i], Cden,   Zeff , 0.6) for i,f in enumerate(fibers)],
                   [cxr[1](65000/3, Temperature[i], Cden,   Zeff , 0.6) for i,f in enumerate(fibers)]])


# intensity[W] divided by the beam density[m^-3] and eff,coff [W*m^3] = []
'''

'''
GF = np.ones(length)
for i in range(GF.size):
    GF[i] = geometrical_factor(Fiber_radius,D_to_Beam[i],Ac_angle[i],Beam_radius/np.sin(AngleLB[i]))

if Line == 'CX':
    coefficient_power = np.sum(Beam_n[mask] * CXcoeff.T, 1) * (np.sqrt(2 * np.pi * Beam_radius ** 2)) / np.sin(AngleLB[mask])
    density_from_power = dataPower[mask] / coefficient_power * 4 * np.pi

    coefficient = np.sum(Beam_n[mask]*CXcoeff.T,1)
    Beam_n2 = Beam_n[mask]*POPcoeff.T*1000000

    coefficient2 = np.sum(Beam_n2*CXcoeff2.T,1)

    CXcoeffE= (CXcoeff + POPcoeff*CXcoeff2)/(1+POPcoeff)
    coefficientEff = np.sum(Beam_n[mask] * CXcoeffE.T, 1)
    density = Integrated_int/GF[mask]/coefficientEff*4*np.pi


if Line == 'BE':
    # Beam density
    Beam_density1 = integrated_BE_2[:,2]/GF[mask]/BEcoeff[0,:]/E_density_input[mask]*4*np.pi#(Beam_radius/np.sin(AngleLB)/0.3989)
    Beam_density2 = integrated_BE_2[:,1]/GF[mask]/BEcoeff[1,:]/E_density_input[mask]*4*np.pi#(Beam_radius/np.sin(AngleLB)/0.3989)
    Beam_density3 = integrated_BE_2[:,0]/GF[mask]/BEcoeff[2,:]/E_density_input[mask]*4*np.pi#(Beam_radius/np.sin(AngleLB)/0.3989)


x = np.arange(0,length)

Fiber_to_R = Fiber_to_R[mask]

fig, [ ax3, ax4, ax5] = plt.subplots(3,1, figsize = (7,15), tight_layout = True)
'''
#ax1.scatter(x,peak_int, label ='peak value')
ax1.scatter(Fiber_to_R,Integrated_int,color = 'red')
#ax1.scatter(x,peak_int_fit,color = 'magenta',marker = 'x',label = 'voigt fit')
ax1.axis(ymin = Integrated_int.min()*0.95,ymax = Integrated_int.max()*1.05)
ax1.set(title = ('power per fiber'), ylabel = ('power [W]'),xlabel = ('major radius [m]'))
ax1.legend(loc = 'upper right')

#ax2.scatter(x,peak_wave,label ='peak value')
#ax2.scatter(x,peak_wave_fit,color = 'r',label = 'gaussian fit')
#ax2.scatter(x,peak_wave_fit2,color = 'm',marker = 'x',label = 'voigt fit')
ax2.set(ylabel = ('Wavelength, [nm]'),xlabel = ('Fiber number'),title = ('Peak frequency'))
ax2.legend(loc = 'upper right')

ax3.scatter(Fiber_to_R,velocity,label ='simulation result velocity')
#ax3.scatter(Fiber_to_R,velocity2,color = 'magenta',marker = 'x',label ='simulation result velocity')
ax3.set(ylabel = ('velocity, [m/s]'),xlabel = ('Major Radius [m]'),title = ('velocity from the shift vs input velocity'))
ax3.scatter(inR,inV,color='red',label='input velocity')
ax3.legend(loc = 'upper right')

ax4.scatter(Fiber_to_R,Temperature,label = 'simulation temperature')
ax4.scatter(inRT,inT,label = 'input temperature')
#ax4.scatter(Fiber_to_R,Temperaturepolated[mask])
ax4.set(ylabel = ('Temperature [a.u]'),xlabel = ('Major Radius [m]'),title = ('Temperature'))
ax4.legend(loc = 'upper right')
'''
if Line == 'CX':
    ax5.scatter(inRden,inden,label = 'Impurity density input')
    ax5.scatter(Fiber_to_R,density_from_power,label = 'Impurity density measurement, SigthLine')
    ax5.scatter(Fiber_to_R,density,label = 'Impurity density measurement, Fiber')
    ax5.set(ylabel = ('Density [m^-3]'),xlabel = ('Major Radius [m]'),title = ('Density profile'))
    ax5.legend(loc = 'best')

if Line == 'BE':
    ax5.plot(Fiber_to_R,Beam_n[mask,0],'xr',label = 'Beam density input 1')
    ax5.plot(Fiber_to_R,Beam_n[mask,1],'xb',label = 'Beam density input 1/2')
    ax5.plot(Fiber_to_R,Beam_n[mask,2],'xg',label = 'Beam density input 1/3')
    ax5.plot(Fiber_to_R,Beam_density1,'or',label = 'Beam density measurement, 1 energy')
    ax5.plot(Fiber_to_R,Beam_density2,'ob',label = 'Beam density measurement, 1/2 energy')
    ax5.plot(Fiber_to_R,Beam_density3,'og',label = 'Beam density measurement, 1/3 energy')
    ax5.set(ylabel = ('Density [m^-3]'),xlabel = ('Major Radius [m]'),title = ('Density profile'))
    ax5.legend(loc = 'best')

'''
fig2, [ax12, ax22, ax32] = plt.subplots(3,1, figsize = (7,12), tight_layout = True)
ax12.scatter(Fiber_to_R,POPcoeff[0,:],color = 'red', label = 'full energy')
ax12.scatter(Fiber_to_R,POPcoeff[1,:],color = 'green',label = 'half energy')
ax12.scatter(Fiber_to_R,POPcoeff[2,:],color = 'blue',label = 'third energy')
ax12.axis(ymin = POPcoeff.min()*0.95,ymax = POPcoeff.max()*1.05)
ax12.set(title = ('R [m]'), ylabel = ('polupation rate, dimesionless'),xlabel = ('major radius [m]'))
ax12.legend(loc = 'upper right')

ax22.scatter(Fiber_to_R,BEcoeff[0,:],color = 'red', label = 'full energy')
ax22.scatter(Fiber_to_R,BEcoeff[1,:],color = 'green',label = 'half energy')
ax22.scatter(Fiber_to_R,BEcoeff[2,:],color = 'blue',label = 'third energy')
ax22.axis(ymin = BEcoeff.min()*0.95,ymax = BEcoeff.max()*1.05)
ax22.set(title = ('R [m]'), ylabel = ('Beam emmision rate'),xlabel = ('major radius [m]'))
ax22.legend(loc = 'upper right')

ax32.scatter(Fiber_to_R,CXcoeff[0,:],color = 'red', label = 'full energy')
ax32.scatter(Fiber_to_R,CXcoeff[1,:],color = 'green',label = 'half energy')
ax32.scatter(Fiber_to_R,CXcoeff[2,:],color = 'blue',label = 'third energy')
#ax32.axis(ymin = CXcoeff.min()*0.95,ymax = CXcoeff.max()*1.05)
ax32.set(title = ('R [m]'), ylabel = ('CX emmision rate'),xlabel = ('major radius [m]'))
ax32.legend(loc = 'upper right')
'''
#fig.suptitle('29880')
plt.show()


