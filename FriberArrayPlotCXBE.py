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
import matplotlib

from ProcessingFunction import *

adas = OpenADAS(permit_extrapolation=True)

SimName = 'file_9209_CX'

#Simulation result
fileBE = 'Data/Outputs/%s.npy' % (SimName[:-2]+'BE') #_18
dataBE = np.load(fileBE)
fileCX = 'Data/Outputs/%s.npy' % SimName #_18
dataCX = np.load(fileCX)

wavelengthBE = dataBE[0,:,:]
signalBE = dataBE[1,:,:]
wavelengthCX = dataCX[0,:,:]
signalCX = dataCX[1,:,:]

signal_variance = dataCX[2,:,:]

# power
filePower = 'Data/Outputs/%sPower.npy' % SimName
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
E_density_input = inputfile['edensity']# second axis is Fiber_to_R
#E_density_input = np.load('Data/Fibers/EDensity2000.npy')
D_density_input = inputfile['ddensity']
#E_density_input = D_density_input

inRden = density_input [1,:]
inden = density_input [0,:]

#Zeff
Zeff = inputfile['zeff']
Bfield = inputfile['b_field']

#Beam density
Beam_n = np.load('Data/Inputs/BeamDensity%s.npy' %SimName)

### geometrical factros
paramfile = np.load('Data/Inputs/LOS%s.npz' %SimName)

#LOS radius and distance
LOS_r = paramfile['r_at_beam_save']

LOS_area = np.pi*LOS_r**2
D_to_Beam = paramfile['d_to_beam']
#print(D_to_Beam)

#LOS and Beam angle
AngleLB = paramfile['angle_LB']
# fiber acceptance angle'''
Ac_angle = paramfile['ac_angle']

# vertical cos factor
cosFV = paramfile['cos_ver']
cosFH = paramfile['cos_hor']

#Fiber number to R mapping
Fiber_to_R = paramfile['R_beam']
###

#color palette
palette = plt.get_cmap('plasma')


#style
plt.style.use('seaborn-darkgrid')

######multiple line plot
### fitting peaks
peak_wave_fit = []
peak_int_fit = []
FWHM = []

#integration
Integrated_intCX = []
integrated_BE_2 = np.empty((0,3),float)

lambda_0 = 529.07 # carbon
lambda_D = 656.28

Line = 'BE'
low_BE,hig_BE = Line_boundaries(Line)
Line = 'CX'
low_CX,hig_CX = Line_boundaries(Line)


fig1 = plt.figure()
spec = fig1.add_gridspec(ncols=1, nrows=2, height_ratios = [2,1])
ax1 = fig1.add_subplot(spec[0])
ax2 = fig1.add_subplot(spec[1])

fibers = np.arange(np.size(wavelengthCX,0))
mask = np.ones(np.size(wavelengthCX,0), dtype= bool)
mask[0:2] = False
fibers = fibers[mask]
print('active fibers',fibers)

for i in fibers: #range(np.size(wavelength,0)):
    ax1.plot(wavelengthCX[i,:],signalCX[i,:], marker = '', color = palette(i*4), linewidth = 1, alpha = 0.9, label =i+1)
    ax2.plot(wavelengthBE[i,:],signalBE[i,:], marker = '', color = palette(i*4), linewidth = 1, alpha = 0.9, label =i+1)
    print(i)
    #GAUSSIAN FITTING
    peekH, peekC, peekW = gaussian_fit(wavelengthCX[i, :],signalCX[i, :])
    peak_int_fit = np.append(peak_int_fit, peekH)
    peak_wave_fit = np.append(peak_wave_fit, peekC)
    FWHM = np.append(FWHM, peekW)

    #INTEGRATION
    Integrated_intCX = np.append(Integrated_intCX,integrate_signal(wavelengthCX[i,:],signalCX[i,:],low_CX,hig_CX))

    #PEAK FINDING, for BE
    peak, peak_v, half_max, edges = find_BE_peak(wavelengthBE[i,:],signalBE[i,:])
    Pho_energy = (1239.8 / (lambda_D))/ ( 1239.8 / (peak))
    print(Pho_energy)
    integrated_BE = area_gaussian(peak_v,half_max)
    ax2.plot(edges.flatten(),np.zeros(6),'x')
    #print(np.append(np.append([0],*integrate_BE_signal(wavelengthBE[i,:],signalBE[i,:],edges)),[0]))
    integrated_BE_2 =  np.append(integrated_BE_2 ,integrate_BE_signal(wavelengthBE[i,:],signalBE[i,:],edges)/Pho_energy,0)
    #integrated_BE_2 = np.append(integrated_BE_2,integrate_BE_signal(wavelengthBE[i, :], signalBE[i, :], edges),0)

    #integrated_allthree = integrate_signal(wavelength[i,:],signal[i,:],658,663)
    #BE_signal_ratios = integrated_BE_2/np.sum(integrated_BE_2)

ax1.legend(loc = 2, ncol = 3, title = 'Fiber')
ax1.set(title = (r'Spectral power'),xlabel = 'Wavelength, [nm]',ylabel= 'Spectral power, [W/nm]' ) #, $C^{5+}(8 \rightarrow 7), \: \lambda = 529.05 nm$
ax2.set(title = (r'Spectral power Standard deviation, $C^{5+}(8 \rightarrow 7), \: \lambda = 529.05 nm$'),xlabel = 'Wavelength, [nm]',ylabel= 'Spectral power, [W/nm]' )
plt.show()

length = np.size(signalCX,0)

####################################

# Velocity calculation
velocity = velocity_cal(peak_wave_fit,lambda_0)

# Temperature calculation
C_mass = 12.0107
D_mass = 2.014
Temperature = Temp_cal(FWHM,C_mass,lambda_0)

# Density calculation
Beam_radius = 0.025
Fiber_radius = 0.0001
cxr = adas.beam_cx_pec(deuterium, carbon, 6, (8, 7))
ber = adas.beam_emission_pec(deuterium,deuterium,1,(3,2))
berC = adas.beam_emission_pec(deuterium,carbon,6,(3,2))
bpr = adas.beam_population_rate(deuterium,2,deuterium,1)
bprC = adas.beam_population_rate(deuterium,2,carbon,6)

Beam_Energy = 65000
POPcoeff = np.array([[bpr(Beam_Energy/1, D_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [bpr(Beam_Energy/2, D_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [bpr(Beam_Energy/3, D_density_input[f], Temperature[i]) for i,f in enumerate(fibers)]])*1000000
POPcoeffC = np.array([[bprC(Beam_Energy/1, inden[f]*6, Temperature[i]) for i,f in enumerate(fibers)],
                   [bprC(Beam_Energy/2, inden[f]*6, Temperature[i]) for i,f in enumerate(fibers)],
                   [bprC(Beam_Energy/3, inden[f]*6, Temperature[i]) for i,f in enumerate(fibers)]])*1000000
BEcoeff = np.array([[ber(Beam_Energy/1, D_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [ber(Beam_Energy/2, D_density_input[f], Temperature[i]) for i,f in enumerate(fibers)],
                   [ber(Beam_Energy/3, D_density_input[f], Temperature[i]) for i,f in enumerate(fibers)]])
BEcoeffC = np.array([[berC(Beam_Energy/1, inden[f]*6, Temperature[i]) for i,f in enumerate(fibers)],
                   [berC(Beam_Energy/2, inden[f]*6, Temperature[i]) for i,f in enumerate(fibers)],
                   [berC(Beam_Energy/3, inden[f]*6, Temperature[i]) for i,f in enumerate(fibers)]])
CXcoeff = np.array([[cxr[0](Beam_Energy/1, Temperature[i], inden[f], Zeff[f], Bfield[f]) for i,f in enumerate(fibers)],
                   [cxr[0](Beam_Energy/2, Temperature[i], inden[f], Zeff[f], Bfield[f]) for i,f in enumerate(fibers)],
                   [cxr[0](Beam_Energy/3, Temperature[i], inden[f], Zeff[f], Bfield[f]) for i,f in enumerate(fibers)]])
CXcoeff2 = np.array([[cxr[1](Beam_Energy/1, Temperature[i],inden[f], Zeff[f], Bfield[f]) for i,f in enumerate(fibers)],
                   [cxr[1](Beam_Energy/2, Temperature[i], inden[f], Zeff[f], Bfield[f]) for i,f in enumerate(fibers)],
                   [cxr[1](Beam_Energy/3, Temperature[i], inden[f], Zeff[f], Bfield[f]) for i,f in enumerate(fibers)]])

POPEff = (D_density_input[None,mask]*POPcoeff + inden[None,mask]*6*POPcoeffC)/E_density_input[None,mask]
print(CXcoeff[0,0] + 1)
# intensity[W] divided by the beam density[m^-3] and eff,coff [W*m^3] = []

GF = np.ones(length)
for i in range(GF.size):
    GF[i] = geometrical_factor(Fiber_radius,D_to_Beam[i],Ac_angle[i],Beam_radius/np.sin(AngleLB[i]))

CXcoeffE = (CXcoeff + POPEff * CXcoeff2) / (1 + POPEff)

coefficientEff = np.sum(Beam_n[mask] * CXcoeffE.T, 1)

density = Integrated_intCX/GF[mask]/coefficientEff*4*np.pi

coefficient_power = np.sum(Beam_n[mask] * CXcoeffE.T, 1) * (np.sqrt(2 * np.pi * Beam_radius ** 2)) / np.sin(AngleLB[mask])
density_from_power = dataPower[mask] / coefficient_power * 4 * np.pi

BEcoeffDC = (D_density_input[None,mask]*BEcoeff + inden[None,mask]*6*BEcoeffC)/E_density_input[None,mask]
BEcoeffEff = BEcoeffDC

# Beam density
Beam_density1 = integrated_BE_2[:,2]/GF[mask]/BEcoeffEff[0,:]/E_density_input[mask]*4*np.pi
Beam_density2 = integrated_BE_2[:,1]/GF[mask]/BEcoeffEff[1,:]/E_density_input[mask]*4*np.pi
Beam_density3 = integrated_BE_2[:,0]/GF[mask]/BEcoeffEff[2,:]/E_density_input[mask]*4*np.pi
density_ratio = E_density_input[mask]*Integrated_intCX/(integrated_BE_2[:,2]*CXcoeffE[0,:]/BEcoeffEff[0,:]+integrated_BE_2[:,1]*CXcoeffE[1,:]/BEcoeffEff[1,:] + integrated_BE_2[:,0]*CXcoeffE[2,:]/BEcoeffEff[2,:])

#print(density_ratio)
x = np.arange(0,length)

Fiber_to_R = Fiber_to_R[mask]
# Save the calculation outputs for futher processing
np.savez('Data/Outputs/'+SimName, Bdensity = Beam_n[mask,0],Bdensity1 = Beam_n[mask,1],Bdensity2 = Beam_n[mask,2], BdensityCal = Beam_density1, BdensityCal1 = Beam_density2, BdensityCal2 = Beam_density3,  density = inden, densityR = inRden, denRatio = density_ratio, denPower = density_from_power, denFib = density, fiberR = Fiber_to_R)

figComp, [[x11, x12, x13], [x21, x22, x23], [x31, x32, x33]]  = plt.subplots(3,3, figsize = (30,20), tight_layout = True)
x11.plot(Fiber_to_R,Integrated_intCX,'o',label = 'CX carbon signal')
x11.set(ylabel = ('Power [W]'),xlabel = ('Major Radius [m]'),title = ('Integrated CX Signal'))
x11.legend(loc = 'best')

x12.plot(Fiber_to_R,integrated_BE_2[:,2],'o',label = 'BE signal, 1st EC.')
x12.plot(Fiber_to_R,integrated_BE_2[:,1],'o',label = 'BE signal, 2nd EC.')
x12.plot(Fiber_to_R,integrated_BE_2[:,0],'o',label = 'BE signal, 3rd EC.')
x12.set(ylabel = ('Power [W]'),xlabel = ('Major Radius [m]'),title = ('Integrated BE Signal'))
x12.legend(loc = 'best')

x13.plot(Fiber_to_R,BEcoeff[0,:],'o',color = 'tab:blue',label = ' D, 1st EC.')
x13.plot(Fiber_to_R,BEcoeff[1,:],'o',color = 'tab:orange',label = ' D, 2nd EC.')
x13.plot(Fiber_to_R,BEcoeff[2,:],'o',color = 'tab:green',label = ' D, 3rd EC.')
x13.plot(Fiber_to_R,BEcoeffDC[0,:],'x',color = 'tab:blue',label = ' D + C, 1st EC.')
x13.plot(Fiber_to_R,BEcoeffDC[1,:],'x',color = 'tab:orange',label = ' D + C, 2nd EC.')
x13.plot(Fiber_to_R,BEcoeffDC[2,:],'x',color = 'tab:green',label = ' D + C, 3rd EC.')
x13.set(ylabel = (r'BECoeff [$Wm^3str^{-1}$]'),xlabel = ('Major Radius [m]'),title = ('Effective BE  coefficient'))
x13.legend(loc = 'best')

x21.plot(Fiber_to_R,CXcoeff[0,:],'o',color = 'tab:blue',label = '(n=1), 1st EC.')
x21.plot(Fiber_to_R,CXcoeff[1,:],'o',color = 'tab:orange',label = '(n=1), 2nd EC.')
x21.plot(Fiber_to_R,CXcoeff[2,:],'o',color = 'tab:green',label = '(n=1), 3rd EC.')
x21.plot(Fiber_to_R,CXcoeffE[0,:],'x',color = 'tab:blue',label = '(n=1,2), 1st EC.')
x21.plot(Fiber_to_R,CXcoeffE[1,:],'x',color = 'tab:orange',label = '(n=1,2), 2nd EC.')
x21.plot(Fiber_to_R,CXcoeffE[2,:],'x',color = 'tab:green',label = '(n=1,2), 3rd EC.')
x21.set(ylabel = (r'CXcoeff [$ Wm^{3}str^{-1}]$'),xlabel = ('Major Radius [m]'),title = ('Effective CX coefficient'))
x21.legend(loc = 'best')

x22.plot(Fiber_to_R,Beam_n[mask,0],'o',color = 'tab:blue',label = 'n_beam in, 1st EC.')
x22.plot(Fiber_to_R,Beam_n[mask,1],'o',color = 'tab:orange',label = 'n_beam in, 2nd EC.')
x22.plot(Fiber_to_R,Beam_n[mask,2],'o',color = 'tab:green',label = 'n_beam in, 3rd EC.')
x22.plot(Fiber_to_R,Beam_density1,'x',color = 'tab:blue',label = 'n_beam out, 1st EC.')
x22.plot(Fiber_to_R,Beam_density2,'x',color = 'tab:orange',label = 'n_beam out, 2nd EC.')
x22.plot(Fiber_to_R,Beam_density3,'x',color = 'tab:green',label = 'n_beam out, 3rd EC.')
x22.set(ylabel = ('Density $[m^{-3}]$'),xlabel = ('Major Radius [m]'),title = ('Beam density components'))
x22.legend(loc = 'best')

x23.plot(Fiber_to_R,E_density_input[mask],'o',color = 'tab:purple',label = 'n_e')
x23.plot(Fiber_to_R,D_density_input[mask],'o',color = 'tab:red',label = 'n_D')
x23.set(ylabel = ('Density $[m^{-3}]$'),xlabel = ('Major Radius [m]'),title = ('Electron and D density'))
x23.legend(loc = 'best')

x31.plot(Fiber_to_R,inden[mask],'o',color = 'tab:blue',label = 'n_C, input')
x31.plot(Fiber_to_R,density_ratio,'o',color = 'tab:red',label = 'n_C, ratio')
x31.plot(Fiber_to_R,density,'o',color = 'tab:green',label = 'n_C, Fiber')
x31.plot(Fiber_to_R,density_from_power,'o',color = 'tab:orange',label = 'n_C, SightLine')
x31.set(ylabel = ('Density $[m^{-3}]$'),xlabel = ('Major Radius [m]'),title = ('Carbon density'))
x31.legend(loc = 'best')

x32.plot(Fiber_to_R,(integrated_BE_2[:,2]*CXcoeff[0,:]/BEcoeffEff[0,:]+integrated_BE_2[:,1]*CXcoeff[1,:]/BEcoeffEff[1,:] + integrated_BE_2[:,0]*CXcoeff[2,:]/BEcoeffEff[2,:]),'o',color = 'tab:blue',label = 'n_C, input')
x32.set(ylabel = ('...'),xlabel = ('Major Radius [m]'),title = ('Intermediet results'))
x32.legend(loc = 'best')

x33.plot(Fiber_to_R, Zeff[mask],'o',color = 'tab:blue',label = 'Zeff, input')
x33.plot(Fiber_to_R, Bfield[mask],'o',color = 'tab:red',label = 'B[T], input')
x33.set(ylabel = ('[a.u]/[T]'),xlabel = ('Major Radius [m]'),title = ('Zeff and B_field'))
x33.legend(loc = 'best')
#fig, [ ax4, ax5] = plt.subplots(2,1, figsize = (7,15), tight_layout = True)
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
ax4.set(ylabel = ('Temperature [a.u]'),xlabel = ('Major Radius [m]'),title = ('Temperature'))
ax4.legend(loc = 'upper right')
'''
'''
ax3.scatter(Fiber_to_R,integrated_BE_2[:,2],label ='Integrated BE, full')
ax3.scatter(Fiber_to_R,integrated_BE_2[:,1],label ='Integrated BE, half')
ax3.scatter(Fiber_to_R,integrated_BE_2[:,0],label ='Integrated BE, third')
ax3.scatter(Fiber_to_R,Integrated_intCX,label ='Integrated CX')
ax3.axis(ymin = 0,ymax = integrated_BE_2[:,2].max()*1.05)
ax3.set(ylabel = ('power [W]'),xlabel = ('Major Radius [m]'),title = ('Integrated signal'))
ax3.legend(loc = 'upper right')
'''
'''''
ax3.scatter(Fiber_to_R,Int1/integrated_BE_2[:,2],label ='full')
ax3.scatter(Fiber_to_R,Int2/integrated_BE_2[:,1],label ='half')
ax3.scatter(Fiber_to_R,Int3/integrated_BE_2[:,0],label ='third')
#ax3.scatter(Fiber_to_R,Integrated_intCX,label ='Integrated CX')
ax3.axis(ymin = 0)
ax3.set(ylabel = ('power ratio, a.u'),xlabel = ('Major Radius [m]'),title = ('Ratio of CXn/BEn power'))
ax3.legend(loc = 'upper right')
'''
'''
ax5.scatter(inRden[mask],inden[mask],label = 'Impurity density input')
ax5.scatter(Fiber_to_R,density_from_power,label = 'Impurity density measurement, SigthLine')
ax5.scatter(Fiber_to_R,density,label = 'Impurity density measurement, Fiber')
ax5.scatter(Fiber_to_R,density_ratio,label = 'Impurity density measurement, Ratio method')
ax5.set(ylabel = ('Density [m^-3]'),xlabel = ('Major Radius [m]'),title = ('Impurity(Carbon) Density profile'))
ax5.legend(loc = 'best')

ax4.plot(Fiber_to_R,Beam_n[mask,0],'xr',label = 'Beam density input 1')
ax4.plot(Fiber_to_R,Beam_n[mask,1],'xb',label = 'Beam density input 1/2')
ax4.plot(Fiber_to_R,Beam_n[mask,2],'xg',label = 'Beam density input 1/3')
ax4.plot(Fiber_to_R,Beam_density1,'or',label = 'Beam density measurement, 1 energy')
ax4.plot(Fiber_to_R,Beam_density2,'ob',label = 'Beam density measurement, 1/2 energy')
ax4.plot(Fiber_to_R,Beam_density3,'og',label = 'Beam density measurement, 1/3 energy')
ax4.set(ylabel = ('Density [m^-3]'),xlabel = ('Major Radius [m]'),title = ('Beam Density profile'))
ax4.legend(loc = 'best')
'''
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
ax32.axis(ymin = CXcoeff.min()*0.95,ymax = CXcoeff.max()*1.05)
ax32.set(title = ('R [m]'), ylabel = ('CX emmision rate'),xlabel = ('major radius [m]'))
ax32.legend(loc = 'upper right')
'''
#fig.suptitle('29880')
plt.show()


