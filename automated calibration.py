# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:10:11 2024

@author: yufei
"""
import numpy as np 
import scipy as scipy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
import pybaselines.polynomial
from pybaselines import Baseline, utils

###  CHANGE THE FOLLOWING PARAMETERS TO OPEN YOUR RUBY DATA FILE  ###
grat    = 1800              # Diffraction grating
mono    = 'triple'          # Monochromator configuration
obj     = 0.5               # Microscope objective
laser   = 3                 # Laser power in uW
binn    = 1                 # Binning factor
date    = '11-18-24'         
path = 'G:/My Drive/Research/PZO pressure/ALL/Nov 2024/{0}/ruby/actually 4K (1128-1198)/'.format(date)
filename = 'ruby_{0}_{2}_{3}uW'.format(grat,mono,laser)

###  CHANGE THE FOLLOWING PARAMETERS TO BEST FIT YOUR RUBY DATA  ###
poly = 1                     # Polynomial order used to fit the baseline
R2prom = 0.5                   # Prominence value used to estimate R2 peak
R1prom = 3                   # Prominence value used to estimate R1 peak
R2start = 30                 # Starting index for R2
R2end = 200                  # Ending index for R2
R1start = 200                # Starting index for R1
R1end = 450                  # Ending index for R1
start = 656                  # First ruby spectra to fit
end = 1094                   # Last ruby spectra to fit 

k = 0.08617                  # Boltzmann constant in meV/K
S = [3.55, 0.2]              # Splitting between R2 and R1 in meV
N = [0.64781, 0.03]          # Quantum efficiency ratio 
ambR1 = [693.85, 0.003]      # R1 position at ambient pressure, from 2-17-24
measured_laser = 514.501     # Measured laser line for calibration, enter 0 for already calibrated data 
R2popt,R1popt,R2pos,R1pos,R2fwhm,R1fwhm,R2int,R1int,iratio,hratio,xdata,ydata,temps,pressures = ([] for i in range(14))

fig, ax = plt.subplots(1, 2, figsize=(20,8))
ax[0].set_xlabel('Wavelength (nm)')
ax[0].set_ylabel('Intensity (a.u.)')
ax[1].set_xlabel('Wavelength (nm)')
ax[1].set_ylabel('Intensity (a.u.)')
ax[0].tick_params(direction='in')
ax[1].tick_params(direction='in')
ax[1].sharey(ax[0])
        

def Lorentz(x, c, w, a):
    return (a/np.pi)*(w/(((x-c)**2)+(w**2)))

def fitpeak(x, y, g):
    popt, pcov = scipy.optimize.curve_fit(Lorentz, x, y, p0=g, bounds=[[690,0,0],[710,np.inf,np.inf]], maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    position = [popt[0],perr[0]]
    fwhm = [2*popt[1],2*perr[1]]
    intensity = [(popt[2]/(np.pi*popt[1])),np.sqrt((perr[2]/(np.pi*popt[1]))**2 + ((popt[2]*perr[1])/(np.pi*(popt[1]**2)))**2)]
    return [position, fwhm, intensity, popt, pcov, perr]

for j in range(start, end+1):
    x, y = ([] for i in range(2))
    file = open(path + filename + '_{0}.txt'.format(j)) 
    data = file.read()
    data = [float(i) for i in data.split()]
    for i in range(0, len(data), 2):
        x.append(data[i] + (514.532 - measured_laser))
        y.append(data[i+1])    
    x = x[:500]
    y = y[:500]
    
    baseline_fitter = Baseline(x_data=x)
    baseline1, params1 = baseline_fitter.modpoly(y, poly_order=poly)
    y -= baseline1
    xdata.append(x)
    ydata.append(y)
    
    R2x = np.array(x[R2start:R2end])
    R2y = np.array(y[R2start:R2end])
    R1x = np.array(x[R1start:R1end])
    R1y = np.array(y[R1start:R1end])
    
    R2peaks, _ = find_peaks(R2y, prominence=R2prom)
    R1peaks, _ = find_peaks(R1y, prominence=R1prom)
    R2p = fitpeak(R2x, R2y, [R2x[R2peaks][0],0.2,R2y[R2peaks][0]])     #R2 parameters
    R1p = fitpeak(R1x, R1y, [R1x[R1peaks][0],0.2,R1y[R1peaks][0]])      #R1 parameters
    
    R2pos.append(R2p[0])
    R1pos.append(R1p[0])
    R2fwhm.append(R2p[1])
    R1fwhm.append(R1p[1])
    R2int.append(R2p[2])
    R1int.append(R1p[2])
    R2popt.append(R2p[3])
    R1popt.append(R1p[3])
    
    i2 = R2p[2][0]
    i2e = R2p[2][1]
    i1 = R1p[2][0]
    i1e = R1p[2][1]
    iratioerr = np.sqrt((i2e/i1)**2+((i2*i1e)/(i1)**2)**2)
    iratio.append([i2/i1,iratioerr])
    hratio.append(max(R2y[R2peaks])/max(R1y[R1peaks]))
    
    #calculating in-situ temperature, using Weinstein S and calculated N at 100K
    z = i2/(N[0]*i1)
    T = -S[0]/(k*np.log(z))
    Terr = np.sqrt((S[1]/(k*np.log(z)))**2 + 
                   ((S[0]*N[1])/(k*N[0]*((np.log(z))**2)))**2 + 
                   ((S[0]*iratioerr)/(k*(i2/i1)*((np.log(z))**2)))**2)
    temps.append([T,Terr])
    
    #calculating temperature induced shift, following Datchi et al. 2007
    if T < 50:
        T_shift = [-0.887, 0.001]
    elif 50 <= T <= 296: 
        dT = T - 296
        a = [0.00664, 0.00004]
        b = [6.76*10**-6, 5.2*10**-7]
        c = [2.33*10**-8, 1.6*10**-9]
        T_shift = [(a[0]*dT) + (b[0]*(dT**2)) - (c[0]*(dT**3)),
                   np.sqrt((dT*a[1])**2 + ((dT**2)*b[1])**2 + ((dT**3)*c[1])**2 + ((a[0] + (2*b[0]*dT) - (3*c[0]*(dT**2)))*Terr)**2 )]
    else: print('The temperature is greater than 296K!')
    
    #calculating pressure using in-situ temperature, following Shen et al. 2020
    d = [1870, 10]
    e = [5.63, 0.03]
    P_shift = [R1p[0][0]-T_shift[0], np.sqrt((R1p[0][1])**2 + (T_shift[1])**2)]
    dR1 = [P_shift[0]-ambR1[0], np.sqrt((P_shift[1])**2 + (ambR1[1])**2)]
    P = ((d[0]*dR1[0])/ambR1[0]) + ((d[0]*e[0]*(dR1[0]**2))/(ambR1[0]**2))
    Perr = np.sqrt((((ambR1[0]*dR1[0]+e[0]*(dR1[0]**2))/(ambR1[0]**2))*d[1])**2 + 
                   ((d[0]*(dR1[0]**2)*e[1])/(ambR1[0]**2))**2 +
                   (((ambR1[0]*d[0]+2*d[0]*e[0]*dR1[0])/(ambR1[0]**2))*dR1[1])**2 + 
                   (((ambR1[0]*d[0]*dR1[0]-2*d[0]*e[0]*(dR1[0]**2))/(ambR1[0]**3))*ambR1[1])**2)
    pressures.append([P,Perr])
    
    R2prom -= 0.005
    
np.savetxt(path + 'R2 positions.txt', R2pos, fmt='%.18f')
np.savetxt(path + 'R1 positions.txt', R1pos, fmt='%.18f')
np.savetxt(path + 'R2 widths.txt', R2fwhm, fmt='%.18f')
np.savetxt(path + 'R1 widths.txt', R1fwhm, fmt='%.18f')
np.savetxt(path + 'R2 intensities.txt', R2int, fmt='%.18f')
np.savetxt(path + 'R1 intensities.txt', R1int, fmt='%.18f')
np.savetxt(path + 'intensity ratios.txt', iratio, fmt='%.18f')
np.savetxt(path + 'height ratios.txt', hratio, fmt='%.18f')
np.savetxt(path + 'in-situ temperatures.txt', temps, fmt='%.18f')
np.savetxt(path + 'pressures.txt', pressures, fmt='%.18f')

tempS = []
colors = plt.cm.coolwarm(np.linspace(0, 1, (abs(end-start))+1))

for i in range(0, len(ydata)):
    ax[0].plot(xdata[0], ydata[(abs(end-start))-i], color=colors[i])
    ax[1].plot(x, Lorentz(x, *R2popt[(abs(end-start))-i]), color=colors[i])
    ax[1].plot(x, Lorentz(x, *R1popt[(abs(end-start))-i]), color=colors[i])
    
for i in range(0, len(temps), 30):
    tempS.append(int(temps[i][0]))
scalarmap = plt.cm.ScalarMappable(norm=None, cmap='coolwarm')
scalarmap.set_array(tempS)
fig.colorbar(scalarmap, ax=ax, ticks=tempS, pad=0.025)

plt.savefig(path + date + " fit plots.png")