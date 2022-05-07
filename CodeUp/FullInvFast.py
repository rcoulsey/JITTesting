# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:34:39 2022

@author: ryanc
"""

import numpy as np
from matplotlib import pyplot as plt 
from numba import njit, prange

##Loop it 

def Plot2d(x, y, Gau2, label=''):
    plt.pcolormesh(x, y, Gau2.T, shading='auto')
    plt.colorbar()
    plt.title(label)
    plt.tight_layout()
    plt.show()

@njit(parallel=True)
def MakePrior(PriorJoint, priormarg, MargData, xv):
    for i in prange(len(xv)):
        PriorJoint[i, :, :] = priormarg[i]*MargData
        
    return PriorJoint

@njit(parallel=True)
def MakeTheory(pxT, Tphi, dxT, Trho, xv, Gau1, Gau2, Theory):
    for i in prange(len(xv)):
        Gau1 = np.exp(-(pxT-Tphi[i])**2/((2*0.15**2)))
        Const1 = 1/(Tphi[i]*np.sqrt(2*np.pi))
        Gau1 = Const1*Gau1

        Gau2 = np.exp(-(dxT-Trho[i])**2/((2*0.08**2)))
        Const2 = 1/(Trho[i]*np.sqrt(2*np.pi))
        Gau2 = Const2*Gau2
        
        Theory[i, :, :] = np.outer(Gau1, Gau2)
        
    return Theory

#@njit
def Cascade(Z, GR, NPhi, RhoB, Cali, xv, c, px, dx, PriorJoint, Theory, pxT, dxT):
    
    vbar = 5.654 - 0.008*GR

    Gau = np.exp(-(xv-vbar)**2/((2*c**2)))
    Const = 1/(c*np.sqrt(2*np.pi))
    Gau = Const*Gau
    priormarg = Gau
    
    #######################################################
    
    aRho = 0.12
    bRho = 0.15
    aPhi = 0.08
    bPhi = 0.2
    
    CREF = 6
    CMAX = 8
    CALI = Cali

    #for 1 depth level make the 2d gaussian
    sigmap = aPhi + (CALI-CREF)*bPhi/(CMAX-CREF)
    sigmad = aRho + (CALI-CREF)*bRho/(CMAX-CREF)
    

    Gau1 = np.exp(-(px-NPhi)**2/((2*sigmap**2)))
    Const1 = 1/(NPhi*np.sqrt(2*np.pi))
    Gau1 = Const1*Gau1
    
    Gau2 = np.exp(-(dx-RhoB)**2/((2*sigmad**2)))
    Const2 = 1/(RhoB*np.sqrt(2*np.pi))
    Gau2 = Const2*Gau2

    MargData = np.outer(Gau1, Gau2)
    
    #######################################################
    
    
    PriorJoint = MakePrior(PriorJoint, priormarg, MargData, xv)
    
    #######################################################
    
    rhom = 2.71
    rhof = 1
    
    Trho = 1.74*xv**(0.25)
    Tphi = (Trho - rhom)/(-rhom + rhof)
    
    ########################################################
    
    Theory = MakeTheory(pxT, Tphi, dxT, Trho, xv, Gau1, Gau2, Theory)

#     for i in range(len(xv)):
#         Gau1 = np.exp(-(pxT-Tphi[i])**2/((2*0.15**2)))
#         Const1 = 1/(Tphi[i]*np.sqrt(2*np.pi))
#         Gau1 = Const1*Gau1

#         Gau2 = np.exp(-(dxT-Trho[i])**2/((2*0.08**2)))
#         Const2 = 1/(Trho[i]*np.sqrt(2*np.pi))
#         Gau2 = Const2*Gau2
        
#         Theory[i, :, :] = np.outer(Gau1, Gau2)

    Theory /= np.sum(Theory) 
    
    ######################################################
    
    Theory = Theory[0:len(xv), 0:len(px), 0:len(dx)]
    Posterior = Theory*PriorJoint

    posteriorMarg = np.sum(Posterior, axis=2)
    posteriorMarg = np.sum(posteriorMarg, axis=1)

    ######################################################
    
    return posteriorMarg


data = np.genfromtxt('Z_GR_NPHI_RHOB_CALI.txt', delimiter=',').T
Z = data[0]
GR = data[1]
NPhi = data[2]
RhoB = data[3]
Cali = data[4]

D = 50

#define value range
xv = np.linspace(2, 7, 520)
#confidence in equation is low so
c = 0.5

#construct a range of values for the density and porosity
px = np.linspace(0, 0.4, 502)
dx = np.linspace(2.2, 3, 504)

FullInv = np.zeros((len(Z), len(xv)))
PriorJoint = np.zeros((len(xv), len(px), len(dx)))
Theory = np.zeros((len(xv), len(xv), len(xv)))
pxT = np.linspace(0, 0.6, len(xv))
dxT = np.linspace(2.2, 3, len(xv))

for d in range(len(Z)):
    FullInv[d, :] = Cascade(Z[d], GR[d], NPhi[d], RhoB[d], Cali[d], xv, c, px, dx, PriorJoint, Theory, pxT, dxT)
    
print(FullInv.shape)

Plot2d(range(520)/10, range(3200), FullInv[0:3200, :].T)