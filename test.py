# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:12:34 2022

@author: QiNova
"""
import numpy as np
import scipy
from matplotlib import pyplot as plt
from math import sqrt
from SBEwithCI import SBE, loadband, Matplot
import time as tp




start=tp.time()

pi=scipy.constants.pi
d=9.32
t=np.linspace(0,41*20,10000)
k=np.linspace(-pi/d, pi/d,201)    
dk=k[1]-k[0]


#set electric field 0
w1=108/27.21       #frequency
tau=41*0.2*sqrt(2) #pulse duration
fi=0               #phase
E0=2e8             #field strength
E0field=np.exp(-4*np.log(2)*((t-41*6)/tau)**2)*np.cos(w1*(t-41*6)+fi)/5.14e11*E0  #field 0

#set electric field 1 (optional)
w2=0.075         #frequency
tau1=41*2*sqrt(2) #pulse duration
fi1=0             #phase
E1=1e10          #field strength
delay=0          #delay between two pulses
E1field=np.exp(-4*np.log(2)*((t-41*6-delay)/tau1)**2)*np.cos(w2*(t-41*6-delay)+fi1)/5.14e11*E1 #field 1

#set band structure
VB1=np.zeros(k.size)
VB=np.array([VB1])

CB1=loadband(k, 'sio2\\CB1.txt')
CB2=loadband(k, 'sio2\\CB2.txt')
CB3=loadband(k, 'sio2\\CB3.txt')
CB4=loadband(k, 'sio2\\CB4.txt')     #load the band from txt file   
CB=np.array([CB1,CB2,CB3,CB4])/27.21 #put the band in a matrix

Ek=np.vstack((VB,CB))                #create a matrix include all the energy band




#set mapping matrix
MP=np.linspace(0,24,25,dtype=int).reshape((5,5))

#set transition dipole matrix
dipole1=np.ones(k.size)*0.1
dipole=np.array([dipole1 for i in range(25)])     #dipole should be a n^2 by k matrix ,n is the number of energy bands

dipole[[0,6,12,18,24]]=np.zeros(k.size)
dipole[[2,10,14,22,4,20,8,16]]=np.zeros(k.size)

dipole[[19,23]]=10*dipole1
dipole[[9,21]]=5*dipole1
dipole[[3,15]]=1.35*dipole1
dipole[[7,11]]=20*dipole1
dipole[[13,17]]=1*dipole1



#set dephasing time
T2=41*0.85 
T1=41*0.85


#calculate energy difference and dephasing term (for saving time) 
ddt=np.array([k*(0+0j) for i in range(25)])
for i in range(len(Ek)):
    for j in range(len(Ek)):
        if i==j:
            pass
        else:
            ddt[MP[i,j]]=Ek[j]-Ek[i]-1j/T2


#set coulomb potential
Vkq=np.ones((k.size,k.size))*(0+0j)
V0=1.8
for p in range(k.size):
    Vkq[p]=V0*dk/2/pi*2*scipy.special.kn(0,np.abs(k[p]-k))   #kn is modified Bessel function of the second kind
Vkq[np.diag_indices_from(Vkq)]=0+0j




#set filter for FFT
# gauss=np.exp(-4*np.log(2)*((t-41*10)/10/41)**12)     

# SBEg=SBE(d,VB,CB,ddt,dipole,k,E0field,E1field,T2,T1,t,Vkq)  #create a SBE object
# SBEg.solve()                                                #solve the PDE 
# SBEg.calculate(gauss)                                       #calculate the P(t),J(t),d(t),P(ω)，J(ω),d(ω), and absorption spectrum
# SBEg.showEfield(rangex=(80,130))                            #plot the electric field in time and frequency, rangex is the range of energy (eV)
# SBEg.showEnergy()                                           #plot all the energy bands
# SBEg.plotelectrondensity(n=0,rangex=(5,7))                  #plot the electron/hole density dynamics on n-th band, rangex is the range of tiem (fs)
# SBEg.plotemission(rangex=(0,600))                           #plot P(t),J(t),P(ω)，J(ω),rangex is the range of energy (eV)
# SBEg.plotabsorption(rangex=(104,112),rangey=(0,-120))       #plot the absorption spectrum,rangex is the range of energy (eV)


# end=tp.time()
# print(end-start)