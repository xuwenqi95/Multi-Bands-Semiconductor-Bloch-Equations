# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:40:42 2023

@author: Wenqi
"""


import numpy as np
import scipy
from matplotlib import pyplot as plt
from math import sqrt
from SBEwithCI import SBE, loadband, Matplot
import time as tp
import copy
from multiprocessing import Pool
import multiprocessing as mp
from matplotlib import cm

from numpy.fft import fft, fftfreq,fftshift,ifft


#A simple SBE simulation should be done before start the scan
def pumpprobe(SBEg,delay,pumpI,probeI,V0,T2,T1): #funcition of scan the parameter (delay,intensity of pump, intensity of probe, Coulomb potential strength, T2, T1)
    

    t=SBEg.time
    gauss=np.exp(-4*np.log(2)*((t-41*200)/180/41)**12)   
    w1=0.057       #frequency
    tau=41*25*sqrt(2) #pulse duration
    fi=0               #phase
    E0=1e9*pumpI             #field strength, pumpI is the intensity factor
    E0field=np.exp(-4*np.log(2)*((t-41*200)/tau)**2)*np.cos(w1*(t-41*200)+fi)/5.14e11*E0  #field 0
    
    #set electric field 1 (optional)
    w2=0.057*2         #frequency
    tau1=41*30*sqrt(2) #pulse duration
    fi1=0             #phase
    E1=1e9*probeI    #field strength, probeI is the intensity factor
    E1field=np.exp(-4*np.log(2)*((t-41*200-delay)/tau1)**2)*np.cos(w2*(t-41*200-delay)+fi1)/5.14e11*E1 #field 1
    
    SBE1=copy.deepcopy(SBEg)
    
    SBE1.T2=T2
    SBE1.T1=T1
    SBE1.Vkq=V0*SBE1.Vkq #Coulomb matrix, V0 is the strength factor
    SBE1.E0=E0field
    SBE1.Efield=E0field+E1field#new Efield
    SBE1.solve()#solve PDEs
    SBE1.calculate(gauss)#calculate the P(t),J(t),d(t),P(ω)，J(ω),d(ω), and absorption spectrum
    return SBE1.SwP     #also can return like (SBE1.SwP,SBE1.SwJ,SBE1.Sw)


if __name__ == '__main__': #similar to main function in C/C++, multiprocessing need this

    
    pi=scipy.constants.pi #pi constant
    d=9.32                  #lattice constant
    t=np.linspace(0,41*400,10000) #time axis
    k=np.linspace(-pi/d, pi/d,201)    #k axis
    dk=k[1]-k[0]
    
    
    #set electric field 0
    w1=0.057       #frequency
    tau=41*25*sqrt(2) #pulse duration
    fi=0               #phase
    E0=1e10             #field strength
    E0field=np.exp(-4*np.log(2)*((t-41*200)/tau)**2)*np.cos(w1*(t-41*200)+fi)/5.14e11*E0  #field 0
    
    #set electric field 1 (optional)
    w2=0.057*2         #frequency
    tau1=41*30*sqrt(2) #pulse duration
    fi1=0             #phase
    E1=1e9         #field strength
    delay=0          #delay between two pulses
    E1field=np.exp(-4*np.log(2)*((t-41*200-delay)/tau1)**2)*np.cos(w2*(t-41*200-delay)+fi1)/5.14e11*E1 #field 1
    
    #load band structure

    Ek=np.loadtxt('jicai_project/bandstructure5.txt')
    Ek=np.hstack((np.flip(Ek[:,1:],axis=1),Ek))/27.21 #symmetry band structure
    
      
    VB=np.array(Ek[0:3])#assign VB
    CB=np.array(Ek[3:])#assign CB
    
    
    
    
    
    #set mapping matrix
    MP=np.linspace(0,len(Ek)**2-1,len(Ek)**2,dtype=int).reshape((len(Ek),len(Ek)))
    
    #set transition dipole matrix
    
    dipole=np.array([np.zeros(k.size) for i in range(len(Ek)**2)])     #dipole should be a n^2 by k matrix ,n is the number of energy bands
   
    #K-p theory
    for p in range(len(Ek)):
        for q in range(len(Ek)):
            if p==q:
                pass
            else:
                if p<=2 and q<=2: #use condition to set different value of dipole for each band
                    dipole[MP[p,q]]=0.1*(Ek[p,100]-Ek[q,100])/(Ek[p]-Ek[q])
                else:
                    dipole[MP[p,q]]=0.1*(Ek[p,100]-Ek[q,100])/(Ek[p]-Ek[q])
    
    
    
    
    
    
    
    
    #set dephasing time
    T2=41*2
    #relaxation time
    T1=41*5
    
    #set coulomb potential(FFT of 1D soft Coulomb)
    Vkq=np.ones((k.size,k.size))*(0+0j)
    V0=0.1
    for p in range(k.size):
        Vkq[p]=V0*dk/2/pi*2*scipy.special.kn(0,np.abs(k[p]-k))   #kn is modified Bessel function of the second kind
    Vkq[np.diag_indices_from(Vkq)]=0+0j
    
    
    #set filter for FFT when calculating Pw and Jw
    gauss=np.exp(-4*np.log(2)*((t-41*200)/180/41)**12)     

    SBEg=SBE(d,VB,CB,dipole,k,E0field,E1field,T2,T1,t,Vkq)  #create a SBE object
    SBEg.solve()                                                #solve the PDE 
    SBEg.calculate(gauss)                                       #calculate the P(t),J(t),d(t),P(ω)，J(ω),d(ω), and absorption spectrum
    SBEg.showEfield(rangex=(1,2))                            #plot the electric field in time and frequency, rangex is the range of energy (eV)
    SBEg.showEnergy()                                           #plot all the energy bands
    SBEg.plotelectrondensity(n=0,rangex=(190,210))                  #plot the electron/hole density dynamics on n-th band, rangex is the range of tiem (fs)
    SBEg.plotemission(rangex=(0,15))                           #plot P(t),J(t),P(ω)，J(ω),rangex is the range of energy (eV)
 

    delaytime=np.linspace(-80,80,41)*41 #set delay time points
    V0=[0,7]                      #set Coulomb strength scan
    pumpI=[1,2,3,4,5,6,7,8,9,10]  #set pump intensity factor scan
    probeI=[3,4,5,6,7,8]        #set probe intensity factor scan
    T2scan=np.array([2])*41     #set T2 scan
    T1scan=np.array([2,5])*41   #set T1 scan
    
    scan=[(SBEg,tpoints,pumpIpoints,probeIpoints,V0points,T2points,T1points) for T1points in T1scan for T2points in T2scan for V0points in V0 for probeIpoints in probeI for pumpIpoints in pumpI for tpoints in delaytime ]
    for i in range(len(V0)*len(pumpI)*len(probeI)*len(T2scan)*len(T1scan)): #do delay scan loop for other parameters setting
        pool=Pool(4) #4 processing
        scanparameter=scan[41*i:41*(i+1)]
        Vpoints=scanparameter[0][4]
        pumpIpoints=scanparameter[0][2]
        probeIpoints=scanparameter[0][3]
        T2points=scanparameter[0][5]
        T1points=scanparameter[0][6]
        pp=pool.starmap(pumpprobe, scanparameter) #multiprocessing calculation, result will be store in a list
        
        pool.close() # pool close
        pool.join() #pool close
        #save the delay scan 
        np.save('jicai_project//5bands//scan3//pumpI='+str(pumpIpoints)+'e9'+'probeI='+str(probeIpoints)+'e9'+'T2='+str('%.2f' % (T2points/41))+'T1='+str('%.2f' % (T1points/41))+'Vkq='+str('%.2f' % (0.1*Vpoints))+'.npy',np.array(pp))

