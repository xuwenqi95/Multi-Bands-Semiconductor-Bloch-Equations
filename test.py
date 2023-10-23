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

#all the value should be convert to atomic unit

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
 
