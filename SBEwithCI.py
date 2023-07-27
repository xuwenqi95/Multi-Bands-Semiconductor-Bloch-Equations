 # -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:26:43 2022

@author: QiNova
"""

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter, blackman
from numpy.fft import fft, fftfreq,fftshift,ifft
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib import cm
from math import sqrt, log, acosh
from finiteSBE import finiteSBE

   

def loadband(k,filename):
    band=np.loadtxt(filename).T                            #load txt data
    band[1]=savgol_filter(band[1],11,1)                    #smooth data
    f=interpolate.interp1d(band[0],band[1])                
    newband=f(k)                                           #Interpolation for your setting k grid
    return newband

def Matplot(M,k):
    for line in M:
        plt.plot(k,line)

def difft(Ek,k):
    x=fftfreq(k.size,k[1]-k[0])*2*pi
    Ex=fft(Ek)
    vk=np.real(ifft(1j*x*Ex))
    return vk
        
                            
class SBE:
    def __init__(self,d,VB,CB,dipole,k,E0field,E1field,T2,T1,time,Vkq):
        N1=len(CB)
        N2=len(VB)
        self.d=d
        self.VB=VB
        self.CB=CB
        
        self.bandN=N1+N2              #Number of band
        self.MP=np.linspace(0, self.bandN**2-1,self.bandN**2,dtype=int).reshape([self.bandN,self.bandN]) #Mapping Matrix
        

        
        
        self.feN=N1                   #Number of CB
        self.fhN=N2                   #Number of VB
        self.dipole=dipole
        self.Efield=E0field+E1field
        self.E0=E0field
        self.T2=T2
        self.T1=T1
        self.time=time
        self.freqs =fftshift(fftfreq(time.size, time[1] - time[0]))   #generate frequency
        self.energy=self.freqs*2*pi*27.21 #eV
        self.k=k
        self.kN=int(self.k.size)                                       #Number of K point
        self.dk=k[1]-k[0]
        self.y0=np.zeros((self.bandN**2+self.bandN)*self.kN,dtype=complex)    #inital value of (pk,fk)
        self.Vgv=difft(self.VB,self.k)
        self.Vgc=difft(self.CB,self.k)
        self.Ek=np.vstack((self.VB,self.CB))                       #Energy of CB and VB
        self.Vg=np.vstack((self.Vgv,self.Vgc))
        self.Vkq=Vkq
   
        
    def calculateddt(self):
        self.ddt=np.array([self.k*(0+0j) for i in range(len(self.Ek)**2)])
        for i in range(len(self.Ek)):
            for j in range(len(self.Ek)):
                if i==j:
                    pass
                else:
                    self.ddt[self.MP[i,j]]=self.Ek[j]-self.Ek[i]-1j/self.T2
        
        
        
        
    def showEfield(self,rangex): #plot the electric field
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(self.time/41,self.Efield)
        plt.title('Electricfield')
        plt.ylabel('A.U.')
        plt.xlabel('Time(fs)')
        plt.subplot(2,1,2)
        plt.plot(self.freqs*2*pi*27.21,np.abs(fftshift(fft(self.Efield))))
        plt.xlabel('eV')
        plt.xlim(rangex[0],rangex[1])
        plt.show()
        
    def showEnergy(self): #plot the bandstructure
        plt.figure()
        Matplot(self.Ek*27.21,self.k/pi*self.d)
        plt.title('Bandstructure')
        plt.xlabel('k(π/d)')
        plt.ylabel('energy(eV)')
        plt.figure()
        Matplot(self.Vg,self.k/pi*self.d)
        plt.title('Group Velocity')
        plt.ylabel('A.U.')
        plt.xlabel('k(π/d)')
        plt.show()
    def solve(self):
        self.calculateddt()
        self.hindex=np.linspace(0, self.fhN-1,self.fhN,dtype=int)                   #set hole/valence band index
        self.eindex=np.linspace(self.fhN, self.feN+self.fhN-1,self.feN,dtype=int)   #set electron/conduct band index
        self.parameter1=(self.bandN,self.kN,self.T1,self.dk)                        #put all the constant parameter in a tuple
        self.parameter2=(self.time,self.Efield,self.dipole,self.ddt,self.MP,self.Vkq,self.k,self.hindex,self.eindex)  #put all the array parameter in a tuple
        self.sol=solve_ivp(finiteSBE, (self.time[0],self.time[-1]),self.y0, t_eval=self.time,atol=1e-10,rtol=1e-12,args=(self.parameter1,self.parameter2)) #use solve_ivp to solve the PDE, change the atol and rtol to control the accuracy

    def calculate(self,Filter):
        y=self.sol.y.T
        self.Pt=np.ones(self.sol.t.size)*(0+0j)
        self.Jt=np.ones(self.sol.t.size)*(0+0j)
        self.fkt=np.zeros((self.sol.t.size,self.bandN*self.kN),dtype=complex)
        self.pkt=np.zeros((self.sol.t.size,self.bandN**2*self.kN),dtype=complex)
        for i in range(len(self.sol.t)):
            yi=np.reshape(y[i],(self.bandN**2+self.bandN,self.kN))
            pk=yi[0:self.bandN**2]
            fk=yi[self.bandN**2:]
            self.Pt[i]=np.real(np.sum(np.sum(self.dipole*pk,1)))
            #self.Pt[i]=np.sum(np.sum(self.dipole*pk,1))
            self.Jt[i]=np.real(np.sum(np.sum(self.Vg*fk,1)))
            #self.Jt[i]=np.sum(np.sum(self.Vg*fk,1))
            self.pkt[i]=pk.reshape(pk.size)
            self.fkt[i]=fk.reshape(fk.size)
        self.Pw=fftshift((fft(self.Pt*Filter)))
        self.Jw=fftshift((fft(self.Jt*Filter)))
        self.SwP=np.abs(self.Pw)**2
        self.SwJ=np.abs(self.Jw)**2
        self.dt=difft(self.Pt,self.time)+self.Jt
        self.dw=fftshift((fft(self.dt*Filter)))
        self.Sw=np.abs(self.dw)**2
        self.Ew=fftshift(fft(self.E0))
        self.absorption=self.freqs*np.imag(self.Pw/self.Ew)
        
    def plotelectrondensity(self,n,rangex):
        plt.figure(2)
        self.fkn=(self.fkt.T)[n*self.kN:(n+1)*self.kN]
        T,K=np.meshgrid(self.time,self.k)
        plt.contourf(T/41,K,self.fkn,cmap=cm.jet,levels=2**7);
        plt.ylabel('k')
        plt.xlabel('Time(fs)')
        plt.xlim(rangex[0],rangex[1])
        plt.colorbar()
        plt.show()
        
    def plotcoherence(self,n,rangex):
        plt.figure(3)
        self.pkn=(self.pkt.T)[n*self.kN:(n+1)*self.kN]
        T,K=np.meshgrid(self.time,self.k)
        plt.contourf(T/41,K,self.pkn,cmap=cm.jet,levels=2**7);
        plt.ylabel('k')
        plt.xlabel('Time(fs)')
        plt.xlim(rangex[0],rangex[1])
        plt.colorbar()
        plt.show()        
    
    def plotemission(self,rangex):
        plt.figure(3)
        plt.subplot(2,2,1)
        plt.plot(self.sol.t/41,self.Jt)
        plt.title("current")
        plt.xlabel("Time(fs)")
        plt.subplot(2,2,2)
        plt.plot(self.sol.t/41,self.Pt)
        plt.title("polarization")
        plt.xlabel("Time(fs)")
        plt.subplot(2,2,3)
        plt.semilogy(self.energy,self.SwJ)
        plt.xlim(rangex[0],rangex[1])
        plt.subplot(2,2,4)
        plt.semilogy(self.energy,self.SwP)
        plt.xlim(rangex[0],rangex[1])
        plt.legend()
        plt.show()
        
    def plotabsorption(self,rangex,rangey):
        plt.figure(4)
        plt.plot(self.freqs*2*pi*27.21,self.absorption)
        plt.xlim(rangex[0],rangex[1])
        plt.ylim(rangey[0],rangey[1])
        plt.show()

pi=scipy.constants.pi


