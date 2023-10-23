# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:00:16 2023

@author: Wenqi
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
from numpy.fft import fft, fftfreq,fftshift,ifft
from matplotlib import cm
import os


def normal(data):#normalize 

    datarange=np.max(data)-np.min(data)
    return (data-np.min(data))/datarange*10







folder_path = 'jicai_project/mgf2/2bands/scan1'
for file_name in os.listdir(folder_path):
    print(file_name)
    if file_name=='figure' or file_name=='pump':
        continue
    scandata=np.load('jicai_project/mgf2/2bands/scan1/'+file_name) #load npy file

    pi=spc.pi #pi
    d=9.32  #lattice constant
    t=np.linspace(0,41*400,10000) #t axis
    k=np.linspace(-pi/d, pi/d,201)  #k axis
    energy =fftshift(fftfreq(t.size, t[1] - t[0]))*2*pi*27.21 #energy axis
    
    
    
    scandata=(scandata).T
    
    #plot each order of harmonics individually
    fig=plt.figure()


    plt.subplots_adjust(hspace=0.5)
    plt.subplot(3,2,1)

    data=normal(scandata[5398:5494]) #normalize for a range
    
    T,K=np.meshgrid(np.linspace(-80,80,41)*41,energy[5398:5494])#make grid
    plt.contourf(T/41,K,data,cmap=cm.jet,levels=np.linspace(0,10,2**7))#plot 2D figure
    

    plt.subplot(3,2,2)

    data=normal(scandata[5547:5643])
    
    T,K=np.meshgrid(np.linspace(-80,80,41)*41,energy[5547:5643])
    plt.contourf(T/41,K,data,cmap=cm.jet,levels=np.linspace(0,10,2**7))
        

    plt.subplot(3,2,3)

    data=normal(scandata[5696:5791])
    
    T,K=np.meshgrid(np.linspace(-80,80,41)*41,energy[5696:5791])
    plt.contourf(T/41,K,data,cmap=cm.jet,levels=np.linspace(0,10,2**7))
    
    plt.subplot(3,2,4)
    

    data=normal(scandata[5844:5940])
    
    T,K=np.meshgrid(np.linspace(-80,80,41)*41,energy[5844:5940])
    im=plt.contourf(T/41,K,data,cmap=cm.jet,levels=np.linspace(0,10,2**7))
    
    plt.subplot(3,2,5)
    

    data=normal(scandata[5993:6089])
    
    T,K=np.meshgrid(np.linspace(-80,80,41)*41,energy[5993:6089])
    im=plt.contourf(T/41,K,data,cmap=cm.jet,levels=np.linspace(0,10,2**7))   
    
    plt.subplot(3,2,6)
    

    data=normal(scandata[6142:6238])
    
    T,K=np.meshgrid(np.linspace(-80,80,41)*41,energy[6142:6238])
    im=plt.contourf(T/41,K,data,cmap=cm.jet,levels=np.linspace(0,10,2**7))
    
    
    

    fig.text(0, 0.5, 'Energy(eV)', va='center', rotation='vertical')
    fig.text(0.45,0, 'Delay (fs)', va='center',)

    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    #save the figure
    plt.savefig('jicai_project\\mgf2\\2bands\\scan1\\figure\\'+file_name[0:-4]+'.jpg',dpi=300,bbox_inches='tight')
    plt.show()
    