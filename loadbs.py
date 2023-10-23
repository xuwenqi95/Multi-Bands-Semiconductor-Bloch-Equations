# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:11:07 2023

@author: Wenqi
"""

import numpy as np
import matplotlib.pyplot  as plt
from scipy import interpolate
VB2=np.loadtxt('jicai_project//mgf2//VB2.txt')
VB1=np.loadtxt('jicai_project//mgf2//VB1.txt')
CB2=np.loadtxt('jicai_project//mgf2//CB2.txt')
CB1=np.loadtxt('jicai_project//mgf2//CB1.txt')
CB3=np.loadtxt('jicai_project//mgf2//CB3.txt')

a=np.where(VB1[:,0]==0.677194017 )
b=np.where(VB1[:,0]==0.5434834008 )


VB2=VB2[150:201,:]
VB1=VB1[150:201,:]
CB2=CB2[150:201,:]
CB1=CB1[150:201,:]

k=np.linspace(VB2[0,0], VB2[-1,0],101)
f=interpolate.interp1d(VB2[:,0],VB2[:,1]) 
VB2=f(k)
f=interpolate.interp1d(VB1[:,0],VB1[:,1]) 
VB1=f(k)
f=interpolate.interp1d(CB2[:,0],CB2[:,1]) 
CB2=f(k)
f=interpolate.interp1d(CB1[:,0],CB1[:,1]) 
CB1=f(k)
f=interpolate.interp1d(CB3[:,0],CB3[:,1]) 
CB3=f(k)



np.savetxt('jicai_project//mgf2//gmVB2.txt', VB2)
np.savetxt('jicai_project//mgf2//gmVB1.txt', VB1)
np.savetxt('jicai_project//mgf2//gmCB2.txt', CB2)
np.savetxt('jicai_project//mgf2//gmCB1.txt', CB1)
np.savetxt('jicai_project//mgf2//gmCB3.txt', CB3)




Ek=np.zeros((101,5))

Ek[:,0]=VB1
Ek[:,1]=VB2
Ek[:,2]=CB1
Ek[:,3]=CB2
Ek[:,4]=CB3
Ek=Ek.T

np.savetxt('jicai_project//mgf2//mgf2bs.txt', Ek)