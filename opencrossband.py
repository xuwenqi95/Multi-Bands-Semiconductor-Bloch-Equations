# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:36:27 2022

@author: Wenqi
"""


import numpy as np

def opencrossband(band): 
    bandN=np.linspace(0,len(band)-1,len(band),dtype=int)
    a=1
    for i in bandN:
        if a==0:
            break
        for j in np.delete(bandN,np.where(bandN<=i)):
                if np.all(band[i]<band[j]):
                    pass
                else:
                    a=0
                    break
    if a==1:
        return band
    else:
        newband=np.sort(band,axis=0)
        newband[0]-=0.002
        newband[1]+=0.002
    return newband