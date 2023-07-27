# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 23:44:12 2023

@author: QiNova
"""


import numpy as np
import torch 



def finiteSBE(t,y,p): 
    # bandN,kN,T1,dk=p1                                              #load parameter1
    # time,Efield,dipole2,ddt1,MP,Vkq,k,hindex,eindex=p2             #load parameter2

    bandN,kN,T1,dk,time,Efield,dipole2,ddt1,MP,Vkq,k,hindex,eindex=p
    
    dipole=torch.zeros(dipole2.shape,dtype=torch.complex128,device='cuda')
    ddt=torch.zeros(ddt1.shape,dtype=torch.complex128,device='cuda')
    
    
    # dipole=np.zeros(dipole2.shape,dtype=np.complex128)             #new rabi energy
    # ddt=np.zeros(ddt1.shape,dtype=np.complex128)                   #new ddt
    y=y.reshape((bandN**2+bandN,kN))                               #reshape the 1D array to a matrix
    pk=y[0:bandN**2]                                               #extract pk
    fk=y[bandN**2:]
    dpk=torch.zeros(pk.shape,dtype=torch.complex128,device='cuda')
    dfk=torch.zeros(fk.shape,dtype=torch.complex128,device='cuda')
    dpkdk=(torch.roll(pk,-1,1)-torch.roll(pk,-1,1))/2/dk
    dfkdk=(torch.roll(fk,-1,1)-torch.roll(fk,-1,1))/2/dk
    


                                                #extract fk
    # dpk=np.zeros(pk.shape,dtype=np.complex128)                     #initialize dpk
    # dfk=np.zeros(fk.shape,dtype=np.complex128)                     #initialize dfk
    # pkleft=np.zeros(pk.shape,dtype=np.complex128)
    # pkright=np.zeros(pk.shape,dtype=np.complex128)
    # fkleft=np.zeros(fk.shape,dtype=np.complex128)
    # fkright=np.zeros(fk.shape,dtype=np.complex128)
    #set Efield
    Et=np.interp(t.cpu(),time,Efield)

    
    #dpk/dk term
    # for i in range(bandN**2):
    #     pkleft[i]=np.roll(pk[i],-1)
    #     pkright[i]=np.roll(pk[i],1)
    # dpkdk=(pkleft-pkright)/2/dk
    # #dfk/dk term
    # for i in range(bandN):
    #     fkleft[i]=np.roll(fk[i],-1)
    #     fkright[i]=np.roll(fk[i],1)
    # dfkdk=(fkleft-fkright)/2/dk

    
    #renormalize rabi energy
    # for i in range(bandN**2):
    #     dipole[i]=(dipole2[i]*Et+torch.mv(Vkq,pk[i]))
    #     #dipole[i]=1/Et*(dipole1[i]*Et)
    dipole=(dipole2*Et+torch.mm(pk,Vkq))

    #renormalize ddt
    for p in range(bandN):
        for q in range(bandN):
            if p!=q:
                if p in hindex and q in eindex:
                    ddt[MP[p,q]]=ddt1[MP[p,q]]-torch.mv(Vkq,(fk[p]+fk[q]))
                else:
                    ddt[MP[p,q]]=ddt1[MP[p,q]]-torch.mv(Vkq,(fk[p]-fk[q]))
                #ddt[MP[p,q]]=ddt1[MP[p,q]]
         
                
    #pkhe term
    for hi in hindex:
        for ej in eindex:
            Ak=torch.ones(k.shape,dtype=torch.complex128,device='cuda')
            for elamda in np.delete(eindex,np.where(eindex==ej)[0]):
                Ak+=dipole[MP[elamda,hi]]*pk[MP[elamda,ej]]-dipole[MP[ej,elamda]]*pk[MP[hi,elamda]]
            for hlamda in np.delete(hindex,np.where(hindex==hi)[0]):
                Ak+=dipole[MP[hlamda,hi]]*pk[MP[hlamda,ej]]-dipole[MP[ej,hlamda]]*pk[MP[hi,hlamda]]
            dpk[MP[hi,ej]]=-1j*(ddt[MP[hi,ej]]*pk[MP[hi,ej]]-dipole[MP[ej,hi]]*(1-fk[ej]-fk[hi])+1j*Et*dpkdk[MP[hi,ej]]+Ak)
            dpk[MP[ej,hi]]=torch.conj(dpk[MP[hi,ej]])

    #pkee term
    for ei in eindex:
        for ej in np.delete(eindex,np.where(eindex<=ei)[0]):
            Ak=torch.ones(k.shape,dtype=torch.complex128,device='cuda')
            for elamda in np.delete(eindex,np.where(eindex==ej)[0]):
                Ak+=dipole[MP[elamda,ei]]*pk[MP[elamda,ej]]
            for elamda in np.delete(eindex,np.where(eindex==ei)[0]):
                Ak-=dipole[MP[ej,elamda]]*pk[MP[ei,elamda]]
            for hlamda in hindex:
                Ak+=dipole[MP[hlamda,ei]]*pk[MP[hlamda,ej]]-dipole[MP[ej,hlamda]]*(pk[MP[ei,hlamda]])
            dpk[MP[ei,ej]]=-1j*(ddt[MP[ei,ej]]*pk[MP[ei,ej]]+dipole[MP[ej,ei]]*(fk[ej]-fk[ei])+1j*Et*dpkdk[MP[ei,ej]]+Ak)  
            dpk[MP[ej,ei]]=torch.conj(dpk[MP[ei,ej]])

      #pkhh term           
    for hi in hindex:
        for hj in np.delete(hindex,np.where(hindex<=hi)[0]):
            Ak=torch.ones(k.shape,dtype=torch.complex128,device='cuda')
            for hlamda in np.delete(hindex,np.where(hindex==hj)[0]):
                Ak+=dipole[MP[hlamda,hi]]*pk[MP[hlamda,hj]]
            for hlamda in np.delete(hindex,np.where(hindex==hi)[0]):
                Ak-=dipole[MP[hj,hlamda]]*pk[MP[hi,hlamda]]
            for elamda in eindex:
                Ak+=dipole[MP[elamda,hi]]*(pk[MP[elamda,hj]])-dipole[MP[hj,elamda]]*pk[MP[hi,elamda]]
            dpk[MP[hi,hj]]=-1j*(ddt[MP[hi,hj]]*pk[MP[hi,hj]]+dipole[MP[hj,hi]]*(fk[hi]-fk[hj])+1j*Et*dpkdk[MP[hi,hj]]+Ak)
            dpk[MP[hj,hi]]=torch.conj(dpk[MP[hi,hj]])
            
    #fke term
    for ei in eindex:
        Ak=torch.ones(k.shape,dtype=torch.complex128,device='cuda')
        for elamda in np.delete(eindex,np.where(eindex==ei)[0]):
            Ak+=dipole[MP[ei,elamda]]*(pk[MP[ei,elamda]])
        for hlamda in hindex:
            Ak+=dipole[MP[ei,hlamda]]*(pk[MP[ei,hlamda]])
        #dfk[ei]=-2*np.imag(Ak*Et)+Et*dfkdk[ei]-0.5/T1*(fk[ei]-np.flip(fk[ei]))
        dfk[ei]=-2*torch.imag(Ak)+Et*dfkdk[ei]-fk[ei]/T1

            
    #fkh term
    for hi in hindex:
        Ak=torch.ones(k.shape,dtype=torch.complex128,device='cuda')
        for hlamda in np.delete(hindex,np.where(hindex==hi)[0]):
            Ak+=dipole[MP[hlamda,hi]]*(pk[MP[hlamda,hi]])
        for elamda in eindex:
            Ak+=dipole[MP[elamda,hi]]*(pk[MP[elamda,hi]])

        # dfk[hi]=-2*np.imag(Ak*Et)+Et*dfkdk[hi]-0.5/T1*(fk[hi]-np.flip(fk[hi]))
        dfk[hi]=-2*torch.imag(Ak)+Et*dfkdk[hi]-fk[hi]/T1

    
    dy=torch.vstack((dpk,dfk))                         #combine the dpk and dfk matrix
    dy=dy.reshape((bandN**2+bandN)*kN)              #reshape it to a 1D array
    return dy
