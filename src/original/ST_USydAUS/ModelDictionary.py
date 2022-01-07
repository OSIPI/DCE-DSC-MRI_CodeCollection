#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:03:20 2018

@author: Sirisha Tadimalla
"""

# Model definitions:
# ve: extracellular volume fraction
# vp: plasma volume fraction
# ca: input function
# conc: measured concentration in voxel or ROI

#Import libraries
import src.original.ST_USydAUS.Tools as tools

######################################
# conc = vp x ca + ktrans x exp(-t(ktrans/ve))*ca
def ExtendedTofts(X, vp, ve, ktrans):
    t = X[:,0]
    ca = X[:,1]

    Tc = ve/ktrans
    
    # expconv calculates convolution of ca and (1/Tc)exp(-t/Tc)
    conc = vp*ca + ve*tools.expconv(Tc, t, ca)
    return(conc)

######################################
# conc = ktrans x exp(-t(ktrans/ve))*ca
def Tofts(X, ve, ktrans):
    t = X[:,0]
    ca = X[:,1]

    Tc = ve/ktrans
    
    # expconv calculates convolution of ca and (1/Tc)exp(-t/Tc)
    conc = ve*tools.expconv(Tc, t, ca)
    return(conc)

###################################### 
# conc = Fp x exp(-t(Fp/vp))*ca 
def OneCompartment(X, vp, Fp):
    t = X[:,0]
    ca = X[:,1]
    
    Tc = vp/Fp

    # expconv calculates convolution of ca and (1/Tc)exp(-t/Tc)
    conc = vp*tools.expconv(Tc,t,ca)
    return(conc)

######################################   
def PatlakModel(X,vp,ki):
    t = X[:,0]
    ca = X[:,1]
    
    conc = ki*tools.integrate(ca,t) + vp*ca
    
    return(conc)

######################################
def conc_HF2CGM(X, ve, kce, kbc):
    t = X[:,0]
    ca = X[:,1]
    
    Tc = (1-ve)/kbc
    
    # expconv calculates convolution of ca and (1/Tc)exp(-t/Tc)
    conc = ve*ca + kce*Tc*tools.expconv(Tc, t, ca)
    return(conc)
    
######################################
def DualInletExtendedTofts(X, fa, fv, vp, ve, ktrans):
    # If vif is not given, fv = 0 and fa = 1, and this model reverts to the
    # single inlet Ext. Tofts model above.
    t = X[:,0]
    ca = X[:,1]
    cv = X[:,2]

    Tc = ve/ktrans
    
    # expconv calculates convolution of input function and (1/Tc)exp(-t/Tc)
    c_if = fa*ca + fv*cv
    conc = vp*c_if + ve*tools.expconv(Tc, t, c_if)
    return(conc)
    
###################################### 
def DualInletOneCompartment(X, fa, fv, vp, Fp):
    # If vif is not given, fv = 0 and fa = 1, and this model reverts to the
    # single inlet One Compartment model above.
    t = X[:,0]
    ca = X[:,1]
    cv = X[:,2]
    
    Tc = vp/Fp

    # expconv calculates convolution of input function and (1/Tc)exp(-t/Tc)
    c_if = fa*ca + fv*cv
    conc = vp*tools.expconv(Tc,t,c_if)
    return(conc)
    
######################################
def DualInletconc_HF2CGM(X, fa, fv, ve, kce, kbc):
    # If vif is not given, fv = 0 and fa = 1, and this model reverts to the
    # single inlet High Flow 2-Compartment Gadoxetate model above.
    t = X[:,0]
    ca = X[:,1]
    cv = X[:,2]
    
    Tc = (1-ve)/kbc
    
    # expconv calculates convolution of ca and (1/Tc)exp(-t/Tc)
    c_if = fa*ca + fv*cv
    conc = ve*c_if + kce*Tc*tools.expconv(Tc, t, c_if)
    return(conc)