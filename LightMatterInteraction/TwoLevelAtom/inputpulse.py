import numpy as np

def Rectangular(t,**kwargs):
    Omega = kwargs.get('Omega',1)
    Mu = kwargs.get('Mu',0)
    t = np.array(t)
    return np.where((( t < (2/Omega + Mu) ) & ( t >= (0 + Mu))), np.sqrt(Omega/2) , 0)
def ExpoDecay(t,**kwargs):
    Omega = kwargs.get('Omega',1)
    Mu = kwargs.get('Mu',0)
    t = np.array(t)
    return np.where( t >= Mu ,np.sqrt(Omega)*np.exp(-Omega*(t-Mu)/2) , 0)
def ExpoRaising(t,**kwargs):
    Omega = kwargs.get('Omega',1)
    Mu = kwargs.get('Mu',0)
    t = np.array(t)
    return np.where( t < Mu ,np.sqrt(Omega)*np.exp(Omega*(t-Mu)/2) , 0)
def ExpoSym(t,**kwargs):
    Omega = kwargs.get('Omega',1)
    Mu = kwargs.get('Mu',0)
    t = np.array(t)
    return np.sqrt(Omega)*np.exp(-Omega*np.abs(t-Mu))
def Gaussian(t,**kwargs):
    Omega = kwargs.get('Omega',1)
    Mu = kwargs.get('Mu',0)
    t = np.array(t)
    return (Omega**2/(2*np.pi))**(1/4) * np.exp(-Omega**2*((t-Mu)**2)/4)
def HyperSec(t,**kwargs):
    Omega = kwargs.get('Omega',1)
    Mu = kwargs.get('Mu',0)
    t = np.array(t)
    return np.sqrt(Omega/2)/np.cosh(Omega*t - Mu)
