from multiprocessing import Pool
import numpy as np
import matplotlib.pylab as plt
from time import time
import random
import sys
from scipy.integrate import odeint, solve_ivp, quad
from scipy.optimize import minimize, minimize_scalar
# from inputpulse import *


class General():
    def __init__(self, xi: callable, Gamma,**kwargs):
        # Here xi is a function. It could be any function but it MUST satisfy the normalization condtion.
        self.Omega = kwargs.get('Omega',1)
        self.Omega_2 = kwargs.get('Omega_2',1)
        self.Mu = kwargs.get('Mu',0)
        self.xi = xi
        self.xi_2 = kwargs.get('wave2',self.xi)
        # if self.xi_2 == None:
        #     self.x_2 = self.xi
        if np.abs(quad(lambda x: abs(self.xi(x))**2, -np.inf, np.inf)[0]-1) > 1e-6:
            raise ValueError("Invalid 'xi' function provided. Check the 'Normalization' constant.")

        
        # Gamma can not be negetive. If there is any Delta (loss), the Gamma will be a complex number.
        self.Gamma = Gamma
        if self.Gamma <= 0:
            raise ValueError("Gamma must be 'positive'.")
        self.Delta = kwargs.get('Delta',0)
        if self.Delta != 0:
            self.Gamma = complex(self.Gamma, 2*self.Delta )

        # In some cases, user may specify the upper and lower range. If not, it would be infinity.
        self.tRange = kwargs.get('tRange',[-np.infty,+np.infty])
        self.lower_limit = self.tRange[0]
        self.upper_limit = self.tRange[1]
    def MeanTime(self):
        t, p = self.P()
        return np.sum(p[1:]*np.diff(t))

    def optimize_single(self, params, parameters_to_optimize, parameter_bounds):
        def objective (params):
            param_values = {param_name: param_value for param_name, param_value in zip(parameters_to_optimize, params)}
            for param_name, param_value in param_values.items():
                setattr(self, param_name, param_value)
            _, p = self.P()
            return -np.max(p) 
        result = minimize(objective, params, method='Nelder-Mead', bounds=parameter_bounds)
        return -result.fun, result.x if result.success else None
    def optimize(self,parameters_to_optimize, **kwargs):
        num_attempts = kwargs.get('num_attempts',1)
        num_processor = kwargs.get('num_processor',4)
        np.random.seed()
        # initial_guess = initial_guess = [np.random.uniform(0 ,5) if param == 'Mu' else
        #                                  np.random.uniform(0.4 , 7) for param in parameters_to_optimize]
        parameter_bounds = [(0, 4) if param == 'Mu'  else (0.4, 7.0) for param in parameters_to_optimize]
        initial_guesses = [[np.random.uniform(lower, upper) for lower, upper in parameter_bounds] for _ in range(num_attempts)]
        with Pool(processes=num_processor) as pool:
            results = pool.starmap(self.optimize_single, [(initial_guess, parameters_to_optimize,  parameter_bounds) for initial_guess in initial_guesses])
        best_result = max(results, key=lambda x: x[0])
        if best_result[1] is not None:
            optimized_P = best_result[0]
            optimized_params = {param_name: param_value for param_name, param_value in zip(parameters_to_optimize, best_result[1])}
            return optimized_P, optimized_params
        else:
            return None

class SinglePhoton (General):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
        if self.upper_limit == np.infty :
            self.upper_limit = 10/ min(self.Omega,self.Gamma) + self.Mu
        if self.lower_limit == -np.infty :
            self.lower_limit = -7/ min(self.Omega,self.Gamma) + self.Mu
        self.nBins = kwargs.get('nBins',1000)
        self.method = kwargs.get('method','vectorize')
        if self.method == 'vectorize':
            self.P = self.P_vect
        elif self.method == 'analytical':
            self.P = self.P_analy
        elif self.method == 'quad':
            self.P = self.P_quad
        else:
            raise ValueError(f'There is not defined such method. "{self.method} was not found!"')
    def P_analy (self, shape: str):
        tSteps,dt = np.linspace( self.lower_limit,self.upper_limit,self.nBins ,retstep=True)
        O = self.Omega
        g = self.Gamma/2
        G = self.Gamma.real
        if shape == 'Rectangular' :
            t1 = tSteps[(0 <= tSteps)&(tSteps <=2/O)]
            t2 = tSteps[tSteps>2/O]
            p_t1 = G*O* np.exp(-G*t1)*np.abs(np.exp(g*t1) - 1)**2 /(2*np.abs(g)**2)
            p_t2 = G*O* np.exp(-G*t2)*np.abs(np.exp(2*g/O) - 1)**2 /(2*np.abs(g)**2)
            p = np.append(p_t1,p_t2 )
            t = np.append(t1,t2)
            return t, p
        elif shape == 'Exponential_decay':
            tSteps = tSteps[tSteps>0]
            if G == O :
                p = G*O*np.exp(-O*tSteps)*tSteps**2
            else:
                p = (4*G*O*np.exp(-G*tSteps)*(np.exp((G-O)*tSteps/2) - 1)**2)/(G-O)**2
            return tSteps, p
        elif shape == 'Exponential_raising':
            t1 = tSteps[tSteps<=0]
            t2 = tSteps[tSteps>0]
            p_t1 = 4*G*O* np.exp(O*t1)/(np.abs(G+O)**2)
            p_t2 = 4*G*O* np.exp(-G*t2)/(np.abs(G+O)**2)
            p = np.append(p_t1,p_t2 )
            ttSteps = np.append(t1,t2)
            return ttSteps, p
        else :
            raise ValueError(f'Invalid wave shape. "{shape}" has no analytical solution!')
    def P_quad (self):
        tSteps = np.linspace( self.lower_limit,self.upper_limit,self.nBins )
        f = lambda t: self.xi(t,Omega = self.Omega,Mu = self.Mu)*np.exp(self.Gamma*t/2)            
        p = [quad(f,-np.infty , t)[0] for t in tSteps]
        p = self.Gamma*np.exp(-self.Gamma*tSteps)*np.abs(p)**2
        return tSteps, p
    def P_vect(self):
        tSteps,dt = np.linspace( self.lower_limit,self.upper_limit,self.nBins ,retstep=True)
        p = np.cumsum(self.xi(tSteps,Omega = self.Omega,Mu = self.Mu)*np.exp(self.Gamma*tSteps/2))*dt
        p = self.Gamma*np.exp(-self.Gamma*tSteps)*np.abs(p)**2
        return tSteps, p

from TwoLevelAtom.general import General     
import numpy as np


class TwoPhoton (General):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
        if self.upper_limit == np.infty :
            self.upper_limit = 10/ min(self.Omega, self.Omega_2, self.Gamma) + self.Mu
        if self.lower_limit == -np.infty :
            self.lower_limit = -7/ min(self.Omega, self.Omega_2, self.Gamma) + self.Mu
        self.nBins = kwargs.get('nBins',1000)
        self.status = kwargs.get('status','indistinguishable')
        if self.Mu != 0:
            self.status = 'distinguishable: same direction'

        if self.status == 'indistinguishable':
            self.P = self.P_indistinguishable
        elif self.status == 'distinguishable: same direction':
            self.P = self.P_dist_unidirection
        elif self.status == 'distinguishable: oppsite direction':
            self.P = self.P_dist_oppositeDirection
        else:
            raise ValueError(f'Currently you can choose either "same" for unidirection and "opposite" for opposite direction.')
    def P_indistinguishable(self):
        t, dt = np.linspace(self.lower_limit,self.upper_limit , self.nBins, retstep=True)
        indx = np.arange(len(t))
        wave = self.xi(t,Omega = self.Omega)
        wave_ij = np.repeat(wave[np.newaxis,:], len(wave), axis = 0)
        ones = np.ones_like(wave_ij)
        F = wave_ij*np.exp(self.Gamma*t/2)
        abs = np.abs( self.Gamma* np.exp(-self.Gamma*t[:,np.newaxis]/2) *np.cumsum(F*np.triu(ones),axis = 1) *dt \
                     *np.cumsum(F* np.tril(ones),axis = 1)*dt - np.cumsum(F,axis = 1)*(wave[:,np.newaxis])*dt )**2

        abs = np.cumsum(abs,axis = 0)*dt
        p1 = np.abs(np.cumsum(F,axis = 1)*dt)**2 * (1-np.cumsum(np.abs(wave_ij)**2, axis = 1)*dt )
        P = 2*self.Gamma*np.exp(-self.Gamma*t)*(abs[indx,indx] + p1[indx,indx])
        return t, P
    def P_dist_unidirection (self):
        t, dt = np.linspace(self.lower_limit,self.upper_limit , self.nBins, retstep=True)
        xi = self.xi(t,Omega = self.Omega,Mu = 0)
        phi = self.xi_2(t,Omega = self.Omega_2, Mu = self.Mu)
        indx = np.arange(len(t))
        xi_ij = np.repeat(xi[np.newaxis,:], len(xi), axis = 0)
        phi_ij = np.repeat(phi[np.newaxis,:], len(phi), axis = 0)
        ones = np.ones_like(xi_ij)
        N = 1 + np.sum(np.abs(np.conjugate(phi)*xi)*dt)**2
        
        P = self.Gamma*(1/N)*np.exp(-self.Gamma*t)*( \
            (np.abs(np.cumsum(xi_ij*np.exp(self.Gamma*t/2),axis=1)*dt)**2 * (1 - np.cumsum(np.abs(phi_ij)**2,axis=1)*dt ) \
             +np.abs(np.cumsum(phi_ij*np.exp(self.Gamma*t/2),axis=1)*dt)**2 * (1 - np.cumsum(np.abs(xi_ij)**2,axis=1)*dt ))[indx,indx] \
            + (np.cumsum(np.conjugate(phi_ij)*np.exp(self.Gamma*t/2) , axis=1)*dt \
                 *np.cumsum( xi_ij*np.exp(self.Gamma*t/2), axis=1)*dt\
                 *np.cumsum(np.conjugate(xi_ij[:,::-1])*phi_ij[:,::-1] , axis=1)[:,::-1]*dt \
                 + np.cumsum(np.conjugate(xi_ij)*np.exp(self.Gamma*t/2) , axis=1)*dt \
                 *np.cumsum( phi_ij*np.exp(self.Gamma*t/2), axis=1)*dt \
                 *np.cumsum(np.conjugate(phi_ij[:,::-1])*xi_ij[:,::-1] , axis=1)[:,::-1]*dt)[indx,indx] \
            + (np.cumsum( (np.abs(self.Gamma* np.exp(-self.Gamma*t[:,np.newaxis]/2) \
                    *np.cumsum(xi_ij*np.exp(self.Gamma*t/2)*np.triu(ones),axis = 1) *dt \
                     *np.cumsum(phi_ij*np.exp(self.Gamma*t/2)* np.tril(ones),axis = 1)*dt \
                    - np.cumsum(phi_ij*np.exp(self.Gamma*t/2),axis = 1)*(xi[:,np.newaxis])*dt \
                     + self.Gamma* np.exp(-self.Gamma*t[:,np.newaxis]/2) *np.cumsum(phi_ij*np.exp(self.Gamma*t/2)*np.triu(ones),axis = 1) *dt \
                     *np.cumsum(xi_ij*np.exp(self.Gamma*t/2)* np.tril(ones),axis = 1)*dt \
                    - np.cumsum(xi_ij*np.exp(self.Gamma*t/2),axis = 1)*(phi[:,np.newaxis])*dt)**2) , axis = 0)*dt) [indx,indx]  )
        return t, P
        
    def P_dist_oppositeDirection(self):
        t, dt = np.linspace(self.lower_limit,self.upper_limit , self.nBins, retstep=True)
        xi = self.xi(t,Omega = self.Omega,Mu = 0)
        phi = self.xi_2(t,Omega = self.Omega_2, Mu = self.Mu)
        indx = np.arange(len(t))
        xi_ij = np.repeat(xi[np.newaxis,:], len(xi), axis = 0)
        phi_ij = np.repeat(phi[np.newaxis,:], len(phi), axis = 0)
        ones = np.ones_like(xi_ij)
        
        P = self.Gamma*np.exp(-2*self.Gamma*t)*( \
            (np.abs(np.cumsum(xi_ij*np.exp(self.Gamma*t),axis=1)*dt)**2 * (1 - np.cumsum(np.abs(phi_ij)**2,axis=1)*dt )\
             +np.abs(np.cumsum(phi_ij*np.exp(self.Gamma*t),axis=1)*dt)**2 * (1 - np.cumsum(np.abs(xi_ij)**2,axis=1)*dt ))[indx,indx] \
            + (np.cumsum( (np.abs(self.Gamma* np.exp(-self.Gamma*t[:,np.newaxis]) \
                              *np.cumsum(xi_ij*np.exp(self.Gamma*t)*np.triu(ones),axis = 1) *dt \
                             *np.cumsum(phi_ij*np.exp(self.Gamma*t)* np.tril(ones),axis = 1)*dt \
                              -np.cumsum(xi_ij*np.exp(self.Gamma*t),axis = 1)*(phi[:,np.newaxis])*dt \
                  + self.Gamma* np.exp(-self.Gamma*t[:,np.newaxis]) *np.cumsum(phi_ij*np.exp(self.Gamma*t)*np.triu(ones),axis = 1) *dt \
                     *np.cumsum(xi_ij*np.exp(self.Gamma*t)* np.tril(ones),axis = 1)*dt )**2),axis=0)*dt) [indx,indx] \
            + (np.cumsum( (np.abs(self.Gamma* np.exp(-self.Gamma*t[:,np.newaxis]) \
                              *np.cumsum(xi_ij*np.exp(self.Gamma*t)*np.triu(ones),axis = 1) *dt \
                             *np.cumsum(phi_ij*np.exp(self.Gamma*t)* np.tril(ones),axis = 1)*dt \
                            - np.cumsum(phi_ij*np.exp(self.Gamma*t),axis = 1)*(xi[:,np.newaxis])*dt \
                     + self.Gamma* np.exp(-self.Gamma*t[:,np.newaxis]) *np.cumsum(phi_ij*np.exp(self.Gamma*t)*np.triu(ones),axis = 1) *dt \
                     *np.cumsum(xi_ij*np.exp(self.Gamma*t)* np.tril(ones),axis = 1)*dt )**2),axis=0)*dt) [indx,indx]  )
        return t, P
