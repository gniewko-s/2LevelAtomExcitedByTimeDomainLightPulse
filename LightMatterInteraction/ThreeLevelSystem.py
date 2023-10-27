from multiprocessing import Pool
import numpy as np
import matplotlib.pylab as plt
from time import time
import random
import sys
from scipy.integrate import odeint, solve_ivp, quad
from scipy.optimize import minimize, minimize_scalar

class General():
    def __init__(self, xi: callable, Gamma_1, Gamma_2,**kwargs):
        # Here xi is a function. It could be any function but it MUST satisfy the normalization condtion.
        self.Omega_1 = kwargs.get('Omega_1',1)
        self.Omega_2 = kwargs.get('Omega_2',1)
        self.Mu = kwargs.get('Mu',0)
        self.xi = xi
        self.phi = kwargs.get('phi',self.xi)
        if np.abs(quad(lambda x: abs(self.xi(x))**2, -np.inf, np.inf)[0]-1) > 1e-6:
            raise ValueError("Invalid 'xi' function provided. Check the 'Normalization' constant.")

        
        # Gamma can not be negetive. If there is any Delta (loss), the Gamma will be a complex number.
        self.Gamma_1 = Gamma_1
        self.Gamma_2 = Gamma_2
        if min(self.Gamma_1,self.Gamma_2) <= 0 :
            raise ValueError("Gamma must be 'positive'.")
        self.Delta = kwargs.get('Delta',0)
        # if self.Delta != 0:
        #     self.Gamma = complex(self.Gamma, 2*self.Delta )

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

class TwoPhoton (General):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.upper_limit == np.infty :
            self.upper_limit = 3/ min(self.Omega_1, self.Omega_2, self.Gamma_1,self.Gamma_2) + self.Mu
        if self.lower_limit == -np.infty :
            self.lower_limit = -1e6
            while True:
                if max(np.abs(self.xi(self.lower_limit , Omega = self.Omega_1)) \
                       ,np.abs(self.phi(self.lower_limit , Omega = self.Omega_2, Mu = self.Mu)) ) > 1e-3:
                    break
                else : 
                    self.lower_limit = self.lower_limit/2 + self.lower_limit*0.01
            # self.lower_limit = -10/ min(self.Omega_1, self.Omega_2, self.Gamma_1,self.Gamma_2) + self.Mu
        self.nBins = kwargs.get('nBins',1000)
        self.status = kwargs.get('status','vectorized')
        if self.status == 'vectorized':
            self.P = self.P_vect
        elif self.status == 'Analytical':
            self.P = self.P_anal

    def P_vect(self):
        t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
        xi = self.xi(t,Omega = self.Omega_1,Mu = 0)
        phi = self.phi(t,Omega = self.Omega_2, Mu = self.Mu)
        ones = np.ones([len(xi),len(xi)])

        P = self.Gamma_1*self.Gamma_2*np.exp(-self.Gamma_2*t[:,np.newaxis])\
        * np.abs(np.cumsum(phi[:,np.newaxis]*np.exp((self.Gamma_2-self.Gamma_1)*t[:,np.newaxis]/2) \
                           * np.cumsum(xi[np.newaxis,:]*np.exp(self.Gamma_1*t/2)* np.tril(ones),axis = 1)*dt ,axis=0)*dt)**2 
        P = np.diag(P)
        return t, P
    def P_anal (self,pulseShape='Exponential_raising'):
        # Analitical solution for exponential raising in the case Mu = 0
        if pulseShape == 'Exponential_raising':
            t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
            t1 = t[t<=0]
            t2 = t[t>0]
            coef =  (16*self.Omega_1*self.Omega_2*self.Gamma_1*self.Gamma_2)/((self.Gamma_1+self.Omega_1)**2 *(self.Gamma_2+self.Omega_1+self.Omega_2)**2 )
            P_t1 = coef*np.exp((self.Omega_1+self.Omega_2)*t1)
            P_t2 = coef* np.exp(-self.Gamma_2*t2)
            P = np.append(P_t1,P_t2 )
            return t, P