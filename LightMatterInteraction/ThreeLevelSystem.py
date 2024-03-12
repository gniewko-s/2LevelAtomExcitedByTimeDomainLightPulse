from multiprocessing import Pool
import numpy as np
import matplotlib.pylab as plt
from time import time
import random
import sys
from scipy.integrate import odeint, solve_ivp, quad
from scipy.optimize import minimize, minimize_scalar
from numba import jit

class General():
    def __init__(self, xi: callable, Gamma_1, Gamma_2, **kwargs):
        """
        Initializes a General class, which is similar to and shared among other classes.

        Parameters:
        xi (callable): The first wave profile.
        Gamma_1 (float): The value of '\Gamma_1' (or '\Gamma_e'). 
        Gamma_2 (float): The value of '\Gamma_2' (or '\Gamma_f').
        **kwargs: Additional keyword arguments.
            phi (callable, optional): The second wave profile. Default is xi.
            Delta_1 (float, optional): The first dispersion value '\Delta_1'. Default is 0.
            Delta_2 (float, optional): The second dispersion value '\Delta_2'. Default is 0.
            tRange (list, optional): The time range in which P should be computed. Default is [-np.infty, +np.infty].
            
        """

        self.Omega_1 = kwargs.get('Omega_1', 1)
        self.Omega_2 = kwargs.get('Omega_2', 1)
        self.Mu = kwargs.get('Mu', 0)
        self.xi = xi
        self.phi = kwargs.get('phi', self.xi)
        

        # Check normalization condition
        if np.abs(quad(lambda x: abs(self.xi(x))**2, -np.inf, np.inf)[0] - 1) > 1e-6:
            raise ValueError("Invalid 'xi' function provided. Check the 'Normalization' constant.")
        self.Gamma_1 = Gamma_1
        self.Gamma_2 = Gamma_2
        # Check if Gamma values are positive
        if min(self.Gamma_1, self.Gamma_2) <= 0:
            raise ValueError("Gamma must be 'positive'.")
        # Calculate complex Gamma if Delta is provided
        self.Delta_1 = kwargs.get('Delta_1', 0)
        if self.Delta_1 != 0:
            self.gamma_1 = complex(self.Gamma_1, 2 * self.Delta_1)
        self.Delta_2 = kwargs.get('Delta_2', 0)
        if self.Delta_2 != 0:
            self.gamma_2 = complex(self.Gamma_2, 2 * self.Delta_2)
        # Set time range
        self.tRange = kwargs.get('tRange', [-np.infty, +np.infty])
        self.lower_limit = self.tRange[0]
        self.upper_limit = self.tRange[1]

    def MeanTime(self):
        """
        Calculates the mean time of excitation.

        Returns:
        float: The mean time.
        """
        t, p = self.P()
        return np.sum(p[1:] * np.diff(t))

    def optimize_single(self, params, parameters_to_optimize, parameter_bounds):
        """
        Optimizes parameters for a single case. This method is defined to enable parallel optimization.
        
        Parameters:
        params (list): Initial guess for parameters.
        parameters_to_optimize (list): Parameters to optimize.
        parameter_bounds (list): Bounds for parameters.
        
        Returns:
        tuple: Optimized P value and parameters if successful; otherwise, returns None.
        """
        def objective(params):
            """
            Objective function for optimization.
            This code utilizes the 'Nelder-Mead' method for optimization. For more information about the optimization method, see:
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>
            
            Parameters:
            params (list): Parameters to optimize.
            
            Returns:
            float: Objective value.
            """
            param_values = {param_name: param_value for param_name, param_value in zip(parameters_to_optimize, params)}
            for param_name, param_value in param_values.items():
                setattr(self, param_name, param_value)
            t, p = self.P()
            return -np.max(p)

        result = minimize(objective, params, method='Nelder-Mead', bounds=parameter_bounds)
        # print(-result.fun, result.x)
        return -result.fun, result.x if result.success else None

    def optimize(self, parameters_to_optimize, **kwargs):
        """
        Parallel optimization of parameters with multiple cores or threads.
        
        Parameters:
        parameters_to_optimize (list): Parameters to optimize.
        **kwargs: Additional keyword arguments.
            num_attempts (int, optional): Number of different optimizations with different initial random guesses. Default is 1.
            num_processor (int, optional): Number of threads involved in optimization. Default is 4.
            Mu_range (list, optional): The bounds of '\mu' in the optimization. Default is [-2, 4].
        
        Returns:
        dict: Optimized P value and parameters if successful; otherwise, returns None.
        """

        num_attempts = kwargs.get('num_attempts', 1)
        num_processor = kwargs.get('num_processor', 4)
        Mu_range = kwargs.get('Mu_range', [-2, 4])
        np.random.seed()
        parameter_bounds = [(Mu_range[0], Mu_range[1]) if param == 'Mu' else (0.2, 20) for param in parameters_to_optimize]
        initial_guesses = [[np.random.uniform(lower, upper) for lower, upper in parameter_bounds] for _ in range(num_attempts)]
        with Pool(processes=num_processor) as pool:
            results = pool.starmap(self.optimize_single, [(initial_guess, parameters_to_optimize, parameter_bounds) for initial_guess in initial_guesses])
        best_result = max(results, key=lambda x: x[0])
        if best_result[1] is not None:
            optimized_P = best_result[0]
            optimized_params = {param_name: param_value for param_name, param_value in zip(parameters_to_optimize, best_result[1])}
            optimized_P = {'P_max': optimized_P}
            return {**optimized_P, **optimized_params}
        else:
            return None

class TwoPhoton(General):
    def __init__(self, *args, **kwargs):
        """
        Initializes a TwoPhoton  with Three level system class.

        Parameters:
        xi (callable): The first wave profile.
        Gamma_1 (float): The value of '\Gamma_1' (or '\Gamma_e'). 
        Gamma_2 (float): The value of '\Gamma_2' (or '\Gamma_f').
        *args: Positional arguments.
        **kwargs: Additional keyword arguments.
            phi (callable, optional): The second wave profile. Default is xi.
            Delta_1 (float, optional): The first dispersion value '\Delta_1'. Default is 0.
            Delta_2 (float, optional): The second dispersion value '\Delta_2'. Default is 0.
            tRange (list, optional): The time range in which P should be computed. Default is [-np.infty, +np.infty].
            tol (float, optional): The desired tolorance. Based this paramtere, we can cut the time range. Default is 1e-5
            nBins (int, optional): The number of bins for P(t). Defualt is 1000.
            status (str, optional): The algorithm in which using that we want to compute the P. defaul is 'vectorized'. Possible values include:
                - 'vectorized': Uses vectorization method or UNCORRELATED case. it make a nBin*nBin matrix and then try to integrate over it. Although this method is more than 10 time faster than quad, it consume more RAM.
                - 'Uncorrelated_quad': Uses scipy.integrate.quad for Uncorrelated case. 
                - 'Correlated': Uses vectorization method or ENTANGLED case. It is similar to 'vectorized' method.
                - 'Correlated_quad': Uses scipy.integrate.quad for ENtangled case. It is slower than previous method but consume less RAM. Also If you are obsest about the integration limmit, it's better to use this limit. 
                - 'Analytical': Uses analytical answers for computation. For now only for UNCORRELATED case of Exponentail Raising and Decay was added.
                
            Default is 'vectorized'.
                    
        """
        super().__init__(*args, **kwargs)
        self.tol = kwargs.get('tol', 1e-5)
        self.nBins = kwargs.get('nBins', 1000)
        self.status = kwargs.get('status', 'vectorized')
        if self.status == 'vectorized':
            self.P = self.P_vect
        elif self.status == 'Correlated':
            self.P = self.P_correlated
        elif self.status == 'Correlated_quad':
            self.P = self.P_correlated_quad
        elif self.status == 'Analytical':
            self.P = self.P_anal
        elif self.status == 'Uncorrelated_quad':
            self.P = self.P_Uncorrelated_quad

    def P_Uncorrelated_quad(self):
        """
        Calculates P for uncorrelated case using quad.

        Returns:
        tuple: Time array and P values.
        """
        def intg(t):
            return quad(lambda t2: self.phi(t2, Omega=self.Omega_2, Mu=self.Mu) * np.exp(-self.Gamma_2 * (t - t2) / 2) * quad(lambda t1: self.xi(t1, Omega=self.Omega_1, Mu=0) * np.exp(-self.Gamma_1 * (t2 - t1) / 2), -np.infty, t2, epsabs=self.tol)[0], -np.infty, t, epsabs=self.tol)[0]
        t = (np.linspace(-4, 10, self.nBins) + self.Mu) / np.min([self.Gamma_1, self.Gamma_2])
        P = self.Gamma_1 * self.Gamma_2 * np.abs(np.array([intg(t1) for t1 in t])) ** 2
        return t, P

    
    def P_vect(self):
        """
        Calculates P using vectorized computation.

        Returns:
        tuple: Time array and P values.
        """
        #  In the first part of the method, it looks for best time range for P (if inf tRange = [-np.infty, +np.infty] using tol)
        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -1e6
            while True:
                if max(np.abs(self.xi(self.lower_limit , Omega = self.Omega_1)) \
                       ,np.abs(self.phi(self.lower_limit , Omega = self.Omega_2, Mu = self.Mu)) ) > self.tol:
                    break
                else : 
                    self.lower_limit = self.lower_limit/2 + self.lower_limit*0.01
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 1e6
            while True:
                if max(np.exp(-self.Gamma_2*self.upper_limit) , np.abs(self.xi(self.upper_limit , Omega = self.Omega_1)) \
                       ,np.abs(self.phi(self.upper_limit , Omega = self.Omega_2, Mu = self.Mu))) > self.tol*1e-1:
                    break
                else:
                    self.upper_limit = self.upper_limit/2 + self.upper_limit*0.01 #+self.Mu
            self.upper_limit = self.upper_limit + self.Mu
        t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
        xi = self.xi(t,Omega = self.Omega_1,Mu = 0)
        phi = self.phi(t,Omega = self.Omega_2, Mu = self.Mu)
        ones = np.ones([len(xi),len(xi)])

        P = self.Gamma_1*self.Gamma_2*np.exp(-self.Gamma_2*t[:,np.newaxis])\
        * np.abs(np.cumsum(phi[:,np.newaxis]*np.exp((self.Gamma_2-self.Gamma_1)*t[:,np.newaxis]/2) \
                           * np.cumsum(xi[np.newaxis,:]*np.exp(self.Gamma_1*t/2)* np.tril(ones),axis = 1)*dt ,axis=0)*dt)**2 
        P = np.diag(P)
        if label_L == True:
            self.lower_limit = -np.infty
        if label_U == True:
            self.upper_limit = np.infty
        return t, P


    def P_correlated_quad(self):
        """
        Calculates P for correlated case using quad.

        Returns:
        tuple: Time array and P values.
        """
        def intg(t):
            return quad( lambda t2: np.exp(-self.Gamma_2*(t - t2)/2)*quad( lambda t1: np.exp(-self.Gamma_1*(t2- t1)/2)*self.psi(t1,t2) , -np.infty, t2,epsabs=self.tol )[0] ,-np.infty, t,epsabs=self.tol)[0]
        # t = np.linspace(-4,10,self.nBins)/np.min([self.Gamma_2,self.Gamma_1,self.Omega_1,self.Omega_2])
        t = (np.linspace(-4,10,self.nBins)+self.Mu)/np.min([self.Gamma_1,self.Gamma_2])
        # t = np.linspace(-2*(1/self.Gamma_1 + 1/self.Gamma_2) - np.abs(self.Mu) , 5*(1/self.Gamma_1 + 1/self.Gamma_2 ) + np.abs(self.Mu) , self.nBins)
        P = self.Gamma_1*self.Gamma_2*np.abs(np.array([intg(t1) for t1 in t]) )**2
        return t, P


    
    def P_correlated(self):
        """
        Calculates P for Entangled case using vectorization method. 

        Returns:
        tuple: Time array and P values.
        """
        # In the first part of the method, it looks for best time range for P (if inf tRange = [-np.infty, +np.infty] using tol)
        if (self.lower_limit == -np.infty) or (self.upper_limit == np.infty) :
            if self.lower_limit == -np.infty :
                self.lower_limit = -10
            if self.upper_limit == np.infty :
                self.upper_limit = 10
            t, dt = np.linspace(self.lower_limit,self.upper_limit,1000,retstep=True)
            f = self.psi(t[np.newaxis:],t[:,np.newaxis])        
            P = self.Gamma_1*self.Gamma_2*np.exp(-self.Gamma_2*t[:,np.newaxis])\
            * np.abs(np.cumsum(np.exp((self.Gamma_2-self.Gamma_1)*t[:,np.newaxis]/2) \
                               * np.cumsum(f*np.exp(self.Gamma_1*t/2)* np.tril(np.ones_like(f)),axis = 1)*dt ,axis=0)*dt)**2 
            P = np.diag(P)
            tt = t[np.abs(P)>self.tol]
            if len(tt) == 0:
                self.lower_limit = t[0]
                self.upper_limit = t[-1]
            else:
                self.lower_limit = tt[0]
                self.upper_limit = tt[-1]
        # After finding the best time range, we can calculate in higher resolution.
        t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
        f = self.psi(t[np.newaxis,:],t[:,np.newaxis])        
        P = self.Gamma_1*self.Gamma_2*np.exp(-self.Gamma_2*t[:,np.newaxis])\
        * np.abs(np.cumsum(np.exp((self.Gamma_2-self.Gamma_1)*t[:,np.newaxis]/2) \
                           * np.cumsum(f*np.exp(self.Gamma_1*t/2)* np.tril(np.ones_like(f)),axis = 1)*dt ,axis=0)*dt)**2 
        P = np.diag(P)
        return t , P

    def psi(self, t1, t2):
        """
        Calculate the entangled Gaussian wave function.
    
        This method calculates the entangled Gaussian wave function using the provided time parameters.
    
        Parameters:
        t1 (float): The mean time of the first photon.
        t2 (float): The mean time of the second photon.
    
        Returns:
        float: The value of the entangled Gaussian wave function for the given time parameters.
        """
        return np.sqrt(self.Omega_1 * self.Omega_2 / (2 * np.pi)) * np.exp(
            -self.Omega_1 ** 2 * (t1 + t2 - self.Mu) ** 2 / 8 - self.Omega_2 ** 2 * (t1 - t2 + self.Mu) ** 2 / 8)

    
    def P_anal(self):
        """
        Calculates P analytically. Only for Uncorrelated case of 'ExpoDecay' and 'ExpoRasinig'

        Returns:
        tuple: Time array and P values.
        """
        pulseShape=self.xi.__name__
        if self.lower_limit == -np.infty :
            self.lower_limit = -1e6
            while True:
                if max(np.abs(self.xi(self.lower_limit , Omega = self.Omega_1)) \
                       ,np.abs(self.phi(self.lower_limit , Omega = self.Omega_2, Mu = self.Mu)) ) > self.tol:
                    break
                else : 
                    self.lower_limit = self.lower_limit/2 + self.lower_limit*0.01
        if self.upper_limit == np.infty :
            self.upper_limit = 1e6
            while True:
                if max(np.exp(-self.Gamma_1*self.upper_limit) , np.exp(-self.Gamma_2*self.upper_limit) ) > self.tol:
                    break
                else:
                    self.upper_limit = self.upper_limit/2 + self.upper_limit*0.01 #+self.Mu
            self.upper_limit = self.upper_limit + self.Mu
        # Analitical solution for exponential raising in the case Mu = 0
        if pulseShape == 'ExpoRaising':
            t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
            t1 = t[t<=0]
            t2 = t[t>0]
            coef =  (16*self.Omega_1*self.Omega_2*self.Gamma_1*self.Gamma_2)/((self.Gamma_1+self.Omega_1)**2 *(self.Gamma_2+self.Omega_1+self.Omega_2)**2 )
            P_t1 = coef*np.exp((self.Omega_1+self.Omega_2)*t1)
            P_t2 = coef* np.exp(-self.Gamma_2*t2)
            P = np.append(P_t1,P_t2 )
            return t, P
        elif pulseShape == 'ExpoDecay':
            t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
            X = complex(self.Delta_1) + self.Gamma_1/2 -self.Omega_1/2
            Y = complex(self.Delta_2) + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
            Z = complex(self.Delta_1 + self.Delta_2) + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2
            coef = self.Gamma_1*self.Gamma_2*self.Omega_1*self.Omega_2*np.exp(-self.Gamma_2*t + self.Omega_2*self.Mu)
            if np.abs(X) < 1e5 :
                self.Omega_1 += self.Omega_1*0.0001
                X = complex(self.Delta_1) + self.Gamma_1/2 -self.Omega_1/2
                Y = complex(self.Delta_2) + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
                Z = complex(self.Delta_1 + self.Delta_2) + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2
            if np.abs(Y) < 1e5 :
                self.Omega_2 += self.Omega_2*0.0001
                X = complex(self.Delta_1) + self.Gamma_1/2 -self.Omega_1/2
                Y = complex(self.Delta_2) + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
                Z = complex(self.Delta_1 + self.Delta_2) + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2
            if np.abs(Z) < 1e5 :
                self.Omega_1 += self.Omega_1*0.0001
                X = complex(self.Delta_1) + self.Gamma_1/2 -self.Omega_1/2
                Y = complex(self.Delta_2) + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
                Z = complex(self.Delta_1 + self.Delta_2) + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2                    
            P =coef /(np.abs(X)**2) *np.abs((np.exp(Z*t) - np.exp(Z*self.Mu))/Z - (np.exp(Y*t) - np.exp(Y*self.Mu))/Y )**2
            P[t<self.Mu] = 0
            return t, P
        
