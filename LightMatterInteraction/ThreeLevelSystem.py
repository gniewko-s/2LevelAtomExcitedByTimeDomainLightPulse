from multiprocessing import Pool
import numpy as np
import matplotlib.pylab as plt
from time import time
import random
import sys
from scipy.integrate import complex_ode, solve_ivp, quad, dblquad
from scipy.optimize import minimize, minimize_scalar
from numba import jit
# import cupy as cp

np.infty = np.inf #there was a change in higher version of numpy.

class General():
    def __init__(self, xi: callable, Gamma_1, Gamma_2, **kwargs):
        r"""
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
            n_1 (float, optional): Mean number of the photon for first profile (For coherent case). Default is 1
            n_2 (float, optional): Mean number of the photon for second profile (For coherent case). Default is 1
            
        """

        self.Omega_1 = kwargs.get('Omega_1', 1)
        self.Omega_2 = kwargs.get('Omega_2', 1)
        self.Mu = kwargs.get('Mu', 0)
        self.xi = xi
        self.phi = kwargs.get('phi', self.xi)
        self.n_1 = kwargs.get('n_1' , 1)   # Mean number of the photon for first profile (For coherent case)
        self.n_2 = kwargs.get('n_2' , 1)   # Mean number of the photon for second profile (For coherent case)

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
        self.Delta_1 = complex(0,self.Delta_1)
        # if self.Delta_1 != 0:
        #     self.gamma_1 = complex(self.Gamma_1, 2 * self.Delta_1)
        self.Delta_2 = kwargs.get('Delta_2', 0)
        self.Delta_2 = complex(0,self.Delta_2)
        # if self.Delta_2 != 0:
        #     self.gamma_2 = complex(self.Gamma_2, 2 * self.Delta_2)
        # Set time range
        self.tRange = kwargs.get('tRange', [-np.infty, +np.infty])
        self.lower_limit = self.tRange[0]
        self.upper_limit = self.tRange[1]
        self.N_threads = kwargs.get('N_threads', 1)

    def MeanTime(self):
        r"""
        Calculates the mean time of excitation.

        Returns:
        float: The mean time.
        """
        t, p = self.P()
        return np.sum(p[1:] * np.diff(t))

    def optimize_single(self, params, parameters_to_optimize, parameter_bounds):
        r"""
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

        result = minimize(objective, params, method='Nelder-Mead', bounds=parameter_bounds )#, options = {'maxiter': 200})
        return -result.fun, result.x if result.success else None

    def optimize(self, parameters_to_optimize, **kwargs):
        r"""
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
        Param_range = kwargs.get('Param_range' , [0.01,10])
        Omega2_range = kwargs.get('Omega2_range',Param_range )
        np.random.seed()
        parameter_bounds = [(Mu_range[0], Mu_range[1]) if param == 'Mu' else (Param_range[0], Param_range[1]) for param in parameters_to_optimize]
        # parameter_bounds = []
        # for param in parameters_to_optimize:
        #     if param == 'Mu':
        #         bounds = (Mu_range[0], Mu_range[1])
        #     elif param == 'Omega_2':
        #         bounds = (Omega2_range[0], Omega2_range[1])
        #     else:
        #         bounds = (Param_range[0], Param_range[1])
        #     parameter_bounds.append(bounds)
            
        initial_guesses = [[np.random.uniform(lower, upper) for lower, upper in parameter_bounds] for _ in range(num_attempts)]
        if num_processor != 1:
            with Pool(processes=num_processor) as pool:
                results = pool.starmap(self.optimize_single, [(initial_guess, parameters_to_optimize, parameter_bounds) for initial_guess in initial_guesses])
        else :
            results = []
            for initial_guess in initial_guesses:
                results.append(self.optimize_single(initial_guess, parameters_to_optimize, parameter_bounds))
            
        best_result = max(results, key=lambda x: x[0])
        if best_result[1] is not None:
            optimized_P = best_result[0]
            optimized_params = {param_name: param_value for param_name, param_value in zip(parameters_to_optimize, best_result[1])}
            optimized_P = {'P_max': optimized_P}
            return {**optimized_P, **optimized_params}
        else:
            optimized_params = {param_name: 0 for param_name in parameters_to_optimize}
            optimized_P = {'P_max': 0}
            return {**optimized_P, **optimized_params}

class TwoPhoton(General):
    def __init__(self, *args, **kwargs):
        r"""
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
            
        elif self.status == 'Uncorrelated_GPU':
            self.P = self.P_unc_gpu
        elif self.status == 'Correlated':
            self.P = self.P_correlated
        elif self.status == 'Correlated_quad':
            self.P = self.P_correlated_quad
        elif self.status == 'Correlated_quad_paral':
            self.P = self.P_correlated_quad_paral     
        elif self.status == 'Uncorrelated_quad_paral':
            self.P = self.P_Uncorrelated_quad_paral
        elif self.status == 'Correlated_GPU':
            self.P = self.P_correlated_GPU
        elif self.status == 'Analytical':
            self.P = self.P_anal
        elif self.status == 'Uncorrelated_quad':
            self.P = self.P_Uncorrelated_quad
        elif self.status == 'Coherent':
            self.P = self.P_coherent
        elif self.status == 'Uni_directional' :
            self.P = self.P_unidirection
    def P_Uncorrelated_quad(self):
        r"""
        Calculates P for uncorrelated case using quad.

        Returns:
        tuple: Time array and P values.
        """
        def intg(t):
            return quad(lambda t2: self.phi(t2, Omega=self.Omega_2, Mu=self.Mu) * np.exp(-(self.Delta_2 + self.Gamma_2/2) * (t - t2)) * quad(lambda t1: self.xi(t1, Omega=self.Omega_1, Mu=0) * np.exp(-self.Gamma_1 * (t2 - t1) / 2) * np.exp(self.Delta_1*t1), -np.infty, t2, epsabs=self.tol, complex_func=True)[0], -np.infty, t, epsrel=self.tol, complex_func=True)[0]
        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -4
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 10

        t = (np.linspace(self.lower_limit, self.upper_limit , self.nBins) + self.Mu) / np.min([self.Gamma_1, self.Gamma_2])
        P = self.Gamma_1 * self.Gamma_2 * np.abs(np.array([intg(t1) for t1 in t])) ** 2
        return t, P


    
    def compute_chunk_uncorrelated(self,chunk):
        results =[]
        for t in chunk:
            intg = quad(lambda t2: self.phi(t2, Omega=self.Omega_2, Mu=self.Mu) * np.exp(-(self.Delta_2 + self.Gamma_2/2) * (t - t2)) * quad(lambda t1: self.xi(t1, Omega=self.Omega_1, Mu=0) * np.exp(-self.Gamma_1 * (t2 - t1) / 2) * np.exp(self.Delta_1*t1), -np.infty, t2, epsabs=self.tol, complex_func=True)[0], -np.infty, t, epsrel=self.tol, complex_func=True)[0]
            result = self.Gamma_1 * self.Gamma_2 * np.abs(intg)**2
            results.append((t,result))
        return results  

    
    def P_Uncorrelated_quad_paral(self):
        r"""
        Calculates P for uncorrelated case using quad.

        Returns:
        tuple: Time array and P values.
        """

        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -2
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 6 + self.Mu

        t = np.linspace(self.lower_limit, self.upper_limit , self.nBins) / np.min([self.Gamma_1, self.Gamma_2])


        # num_processes = 20
        num_processes = self.N_threads
        chunk_size = len(t) // num_processes
        chunks = [t[i:i+chunk_size] for i in range(0, len(t), chunk_size)]

        pool = Pool(processes=num_processes)
        results = pool.map(self.compute_chunk_uncorrelated, chunks)
        pool.close()
        pool.join()
        all_results = [item for sublist in results for item in sublist]
        t_val, P_val = zip(*all_results)

        # P = self.Gamma_1 * self.Gamma_2 * np.abs(np.array([intg(t1) for t1 in t])) ** 2
        return np.array(t_val), np.array(P_val)


    
    def P_unc_gpu(self):
        if self.xi.__name__ == 'Rectangular':
            label_L , label_U = False , False
            if self.lower_limit == -np.infty :
                label_L = True
                self.lower_limit = min(-1/self.Omega_1 , -1/ self.Omega_2 + self.Mu )
            if self.upper_limit == np.infty :
                label_U = True
                # self.upper_limit = max(1/self.Omega_1 ,1/ self.Omega_2 + self.Mu)*1.2
                self.upper_limit = (1/ self.Omega_2 + self.Mu)*1.1
            
            t = cp.linspace(self.lower_limit,self.upper_limit,self.nBins)
            if (-1/self.Omega_1 > -1/self.Omega_2 + self.Mu):
                return cp.asnumpy(t), cp.asnumpy(cp.zeros_like(t))
        else:
            label_L , label_U = False , False
            if self.lower_limit == -np.infty :
                label_L = True
                self.lower_limit = -4
            if self.upper_limit == np.infty :
                label_U = True
                self.upper_limit = 10
            self.upper_limit = self.upper_limit + self.Mu
            t = cp.linspace(self.lower_limit,self.upper_limit,self.nBins)/min(self.Gamma_1,self.Gamma_2)
        dt = cp.diff(t)[0]
        xi = cp.asarray(self.xi(cp.asnumpy(t),Omega = self.Omega_1,Mu = 0))
        phi = cp.asarray(self.phi(cp.asnumpy(t),Omega = self.Omega_2, Mu = self.Mu))
        ones = cp.ones([len(xi),len(xi)])

        P = self.Gamma_1*self.Gamma_2*cp.exp(-self.Gamma_2*t[:,cp.newaxis])\
        * cp.abs(cp.cumsum(phi[:,cp.newaxis]*cp.exp((self.Delta_2+(self.Gamma_2-self.Gamma_1)/2)*t[:,cp.newaxis]) \
                           * cp.cumsum(xi[cp.newaxis,:]*cp.exp((self.Delta_1 + self.Gamma_1/2)*t)* cp.tril(ones),axis = 1)*dt ,axis=0)*dt)**2 
        P = cp.diag(P)
        if label_L == True:
            self.lower_limit = -np.infty
        if label_U == True:
            self.upper_limit = np.infty
        return cp.asnumpy(t), cp.asnumpy(P)



    
    def P_vect(self):
        r"""
        Calculates P using vectorized computation.

        Returns:
        tuple: Time array and P values.
        """
        if self.xi.__name__ == 'Rectangular':
            label_L , label_U = False , False
            if self.lower_limit == -np.infty :
                label_L = True
                self.lower_limit = min(-1/self.Omega_1 , -1/ self.Omega_2 + self.Mu )
            if self.upper_limit == np.infty :
                label_U = True
                # self.upper_limit = max(1/self.Omega_1 ,1/ self.Omega_2 + self.Mu)*1.2
                self.upper_limit = (1/ self.Omega_2 + self.Mu)*1.1
            
            t = np.linspace(self.lower_limit,self.upper_limit,self.nBins)
            if (-1/self.Omega_1 > -1/self.Omega_2 + self.Mu):
                return t, np.zeros_like(t)
        else:
            label_L , label_U = False , False
            if self.lower_limit == -np.infty :
                label_L = True
                self.lower_limit = -4
            if self.upper_limit == np.infty :
                label_U = True
                self.upper_limit = 10
                self.upper_limit = self.upper_limit + self.Mu
            t = (np.linspace(self.lower_limit,self.upper_limit,self.nBins)+self.Mu)/np.min([self.Gamma_1,self.Gamma_2])        
        dt = np.diff(t)[0]
        
        xi = self.xi(t,Omega = self.Omega_1,Mu = 0)
        phi = self.phi(t,Omega = self.Omega_2, Mu = self.Mu)
        ones = np.ones([len(xi),len(xi)])

        P = self.Gamma_1*self.Gamma_2*np.exp(-self.Gamma_2*t[:,np.newaxis])\
        * np.abs(np.cumsum(phi[:,np.newaxis]*np.exp((self.Delta_2+(self.Gamma_2-self.Gamma_1)/2)*t[:,np.newaxis]) \
                           * np.cumsum(xi[np.newaxis,:]*np.exp((self.Delta_1 + self.Gamma_1/2)*t)* np.tril(ones),axis = 1)*dt ,axis=0)*dt)**2 
        P = np.diag(P)
        if label_L == True:
            self.lower_limit = -np.infty
        if label_U == True:
            self.upper_limit = np.infty
        return t, P
        # label_L , label_U = False , False
        # if self.lower_limit == -np.infty :
        #     label_L = True
        #     self.lower_limit = -4
        # if self.upper_limit == np.infty :
        #     label_U = True
        #     self.upper_limit = 10
        #     self.upper_limit = self.upper_limit + self.Mu
        # while True:

        #     t = (np.linspace(self.lower_limit,self.upper_limit,self.nBins)+self.Mu)/np.min([self.Gamma_1,self.Gamma_2])        
        #     dt = np.diff(t)[0]
            
        #     xi = self.xi(t,Omega = self.Omega_1,Mu = 0)
        #     phi = self.phi(t,Omega = self.Omega_2, Mu = self.Mu)
        #     ones = np.ones([len(xi),len(xi)])
    
        #     P = self.Gamma_1*self.Gamma_2*np.exp(-self.Gamma_2*t[:,np.newaxis])\
        #     * np.abs(np.cumsum(phi[:,np.newaxis]*np.exp((self.Delta_2+(self.Gamma_2-self.Gamma_1)/2)*t[:,np.newaxis]) \
        #                        * np.cumsum(xi[np.newaxis,:]*np.exp((self.Delta_1 + self.Gamma_1/2)*t)* np.tril(ones),axis = 1)*dt ,axis=0)*dt)**2 
        #     P = np.diag(P)
        #     DT = t[P>self.tol]
        #     if len(DT) != 0:
        #         DT = abs(DT[-1] -DT[0])
        #     else :
        #         DT = np.array([0])
        #     DTR = DT/(t[-1] - t[0])
        #     if DTR>0.1:
        #         if label_L == True:
        #             self.lower_limit = -np.infty
        #         if label_U == True:
        #             self.upper_limit = np.infty
        #         print(f'$$$$$$$$ {t[0]}')
        #         return t, P
        #         break
        #     else :
        #         # self.upper_limit = np.mean([self.upper_limit , DT[-1]])
        #         # self.lower_limit = np.mean([self.lower_limit , -DT[0]])
        #         self.upper_limit /= 2
        #         self.lower_limit /= 2


    def compute_chunk_correlated(self,chunk):
        results =[]
        for t in chunk:
            intg = quad( lambda t2: np.exp(-(self.Delta_2+self.Gamma_2/2)*(t - t2))*quad( lambda t1: np.exp(-self.Gamma_1*(t2- t1)/2 + self.Delta_1*t1)*self.psi(t1,t2) , -np.infty, t2,epsabs=self.tol, complex_func=True )[0] ,-np.infty, t,epsabs=self.tol, complex_func=True)[0]
            # result = self.Gamma_1 * self.Gamma_2 * np.abs(intg)**2
            result = self.Gamma_1 * self.Gamma_2 * np.abs(intg)**2
            results.append((t,result))
        return results  

    
    def P_correlated_quad_paral(self):
        r"""
        Calculates P for uncorrelated case using quad.

        Returns:
        tuple: Time array and P values.
        """

        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -2
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 6 + self.Mu

        t = np.linspace(self.lower_limit, self.upper_limit , self.nBins) / np.min([self.Gamma_1, self.Gamma_2])


        num_processes = self.N_threads
        chunk_size = len(t) // num_processes
        chunks = [t[i:i+chunk_size] for i in range(0, len(t), chunk_size)]

        pool = Pool(processes=num_processes)
        results = pool.map(self.compute_chunk_correlated, chunks)
        pool.close()
        pool.join()
        all_results = [item for sublist in results for item in sublist]
        t_val, P_val = zip(*all_results)
        # print('jhsdghgfhgdhgf',P_val)
        # P = self.Gamma_1 * self.Gamma_2 * np.abs(np.array([intg(t1) for t1 in t])) ** 2
        return np.array(t_val), np.array(P_val)


    

    def P_correlated_quad(self):
        """
        Calculates P for correlated case using quad.

        Returns:
        tuple: Time array and P values.
        """
        def intg(t):
            return quad( lambda t2: np.exp(-(self.Delta_2+self.Gamma_2/2)*(t - t2))*quad( lambda t1: np.exp(-self.Gamma_1*(t2- t1)/2 + self.Delta_1*t1)*self.psi(t1,t2) , -np.infty, t2,epsabs=self.tol, complex_func=True )[0] ,-np.infty, t,epsabs=self.tol, complex_func=True)[0]
        # t = np.linspace(-4,10,self.nBins)/np.min([self.Gamma_2,self.Gamma_1,self.Omega_1,self.Omega_2])
        t = (np.linspace(-4,10,self.nBins)+self.Mu)/np.min([self.Gamma_1,self.Gamma_2])
        # t = np.linspace(-2*(1/self.Gamma_1 + 1/self.Gamma_2) - np.abs(self.Mu) , 5*(1/self.Gamma_1 + 1/self.Gamma_2 ) + np.abs(self.Mu) , self.nBins)
        # P = self.Gamma_1*self.Gamma_2*np.abs(np.array([intg(t1) for t1 in t]) )**2
        P = self.Gamma_1*self.Gamma_2*np.abs(np.array([intg(t1) for t1 in t]) )**2
        return t, P


    
    def P_correlated_GPU(self):
        """
        GPU
        Calculates P for Entangled case using vectorization method. 

        Returns:
        tuple: Time array and P values.
        """
        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -4
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 10
            self.upper_limit = self.upper_limit + self.Mu
        t = cp.linspace(self.lower_limit,self.upper_limit,self.nBins)
        t = (t  + self.Mu)/min(self.Gamma_1,self.Gamma_2)
        dt = cp.diff(t)[0]
        
        f = cp.asarray(self.psi( cp.asnumpy(t[cp.newaxis,:]), cp.asnumpy(t[:,cp.newaxis])) )       
        P = self.Gamma_1*self.Gamma_2*cp.exp(-self.Gamma_2*t[:,cp.newaxis])\
        * cp.abs(cp.cumsum(cp.exp((self.Delta_2 + (self.Gamma_2-self.Gamma_1)/2)*t[:,cp.newaxis]) \
                           * cp.cumsum(f*cp.exp((self.Delta_1 + self.Gamma_1/2)*t)* cp.tril(cp.ones_like(f)),axis = 1)*dt ,axis=0)*dt)**2 
        P = cp.diag(P)
        if label_L == True:
            self.lower_limit = -np.infty
        if label_U == True:
            self.upper_limit = np.infty
        return cp.asnumpy(t), cp.asnumpy(P)

    
    def P_correlated(self):
        """
        Calculates P for Entangled case using vectorization method. 

        Returns:
        tuple: Time array and P values.
        """
        # In the first part of the method, it looks for best time range for P (if inf tRange = [-np.infty, +np.infty] using tol)
        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -4
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 10
            self.upper_limit = self.upper_limit + self.Mu
        t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
        t = (t  + self.Mu)/min(self.Gamma_1,self.Gamma_2)

        f = self.psi(t[np.newaxis,:],t[:,np.newaxis])        
        P = self.Gamma_1*self.Gamma_2*np.exp(-self.Gamma_2*t[:,np.newaxis])\
        * np.abs(np.cumsum(np.exp((self.Delta_2 + (self.Gamma_2-self.Gamma_1)/2)*t[:,np.newaxis]) \
                           * np.cumsum(f*np.exp((self.Delta_1 + self.Gamma_1/2)*t)* np.tril(np.ones_like(f)),axis = 1)*dt ,axis=0)*dt)**2 
        P = np.diag(P)
        if label_L == True:
            self.lower_limit = -np.infty
        if label_U == True:
            self.upper_limit = np.infty
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
        # Analitical solution for exponential raising in the case Mu = 0
        if pulseShape == 'ExpoRaising':
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

            t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
            t1 = t[t<=0]
            t2 = t[t>0]
            coef =  (16*self.Omega_1*self.Omega_2*self.Gamma_1*self.Gamma_2)/((self.Gamma_1+self.Omega_1)**2 *(self.Gamma_2+self.Omega_1+self.Omega_2)**2 )
            P_t1 = coef*np.exp((self.Omega_1+self.Omega_2)*t1)
            P_t2 = coef* np.exp(-self.Gamma_2*t2)
            P = np.append(P_t1,P_t2 )
            return t, P

        
        elif pulseShape == 'ExpoDecay':
            label_L , label_U = False , False
            if self.lower_limit == -np.infty :
                label_L = True
                self.lower_limit = 0
            if self.upper_limit == np.infty :
                label_U = True
                self.upper_limit = 6 
            self.upper_limit = self.upper_limit + self.Mu

            # t = np.linspace(self.lower_limit, self.upper_limit , self.nBins) / np.min([self.Gamma_1, self.Gamma_2])

            # if self.lower_limit == -np.infty :
            #     self.lower_limit = -1e6
            #     while True:
            #         if max(np.abs(self.xi(self.lower_limit , Omega = self.Omega_1)) \
            #                ,np.abs(self.phi(self.lower_limit , Omega = self.Omega_2, Mu = self.Mu)) ) > self.tol:
            #             break
            #         else : 
            #             self.lower_limit = self.lower_limit/2 + self.lower_limit*0.01
            # if self.upper_limit == np.infty :
            #     self.upper_limit = 1e6
            #     while True:
            #         if max(np.exp(-self.Gamma_1*self.upper_limit) , np.exp(-self.Gamma_2*self.upper_limit) ) > self.tol:
            #             break
            #         else:
            #             self.upper_limit = self.upper_limit/2 + self.upper_limit*0.01 #+self.Mu
            #     self.upper_limit = self.upper_limit + self.Mu

            
            def inner_comp(t):
                X = complex(self.Delta_1) + self.Gamma_1/2 -self.Omega_1/2
                Y = complex(self.Delta_2) + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
                Z = complex(self.Delta_1 + self.Delta_2) + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2
                coef = self.Gamma_1*self.Gamma_2*self.Omega_1*self.Omega_2*np.exp(-self.Gamma_2*t + self.Omega_2*self.Mu)
                if np.abs(X) < 1e5 :
                    self.Omega_1 += self.Omega_1*0.0001
                    X = self.Delta_1 + self.Gamma_1/2 -self.Omega_1/2
                    Y = self.Delta_2 + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
                    Z = self.Delta_1 + self.Delta_2 + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2
                if np.abs(Y) < 1e5 :
                    self.Omega_2 += self.Omega_2*0.0001
                    X = self.Delta_1 + self.Gamma_1/2 -self.Omega_1/2
                    Y = self.Delta_2 + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
                    Z = self.Delta_1 + self.Delta_2 + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2
                if np.abs(Z) < 1e5 :
                    self.Omega_1 += self.Omega_1*0.0001
                    X = self.Delta_1 + self.Gamma_1/2 -self.Omega_1/2
                    Y = self.Delta_2 + (self.Gamma_2 - self.Gamma_1)/2 -self.Omega_2/2
                    Z = self.Delta_1 + self.Delta_2 + self.Gamma_2/2 - (self.Omega_1 + self.Omega_2)/2                    
                P =coef /(np.abs(X)**2) *np.abs((np.exp(Z*t) - np.exp(Z*self.Mu))/Z - (np.exp(Y*t) - np.exp(Y*self.Mu))/Y )**2
                P[t<self.Mu] = 0
                return t, P
            t = np.linspace(self.lower_limit, self.upper_limit , self.nBins) / np.min([self.Gamma_1, self.Gamma_2])
            t, p = inner_comp(t)
            t = t[p>0.0001]
            if len(t) == 0:
                return np.zeros(1000),np.zeros(1000)
            else:
                t = np.linspace(0, t[-1] , self.nBins) #/ np.min([self.Gamma_1, self.Gamma_2])
                t, P = inner_comp(t)
                return t, P

        
        elif pulseShape == 'Rectangular':
            if self.lower_limit == -np.infty :
                self.lower_limit = 0
            if self.upper_limit == np.infty :
                self.upper_limit = max(2/self.Omega_1 ,  self.Mu + 2/self.Omega_2)*1.5
            t, dt = np.linspace(self.lower_limit,self.upper_limit,self.nBins,retstep=True)
            
            if self.Gamma_1 == self.Gamma_2:
                self.Gamma_1 += self.Gamma_1*0.0001     
            X = self.Delta_1*1j + self.Gamma_1/2
            Y = self.Delta_2 *1j + self.Gamma_2/2 - self.Gamma_1/2
            Z= (self.Delta_1 + self.Delta_2)*1j + self.Gamma_2/2
            const = self.Gamma_1*self.Gamma_2*self.Omega_1*self.Omega_2/4
            if self.Mu <(2/self.Omega_1) and (2/self.Omega_1) <= (2/self.Omega_2 + self.Mu):
                # print('CASE 1')
                t1 = t[np.where(t<self.Mu)]
                P1 = np.zeros_like(t1)
            
                t2 = t[np.where((self.Mu<= t) & (t<= (2/self.Omega_1)))]
                P2 = (const*np.exp(-self.Gamma_2*t2))*np.abs( (np.exp(Y*t2) - np.exp(Y*self.Mu))/(Y*X) - (np.exp(Z*t2) - np.exp(Z*self.Mu))/(X*Z) )**2
            
                t3 = t[np.where( ((2/self.Omega_1)< t)& (t<= (2/self.Omega_2 + self.Mu)) )]   
                P3 = const*np.exp(-self.Gamma_2*t3) * np.abs((np.exp(Z*2/self.Omega_1) - np.exp(Z*self.Mu))/(X*Z) - ( (np.exp(Y*2/self.Omega_1) -np.exp(Y*self.Mu))/(X*Y) ) + ((np.exp(X*2/self.Omega_1) -1)*(np.exp(Y*t3) - np.exp(Y*2/self.Omega_1)) )/(X*Y) )**2
        
                
                t4 = t[np.where(t>(2/self.Omega_2 + self.Mu))]
                P4 = const*np.exp(-self.Gamma_2*t4) * np.abs((np.exp(Z*2/self.Omega_1) - np.exp(Z*self.Mu))/(X*Z) - ( (np.exp(Y*2/self.Omega_1) - np.exp(Y*self.Mu))/(X*Y) )  + ((np.exp(X*2/self.Omega_1) -1)*(np.exp(Y*(2/self.Omega_2 +self.Mu)) - np.exp(Y*2/self.Omega_1)) )/(X*Y) )**2
        
                t = np.append(t1,t2)
                t = np.append(t,t3)
                t = np.append(t,t4)
                P = np.append(P1,P2)
                P = np.append(P,P3)
                P = np.append(P,P4)
                
            elif self.Mu<=(2/self.Omega_2+self.Mu) and (2/self.Omega_2+self.Mu) <= (2/self.Omega_1):
                # print('CASE 2')
                
                t1 = t[np.where(t<self.Mu)]
                P1 = np.zeros_like(t1)
        
                t2 = t[np.where((self.Mu<= t) & (t<= 2/self.Omega_2+self.Mu))]
                P2 = const*np.exp(-self.Gamma_2*t2)*np.abs( (np.exp(Z*t2) - np.exp(Z*self.Mu))/(X*Z) - (np.exp(Y*t2) - np.exp(Y*self.Mu))/(X*Y) )**2
                
                t3 = t[np.where((self.Mu+2/self.Omega_2<= t))]
                P3 = const*np.exp(-self.Gamma_2*t3)*np.abs( (np.exp(Z*(self.Mu+2/self.Omega_2)) - np.exp(Z*self.Mu))/(X*Z) - (np.exp(Y*(self.Mu+2/self.Omega_2)) - np.exp(Y*self.Mu))/(X*Y))**2
        
                t = np.append(t1,t2)
                t = np.append(t,t3)
                P = np.append(P1,P2)
                P = np.append(P,P3)
                
            elif (2/self.Omega_1 <= self.Mu) and (self.Mu<= 2/self.Omega_2+self.Mu):        
                # print('CASE 3')
                
                t1 = t[np.where(t<self.Mu)]
                P1 = np.zeros_like(t1)
        
                t2 = t[np.where((self.Mu<= t) & (t<= 2/self.Omega_2+self.Mu))]
                P2 = const*np.exp(-self.Gamma_2*t2)*np.abs(  (np.exp(2*X/self.Omega_1) - 1)*(np.exp(Y*t2) - np.exp(Y*self.Mu))/(X*Y) )**2
                
                t3 = t[np.where((self.Mu+2/self.Omega_2<= t))]
                P3 = const*np.exp(-self.Gamma_2*t3)*np.abs(  (np.exp(2*X/self.Omega_1) - 1)*(np.exp(Y*(2/self.Omega_2+self.Mu)) - np.exp(Y*self.Mu))/(X*Y) )**2
        
                t = np.append(t1,t2)
                t = np.append(t,t3)  
                P = np.append(P1,P2)
                P = np.append(P,P3)

            return t, P



    def PHI (self,t2,t1):
            return self.xi(t2, Omega=self.Omega_2, Mu=self.Mu) * self.xi(t1, Omega=self.Omega_1, Mu=0)
    def P_unidirection_chunk(self,chunk):
            results=[]
            for t in chunk:
                intg = quad( lambda t2: np.exp(-(self.Delta_2+self.Gamma_2/2)*(t - t2))*
                            quad( lambda t1: np.exp(-self.Gamma_1*(t2- t1)/2 + self.Delta_1*t1)* (self.PHI(t2,t1) + self.PHI(t1,t2) )  
                                 , self.lower_limit, t2,epsabs=self.tol, complex_func=True )[0] 
                            ,self.lower_limit, t , epsabs=self.tol, complex_func=True)[0]
                result = (self.Gamma_1 * self.Gamma_2 ) * np.abs(intg)**2
                results.append((t,result))
                # print(results)
            return results  

    def P_unidirection (self):
        N_PHI = quad(lambda t2: quad( lambda t1: np.abs(self.PHI(t2,t1))**2 +  np.conjugate(self.PHI(t1,t2))*self.PHI(t2,t1)
                                     , self.lower_limit, np.infty, complex_func=True)[0], self.lower_limit, np.infty, epsrel=self.tol, complex_func=True)[0]
        N_PHI = np.abs(N_PHI)

        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -2
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 6 + self.Mu
        t = np.linspace(self.lower_limit, self.upper_limit , self.nBins) / np.min([self.Gamma_1, self.Gamma_2])
        
        num_processes = self.N_threads
        chunk_size = len(t) // num_processes
        chunks = [t[i:i+chunk_size] for i in range(0, len(t), chunk_size)]
        pool = Pool(processes=num_processes)
        results = pool.map(self.P_unidirection_chunk, chunks)
        pool.close()
        pool.join()
        all_results = [item for sublist in results for item in sublist]
        t_val, P_val = zip(*all_results)
        return np.array(t_val), np.array(P_val)/N_PHI


    

    def P_coherent(self):
        def rhs(t, initial):
            Rho_ff, Rho_ef, Rho_gf, Rho_ee, Rho_ge, Rho_gg = initial 
            alpha_1 = self.xi(t,Omega = self.Omega_1,Mu = 0)
            alpha_1 =  np.sqrt(self.Gamma_1 * self.n_1) * alpha_1
            alpha_2 = self.phi(t,Omega = self.Omega_2,Mu = self.Mu)
            alpha_2 = np.sqrt(self.Gamma_2 * self.n_2) * alpha_2
            dRho_ffdt = - alpha_2 * (Rho_ef + np.conjugate(Rho_ef)) - self.Gamma_2 * Rho_ff
            dRho_efdt = self.Delta_2 * Rho_ef - alpha_1 * Rho_gf +  alpha_2 * (Rho_ff - Rho_ee) - (self.Gamma_1 + self.Gamma_2)*Rho_ef/2
            dRho_gfdt = (self.Delta_1 + self.Delta_2) * Rho_gf + alpha_1 * Rho_ef - alpha_2 * Rho_ge - self.Gamma_2 * Rho_gf/2
            dRho_eedt = - alpha_1 * (Rho_ge + np.conjugate(Rho_ge)) + alpha_2 * (Rho_ef + np.conjugate(Rho_ef)) - self.Gamma_1*Rho_ee  + self.Gamma_2*Rho_ff
            dRho_gedt = self.Delta_1 * Rho_ge + alpha_2 * Rho_gf + alpha_1 * (Rho_ee - Rho_gg) - self.Gamma_1 * Rho_ge / 2
            dRho_ggdt = alpha_1 * (Rho_ge + np.conjugate(Rho_ge)) + self.Gamma_1 * Rho_ee
            return [dRho_ffdt, dRho_efdt, dRho_gfdt, dRho_eedt, dRho_gedt, dRho_ggdt]
            
        initial_condition = [0 , 0 , 0 , 0 , 0 , 1 ]
        label_L , label_U = False , False
        if self.lower_limit == -np.infty :
            label_L = True
            self.lower_limit = -2
        if self.upper_limit == np.infty :
            label_U = True
            self.upper_limit = 6 + self.Mu

        t = np.linspace(self.lower_limit, self.upper_limit , self.nBins) / np.min([self.Gamma_1, self.Gamma_2])
        # t = (np.linspace(-4,10,self.nBins)+self.Mu)/np.min([self.Gamma_1,self.Gamma_2])
        solver = complex_ode(rhs)
        solver.set_initial_value(initial_condition, t[0])
        r = []
        for time in t[1:]:
            r.append(solver.integrate(time))
        r.insert(0, initial_condition)
        r = np.array(r)
        return t,  r[:,0]
        
        
