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
