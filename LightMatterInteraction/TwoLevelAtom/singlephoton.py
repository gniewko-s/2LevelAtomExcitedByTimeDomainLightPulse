     
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
            t1 = ttSteps[ttSteps<=0]
            t2 = ttSteps[ttSteps>0]
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
