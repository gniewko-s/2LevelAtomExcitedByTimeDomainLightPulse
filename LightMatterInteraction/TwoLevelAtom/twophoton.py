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
