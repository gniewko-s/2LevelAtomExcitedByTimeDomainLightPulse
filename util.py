import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from collections.abc import Callable
from operator import itemgetter
import matplotlib.pyplot as plt
from multiprocessing import Pool


def print_args(f):
    def new(*args,**kwargs):
        v = f(*args,**kwargs)
        print(args[1:],end=' ')
        print(v)
        return v
    return new

class General:
    
    def __init__(self, xi:Callable, Gamma, trange:np.array, **kwargs):
        self.Gamma = Gamma
        self.xi = xi            # xi: *args: PhotonPulseShape(t)
        self.trange = trange
        self.kwargs=kwargs
    
    @staticmethod
    def irange(*list_of_supports):
        if len(list_of_supports) == 0:
            return (-np.inf,np.inf)
        else:
            support1 = list_of_supports[0]
            support2 = General.irange(*list_of_supports[1:])
            if support1 is None:
                return support2
            else:
                tmin = np.maximum(support1[0],support2[0])
                tmax = np.minimum(support1[1],support2[1])
                return (tmin,tmax) if tmin < tmax else (0,0) # better universal empty set
    
    @staticmethod
    def quads(f,*list_of_supports, complex_func=False):
        r = General.irange(*list_of_supports)
        if r == (0,0):
            return (0,0)
        else:
            return quad(f,*r, complex_func=complex_func)
    
    @staticmethod
    def _norm(f,si,sf):
        return (quad(lambda s: np.abs(f(s))**2,si,sf)[0])**.5
    
    def norm(self,*args):
        si, sf = s if not ( s := self.xi.support(*args) ) is None else (-np.inf,np.inf)
        return self._norm(self.xi(*args),si,sf)
    
    def P(self,*args,trange=None):
        if trange is None:
            trange = self.trange
        raise NotImplementedError
    
    def P_onepoint(self,*args,t):
        return self.P(*args,trange=np.array([self.trange[0],t]))[-1]
    
    @staticmethod
    def antiderivative(fun: Callable, a: np.array, C: 'integration constant' = 0, support = None, complex_func=False):
        if support is None:
            return np.cumsum([C]+[quad(fun,x,y,complex_func=complex_func)[0] for x,y in zip(a[:-1],a[1:])])
        else:
            si,sf = support
            if sf <= a[-1]:
                before, rrange, after = np.split(a,[np.argmax(a>=si),np.argmax(a>=sf)])
            else:
                before, rrange = np.split(a,[np.argmax(a>=si)])
                after = np.array([])
            before = C*np.ones(before.shape)
            if rrange.shape[0]:
                Ci = quad(fun,si,rrange[0],complex_func=complex_func)[0] if before.shape[0] else 0
                Cf = quad(fun,rrange[-1],sf,complex_func=complex_func)[0] if after.shape[0] else 0
                rrange = General.antiderivative(fun, rrange, Ci+C, complex_func=complex_func)
                after = (rrange[-1]+Cf)*np.ones(after.shape) 
            else:
                after = quad(fun,si,sf,complex_func=complex_func)[0]*np.ones(after.shape)
            return np.concatenate((before,rrange,after))
   
    """@print_args
    def Pmax(self,*args, M: int = 5):
        #t = time()
        def gen():
            for _ in range(M):
                while True:
                    res = minimize(lambda x: -self.P_onepoint(*args,t=x[0]),
                                   self.trange[0] + self.trange[-1] * np.random.rand(), method='Nelder-Mead')
                    if res.success:
                        break
                yield res.fun
        m = min(gen())
        #print(m,time()-t)
        return -m"""
    
    def Pmax(self,*args, M: int = 5):
        global success
        def success(i):
            while True:
                res = minimize(lambda x: -self.P_onepoint(*args,t=x[0]),
                    self.trange[0] + self.trange[-1] * np.random.rand(), method='Nelder-Mead')
                if res.success:
                    return res.fun
        return -min(Pool().map(success, range(M)))
        
    @print_args
    def best_shape(self,init: Callable, M: int = 5):
        global success
        def success(i):
            while True:
                res = minimize(lambda x: -self.P_onepoint(*x[:-1],t=x[-1]), init(), method='Nelder-Mead')
                if res.success:
                    return res
                print(res.message)
        return min(Pool().map(success, range(M)), key = itemgetter('fun'))

            
class SinglePhotonState(General):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not hasattr(self.xi,'support'):
            self.xi.support = lambda *args: None
        if 'xiexp_int' in self.kwargs:
            self.P = self.P1
        else:
            self.P = self.P0
    
    def P0(self, *args, trange=None):
        if trange is None:
            trange = self.trange
        N = self.norm(*args)
        xiexp = lambda s: self.xi(*args)(s)*np.exp(self.Gamma*s/2)
        int1 = self.antiderivative(
                                    xiexp, 
                                    trange, 
                                    support = self.xi.support(*args) if hasattr(self.xi,'support') else None
                                )
        return self.Gamma * np.exp(-self.Gamma*trange) * np.abs(int1 / N)**2
    
    def P1(self, *args, trange=None):
        if trange is None:
            trange = self.trange
        N = self.norm(*args)
        int1 = self.kwargs['xiexp_int'](*args)(trange)
        return self.Gamma * np.exp(-self.Gamma*trange) * np.abs(int1 / N)**2
    
class TwoPhotonState(SinglePhotonState):
    
    def P0(self, *args, trange=None):
        support = self.xi.support(*args)
        if trange is None:
            trange = self.trange
        t0 = self.trange[0]
        N = self.norm(*args)
        xi = lambda s: self.xi(*args)(s) / N
        xiexp = lambda s: xi(s)*np.exp(self.Gamma*s/2)
        int11 = self.antiderivative(xiexp,trange,support=support)
        C = self.quads(lambda s: np.abs(xi(s))**2,(-np.inf,t0),support, complex_func=True)[0]
        int12 = self.antiderivative(lambda s: np.abs(xi(s))**2,trange,C,support=support)
        int1 = np.abs( int11 )**2 * (1 - int12)
        quad_exp = lambda t0,t1: self.quads(xiexp,(t0,t1),support, complex_func=True)[0]
        mask = (lambda t1: 1) if support is None else lambda t1: (t1>support[0]) * (t1<support[1])
        int2_expr = lambda t, t1: np.exp(-self.Gamma*t1) * np.abs( 
                                        self.Gamma *\
                                        (q1 := quad_exp(t1,t)) * (q2 := quad_exp(t0,t1)) -\
                                        xiexp(t1) * (q1 + q2) * mask(t1) 
                                    )**2
        int2 = np.array([ quad(lambda t1: int2_expr(t,t1), t0, t)[0] for t in trange])
        return 2*self.Gamma*np.exp(-self.Gamma*trange) * ( int1 + int2 )
    
    def P1(self, *args, trange=None):
        support = self.xi.support(*args)
        if trange is None:
            trange = self.trange
        t0 = self.trange[0]
        N = self.norm(*args)
        xi = lambda s: self.xi(*args)(s) / N
        xiexp = lambda s: xi(s)*np.exp(self.Gamma*s/2)
        xiexp_int = self.kwargs['xiexp_int'](*args)
        int11 = xiexp_int(trange) / N
        C = self.quads(lambda s: np.abs(xi(s))**2,(-np.inf,t0),support, complex_func=True)[0]
        int12 = self.antiderivative(lambda s: np.abs(xi(s))**2,trange,C,support=support)
        int1 = np.abs( int11 )**2 * (1 - int12)
        quad_exp = lambda t0,t1: (xiexp_int(t1)-xiexp_int(t0)) / N
        mask = (lambda t1: 1) if support is None else lambda t1: (t1>support[0]) * (t1<support[1])
        int2_expr = lambda t, t1: np.exp(-self.Gamma*t1) * np.abs( 
                                        self.Gamma *\
                                        (q1 := quad_exp(t1,t)) * (q2 := quad_exp(t0,t1)) -\
                                        xiexp(t1) * (q1 + q2) * mask(t1) 
                                    )**2
        int2 = np.array([ quad(lambda t1: int2_expr(t,t1), t0, t)[0] for t in trange])
        return 2*self.Gamma*np.exp(-self.Gamma*trange) * ( int1 + int2 )


class TwoPhotonDistinguishable(General):
    
    # self.xi - lambda *args: (lambda s: ..., lambda s: ...)
    # self.xi.support = lambda *args: (( float, float ), ( float, float ) )
    def __init__(self,*iargs, **kwargs):
        super().__init__(*iargs, **kwargs)
        if not hasattr(self.xi,'support'):
            self.xi.support = lambda *args: (None,None)
        self.xi.mask = lambda *args: \
            [(lambda s:1) if su is None else (lambda s, su=su: (s>su[0]) * (s<su[1])) \
             for su in self.xi.support(*args)]
    
    def norm(self,*args):
        return [self._norm(f,*((-np.inf, np.inf) if s is None else s)) for f,s in zip(self.xi(*args), self.xi.support(*args))]
        
    def P(self, *args, trange=None):
        support_xi, support_phi = self.xi.support(*args)
        if trange is None:
            trange = self.trange
        t0 = self.trange[0]
        Nxi, Nphi = self.norm(*args)
        xi = lambda s: self.xi(*args)[0](s) / Nxi
        phi = lambda s: self.xi(*args)[1](s) / Nphi
        xiexp = lambda s: xi(s)*np.exp(self.Gamma*s/2)
        phiexp = lambda s: phi(s)*np.exp(self.Gamma*s/2)
        
        if 'xiexp_int' in self.kwargs:
            xiexp_int = self.kwargs['xiexp_int'](*args)
            int11xi = xiexp_int(trange) / Nxi
            quad_xiexp = lambda t0,t1: (xiexp_int(t1)-xiexp_int(t0)) / Nxi
        else:
            int11xi = self.antiderivative(xiexp,trange,support=support_xi)
            quad_xiexp = lambda t0,t1: self.quads(xiexp,(t0,t1),support_xi)[0]
        
        if 'phiexp_int' in self.kwargs:
            phiexp_int = self.kwargs['phiexp_int'](*args)
            int11phi = phiexp_int(trange) / Nphi
            quad_phiexp = lambda t0,t1: (phiexp_int(t1)-phiexp_int(t0)) / Nphi
        else:
            int11phi = self.antiderivative(phiexp,trange,support=support_phi)
            quad_phiexp = lambda t0,t1: self.quads(phiexp,(t0,t1),support_phi)[0]
        
        inner = self.quads(lambda s: xi(s).conjugate()*phi(s), (-np.inf, np.inf),support_xi,support_phi)[0]
        Cinner = self.quads(lambda s: xi(s).conjugate()*phi(s), (-np.inf,t0),support_xi,support_phi)[0]
        Cxi = self.quads(lambda s: np.abs(xi(s))**2,(-np.inf,t0),support_xi)[0]
        Cphi = self.quads(lambda s: np.abs(phi(s))**2,(-np.inf,t0),support_phi)[0]
        int12xi = self.antiderivative(lambda s: np.abs(xi(s))**2,trange,Cxi,support=support_xi)
        int12phi = self.antiderivative(lambda s: np.abs(phi(s))**2,trange,Cphi,support=support_phi)
        int12inner = self.antiderivative(lambda s: xi(s).conjugate()*phi(s),
                                         trange,
                                         Cinner,
                                         support=self.irange(support_xi,support_phi),
                                         complex_func=True
                                        )
        
        int1 = np.abs( int11xi )**2 * (1 - int12phi)\
                + np.abs( int11phi )**2 * (1 - int12xi)\
                + 2 * ( int11phi.conjugate() * int11xi * (inner - int12inner) ).real
        
        mask_xi, mask_phi = self.xi.mask(*args)
        int2_expr = lambda t, t1: np.exp(-self.Gamma*t1) * np.abs( 
                                        self.Gamma * (q1 := quad_xiexp(t1,t)) * (q2 := quad_phiexp(t0,t1)) +\
                                        self.Gamma * (q3 := quad_phiexp(t1,t)) * (q4 := quad_xiexp(t0,t1)) -\
                                        phiexp(t1) * (q1 + q4) * mask_phi(t1) -\
                                        xiexp(t1) * (q2 + q3) * mask_xi(t1)
                                    )**2
        int2 = np.array([ quad(lambda t1: int2_expr(t,t1), t0, t)[0] for t in trange])
        return self.Gamma*np.exp(-self.Gamma*trange) * ( int1 + int2 ) / (1 + np.abs(inner))
