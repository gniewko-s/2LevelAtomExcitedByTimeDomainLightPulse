from time import time
from util import *

import warnings 
warnings.filterwarnings('ignore')


def Pe_graphs():
	
	fig, ax = plt.subplots(2,2, figsize=(20, 15))
	fig.tight_layout(pad=8.0)
	
	#################################################################################################
	title = "Two distingishable photons, rectangular pulse implemented by continuous approximation:"#
	#################################################################################################

	w = .01
	rectangular = lambda a: 2*(lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0 ,)
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51))

	t=time()
	for a in range(1,4):
		ax[0,0].plot(tphs.trange, tphs.P(a), label=f'$a={a}$')
	ax[0,0].title.set_text(title)
	ax[0,0].legend()
	ax[0,0].text(0,-.15,f'time = {time()-t}')

	##############################################################################################
	title = "Two distiguishable photons, rectangular pulse implemented by the support attribute:"#
	##############################################################################################

	rectangular = lambda a: 2*(lambda s: 1,)
	rectangular.support = lambda a: 2*((0,a),)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51))

	t=time()
	for a in range(1,4):
		ax[0,1].plot(tphs.trange, tphs.P(a), label=f'$a={a}$')
	ax[0,1].title.set_text(title)
	ax[0,1].legend()
	ax[0,1].text(0,-.15,f'time = {time()-t}')

	######################################################################################################################
	title = "Two distinguishable photons, rectangular pulse implemented by continuous approximation, xiexp_int provided:"#
	######################################################################################################################

	w = .01
	rectangular = lambda a: 2*(lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0 ,)
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51),
								   xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   phiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   )

	t=time()
	for a in range(1,4):
		ax[1,0].plot(tphs.trange, tphs.P(a), label=f'$a={a}$')
	ax[1,0].title.set_text(title)
	ax[1,0].legend()
	ax[1,0].text(0,-.15,f'time = {time()-t}')

	###################################################################################################
	title = "Two photons, rectangular pulse implemented by the support attribute, xiexp_int provided:"#
	###################################################################################################

	rectangular = lambda a: 2*(lambda s: 1,)
	rectangular.support = lambda a: 2*((0,a),)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51),
								   xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   phiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   )

	t=time()
	for a in range(1,4):
		ax[1,1].plot(tphs.trange, tphs.P(a), label=f'$a={a}$')
	ax[1,1].title.set_text(title)
	ax[1,1].legend()
	ax[1,1].text(0,-.15,f'time = {time()-t}')

	plt.show()


def Pmax_graphs():
	
	fig, ax = plt.subplots(2,2, figsize=(20, 15))
	fig.tight_layout(pad=8.0)
	
	#################################################################################################
	title = "Two distingishable photons, rectangular pulse implemented by continuous approximation:"#
	#################################################################################################

	w = .01
	rectangular = lambda a: 2*(lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0 ,)
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51))

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[0,0].plot(a_range,y,'.')
	ax[0,0].title.set_text(title)
	ax[0,0].text(0,.25,f"time = {time()-t}")

	##############################################################################################
	title = "Two distiguishable photons, rectangular pulse implemented by the support attribute:"#
	##############################################################################################

	rectangular = lambda a: 2*(lambda s: 1,)
	rectangular.support = lambda a: 2*((0,a),)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51))

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[0,1].plot(a_range,y,'.')
	ax[0,1].title.set_text(title)
	ax[0,1].text(0,.25,f"time = {time()-t}")

	######################################################################################################################
	title = "Two distinguishable photons, rectangular pulse implemented by continuous approximation, xiexp_int provided:"#
	######################################################################################################################

	w = .01
	rectangular = lambda a: 2*(lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0 ,)
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51),
								   xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   phiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   )

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[1,0].plot(a_range,y,'.')
	ax[1,0].title.set_text(title)
	ax[1,0].text(0,.25,f"time = {time()-t}")

	###################################################################################################
	title = "Two photons, rectangular pulse implemented by the support attribute, xiexp_int provided:"#
	###################################################################################################

	rectangular = lambda a: 2*(lambda s: 1,)
	rectangular.support = lambda a: 2*((0,a),)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51),
								   xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   phiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   )

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[1,1].plot(a_range,y,'.')
	ax[1,1].title.set_text(title)
	ax[1,1].text(0,.25,f"time = {time()-t}")
	
	plt.show()


def best_parameters():
	
	################################################################################################
	print("Two distingishable photons, rectangular pulse implemented by continuous approximation:")#
	################################################################################################

	w = .01
	rectangular = lambda a: 2*(lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0 ,)
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51))

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	#############################################################################################
	print("Two distiguishable photons, rectangular pulse implemented by the support attribute:")#
	#############################################################################################

	rectangular = lambda a: 2*(lambda s: 1,)
	rectangular.support = lambda a: 2*((0,a),)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51))

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	#####################################################################################################################
	print("Two distinguishable photons, rectangular pulse implemented by continuous approximation, xiexp_int provided:")#
	#####################################################################################################################

	w = .01
	rectangular = lambda a: 2*(lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0 ,)
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51),
								   xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   phiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   )

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	##################################################################################################
	print("Two photons, rectangular pulse implemented by the support attribute, xiexp_int provided:")#
	##################################################################################################

	rectangular = lambda a: 2*(lambda s: 1,)
	rectangular.support = lambda a: 2*((0,a),)
	tphs = TwoPhotonDistinguishable(rectangular, 1, np.linspace(0,5,51),
								   xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   phiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1),
								   )

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)


def optimisation_3par():
	
	Gamma = 1
	rectangular = lambda a,b,c: 2*(lambda s: 1,)
	rectangular.support = lambda a,b,c: ((c,c+b),(0,a))[::1]
	tphs = TwoPhotonDistinguishable(rectangular, Gamma, np.linspace(0,5,101),
									phiexp_int = lambda a,b,c, Gamma=Gamma: lambda s: 
										(s>=0) * 2/Gamma * (np.exp(Gamma * np.minimum(s,a)/2) - 1),
									xiexp_int = lambda a,b,c, Gamma=Gamma: lambda s: 
										(s>=c) * 2/Gamma * (np.exp(Gamma * np.minimum(s,b+c)/2) - np.exp(Gamma * c/2)),
								   )                              

	t=time()
	res = tphs.best_shape(lambda: np.array([5,1,1,3])*np.random.rand(4))
	print(res)
	print(time()-t)
	
	t=time()
	a,b,c = res.x[:-1]
	plt.plot(tphs.trange, tphs.P(a,b,c),label=f'$a={a}, b={b}, c={c}$')
	plt.legend()
	plt.show()
	print(time()-t)


if __name__ == "__main__":
	
	#Pe_graphs()
	#Pmax_graphs()
	#best_parameters()
	optimisation_3par()

