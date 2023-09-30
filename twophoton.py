from time import time
from util import *

import warnings 
warnings.filterwarnings('ignore')


def Pe_graphs():
	
	fig, ax = plt.subplots(2,2, figsize=(20, 15))
	fig.tight_layout(pad=8.0)
	
	##################################################################################
	title = "Two photons, rectangular pulse implemented by continuous approximation:"#
	##################################################################################

	w = .01
	rectangular = lambda a: lambda s: a**-.5 * ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51))

	t=time()
	for a in range(1,4):
		ax[0,0].plot(tphs.trange, tphs.P(a), label=f'$a={a}$')
	ax[0,0].title.set_text(title)
	ax[0,0].legend()
	ax[0,0].text(0,-.15,f'time = {time()-t}')

	###############################################################################
	title = "Two photons, rectangular pulse implemented by the support attribute:"#
	###############################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51))

	t=time()
	for a in range(1,4):
		ax[0,1].plot(tphs.trange, tphs.P(a), label=f'$a={a}$')
	ax[0,1].title.set_text(title)
	ax[0,1].legend()
	ax[0,1].text(0,-.15,f'time = {time()-t}')


	######################################################################################################
	title = "Two photons, rectangular pulse implemented by continuous approximation, xiexp_int provided:"#
	######################################################################################################

	w = .01
	rectangular = lambda a: lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51),
						 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1) )

	t=time()
	for a in range(1,4):
		ax[1,0].plot(tphs.trange, tphs.P(a), label=f'$a={a}$')
	ax[1,0].title.set_text(title)
	ax[1,0].legend()
	ax[1,0].text(0,-.15,f'time = {time()-t}')

	###################################################################################################
	title = "Two photons, rectangular pulse implemented by the support attribute, xiexp_int provided:"#
	###################################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51),
						 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1) )

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
	
	##################################################################################
	title = "Two photons, rectangular pulse implemented by continuous approximation:"#
	##################################################################################

	w = .01
	rectangular = lambda a: lambda s: a**-.5 * ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51))

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[0,0].plot(a_range,y,'.')
	ax[0,0].title.set_text(title)
	ax[0,0].text(0,.25,f"time = {time()-t}")

	###############################################################################
	title = "Two photons, rectangular pulse implemented by the support attribute:"#
	###############################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51))

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[0,1].plot(a_range,y,'.')
	ax[0,1].title.set_text(title)
	ax[0,1].text(0,.25,f"time = {time()-t}")

	######################################################################################################
	title = "Two photons, rectangular pulse implemented by continuous approximation, xiexp_int provided:"#
	######################################################################################################

	w = .01
	rectangular = lambda a: lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51),
						 xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1) )

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[1,0].plot(a_range,y,'.')
	ax[1,0].title.set_text(title)
	ax[1,0].text(0,.25,f"time = {time()-t}")

	###################################################################################################
	title = "Two photons, rectangular pulse implemented by the support attribute, xiexp_int provided:"#
	###################################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51),
						 xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1) )

	t=time()
	a_range = np.linspace(.2,3,15)
	y = np.vectorize(tphs.Pmax)(a_range)
	ax[1,1].plot(a_range,y,'.')
	ax[1,1].title.set_text(title)
	ax[1,1].text(0,.25,f"time = {time()-t}")
	
	plt.show()


def best_parameters():
	
	#################################################################################
	print("Two photons, rectangular pulse implemented by continuous approximation:")#
	#################################################################################

	w = .01
	rectangular = lambda a: lambda s: a**-.5 * ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51))

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	##############################################################################
	print("Two photons, rectangular pulse implemented by the support attribute:")#
	##############################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51))

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	#####################################################################################################
	print("Two photons, rectangular pulse implemented by continuous approximation, xiexp_int provided:")#
	#####################################################################################################

	w = .01
	rectangular = lambda a: lambda s: ( 1 + ((a-s)/w)/(((a-s)/w)**2+1)**.5 ) / 2 if s>0 else 0
	#rectangular = lambda a: lambda s: 0 if s<0 or s>a else (1 if s<.9*a else 10*(a-s)/a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51),
						 xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1) )

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	##################################################################################################
	print("Two photons, rectangular pulse implemented by the support attribute, xiexp_int provided:")#
	##################################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	tphs = TwoPhotonState(rectangular, 1, np.linspace(0,5,51),
						 xiexp_int = lambda a: lambda s: (s>0) * 2*(np.exp(np.minimum(s,a)/2) - 1) )

	t=time()
	print(tphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)


if __name__ == "__main__":
	
	#Pe_graphs()
	#Pmax_graphs()
	best_parameters()
