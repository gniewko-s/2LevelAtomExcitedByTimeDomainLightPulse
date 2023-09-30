from time import time
from util import *

import warnings 
warnings.filterwarnings('ignore')


def Pe_graphs():
	
	fig, ax = plt.subplots(2,2, figsize=(20, 15))
	fig.tight_layout(pad=8.0)
	
	##################################################################################
	title = "Single photon, rectangular pulse implemented by conditional expression:"#
	##################################################################################

	rectangular = lambda a: lambda s: a**-.5 if s>0 and s<a else 0
	# rectangular pulse, one parameter a = width of the pulse

	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101))

	t=time()
	for a in range(1,9):
		ax[0,0].plot(sphs.trange, sphs.P(a), label=f'$a={a}$')
	ax[0,0].title.set_text(title)
	ax[0,0].legend()
	ax[0,0].text(0,-.15,f'time = {time()-t}')

	#################################################################################
	title = "Single photon, rectangular pulse implemented by the support attribute:"#
	#################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101))

	t=time()
	for a in range(1,9):
		ax[0,1].plot(sphs.trange, sphs.P(a), label=f'$a={a}$')
	ax[0,1].title.set_text(title)
	ax[0,1].legend()
	ax[0,1].text(0,-.15,f'time = {time()-t}')

	######################################################################################################
	title = "Single photon, rectangular pulse implemented by conditional expression, xiexp_int provided:"#
	######################################################################################################

	rectangular = lambda a: lambda s: 1 if s>0 and s<a else 0
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101), 
							 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1))

	t=time()
	for a in range(1,9):
		ax[1,0].plot(sphs.trange, sphs.P(a), label=f'$a={a}$')
	ax[1,0].title.set_text(title)
	ax[1,0].legend()
	ax[1,0].text(0,-.15,f'time = {time()-t}')

	#####################################################################################################
	title = "Single photon, rectangular pulse implemented by the support attribute, xiexp_int provided:"#
	#####################################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101), 
							 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1))

	t=time()
	for a in range(1,9):
		ax[1,1].plot(sphs.trange, sphs.P(a), label=f'$a={a}$')
	ax[1,1].title.set_text(title)
	ax[1,1].legend()
	ax[1,1].text(0,-.15,f'time = {time()-t}')
	
	plt.show()


def Pmax_graphs():
	
	fig, ax = plt.subplots(2,2, figsize=(20, 15))
	fig.tight_layout(pad=8.0)

	##################################################################################
	title = "Single photon, rectangular pulse implemented by conditional expression:"#
	##################################################################################

	rectangular = lambda a: lambda s: a**-.5 if s>0 and s<a else 0
	# rectangular pulse, one parameter a = width of the pulse
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101))

	t=time()
	a_range = np.linspace(.1,5,99)
	y = np.vectorize(sphs.Pmax)(a_range)
	ax[0,0].plot(a_range,y,'.')
	ax[0,0].title.set_text(title)
	ax[0,0].text(0,-.05,f"time = {time()-t}")

	#################################################################################
	title = "Single photon, rectangular pulse implemented by the support attribute:"#
	#################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101))

	t=time()
	a_range = np.linspace(.1,5,99)
	y = np.vectorize(sphs.Pmax)(a_range)
	ax[0,1].plot(a_range,y,'.')
	ax[0,1].title.set_text(title)
	ax[0,1].text(0,-.05,f"time = {time()-t}")

	######################################################################################################
	title = "Single photon, rectangular pulse implemented by conditional expression, xiexp_int provided:"#
	######################################################################################################

	rectangular = lambda a: lambda s: 1 if s>0 and s<a else 0
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101), 
							 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1))

	t=time()
	a_range = np.linspace(.1,5,99)
	y = np.vectorize(sphs.Pmax)(a_range)
	ax[1,0].plot(a_range,y,'.')
	ax[1,0].title.set_text(title)
	ax[1,0].text(0,-.05,f"time = {time()-t}")

	#####################################################################################################
	title = "Single photon, rectangular pulse implemented by the support attribute, xiexp_int provided:"#
	#####################################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101), 
							 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1))

	t=time()
	a_range = np.linspace(.1,5,99)
	y = np.vectorize(sphs.Pmax)(a_range)
	ax[1,1].plot(a_range,y,'.')
	ax[1,1].title.set_text(title)
	ax[1,1].text(0,-.05,f"time = {time()-t}")
	
	plt.show()


def best_parameters():
	
	#################################################################################
	print("Single photon, rectangular pulse implemented by conditional expression:")#
	#################################################################################

	rectangular = lambda a: lambda s: a**-.5 if s>0 and s<a else 0
	# rectangular pulse, one parameter a = width of the pulse
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101))

	t=time()
	print(sphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	################################################################################
	print("Single photon, rectangular pulse implemented by the support attribute:")#
	################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101))

	t=time()
	print(sphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	#####################################################################################################
	print("Single photon, rectangular pulse implemented by conditional expression, xiexp_int provided:")#
	#####################################################################################################

	rectangular = lambda a: lambda s: 1 if s>0 and s<a else 0
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101), 
							 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1))

	t=time()
	print(sphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)

	#####################################################################################################
	print("Single photon, rectangular pulse implemented by the support attribute, xiexp_int provided:")#
	#####################################################################################################

	rectangular = lambda a: lambda s: 1
	rectangular.support = lambda a:(0,a)
	sphs = SinglePhotonState(rectangular, 1, np.linspace(0,10,101), 
							 xiexp_int = lambda a: lambda s: (s>0) * 2 * (np.exp(np.minimum(s,a)/2) - 1))

	t=time()
	print(sphs.best_shape(lambda: 3*np.random.rand(2)))
	print(time()-t)


if __name__=="__main__":
	
	#Pe_graphs()
	Pmax_graphs()
	#best_parameters()
