import pandas as pd
import scipy.stats as sctat
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans


class coclustering_1:
# co-clustering without preferences value

## zone de test
##ber = 0.2
##berno = sctat.bernoulli(ber)
##data = berno.rvs((3,10))
##data1 = berno.rvs((20,10))

	def __init__(self, phi, data, nx, ny, 
			K, ,L,
			Q_xc, Q_yd, p_phi, phi,
			p_c, p_d, maxIter):
		self.maxIter = maxIter
		self.data = data
		(self.N, self.M) = self.data.shape
		## count marginal
		self.nx = np.sum(self.data, 1)
		self.ny = np.sum(self.data, 0)
		self.K = K
		self.L = L
		### intialization of p_phi  ###########################
		## init clusters for X with kmeans
 		initClusters = KMeans(n_clusters=K)	
		initClusters.fit(self.data)	
		C = self.initClusters.predict(self.data)
		## init clusters for Y with kmeans
		initClusters = KMeans(n_clusters=L)	
		initClusters.fit(np.t(self.data))
		D = self.initClusters.predict(self.data)
		## intialization of q 
		self.Q_xc = np.zeros((self.N, K)) # = Q(x,c)		
		self.Q_yd = np.zeros((self.M, L)) # = Q(y,d)
		for c in range(self.K):
			self.Q_xc[:,j] = (C == c)
		for d in range(self.L):
			self.Q_yd(i,j) = (D == d)
		## fill p_phi
		self.p_phi = np.zeros((self.N,self.M,self.K,self.L))
		for j in range(self.M):
			for c in range(self.K):
				self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c], self.Q_yd[j,:]
		## intializations proba cluster  ###########################
		self.p_c = np.zeros((self.K,1))
		self.p_d = np.zeros((self.L,1))
		for c in range(self.K):
			self.p_c[c] = C[C == c].shape[0]/self.N
		for d in range(self.L):
			self.p_d[d] = D[D == d].shape[0]/self.M
		## intialization of phi  ###########################
		self.phi = np.zeros((self.K,self.L))
		DenominatorX = np.sum(np.sum(self.p_phi, axis=(1,3)) * nx, axis=0)
		DenominatorY = np.sum(np.sum(self.p_phi, axis=(0,2)) * ny, axis=1)
		for c in range(self.K):
			for d in range(self.L):
				self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d])

	def EM_step(self,maxIter):
		for i in range(maxIter)]:
			## Mise à jour des proba cluster
			self.Q_xc = np.dot(self.data, np.dot(self.Q_yd, np.log(self.phi).T)) ### CAREFUL if null value in phi
			self.Q_yd = np.dot(self.data.T, np.dot(self.Q_xc, np.log(self.phi)))
			## Update p_phi -> ÇA COUTE UNE BURNE
			for j in range(self.M):
				for c in range(self.K):
					self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c], self.Q_yd[j,:]
			## inner M-step:
			DenominatorX = np.sum(np.sum(self.p_phi, axis=(1,3)) * nx, axis=0)
			DenominatorY = np.sum(np.sum(self.p_phi, axis=(0,2)) * ny, axis=1)
			for c in range(self.K):
				for d in range(self.L):
					self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d])
			## Update of p_c and p_d:
			self.p_c = np.sum(self.p_phi, axis=(0,1,3))
			self.p_c = self.p_c / np.sum(self.p_c)
			self.p_d = np.sum(self.p_phi, axis=(0,1,2))
			self.p_d = self.p_d / np.sum(self.p_d)

	def getClusters(self):
		D = np.sum(self.p_phi, axis=(0,2))
		C = np.sum(self.p_phi, axis=(1,3))
		return(C,D)






















	
