import numpy as np
from sklearn.cluster import KMeans

class coClusteringAdjacency(object):
	"""Co-clustering without preferences value (adjacency matrix)"""
	def __init__(self, data, K, L, maxIter):
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
		C = initClusters.predict(self.data)
		## init clusters for Y with kmeans
		initClusters.n_clusters=L
		initClusters.fit(self.data.T)
		D = initClusters.predict(self.data.T)
		## intialization of q 
		self.Q_xc = np.zeros((self.N, K)) # = Q(x,c)
		self.Q_yd = np.zeros((self.M, L)) # = Q(y,d)
		for c in range(self.K):
			self.Q_xc[:,c] = (C == c)
		for d in range(self.L):
			self.Q_yd[:,d] = (D == d)
		## fill p_phi
		self.p_phi = np.zeros((self.N,self.M,self.K,self.L))
		for j in range(self.M):
			for c in range(self.K):
				self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c].reshape(self.N, 1), self.Q_yd[j,:].reshape(1, self.L))
		self.p_phi = self.p_phi / np.sum(self.p_phi) # Normalization
		## intializations proba cluster  ###########################
		self.p_c = np.zeros((self.K,1))
		self.p_d = np.zeros((self.L,1))
		for c in range(self.K):
			self.p_c[c] = C[C == c].shape[0]/self.N
		for d in range(self.L):
			self.p_d[d] = D[D == d].shape[0]/self.M
		## intialization of phi  ###########################
		self.phi = np.zeros((self.K,self.L))
		DenominatorX = np.sum(np.sum(self.p_phi, axis=(1,3)) * self.nx.reshape(self.N,1), axis=0)
		DenominatorY = np.sum(np.sum(self.p_phi, axis=(0,2)) * self.ny.reshape(self.M,1), axis=1)
		for c in range(self.K):
			for d in range(self.L):
				self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d]) + 1e-22

	def fit(self):
		for i in range(self.maxIter):
			self.i = i
			## Mise à jour des proba cluster
			self.Q_xc = np.dot(self.data, np.dot(self.Q_yd, np.log(self.phi).T)) ### CAREFUL if null value in phi
			# self.Q_xc = self.Q_xc / np.sum(self.Q_xc)
			self.Q_yd = np.dot(self.data.T, np.dot(self.Q_xc, np.log(self.phi)))
			# print(self.Q_xc)
			# print(self.Q_yd)
			## Update p_phi
			for j in range(self.M):
				for c in range(self.K):
					self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c].reshape(self.N, 1), self.Q_yd[j,:].reshape(1, self.L))
			self.p_phi = self.p_phi / np.sum(self.p_phi) # Normalization
			#if(np.sum(self.p_phi) != 1):
			#	raise Exception("np.sum(self.p_phi) = %i"%np.sum(self.p_phi))
			## inner M-step:
			DenominatorX = np.sum(np.sum(self.p_phi, axis=(1,3)) * self.nx.reshape(self.N,1), axis=0)
			DenominatorY = np.sum(np.sum(self.p_phi, axis=(0,2)) * self.ny.reshape(self.M,1), axis=1)
			for c in range(self.K):
				for d in range(self.L):
					self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d]) + 1e-22
			## Update of p_c and p_d:
			self.p_c = np.sum(self.p_phi, axis=(0,1,3))
			self.p_c = self.p_c / np.sum(self.p_c)
			self.p_d = np.sum(self.p_phi, axis=(0,1,2))
			self.p_d = self.p_d / np.sum(self.p_d)

	def getClusters(self):
		D = np.sum(self.p_phi, axis=(0,2))
		C = np.sum(self.p_phi, axis=(1,3))
		return(C,D)






















	
