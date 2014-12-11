import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer

class coClusteringAdjacency(object):
	"""Co-clustering without preferences value (adjacency matrix)"""
	def __init__(self, data, K, L, maxIter, tol):
		self.tol = tol
		self.nbIter = 0
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
		C = initClusters.fit_predict(self.data)
		## init clusters for Y with kmeans
		initClusters.n_clusters=L
		D = initClusters.fit_predict(self.data.T)
		## intialization of q 
		binarizer = LabelBinarizer()
		self.Q_xc = binarizer.fit_transform(C)
		self.Q_yd = binarizer.fit_transform(D)
		# self.Q_xc = np.zeros((self.N, K)) # = Q(x,c)
		# self.Q_yd = np.zeros((self.M, L)) # = Q(y,d)
		# for c in range(self.K):
			# self.Q_xc[:,c] = (C == c)
		# for d in range(self.L):
			# self.Q_yd[:,d] = (D == d)
		## intializations proba cluster  ###########################
		self.P_c = np.sum(self.Q_xc, axis=0) / self.N
		self.P_d = np.sum(self.Q_yd, axis=0) / self.M
		# self.P_c = np.zeros((self.K,1))
		# self.P_d = np.zeros((self.L,1))
		# for c in range(self.K):
		# 	self.P_c[c] = C[C == c].shape[0]/self.N
		# for d in range(self.L):
		# 	self.P_d[d] = D[D == d].shape[0]/self.M
		## fill p_phi ###########################
		self.p_phi = np.zeros((self.N,self.M,self.K,self.L))
		for j in range(self.M):
			for c in range(self.K):
				self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c].reshape(self.N, 1), self.Q_yd[j,:].reshape(1, self.L))
		#self.p_phi = self.p_phi / np.sum(self.p_phi) # Normalization
		## intialization of phi  ###########################
		self.phi = np.zeros((self.K,self.L))
		self.P_xc = np.sum(self.p_phi, axis=(1,3))
		self.P_xc = self.P_xc / np.sum(self.P_xc, axis=(1)).reshape(self.N,1)
		self.P_yd = np.sum(self.p_phi, axis=(0,2)) 
		self.P_yd = self.P_yd / np.sum(self.P_yd, axis=(1)).reshape(self.M,1)
		DenominatorX = np.sum(self.P_xc * self.nx.reshape(self.N,1), axis=0)
		DenominatorY = np.sum(self.P_yd * self.ny.reshape(self.M,1), axis=1)
		for c in range(self.K):
			for d in range(self.L):
				self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d]) + 1e-22

	def fit(self):
		old_Pc = self.P_c + 1
		old_Pd = self.P_d + 1
		while(np.sum(np.abs(self.P_c - old_Pc)) + np.sum(np.abs(self.P_d - old_Pd)) and self.nbIter < self.maxIter):
			self.nbIter = self.nbIter + 1
			## E-Step
			self.Q_xc = self.P_c * np.exp(np.dot(self.data, np.dot(self.Q_yd, np.log(self.phi).T)))
			self.Q_xc = self.Q_xc / np.sum(self.Q_xc,1).reshape(self.N,1) ## normalization
			self.Q_yd = self.P_d * np.exp(np.dot(self.data.T, np.dot(self.Q_xc, np.log(self.phi))))
			self.Q_yd = (self.Q_yd / np.sum(self.Q_yd,1).reshape(self.M,1)) ## normalization
			## Update p_phi
			for j in range(self.M):
				for c in range(self.K):
					self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c].reshape(self.N, 1), self.Q_yd[j,:].reshape(1, self.L))
			## inner M-step:
			DenominatorX = np.sum(np.sum(self.p_phi, axis=(1,3)) * self.nx.reshape(self.N,1), axis=0)
			DenominatorY = np.sum(np.sum(self.p_phi, axis=(0,2)) * self.ny.reshape(self.M,1), axis=1)
			for c in range(self.K):
				for d in range(self.L):
					self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d]) + 1e-22 ## Use some rounding ?
			## Update of P_c and P_d:
			old_Pc = self.P_c
			self.P_c = np.sum(self.p_phi, axis=(0,1,3))
			self.P_c = self.P_c / np.sum(self.P_c)
			old_Pd = self.P_d
			self.P_d = np.sum(self.p_phi, axis=(0,1,2))
			self.P_d = self.P_d / np.sum(self.P_d)
		if(self.nbIter == self.maxIter):
			print("The algorithm reached the max. number of iteration without achieving convergence")
		else:
			print("The algorithm converged in %i iterations" %self.nbIter)

	# def getClusters(self):
	# 	coord = np.where(np.round(self.p_phi) == 1)
	# 	C = 
	# 	return(C,D)






















	
