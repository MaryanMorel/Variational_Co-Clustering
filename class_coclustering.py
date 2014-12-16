import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer


class coClusteringAdjacency(object):
	"""Co-clustering without preferences value (adjacency matrix)"""
	def __init__(self, data, K, L, maxIter, tol, random_state=3):
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
		initClusters = KMeans(n_clusters=K, random_state=random_state)
		C = initClusters.fit_predict(self.data)
		## init clusters for Y with kmeans
		initClusters.n_clusters=L
		D = initClusters.fit_predict(self.data.T)
		## intialization of q 
		binarizer = LabelBinarizer()
		self.Q_xc = binarizer.fit_transform(C)
		self.Q_yd = binarizer.fit_transform(D)
		## intializations proba cluster  ###########################
		self.P_c = np.sum(self.Q_xc, axis=0) / self.N
		self.P_d = np.sum(self.Q_yd, axis=0) / self.M
		## fill p_phi ###########################
		self.p_phi = np.zeros((self.N,self.M,self.K,self.L))
		for j in range(self.M):
			for c in range(self.K):
				self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c].reshape(self.N, 1), self.Q_yd[j,:].reshape(1, self.L))
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
				self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d]) + 1e-50

	def fit(self):
		old_Pc = self.P_c + 1
		old_Pd = self.P_d + 1
		while(np.sum(np.abs(self.P_c - old_Pc)) + np.sum(np.abs(self.P_d - old_Pd)) > self.tol and self.nbIter < self.maxIter):
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
			self.P_xc = np.sum(self.p_phi, axis=(1,3))
			self.P_xc = self.P_xc / np.sum(self.P_xc, axis=(1)).reshape(self.N,1)
			self.P_yd = np.sum(self.p_phi, axis=(0,2)) 
			self.P_yd = self.P_yd / np.sum(self.P_yd, axis=(1)).reshape(self.M,1)
			DenominatorX = np.sum(self.P_xc * self.nx.reshape(self.N,1), axis=0)
			DenominatorY = np.sum(self.P_yd * self.ny.reshape(self.M,1), axis=1)
			for c in range(self.K):
				for d in range(self.L):
					self.phi[c,d] = np.sum(self.p_phi[:,:,c,d] * self.data)/(DenominatorX[c]*DenominatorY[d]) + 1e-50
			## Update of P_c and P_d:
			old_Pc = self.P_c
			self.P_c = np.sum(self.p_phi, axis=(0,1,3))
			self.P_c = self.P_c / np.sum(self.P_c)
			old_Pd = self.P_d
			self.P_d = np.sum(self.p_phi, axis=(0,1,2))
			self.P_d = self.P_d / np.sum(self.P_d)
		if(self.nbIter == self.maxIter):
			print("The algorithm has reached the max. number of iteration without achieving convergence")
		else:
			print("The algorithm has converged in %i iterations" %self.nbIter)

	def getClusters(self, rounding=1):
		C = np.argmax(np.round(self.P_xc, rounding), axis=1)
		D = np.argmax(np.round(self.P_yd, rounding), axis=1)
		return(C,D)

	def getClusterAssociation(self):
		C_assoc = np.argmax(self.phi, axis=1)
		D_assoc = np.argmax(self.phi, axis=0)
		return(C_assoc, D_assoc)

	def getLikelihood(self):
		ll = np.zeros((self.N,self.M,self.K,self.L))
		for i in range(self.N):
			for j in range(self.M):
				ll[i,j,:,:] = self.p_phi[i,j,:,:] * self.phi
		ll = np.sum(self.data * np.sum(ll, axis=(2,3)))
		return(ll)





















	
