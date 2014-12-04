import pandas as pd
import scipy.stats as sctat
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans


class coclustering_1:
# co-clustering without preferences value

## zone de test
ber = 0.2
berno = sctat.bernoulli(ber)
data = berno.rvs((3,10))
data1 = berno.rvs((20,10))

	def __init__(self, phi, data, mu, nx, ny, 
			n_initx,n_clusterx, n_inity,n_clustery,
			q_x, self.q_y, p_phi, phi,
			EM_n_cluster):
		self.EM_n_cluster = EM_n_cluster
		self.data =  # we need to generate some 
		self.data_col = self.data.shape[1]
		self.data_row = self.data.shape[0]
		## count marginal
		self.nx =  np.zeros((self.data_row,1))
		for i in [1:self.data_row]:
			self.nx(i,1) = sum(self.data(i,:))
		self.ny =  np.zeros((self.data_col,1))
		for i in [1:self.data_col]:
			self.ny(i,1) = sum(self.data(:,i))

		## intialization of p_phi  ###########################
			## param of kmeans		
		self.n_initx = n_initx
		self.n_inity = n_inity
		self.n_clusterx = n_clusterx
		self.n_clustery = n_clustery

			## fitting kmeans
 		kmeanx = KMeans(n_init = n_initx, n_cluster = n_clusterx)	
		kmeanx.fit(self.data)	
		indic_x = self.kmeanx.predict(self.data)	
		kmeany = KMeans(n_init = n_inity, n_cluster = n_clustery)	
		kmeany.fit(self.data)	
		indic_y = self.kmeany.predict(self.data)

			## intialization of q 
		self.q_x = np.zeros((self.data_row, n_clusterx))		
		self.q_y = np.zeros((self.data_col, n_clustery))
		for i in [1:self.data_row]:
			for j in [1:self.n_clusterx]:
				if ( indic_x(i) == j):
					self.q_x(i,j) = 1
				else:
					self.q_x(i,j) = 0
		for i in [1:self.data_col]:
			for j in [1:self.n_clustery]:
				if ( indic_y(i) == j):
					self.q_y(i,j) = 1
				else:
					self.q_y(i,j) = 0
			## filling p_phi
		self.phi = np.zeros((self.data_row,self.data_col,self.n_clusterx,self.n_clustery))
		for i in [1:self.data_row]:
			for j in [1:self.data_col]:
				for i1 in [1:self.n_clusterx]:
					for j1 in [1:self.n_clustery]:
						self.p_phi(i,j,i1,j1) = self.q_x(i,i1)*self.q_y(j,j1)
		###########################

		## intialization of phi  ###########################
		self.phi = np.zeros((self.n_clusterx,self.n_clustery))
		for i1 in [1:self.n_clusterx]:
			for j1 in [1:self.n_clustery]:
				Numerator = 0
				for i in [1:self.data_row]:
					for j in [1:self.data_col]:
						Numerator = Numerator + \
							    self.p_phi(i,j,i1,j1)*self.data(i,j)
				Denominator1 = 0	
				for i in [1:self.data_row]:
				### TODO : vérifier que l' on a bien une proba borné
					Denominator1 = Denominator1 + np.sum(self.p_phi(i,:,i1,:), \
										axis=[1,3])*self.nx(i)
				Denominator2 = 0	
				for j in [1:self.data_col]:
				### TODO : vérifier que l' on a bien une proba borné
					Denominator2 = Denominator2 + np.sum(self.p_phi(:,j,:,j1), \
										axis=[0,2])*self.ny(j)


	def EM_step(self,EM_n_cluster):
	## for i in [1:EM_n_cluster]

	## inner E-step:
		## self.q_x part
		for i in [1:self.data_row]:
			for i1 in [1:self.n_clusterx]:
				self.q_x(i,i1) = 

	## inner M-step:
		for i1 in [1:self.n_clusterx]:
			for j1 in [1:self.n_clustery]:
				Numerator = 0
				for i in [1:self.data_row]:
					for j in [1:self.data_col]:
						Numerator = Numerator + \
							    self.p_phi(i,j,i1,j1)*self.data(i,j)
				Denominator1 = 0	
				for i in [1:self.data_row]:
				### TODO : vérifier que l' on a bien une proba borné
					Denominator1 = Denominator1 + np.sum(self.p_phi(i,:,i1,:), \
										axis=[1,3])*self.nx(i)
				Denominator2 = 0	
				for j in [1:self.data_col]:
				### TODO : vérifier que l' on a bien une proba borné
					Denominator2 = Denominator2 + np.sum(self.p_phi(:,j,:,j1), \
										axis=[0,2])*self.ny(j)

	
