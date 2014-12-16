import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import coo_matrix

class SBM(object):
    """Stochastic Block Model"""
    def __init__(self, data, K, L, maxIter, tol, random_state=3):
        self.tol = tol
        self.nbIter = 0
        self.maxIter = maxIter
        self.data = data
        self.dataS = coo_matrix(data)
        (self.N, self.M) = self.data.shape
        ## count marginal
        self.K = K
        self.L = L
        ### intialization of p_phi  ###########################
        ## init clusters for X with kmeans
        initClusters = KMeans(n_clusters=K, random_state=random_state)
        C = initClusters.fit_predict(self.dataS)
        ## init clusters for Y with kmeans
        initClusters.n_clusters=L
        D = initClusters.fit_predict(self.dataS.T)
        ## intialization of q 
        binarizer = LabelBinarizer()
        self.Q_xc = binarizer.fit_transform(C)
        self.Q_yd = binarizer.fit_transform(D)
        if(K == 2):
            self.Q_xc = np.hstack((self.Q_xc, 1 - self.Q_xc))
        if(L == 2):
            self.Q_yd = np.hstack((self.Q_yd, 1 - self.Q_yd))
        ## intializations proba cluster  ###########################
        self.P_c = np.sum(self.Q_xc, axis=0) / self.N
        self.P_d = np.sum(self.Q_yd, axis=0) / self.M
        ## fill p_phi ###########################
        self.p_phi = np.zeros((self.N,self.M,self.K,self.L))
        for j in range(self.M):
            for c in range(self.K):
                self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c].reshape(self.N, 1), self.Q_yd[j,:].reshape(1, self.L))
        ## intialization of P_v_cd ###########################
        self.P_v_cd = np.zeros((2,K,L))
        denom = 0
        for i,j,v in zip(self.dataS.row, self.dataS.col, self.dataS.data):
            if(v == 1):
                self.P_v_cd[1,:,:] += self.p_phi[i,j,:,:]
            if(v == -1):
                self.P_v_cd[0,:,:] += self.p_phi[i,j,:,:]
            denom += self.p_phi[i,j,:,:]
        denom += 1e-50
        self.P_v_cd[1,:,:] = self.P_v_cd[1,:,:] / denom
        self.P_v_cd[1,:,:] = self.P_v_cd[1,:,:] / np.sum(self.P_v_cd[1,:,:]) # normalization
        self.P_v_cd[0,:,:] = self.P_v_cd[0,:,:] / denom
        self.P_v_cd[0,:,:] = self.P_v_cd[0,:,:] / np.sum(self.P_v_cd[0,:,:]) # normalization
        self.P_v_cd += 1e-50



    def fit(self):
        old_Pc = self.P_c + 1
        old_Pd = self.P_d + 1
        while(np.sum(np.abs(self.P_c - old_Pc)) + np.sum(np.abs(self.P_d - old_Pd)) > self.tol and self.nbIter < self.maxIter):
            self.nbIter = self.nbIter + 1
            ## E-Step
            self.Q_xc = self.P_c * np.exp((self.dataS==1).dot(self.Q_yd.dot(np.log(self.P_v_cd[1,:,:].T))) + \
                (self.dataS==-1).dot(self.Q_yd.dot(np.log(self.P_v_cd[0,:,:].T))))
            self.Q_xc = self.Q_xc / np.sum(self.Q_xc,1).reshape(self.N,1) ## normalization
            self.Q_yd = self.P_d * np.exp((self.dataS==1).T.dot(self.Q_xc.dot(np.log(self.P_v_cd[1,:,:]))) + \
                (self.dataS==-1).T.dot(self.Q_xc.dot(np.log(self.P_v_cd[0,:,:]))))
            self.Q_yd = self.Q_yd / np.sum(self.Q_yd,1).reshape(self.M,1) ## normalization
            ## Update p_phi
            for j in range(self.M):
                for c in range(self.K):
                    self.p_phi[:,j,c,:] = np.dot(self.Q_xc[:,c].reshape(self.N, 1), self.Q_yd[j,:].reshape(1, self.L))
            ## M-step:
            denom = 0
            for i,j,v in zip(self.dataS.row, self.dataS.col, self.dataS.data):
                if(v == 1):
                    self.P_v_cd[1,:,:] += self.p_phi[i,j,:,:]
                if(v == -1):
                    self.P_v_cd[0,:,:] += self.p_phi[i,j,:,:]
                denom += self.p_phi[i,j,:,:]
            denom += 1e-50
            self.P_v_cd[1,:,:] = self.P_v_cd[1,:,:] / denom
            self.P_v_cd[1,:,:] = self.P_v_cd[1,:,:] / np.sum(self.P_v_cd[1,:,:]) # normalization
            self.P_v_cd[0,:,:] = self.P_v_cd[0,:,:] / denom
            self.P_v_cd[0,:,:] = self.P_v_cd[0,:,:] / np.sum(self.P_v_cd[0,:,:]) # normalization
            self.P_v_cd += 1e-50
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
        P_xc = np.sum(self.p_phi, axis=(1,3))
        P_xc = P_xc / np.sum(P_xc, axis=(1)).reshape(self.N,1)
        P_yd = np.sum(self.p_phi, axis=(0,2)) 
        P_yd = P_yd / np.sum(P_yd, axis=(1)).reshape(self.M,1)
        C = np.argmax(np.round(P_xc, rounding), axis=1)
        D = np.argmax(np.round(P_yd, rounding), axis=1)
        return(C,D)

    def getClusterAssociation(self):
        dataABS = np.abs(self.data)
        nx = np.sum(dataABS, 1)
        ny = np.sum(dataABS, 0)
        phi = np.zeros((self.K,self.L))
        P_xc = np.sum(self.p_phi, axis=(1,3))
        P_xc = P_xc / np.sum(P_xc, axis=(1)).reshape(self.N,1)
        P_yd = np.sum(self.p_phi, axis=(0,2)) 
        P_yd = P_yd / np.sum(P_yd, axis=(1)).reshape(self.M,1)
        DenominatorX = np.sum(P_xc * nx.reshape(self.N,1), axis=0)
        DenominatorY = np.sum(P_yd * ny.reshape(self.M,1), axis=1)
        for c in range(self.K):
            for d in range(self.L):
                phi[c,d] = np.sum(self.p_phi[:,:,c,d] * dataABS)/(DenominatorX[c]*DenominatorY[d]) + 1e-50
        C_assoc = np.argmax(phi, axis=1)
        D_assoc = np.argmax(phi, axis=0)
        return(C_assoc, D_assoc, phi)

    def getLikelihood(self):
        ll = np.zeros((self.N,self.M,self.K,self.L))
        for i in range(self.N):
            for j in range(self.M):
                ll[i,j,:,:] = self.p_phi[i,j,:,:] * self.phi
        ll = np.sum(self.data * np.sum(ll, axis=(2,3)))
        return(ll)





















    
