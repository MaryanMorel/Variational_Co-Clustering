#!/usr/local/opt/python3/bin/python3

import MatrixGenerator as mg
import class_coclustering as CC

if __name__ == '__main__':
	nrows = 100 
	ncols = 20
	nC = 8
	nD = 4

	(C,D,M) = mg.sampleMatrix(nrows, ncols, nC, nD)

	clf = CC.coClusteringAdjacency(data=M, K=8, L=4, maxIter=10)
	clf.fit() ## At some point, p_phi becomes NA
	print(clf.getClusters()) ## Returns NAs (probably because of some 0 or negative value in a log)