from numpy.random import choice, shuffle, random_integers, rand
from numpy import sort, arange, zeros, max
from functools import partial
from itertools import zip_longest

def replaceNone(x, N):
    if(x[0] == None):
        x = (choice(arange(N), 1)[0], x[1])
    if(x[1] == None):
        x = (x[0], choice(arange(N), 1)[0])
    return x

def sampleMatrix(nrows, ncols, nC, nD, preference_value=False):
	## random clusters for rows
	Xindexes = list(range(nrows))
	cuts = choice(Xindexes[1:], nC-1, replace=False)
	shuffle(Xindexes)
	C = []
	low = 0
	for sup in sort(cuts):
		C.append(Xindexes[low:sup])
		low = sup
	C.append(Xindexes[sup:])
	## Random clusters for columns 
	Yindexes = list(range(ncols))
	cuts = choice(Yindexes[1:], nD-1, replace=False)
	shuffle(Yindexes)
	D = []
	low = 0
	for sup in sort(cuts):
		D.append(Yindexes[low:sup])
		low = sup
	D.append(Yindexes[sup:])
	## Association between C and D
	if(nC > nD):
		mapfun = partial(replaceNone, N=nD)
		association = list(map(mapfun, list(zip_longest(choice(arange(nC), nC, replace=False), choice(arange(nD), nD, replace=False)))))
	else:
		raise Exception("We assume nC >= nD")
	## Generate observations 
	M = zeros((nrows, ncols))
	for (c, d) in association:
		#print(c)
		#print(d)
		## Law for a specified cluster
		p = rand(1, ncols)[0]
		k = sum(p)
		for i in D[d]:
			p[i] += k*3
		p = p / sum(p)
		for i in C[c]:
			index_UI = choice(arange(ncols), len(D[d])+random_integers(0,2,1)[0],replace=False, p=p)
			# print((c, index_UI))
			for j in index_UI:
				M[i, j] = 1
	return (C,D,M)