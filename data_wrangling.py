import pandas as pd
import numpy as np
import csv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import networkx as nx

## Data format
## user id | item id | rating | timestamp

## user-item matrix
data = pd.read_csv('ml-100k/u1.base', sep='\t', header=None, engine='c')
data.columns = ['user', 'item', 'rating', 'timestamp']
#nu = data.user.nunique()
#ni = data.item.nunique()

## To sparse matrix
UI = coo_matrix((data.rating,(data.user-1,data.item-1))) # '-1' in order to cope with 0-indexing
## BE CAREFUL when merging with IDs later on !

## Directly to graph format 
with open('ml-100k/u1.base') as f:

G = nx.readwrite.edgelist.parse_edgelist(lines, comments='#', delimiter=None, create_using=None, nodetype=None, data=True)

G=nx.read_weighted_edgelist('ml-100k/u1.base')