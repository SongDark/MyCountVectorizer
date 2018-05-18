from scipy import sparse

rows = [0,1,2,2]
cols = [0,1,2,2]
vals = [1,1,1,1]
mat = sparse.csr_matrix((vals,(rows,cols)), shape=(3,3))
print mat.toarray()