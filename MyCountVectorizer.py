import numpy as np
from scipy import sparse

class MyCountVectorizer():
    def __init__(self, pass_stop=True):
        self.pass_stop = pass_stop
    
    def fit(self, data):
        data = map(lambda x:str(x).split(" "), data)
        self.elements_ = set()
        for line in data:
            for x in line:
                if self.pass_stop:
                    if len(x)==1:
                        continue
                self.elements_.add(x)
        self.elements_ = np.sort(list(self.elements_))
        self.labels_ = np.arange(len(self.elements_)).astype(int)
        self.dict_ = {}
        for i in range(len(self.elements_)):
            self.dict_[str(self.elements_[i])] = self.labels_[i]
    
    def transform(self, data):
        rows = []
        cols = []
        data = map(lambda x:str(x).split(" "), data)
        for i in range(len(data)):
            for x in data[i]:
                if self.pass_stop:
                    if len(x)==1:
                        continue
                rows.append(i)
                cols.append(self.dict_[x])
        vals = np.ones((len(rows),)).astype(int)

        return sparse.csc_matrix((vals, (rows, cols)), shape=(len(data), len(self.labels_)))
        
        
