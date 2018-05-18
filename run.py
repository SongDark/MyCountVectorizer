from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy import sparse
from MyCountVectorizer import MyCountVectorizer
import time
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("data/test.csv")

records = []
for n in range(100, 10100, 100):
    seq = np.tile(data['interest'].values, (n,1)).ravel()
    
    # CountVectorizer in sklearn
    start1 = time.time()
    cv = CountVectorizer()
    cv.fit(seq)
    res1 = cv.transform(seq).toarray()
    end1 = time.time()

    # MyCountVectorizer
    start2 = time.time()
    mcv = MyCountVectorizer()
    mcv.fit(seq)
    res2 = mcv.transform(seq).toarray()
    end2 = time.time()

    assert np.sum(res1-res2) == 0

    records.append((n, end1-start1, end2-start2))
records = np.array(records)

plt.plot(records[:,0], records[:,1], label="CountVectorizer")
plt.plot(records[:,0], records[:,2], label="MyCountVectorizer")
plt.legend()
plt.savefig('pictures/result.png')