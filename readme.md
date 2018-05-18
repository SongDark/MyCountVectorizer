# MyCountVectorizer

## Introduction

A simpler version of `sklearn.feature_extraction.text.CountVectorizer`, it realizes two major functions :

(1) `fit`: split sentences separated by space, and build an vocabulary

(2) `transform`: split sentences separated by space, and encode them, and return a `csc_matrix`

It is faster than the sklearn version.

## Usage

```python 
from MyCountVectorizer import MyCountVectorizer

# data is like ["AA BB CC", "A B"]

mcv = MyCountVectorizer()
mcv.fit(data)
mcv.transform(data)
```

## Comparison of time consumption

Please try it yourself.

```python 
python run.py
```

<center>
<img src="https://github.com/SongDark/MyCountVectorizer/blob/master/pictures/result.png?raw=true" width="70%">

