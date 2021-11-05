#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import os
import json
import string
from sklearn.base import BaseEstimator, TransformerMixin
from nlp import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[2]:


###https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/


# In[8]:


class TextDecodeXFormer(BaseEstimator, TransformerMixin):
    """Wrapper for using encoding and decoding to clean and transform unstructured text"""
    
    #the constructor
    def __init__(self, text_cols, encoding_utf8 = True, encoding_replace=True, verbose=True):
        self.text_cols = text_cols
        self.encoding_utf8 = encoding_utf8
        self.encoding_replace = encoding_replace
        self.verbose = verbose
        
    #estimator method
    def fit(self, X, y = None):
        return self
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    #transformation
    def transform(self, X, y = None):
        process = True
        start = datetime.datetime.now()
        try:
            if len(self.text_cols)==1:
                if self.verbose:
                    print('Columns: ', self.text_cols[0])
                if self.encoding_utf8 and self.encoding_replace:
                    X[self.text_cols[0]] = [str(x.encode('utf-8','replace')) for x in X[self.text_cols[0]]]
                elif self.encoding_utf8:
                    X[self.text_cols[0]] = [str(x.encode('utf-8','ignore')) for x in X[self.text_cols[0]]]
            else:
                for col_name in self.text_cols:
                    if self.verbose:
                        print('Columns: ', col_name)
                    if self.encoding_utf8 and self.encoding_replace:
                        X[col_name] = [str(x.encode('utf-8','replace')) for x in X[col_name]]
                    elif self.encoding_utf8:
                        X[col_name] = [str(x.encode('utf-8','ignore')) for x in X[col_name]]
        except Exception as err:
            if self.verbose:
                print('Error: ', err)
        end = datetime.datetime.now()
        diff = end-start
        if self.verbose:
            print(diff.seconds)
        return X
    
class RemovePunctuationXFormer(BaseEstimator, TransformerMixin):
    """Wrapper for using encoding and decoding to clean and transform unstructured text"""
    #the constructor
    def __init__(self, punc_list = [], ignore_list=[], replace_string='\s', verbose = True):
        self.ignore_list = ignore_list
        self.replace_string = replace_string
        self.verbose = verbose
        if len(punc_list)==0:
            self.punc_list = []
            for character in string.punctuation:
                self.punc_list.append(str(character))
        else:
            self.punc_list = punc_list
        
    #estimator method
    def fit(self, X, y = None):
        return self
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    #transformation
    def transform(self, X, y = None):
        process = True
        start = datetime.datetime.now()
        self.text_cols = X.columns
        try:
            if len(self.text_cols)==1:
                if self.verbose:
                    print('Columns: ', self.text_cols[0])
                X[self.text_cols[0]] = ["".join([i if (i not in self.punc_list or i in self.ignore_list) else self.replace_string for i in x]) for x in X[self.text_cols[0]]]  
            else:
                for col_name in self.text_cols:
                    if self.verbose:
                        print('Columns: ', col_name)
                    X[col_name] = ["".join([i if (i not in self.punc_list or i in self.ignore_list) else self.replace_string for i in x]) for x in X[col_name]]   
        except Exception as err:
            if self.verbose:
                print('Error: ', err)
        end = datetime.datetime.now()
        diff = end-start
        if self.verbose:
            print(diff.seconds)
        return X
    
class LowerTextXFormer(BaseEstimator, TransformerMixin):
    
    def __init__(self, verbose = True):
        self.verbose = verbose
    
    def fit(self, X, y=None):
        return self 
    
    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def transform(self, X, y=None):
        process = True
        start = datetime.datetime.now()
        self.text_cols = X.columns
        try:
            if len(self.text_cols)==1:
                if self.verbose:
                    print('Columns: ', self.text_cols[0])
                X[self.text_cols[0]] = X[self.text_cols[0]].apply(lambda x: x.lower()) 
            else:
                for col_name in self.text_cols:
                    if self.verbose:
                        print('Columns: ', col_name)
                    X[col_name] = X[col_name].apply(lambda x: x.lower())
        except Exception as err:
            if self.verbose:
                print('Error: ', err)
        end = datetime.datetime.now()
        diff = end-start
        if self.verbose:
            print(diff.seconds)
        return X


# In[9]:


def test_case():
    dataset = load_dataset('glue', 'mrpc', split='train')
    eg_dataset_df = pd.DataFrame([dataset['sentence1'],dataset['sentence2'],dataset['label']]).T
    eg_dataset_df.columns = ['sentence1','sentence2','label']
    print(eg_dataset_df.head(1).values)
    #the numeric attributes transformation pipeline
    punc_list = []
    for character in string.punctuation:
        punc_list.append(str(character))
    text_pipeline = Pipeline([
        ('text_decode', TextDecodeXFormer(text_cols = ['sentence1','sentence2'],
                                              encoding_utf8 = True, 
                                              encoding_replace = False)),
        ('remove_punctuation',RemovePunctuationXFormer(punc_list = punc_list,
                                                        ignore_list = [',','?','.'],
                                                        replace_string='*')),
        ('lower_text',LowerTextXFormer())
    ])
    #perform the fit transform
    eg_dataset_df_clean = text_pipeline.fit_transform(eg_dataset_df)
    print(eg_dataset_df_clean.head(1).values)


# In[10]:


test_case()


# In[ ]:


#Test functions separately to get the right outputs

string_data = 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .'
punc_list = []
for character in string.punctuation:
    punc_list.append(str(character))
print(punc_list)
ignore_list = [',','?','.']
replace_string = '*'
string_clean= ["".join([i if (i not in punc_list or i in ignore_list) else replace_string for i in string_data])]
print(string_data)
print(string_clean)


# In[ ]:


from sklearn.neighbors import KNeighborsTransformer
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix

class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric="euclidean", method="sw-graph", n_jobs=1):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        self.n_samples_fit_ = X.shape[0]

        # see more metric in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        space = {
            "euclidean": "l2",
            "cosine": "cosinesimil",
            "l1": "l1",
            "l2": "l2",
        }[self.metric]

        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(X)
        self.nmslib_.createIndex()
        return self

    def transform(self, X):
        n_samples_transform = X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(X, k=n_neighbors, num_threads=self.n_jobs)
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        return kneighbors_graph

#https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html#sphx-glr-auto-examples-neighbors-approximate-nearest-neighbors-py
class AnnoyTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using annoy.AnnoyIndex as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric="euclidean", n_trees=10, search_k=-1):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = metric

    def fit(self, X):
        self.n_samples_fit_ = X.shape[0]
        self.annoy_ = annoy.AnnoyIndex(X.shape[1], metric=self.metric)
        for i, x in enumerate(X):
            self.annoy_.add_item(i, x.tolist())
        self.annoy_.build(self.n_trees)
        return self

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X=None)

    def _transform(self, X):
        """As `transform`, but handles X is None for faster `fit_transform`."""

        n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        indices = np.empty((n_samples_transform, n_neighbors), dtype=int)
        distances = np.empty((n_samples_transform, n_neighbors))

        if X is None:
            for i in range(self.annoy_.get_n_items()):
                ind, dist = self.annoy_.get_nns_by_item(
                    i, n_neighbors, self.search_k, include_distances=True
                )

                indices[i], distances[i] = ind, dist
        else:
            for i, x in enumerate(X):
                indices[i], distances[i] = self.annoy_.get_nns_by_vector(
                    x.tolist(), n_neighbors, self.search_k, include_distances=True
                )

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        return kneighbors_graph

