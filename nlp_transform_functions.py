#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import datetime
import os
import json
import string
import sklearn
import re
from sklearn.base import BaseEstimator, TransformerMixin
import nlp
import nltk
from nltk.stem import WordNetLemmatizer
from nlp import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import fuzzywuzzy
from fuzzywuzzy import fuzz
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
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
                X[self.text_cols[0]] = [str("".join([i if (i not in self.punc_list or i in self.ignore_list) else self.replace_string for i in str(x)])) for x in X[self.text_cols[0]]]
            else:
                for col_name in self.text_cols:
                    if self.verbose:
                        print('Columns: ', col_name)
                    X[col_name] = [str("".join([i if (i not in self.punc_list or i in self.ignore_list) else self.replace_string for i in str(x)])) for x in X[col_name]]
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
                X[self.text_cols[0]] = X[self.text_cols[0]].apply(lambda x: str(x.lower()))
            else:
                for col_name in self.text_cols:
                    if self.verbose:
                        print('Columns: ', col_name)
                    X[col_name] = X[col_name].apply(lambda x: str(x.lower()))

        except Exception as err:
            if self.verbose:
                print('Error: ', err)
        end = datetime.datetime.now()
        diff = end-start
        if self.verbose:
            print(diff.seconds)
        return X

class TokenizeTextXFormer(BaseEstimator, TransformerMixin):

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
                X[self.text_cols[0]] = X[self.text_cols[0]].apply(lambda x: re.split('w+',x))
            else:
                for col_name in self.text_cols:
                    if self.verbose:
                        print('Columns: ', col_name)
                    X[col_name] = X[col_name].apply(lambda x: re.split('w+',x))
        except Exception as err:
            if self.verbose:
                print('Error: ', err)
        end = datetime.datetime.now()
        diff = end-start
        if self.verbose:
            print(diff.seconds)
        return X

class StopwordsRemoveXFormer(BaseEstimator, TransformerMixin):

    def __init__(self, verbose = True):
        self.verbose = verbose
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.wordnet_lemmatizer = WordNetLemmatizer()

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
                X[self.text_cols[0]] = X[self.text_cols[0]].apply(lambda x: [self.wordnet_lemmatizer.lemmatize(i) for i in x if i not in self.stopwords])
            else:
                for col_name in self.text_cols:
                    if self.verbose:
                        print('Columns: ', col_name)
                    X[col_name] = X[col_name].apply(lambda x: [self.wordnet_lemmatizer.lemmatize(i) for i in x if i not in self.stopwords])
        except Exception as err:
            if self.verbose:
                print('Error: ', err)
        end = datetime.datetime.now()
        diff = end-start
        if self.verbose:
            print(diff.seconds)
        return X

# In[9]:
class TwoColumnSimilarityMatch:

    def __init__(self):
        dataset = load_dataset('glue', 'mrpc', split='train')
        self.eg_dataset_df = pd.DataFrame([dataset['sentence1'],dataset['sentence2'],dataset['label']]).T
        self.eg_dataset_df.columns = ['sentence1','sentence2','label']
        nltk.download('stopwords')

    def test_transforms(self):
        process = True
        start = datetime.datetime.now()
        try:
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
                ('lower_text', LowerTextXFormer()),
                ('text_tokenize', TokenizeTextXFormer()),
                ('stopwords_remove', StopwordsRemoveXFormer())
            ])
            self.eg_dataset_transform = text_pipeline.fit_transform(self.eg_dataset_df)
        except Exception as err:
                print('Error: ', err)
                process = False
        end = datetime.datetime.now()
        diff = end-start
        print(diff.seconds)
        return self.eg_dataset_transform, process

    def matcher_score(self):
        similarity_score_lst = [fuzz.ratio(row['sentence1'],row['sentence2']) for index,row in self.eg_dataset_transform.iterrows()]
        self.eg_dataset_transform['similar_op_flag'] = [1 if x>50 else 0 for x in similarity_score_lst]
        self.eg_dataset_transform['similar_prob'] = [float(x)/100 for x in similarity_score_lst]
        cutoff = [0,10,20,30,40,50,60,70,80,90]
        y_val = [int(x[0]) for x in self.eg_dataset_transform['label'].values]
        for value in cutoff:
            print("Cutoff: ", value)
            self.eg_dataset_transform['similar_pred_flag'] = [1 if x>=value else 0 for x in similarity_score_lst]
            prediction_flags = self.eg_dataset_transform['similar_pred_flag'].values.tolist()
            print(confusion_matrix(y_val,prediction_flags))
        self.eg_dataset_transform['similar_pred_flag'] = [1 if x>70 else 0 for x in similarity_score_lst]
        prediction_flags = self.eg_dataset_transform['similar_pred_flag'].values.tolist()
        print(classification_report(y_val, prediction_flags , labels=[0,1], target_names = ['Value_0 (Non-Similar)','Value_1 (Similar)']))
        fpr, tpr, _ = roc_curve(y_val, self.eg_dataset_transform['similar_prob'].values.tolist(), pos_label=1)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    def matcher_score_token(self):
        similarity_score_lst = [fuzz.token_sort_ratio(row['sentence1'],row['sentence2']) for index,row in self.eg_dataset_transform.iterrows()]
        self.eg_dataset_transform['similar_op_flag'] = [1 if x>50 else 0 for x in similarity_score_lst]
        self.eg_dataset_transform['similar_prob'] = [float(x)/100 for x in similarity_score_lst]
        cutoff = [0,10,20,30,40,50,60,70,80,90]
        y_val = [int(x[0]) for x in self.eg_dataset_transform['label'].values]
        for value in cutoff:
            print("Cutoff: ", value)
            self.eg_dataset_transform['similar_pred_flag'] = [1 if x>=value else 0 for x in similarity_score_lst]
            prediction_flags = self.eg_dataset_transform['similar_pred_flag'].values.tolist()
            print(confusion_matrix(y_val,prediction_flags))
        self.eg_dataset_transform['similar_pred_flag'] = [1 if x>70 else 0 for x in similarity_score_lst]
        prediction_flags = self.eg_dataset_transform['similar_pred_flag'].values.tolist()
        print(classification_report(y_val, prediction_flags , labels=[0,1], target_names = ['Value_0 (Non-Similar)','Value_1 (Similar)']))
        fpr, tpr, _ = roc_curve(y_val, self.eg_dataset_transform['similar_prob'].values.tolist(), pos_label=1)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

import matplotlib.pyplot as plt
matcher = TwoColumnSimilarityMatch()
matcher.test_transforms()
matcher.matcher_score()
matcher.matcher_score_token()
plt.show()

#URL removal, HTML tags removal, Rare words removal, Frequent words removal, Spelling checking
# In[ ]:


#Test functions separately to get the right outputs

# string_data = 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .'
# punc_list = []
# for character in string.punctuation:
#     punc_list.append(str(character))
# print(punc_list)
# ignore_list = [',','?','.']
# replace_string = '*'
# string_clean= ["".join([i if (i not in punc_list or i in ignore_list) else replace_string for i in string_data])]
# print(string_data)
# print(string_clean)
