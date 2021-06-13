#!/usr/bin/env python
# coding: utf-8

# In[58]:





# In[59]:


import jieba
import gensim

import matplotlib.pyplot as plt
import numpy as np

from utils.data_loader import load_data
from tqdm import tqdm


# In[60]:


def load_stopwords(path_to_file):
    stop_words=set()
    with open(path_to_file,encoding="utf-8") as f:
        content=f.readlines()
    for word in content:
        stop_words.add(word.strip('\n'))
        
    return stop_words


# In[61]:


def cut_sentence(sentence):
    """
    分词，去停用词，返回一个列表
    """
    result=[]
    for word in jieba.lcut(sentence):
        if word not in stop_words:
            result.append(word)
            
    return result


# In[62]:


def load_word2vec(path_to_file):
    print("加载词向量...")
    return gensim.models.KeyedVectors.load_word2vec_format(path_to_file,binary=False)


# In[64]:


def random_vector(seed):
    np.random.seed(seed)
    vec=-1 + 2*np.random.random((300))
    vec=vec.astype('float64')
    
    return vec


# In[65]:


def get_vector(word,seed=108):
    try:
        return word2vec[word]
    except:
        return random_vector(seed)


# In[66]:


def sentence2vector(word_list):
    result=[]
    for w in word_list:
        result.append(get_vector(w))
        
    return result


# In[68]:


def get_label(dataset):
    label=[]
    for d in dataset:
        if(d[1]!=5):
            label.append(d[1])
        else:
            label.append(0)
    label=np.array(label,dtype='uint8')
    
    return label


# In[91]:


# 向量化
def vectorize(dataset):
    dataset_new=[]
    for d in tqdm(dataset):
        dataset_new.append(sentence2vector(cut_sentence(d[0])))
        
    return dataset_new

# 截断和补0
padding=np.zeros(300,dtype='float64')

def unify(dataset,max_len):
    for i in tqdm(range(len(dataset))):
        if len(dataset[i])==max_len:
            pass
        elif len(dataset[i])<max_len:
            while(len(dataset[i])!=max_len):
                dataset[i].append(padding)
        else:
            dataset[i]=dataset[i][:max_len]
            
def array2np(dataset):
    for index in tqdm(range(len(dataset))):
        dataset[index]=np.array(dataset[index])


# In[92]:


def load_data_mlp():
    train,test=load_data('./小组作业-语料/')
    train_text=vectorize(train)
    unify(train_text,30)
    train_label=get_label(train)
    
    test_text=vectorize(test)
    unify(test_text,30)
    test_label=get_label(test)
    
    array2np(train_text)
    array2np(test_text)
    
    return (np.array(train_text),train_label),(np.array(test_text),test_label)


# In[70]:


stop_words=load_stopwords('./src/hit_stopwords.txt')
word2vec=load_word2vec('../sgns.zhihu.word.bz2')