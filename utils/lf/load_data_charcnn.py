#!/usr/bin/env python
# coding: utf-8

# In[42]:


import sys



# In[3]:


import copy

import matplotlib.pyplot as plt
import numpy as np

from utils.data_loader import load_data
from tqdm import tqdm


# In[4]:


global word_dict,index_dict
word_dict={}
index_dict={}


# In[5]:


def count_freq():
    """
    统计字频
    """
    global word_dict
    
    for t in tqdm(train):
        for char in t[0]:
            if char in word_dict:
                word_dict[char]+=1
            else:
                word_dict[char]=1


# In[6]:


def make_index():
    """
    为每个单词打上标号
    """
    global index_dict,word_dict
    
    temp=sorted(word_dict.items(),key=lambda x:x[1],reverse=True)
    for index,content in tqdm(enumerate(temp)):
        index_dict[content[0]]=index+1


# In[7]:


def show_length():
    """
    统计文章字长
    """
    global train
    
    l=[]
    for t in train:
        l.append(len(t[0]))
        
    a=0
    b=0
    c=0
    d=0
    e=0
    
    for i in l:
        if 0<=i<50:
            a+=1
        elif 50<=i<100:
            b+=1
        elif 100<=i<150:
            c+=1
        elif 150<=i<200:
            d+=1
        else:
            e+=1
        
    print([a,b,c,d,e])


# In[8]:


def to_one_hot(dataset):
    """
    数据集转换为one-hot编码
    """
    global index_dict
        
    dataset_new=copy.deepcopy(dataset)
    for index,content in enumerate(dataset_new):
        temp=[]
        for char in content[0]:
            if char in index_dict:
                temp.append(index_dict[char])
            else:
                temp.append(0)
        dataset_new[index][0]=temp
        
    return dataset_new


# In[9]:


def unify(dataset,max_len=150):
    """
    截断和补零
    """
    
    for index,content in enumerate(dataset):
        if len(content[0])==max_len:
            pass
        elif len(content[0])>max_len:
            content[0]=content[0][:max_len]
            dataset[index]=content
        else:
            while(len(content[0])!=150):
                content[0].append(0)
            dataset[index]=content


# In[10]:


def array2numpy(dataset):
    x=[]
    y=[]
    for d in dataset:
        x.append(np.array(d[0]))
        if d[1]==5:
            y.append(np.array(0))
        else:
            y.append(np.array(d[1]))
    x=np.array(x,dtype='float32')
    y=np.array(y,dtype='uint8')
    return x,y


# In[18]:


def load_data_charcnn():
    global train,test
    train,test=load_data('./小组作业-语料/')
    count_freq()
    make_index()
    
    train_one_hot=to_one_hot(train)
    test_one_hot=to_one_hot(train)
    
    unify(train_one_hot)
    unify(test_one_hot)
    
    x_train,y_train=array2numpy(train_one_hot)
    x_test,y_test=array2numpy(test_one_hot)
    
    x_train=x_train.reshape(x_train.shape[0],150,1)
    x_test=x_test.reshape(x_test.shape[0],150,1)
    
    return (x_train,y_train),(x_test,y_test)

