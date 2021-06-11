#!/usr/bin/env python
# coding: utf-8

# In[186]:


try:
    from IPython.display import display
except:
    print("缺少IPython模块，结果将不会进行可视化展示")

import matplotlib.pyplot as plt
import pandas as pd


# In[31]:


def base(label_truth,label_predict):
    global data
    data={
        1:{'tp':0,'fp':0,'fn':0},
        2:{'tp':0,'fp':0,'fn':0},
        3:{'tp':0,'fp':0,'fn':0},
        5:{'tp':0,'fp':0,'fn':0},
        'all':{'tp':0,'fp':0,'fn':0},
    }
    for index,predict in enumerate(label_predict):
        if predict==label_truth[index]:
            data[label_truth[index]]['tp']+=1
        else:
            data[predict]['fp']+=1
            data[label_truth[index]]['fn']+=1
            
    for genre in [1,2,3,5]:
        data['all']['tp']+=data[genre]['tp']
        data['all']['fp']+=data[genre]['fp']
        data['all']['fn']+=data[genre]['fn']


# In[104]:


def calculate():
    global data,result
    result={
        1:{'precision':None,'recall':None,'f1':None},
        2:{'precision':None,'recall':None,'f1':None},
        3:{'precision':None,'recall':None,'f1':None},
        5:{'precision':None,'recall':None,'f1':None},
        'all':{'precision':None,'recall':None,'f1':None}
    }
    for genre in [1,2,3,5,'all']:
        result[genre]['precision']=float(data[genre]['tp']/(data[genre]['tp']+data[genre]['fp']))
        result[genre]['recall']=float(data[genre]['tp']/(data[genre]['tp']+data[genre]['fn']))
        result[genre]['f1']=2*result[genre]['precision']*result[genre]['recall']/(result[genre]['recall']+result[genre]['precision'])


# In[187]:


def show():
    global df
    pd.set_option('precision', 2)
    scores = {
    'Precision' : [result[genre]['precision'] for genre in [1,2,3,5,'all']],
    'Recall' : [result[genre]['recall'] for genre in [1,2,3,5,'all']],
    'F1' : [result[genre]['f1'] for genre in [1,2,3,5,'all']],
    }
    df = pd.DataFrame(scores,index=['军事新闻', '体育新闻', '晋江小说', '电影短评','综合'])
    try:
        display(df)
    except:
        pass


# In[182]:


def save_to_file(output_filename):
    global df
    df.to_csv(output_filename, sep='\t',index=True, header=True,encoding='gbk',float_format='%.2f')
    print("\n结果保存在%s"%output_filename)


# In[183]:


def score(label_truth,label_predict,output_filename=None):
    if len(label_truth)!=len(label_predict):
        print("Error: 预测值个数和真实值个数不匹配！")
        return
    
    base(label_truth,label_predict)
    calculate()
    show()
    if output_filename:
        save_to_file(output_filename)


# In[188]:



# In[ ]:




