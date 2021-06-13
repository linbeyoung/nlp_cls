import glob
import numpy as np
from math import floor


test_ratio=0.2 # 测试集比例
seed=107

def method_1(raw_data_path):
    """
    划分方法一：所有数据打乱，按比例切割训练、测试集
    默认测试集比例0.2
    """
    all_data=[]

    for file in glob.glob(raw_data_path+"*"):
        with open(file,encoding="utf-8") as f:
            content=f.readlines()
            for line in content:
                all_data.append(line.rstrip().split('\t'))
                
    for index,line in enumerate(all_data):
        all_data[index][1]=int(line[1]) # convert string to int
    
    np.random.seed(seed)
    np.random.shuffle(all_data)
    
    split=floor(len(all_data)*(1-test_ratio))
    train=all_data[:split]
    test=all_data[split:]

    print("加载完成，测试集比例%s"%(test_ratio))
    print("训练集%s条"%(len(train)))
    print("测试集%s条"%(len(test)))
    
    return train,test

def load_data(raw_data_path):
    return method_1(raw_data_path)