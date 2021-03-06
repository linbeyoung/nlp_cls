import glob
from math import floor

import numpy as np

raw_data_path = "./小组作业-语料/"

test_ratio = 0.2  # 测试集比例
seed = 107


def method_1(OUTPUT=False):
    """
    划分方法一：所有数据打乱，按比例切割训练、测试集
    默认测试集比例0.2
    """
    all_data = []

    for file in glob.glob(raw_data_path + "*"):
        with open(file, encoding="utf-8") as f:
            content = f.readlines()
            for line in content:
                # all_data.append(line.rstrip().split('\t'))
                all_data.append(line.rstrip())

    # for index,line in enumerate(all_data):
    #     all_data[index][1]=int(line[1]) # convert string to int

    np.random.seed(seed)
    np.random.shuffle(all_data)

    split = floor(len(all_data) * (1 - test_ratio))
    train = all_data[:split]
    # print(train)
    test = all_data[split:]

    if OUTPUT:
        with open('datasets/train.txt', 'a') as train_set:
            for line in train:
                train_set.writelines(line + '\n')

        with open('datasets/test.txt', 'a') as test_set:
            for line in test:
                test_set.writelines(line + '\n')

    print("加载完成，测试集比例%s" % (test_ratio))
    print("训练集%s条" % (len(train)))
    print("测试集%s条" % (len(test)))

    return train, test


def load_data():
    return method_1()


method_1(OUTPUT=True)
