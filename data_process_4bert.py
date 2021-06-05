# -*- coding: utf-8 -*-
# @Time   : 2021/6/5 11:45
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : data_process_4bert.py

import os


def data_process_bert(file, outfile=None):
    if not outfile:
        outfile = 'datasets/bert/' + os.path.split(file)[1][:-4] + '_bert.txt'

    with open(file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()

        for line in lines:
            line = line.replace('\n', '')

            sen, tag = line.split('\t')[0], line.split('\t')[1]

            with open(outfile, 'a', encoding='UTF-8') as of:
                # print(tag, sen)
                print(tag + '\t' + sen + '\n')
                of.writelines(tag + '\t' + sen + '\n')

    return


# def get_all_files(dir):
#     files = os.listdir(dir)
#     for
#     return
root_dir = '小组作业-语料'
files = os.listdir(root_dir)
for file in files:
    if file.endswith('txt'):
        data_process_bert(os.path.join(root_dir, file))
