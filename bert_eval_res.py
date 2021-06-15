# -*- coding: utf-8 -*-
# @Time   : 2021/6/14 15:48
# @Author : beyoung
# @Email  : linbeyoung@stu.pku.edu.cn
# @File   : bert_eval_res.py

from utils.score_4bert import score




def str2list(str_old):
    list_new = list(str_old)
    list_new = [i for i in list_new if i != '[']
    list_new = [i for i in list_new if i != ']']
    list_new = [i for i in list_new if i != ',']
    list_new = [i for i in list_new if i != ' ']
    list_new = [i for i in list_new if i != '\n']
    list_new = [int(i) for i in list_new]

    return list_new

if __name__ == '__main__':

    with open("/Users/Beyoung/Desktop/Projects/nlp_cls/output/electra_pred.txt", 'r') as pred_txt:
        res = pred_txt.readlines()

    label_predict = str2list(res[-2])
    # print(label_predict)
    label_truth = str2list(res[-1])
    # print(len(label_truth), len(label_predict))
    # print(label_predict)
    print(score(label_truth, label_predict, output_filename='electra.txt'))
