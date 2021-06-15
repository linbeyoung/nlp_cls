# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from model.model import *
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


# 全局变量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset_path = 'dataset/test.txt'
test_features_path = 'features/Topics/topics_test.txt'
vector_save_path = 'Final model/word2vec.txt'
model_save_path = 'Final model/model_24.pth'


class MyDataset(Dataset):
    def __init__(self, embed_size=128, artificial_features=False):
        self.artificial_features = artificial_features
        dataset = []
        labels = []
        label_dict = {1: 0, 2: 1, 3: 2, 5: 3}
        vectors = np.zeros((4736, 200, embed_size), dtype='float32')
        with open(test_dataset_path, 'r', encoding='utf-8') as f_dataset:
            for text in f_dataset:
                label = int(text[-2])  # 原标签
                labels.append(label_dict[label])
                filtered_text = re.sub(r'\s+|\r|#', '', text[0:-2])  # 文本
                seg_text = jieba.lcut(filtered_text, cut_all=False)
                dataset.append(seg_text)

        # 加载训练好的词向量
        model = KeyedVectors.load_word2vec_format(vector_save_path)
        vocab = model.wv.vocab
        for x, text in enumerate(dataset, 0):
            for y, word in enumerate(text, 0):
                if y >= 200:
                    break
                if word in vocab:
                    vectors[x][y] = model[word]
                else:
                    vectors[x][y] = np.zeros(embed_size)

        self.dataset = torch.tensor(vectors).view(4736, 1, 200, embed_size)
        self.label = torch.tensor(labels)

        if artificial_features:
            features = np.zeros((4736, 64), dtype='float32')
            with open(test_features_path, 'r', encoding='utf-8') as f_features:
                file = f_features.readlines()
                for i, feature in enumerate(file, 0):
                    features[i] = feature.split()
            self.features = torch.tensor(features)

    def __getitem__(self, item):
        if self.artificial_features:
            return self.dataset[item], self.label[item], self.features[item]
        return self.dataset[item], self.label[item]

    def __len__(self):
        return len(self.dataset)


def test(test_loader, model, criterion, name, artificial_features=False):
    model.eval()
    correct = 0
    total = 0
    target_list = []
    pred_list = []
    with torch.no_grad():
        running_loss = 0
        for data in test_loader:
            # artificial_features=True
            if artificial_features:
                inputs, target, features = data
                inputs, target, features = inputs.to(device), target.to(device), features.to(device)
                outputs = model(inputs, features)
            # artificial_features=False
            else:
                inputs, target = data
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, target)
            pred = torch.max(outputs, dim=1)[1]

            for x in target:
                target_list.append(x.item())
            for x in pred:
                pred_list.append(x.item())

            correct += (pred == target).sum().item()
            total += target.size(0)
            running_loss += loss.item()
        print('test loss: {:.6f} Accuracy on {} set: {:.2f}% [{}/{}]'
              .format(running_loss / len(test_loader), name, 100 * correct / total, correct, total))
        return target_list, pred_list


def main():
    dataset = MyDataset()
    test_loader = DataLoader(dataset, shuffle=False, batch_size=256)
    model = TextRCNN()
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    target_list, pred_list = test(test_loader, model, criterion, 'test')
    # 打印P,R,F1,Acc
    print(metrics.classification_report(target_list, pred_list, digits=3))


if __name__ == '__main__':
    main()
