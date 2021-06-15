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

# 全局变量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = 'dataset/train_valid.txt'
result_save_path = 'result_save/result.txt'
features_path = 'features/Topics/topics_train_valid.txt'


class MyDataset(Dataset):
    def __init__(self, embed_size=128, artificial_features=False):
        self.artificial_features = artificial_features
        dataset = []
        labels = []
        label_dict = {1: 0, 2: 1, 3: 2, 5: 3}
        vectors = np.zeros((42620, 200, embed_size), dtype='float32')
        with open(dataset_path, 'r', encoding='utf-8') as f_dataset:
            for text in f_dataset:
                label = int(text[-2])  # 原标签
                labels.append(label_dict[label])
                filtered_text = re.sub(r'\s+|\r|#', '', text[0:-2])  # 文本
                seg_text = jieba.lcut(filtered_text, cut_all=False)
                dataset.append(seg_text)

        # 自己训练词向量
        model = Word2Vec(dataset[0:37885], window=10, min_count=3, vector_csize=embed_size, sg=1)
        model.save('vector_save/word2vec.model')
        model.wv.save_word2vec_format('vector_save/word2vec.txt')

        # 加载训练好的词向量
        # model = KeyedVectors.load_word2vec_format('Pre-Training_vector/word2vec_wiki.txt')

        vocab = model.wv.vocab
        for x, text in enumerate(dataset, 0):
            for y, word in enumerate(text, 0):
                if y >= 200:
                    break
                if word in vocab:
                    vectors[x][y] = model[word]
                else:
                    vectors[x][y] = np.zeros(embed_size)

        self.dataset = torch.tensor(vectors).view(42620, 1, 200, embed_size)
        self.label = torch.tensor(labels)

        if artificial_features:
            features = np.zeros((42620, 64), dtype='float32')
            with open(features_path, 'r', encoding='utf-8') as f_features:
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


def train(train_loader, model, criterion, optimizer, epoch, artificial_features=False):
    model.train()
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
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
        optimizer.zero_grad()   # 梯度清零
        loss.backward()         # 反向传播求解梯度
        # nn.utils.clip_grad_value_(model.parameters(), clip_value=5)  # 梯度裁剪
        optimizer.step()        # 更新权重参数

        running_loss += loss.item()
        if batch_idx % 30 == 29:
            print('[{}, {:>5}] train loss: {:.6f}'.format(epoch + 1, batch_idx + 1, running_loss / 30))
            running_loss = 0


def test(test_loader, model, criterion, name, epoch, artificial_features=False):
    model.eval()
    correct = 0
    total = 0
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
            correct += (pred == target).sum().item()
            total += target.size(0)
            running_loss += loss.item()
        print('test loss: {:.6f} Accuracy on {} set: {:.2f}% [{}/{}]'
              .format(running_loss / len(test_loader), name, 100 * correct / total, correct, total))
        # 保存模型
        torch.save(model.state_dict(), 'model_save/model' + '_%d.pth' % (epoch + 1))
        return 100 * correct / total


def main():
    dataset = MyDataset()
    dataset = list(dataset)
    train_dataset = dataset[0:37885]
    valid_dataset = dataset[37885:42620]
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=256)

    model = TextCNN()
    # model = VariableCNNBiLSTMAttentionWithFeatures()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # CrossEntropy:alpha=None, gamma=0, eps=0.0
    # LabelSmoothingCrossEntropy: alpha=None, gamma=0
    # criterion = FocalLossLabelSmoothingCrossEntropy(alpha=[0.20, 0.4, 0.2, 0.2], gamma=2, eps=0.2)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    f = open(result_save_path, 'w', encoding='utf-8')
    for epoch in range(100):
        train(train_loader, model, criterion, optimizer, epoch)
        test(train_loader, model, criterion, 'train', epoch)
        valid_accuracy = test(valid_loader, model, criterion, 'valid', epoch)
        accuracy = round(valid_accuracy, 2)
        f.write(str(accuracy) + '\n')

    f.close()


if __name__ == '__main__':
    main()
