# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextCNN(nn.Module):
    def __init__(self, embed_size=128, kernel_num=64, outputs_size=4):
        super(TextCNN, self).__init__()
        self.kernel_num = kernel_num
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(2, embed_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(3, embed_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(4, embed_size))
        self.pooling1 = nn.MaxPool2d((200 - 1, 1))
        self.pooling2 = nn.MaxPool2d((200 - 2, 1))
        self.pooling3 = nn.MaxPool2d((200 - 3, 1))
        self.fc = nn.Linear(kernel_num * 3, outputs_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.pooling1(F.relu(self.conv1(x)))
        x2 = self.pooling2(F.relu(self.conv2(x)))
        x3 = self.pooling3(F.relu(self.conv3(x)))
        x = torch.cat((x1.view(-1, self.kernel_num), x2.view(-1, self.kernel_num), x3.view(-1, self.kernel_num)), dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x

class TextRNN(nn.Module):
    def __init__(self, embed_size=128, hidden_size=64, outputs_size=4):
        super(TextRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, outputs_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.squeeze(1)
        outputs, (h_n, c_n) = self.lstm(x)
        x_f = h_n[-2, :, :]
        x_b = h_n[-1, :, :]
        x = torch.cat((x_f, x_b), dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        return x

class TextRCNN(nn.Module):
    def __init__(self, embed_size=128, hidden_size=64, outputs_size=4):
        super(TextRCNN, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.pooling = nn.MaxPool2d((200, 1))
        self.fc = nn.Linear(hidden_size * 2 + embed_size, outputs_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.squeeze(1)
        outputs, (h_n, c_n) = self.lstm(x)
        x = torch.cat((outputs, x), dim=2)
        x = torch.tanh(x)

        x = self.pooling(x)
        x = x.squeeze(1)

        x = self.dropout(x)
        x = self.fc(x)
        return x



