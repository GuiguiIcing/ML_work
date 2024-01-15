# -*- coding: utf-8 -*-
# @Time    : 2023/12/8 17:06
# @Author  : wxb
# @File    : pre_process.py

import csv
import random
import sys

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def split_into_pieces(data, length=96):
    if length not in (96, 336):
        assert False, f'{length} not in (96, 336).'

    split_data = []
    total_length = len(data)
    print('data length:', total_length, end=' ')
    x_len = 96
    for i in range(total_length - x_len - length + 1):
        X = data[i: i + x_len]
        Y_in = data[i + x_len - 1: i + x_len + length - 1]
        Y = data[i + x_len: i + x_len + length]

        split_data.append((X, Y_in, Y))
    total_length = len(split_data)

    print('split_data length:', total_length)

    return split_data


def collate_3(batch):
    x, y_in, y = zip(*batch)
    x, y_in, y = np.array(x),  np.array(y_in), np.array(y)
    x = torch.tensor(x, dtype=torch.float)
    y_in = torch.tensor(y_in, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    return x, y_in, y


class loader:
    def __init__(self, length, pic_path, path='ETT-small/'):
        # load all data
        self.train_set = path + 'train_set.csv'
        self.eval_set = path + 'validation_set.csv'
        self.test_set = path + 'test_set.csv'

        self.length = length
        self.save_pic_path = pic_path

        self.header = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        self.standard = StandardScaler()
        self.open_load()

    def open_load(self):

        dataset = pd.read_csv(self.train_set).drop(labels='date', axis=1)
        dataset = self.standard.fit_transform(dataset)
        self.train_set = split_into_pieces(dataset, length=self.length)

        dataset = pd.read_csv(self.eval_set).drop(labels='date', axis=1)
        dataset = self.standard.fit_transform(dataset)
        self.eval_set = split_into_pieces(dataset, length=self.length)

        dataset = pd.read_csv(self.test_set).drop(labels='date', axis=1)
        dataset = self.standard.fit_transform(dataset)
        self.test_set = split_into_pieces(dataset, length=self.length)

    def de_standardize_draw(self, target, predict):
        index = 0
        for index in range(96):
            target = target[index].reshape(-1, 7).cpu().numpy()
            predict = predict[index].reshape(-1, 7).cpu().numpy()
            # (96, 7)
            target = self.standard.inverse_transform(target)
            # print('target')
            # print(target[:, 0])
            predict = self.standard.inverse_transform(predict)
            # print('predict')
            # print(predict[:, 0])

            for i in range(7):
                x = range(target.shape[0])
                trg = target[:, i]
                pred = predict[:, i]

                plt.figure()
                plt.plot(x, trg, label='target', color='orange')
                plt.plot(x, pred, label='predict', color='green')
                plt.legend()

                plt.xlabel('hour')
                plt.ylabel('value')
                plt.title(f'{self.header[i]}')
                plt.savefig(f'{self.save_pic_path}/{self.header[i]}.{index}.png')
                plt.close()
            print(f'Save figures in {self.save_pic_path}')

        # import sys
        # sys.exit(1)
