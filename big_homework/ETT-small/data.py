# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 14:53
# @Author  : wxb
# @File    : data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


file = 'ETTh1.csv'
standard = StandardScaler()

dataset = pd.read_csv(file).drop(labels='date', axis=1)

dataset = np.array(dataset)

a = np.zeros((3, 7))
for i in range(7):
    a[1][i] = float('inf')

for each in dataset:
    for i in range(7):
        a[0][i] = each[i] if a[0][i] <= each[i] else a[0][i]   # max
        a[1][i] = each[i] if a[1][i] >= each[i] else a[1][i]   # min
        a[2][i] += each[i]

for i in range(7):
    a[2][i] /= dataset.shape[0]

for each in a:
    for k in each:
        print(f'{k:6.2f}', end=' ')
    print()
