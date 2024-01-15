# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 17:23
# @Author  : wxb
# @File    : draw.py

import matplotlib.pyplot as plt
import numpy as np


f = open('train.96.last.0.log', mode='r', encoding='utf-8')
# f = open('train.336.last.24.log', mode='r', encoding='utf-8')

loss = []

for line in f:
    if 'train: Loss:' in line:
        loss.append(float(line.split()[-1]))

x = range(1, len(loss)+1)

plt.figure()
plt.plot(x, loss, label='loss', color='orange', marker='*')

plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.yticks(np.arange(0, 0.25, step=0.05))
plt.title(f'Changes in training loss (length=96)')
# plt.title(f'Changes in training loss (length=336)')
plt.show()
