# -*- coding: utf-8 -*-
# @Time    : 2023/12/8 20:28
# @Author  : wxb
# @File    : run.py

import random
import argparse
import torch
import os

from train import Train
from evaluate import Evaluate


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode',
                      type=str,
                      choices=['train', 'evaluate', 'only'])
    args.add_argument('--model-name',
                      type=str, default='transformer',
                      choices=['lstm', 'transformer', 'new'])
    args.add_argument('--length',
                      type=int, default=96,
                      choices=[96, 336])
    args.add_argument('--batch-size',
                      type=int, default=32)
    args.add_argument('--epochs',
                      type=int, default=1000)
    args.add_argument('--save-path',
                      type=str, default='exp/transformer/model.pt')
    args.add_argument('--save-pic-path',
                      type=str, default='pic.96/')
    args.add_argument('--patience',
                      type=int, default=20)
    args.add_argument('--device',
                      type=str, default='4')
    args.add_argument('--seed',
                      type=int, default=24, choices=[24, 0, 1, 2, 3])

    args = args.parse_args()

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.save_pic_path):
        os.mkdir(args.save_pic_path)

    print(args)
    if args.mode == 'train':
        train = Train()
        train(args)
    elif args.mode == 'evaluate':
        eval = Evaluate()
        eval(args)