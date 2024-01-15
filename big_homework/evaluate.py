# -*- coding: utf-8 -*-
# @Time    : 2023/12/21 16:09
# @Author  : wxb
# @File    : evaluate.py

import sys
from datetime import datetime

from cmd import CMD
from pre_process import loader, collate_3
from models import transformer_model

from torch.utils.data import DataLoader


class Evaluate(CMD):
    def __call__(self, args):
        super(Evaluate, self).__call__(args)
        print('Preprocess the data...')
        load = loader(args.length, args.save_pic_path)

        train, dev, test = (
            load.train_set, load.eval_set, load.test_set)
        self.trainset = DataLoader(train, batch_size=args.batch_size,
                                   shuffle=True, collate_fn=collate_3)
        self.devset = DataLoader(dev, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_3)
        self.testset = DataLoader(test, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_3)
        # print('train', len(self.trainset))
        print('dev', len(self.devset))
        print('test', len(self.testset))

        start = datetime.now()
        self.model = transformer_model.load(args.save_path)
        loss = self.evaluate(self.devset, load)
        print(f"the loss of dev is {loss:.4f}")

        # loss = self.evaluate(self.trainset)
        # print(f"the loss of train is {loss:.4f}")

        loss = self.evaluate(self.testset)
        print(f"the loss of test is {loss:.4f}")

        total_time = datetime.now() - start

        print(f"{total_time}s elapsed")
