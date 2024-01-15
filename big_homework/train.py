# -*- coding: utf-8 -*-
# @Time    : 2023/12/18 15:19
# @Author  : wxb
# @File    : train.py
import sys
from datetime import datetime, timedelta

from cmd import CMD
from pre_process import loader, collate_3
from models import transformer_model

from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

import random


class Train(CMD):
    def __call__(self, args):
        super(Train, self).__call__(args)
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
        print('train', len(self.trainset))
        print('dev', len(self.devset))
        print('test', len(self.testset))
        # create the model
        print("Create the model.")
        self.model = transformer_model(n_encoder_inputs=7,
                                       n_decoder_inputs=7
                                       ).to(args.device)
        # print(f"{self.model}")

        lr = 1e-4
        mu, nu = 0.9, 0.9
        epsilon = 1e-12
        decay = 0.01
        decay_epochs = 45
        print(f'lr={lr}')
        self.optimizer = Adam(self.model.parameters(),
                              lr,
                              (mu, nu),
                              epsilon)

        # self.scheduler = lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, patience=10, factor=0.1
        # )
        # decay_steps = decay_epochs * len(self.trainset)
        # self.scheduler = lr_scheduler.ExponentialLR(
        #     self.optimizer, decay ** (1 / decay_steps)
        # )

        total_time = timedelta()
        best_e, least_loss = 1, float('inf')

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            print(f"Epoch {epoch} / {args.epochs}:")
            self.train(self.trainset)

            # dev_loss = self.evaluate(self.devset, load)
            # print(f"{'dev:':6} MSE-Loss: {dev_loss:.4f}")
            # test_loss = self.evaluate(self.testset)
            # print(f"{'test:':6} Loss: {test_loss:.4f}")

            t = datetime.now() - start
            # # save the model if it is the best so far
            # if dev_loss < least_loss and epoch > args.patience // 5:
            #     best_e, least_loss = epoch, dev_loss
            #     self.model.save(args.save_path)
            #     print(f"{t}s elapsed (saved)\n")
            # else:
            #     print(f"{t}s elapsed\n")
            total_time += t
            # if epoch - best_e >= args.patience:
            #     break
        self.model.save(args.save_path)
        print(f"save model in {args.save_path}")
        self.model = transformer_model.load(args.save_path)
        dev_loss = self.evaluate(self.devset, load)
        print(f"{'dev:':6} MSE-Loss: {dev_loss:.4f}")
        loss = self.evaluate(self.testset)

        print(f"min loss of dev is {least_loss:.4f} at epoch {best_e}")
        print(f"the loss of test at epoch {best_e} is {loss:.4f}")

        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
