# -*- coding: utf-8 -*-
# @Time    : 2023/12/8 17:38
# @Author  : wxb
# @File    : cmd.py

import argparse
import random
import sys

from tqdm import tqdm

import torch
import torch.nn as nn


class CMD(object):
    def __call__(self, args: argparse.ArgumentParser):
        self.args = args
        self.criterion = nn.MSELoss()

    def train(self, loader):
        self.model.train()
        torch.set_grad_enabled(True)
        total_loss = 0
        for data in loader:
            self.optimizer.zero_grad()

            batch_input, batch_target_in, batch_target = data

            batch_input = batch_input.to(self.args.device)
            batch_target = batch_target.to(self.args.device)
            batch_target_in = batch_target_in.to(self.args.device)

            output = self.model((batch_input, batch_target_in))

            # loss = self.get_loss(output, batch_target)
            loss = self.criterion(output, batch_target)
            total_loss += loss.item()
            loss.backward()

            self.optimizer.step()

        print(f"{'train:':6} Loss: {total_loss/len(loader):.5f}")

    @torch.no_grad()
    def evaluate(self, loader, load=None):
        self.model.eval()
        total_loss_MSE, total_loss_smape = 0, 0
        index = 10
        print('evaluate and the draw index:', index)
        j = 0
        for data in loader:
            batch_input, batch_target_in, batch_target = data

            batch_size, seq_len, feature = batch_target_in.shape

            batch_input = batch_input.to(self.args.device)
            batch_target = batch_target.to(self.args.device)
            batch_target_in = batch_target_in.to(self.args.device)
            # output = self.model((batch_input, batch_target_in))
            output = self.model((batch_input, batch_target_in[:, :1, :]))
            for i in range(1, seq_len):
                batch_target_in[:, i, :] = output[:, -1, :]   # new test
                output = self.model((batch_input, batch_target_in[:, :(i+1), :]))

            if load and j == index:
                load.de_standardize_draw(batch_target, output)

            loss = self.criterion(output, batch_target)
            total_loss_MSE += loss.item()
            j += 1
            # break

        total_loss_MSE /= len(loader)
        # total_loss_MSE /= j
        return total_loss_MSE

    @staticmethod
    def get_loss(y_pred, target):
        y_pred = y_pred.view(-1)
        target = target.view(-1)
        loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
        return loss.mean()
