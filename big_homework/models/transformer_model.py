# -*- coding: utf-8 -*-
# @Time    : 2023/12/8 17:06
# @Author  : wxb
# @File    : transformer_model.py

import torch
import torch.nn as nn


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class transformer_model(nn.Module):
    def __init__(self,
                 n_encoder_inputs,
                 n_decoder_inputs,
                 channels=512,
                 dropout=0.1,
                 ):
        super(transformer_model, self).__init__()
        self.n_encoder_inputs = n_encoder_inputs
        self.n_decoder_inputs = n_decoder_inputs

        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = nn.Linear(n_encoder_inputs, channels)
        self.output_projection = nn.Linear(n_decoder_inputs, channels)

        self.linear = nn.Linear(channels, 7)

        self.do = nn.Dropout(p=self.dropout)

    def forward(self, x):
        src, trg = x
        src = self.encode_src(src)
        out = self.decode_trg(trg=trg, memory=src)

        return out

    def encode_src(self, src):

        src_start = self.input_projection(src).permute(1, 0, 2)   # (seq_len, b, 512)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)  # (seq_len, b, 512)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory):

        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['n_encoder_inputs'], state['n_decoder_inputs'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)
        return model

    def save(self, path):
        state_dict = self.state_dict()
        state = {
            # 'args': self.args,
            'n_encoder_inputs': self.n_encoder_inputs,
            'n_decoder_inputs': self.n_decoder_inputs,
            'state_dict': state_dict,
        }
        torch.save(state, path)
