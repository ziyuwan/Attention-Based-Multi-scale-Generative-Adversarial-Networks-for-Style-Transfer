import functools
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Adam

from .base_model import BaseModel
from .networks import VGGNet, init_net


class DisModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parent_parser, isTrain=True):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # AdaIn config
        if isTrain:
            # Losses
            parser.add_argument('--dis_weight', type=float, default=1.0)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        encoder = VGGNet()
        decoder = StyleDecoder(1000)
        init_network = functools.partial(
            init_net,
            init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids
        )

        self.net_encoder = nn.DataParallel(encoder, opt.gpu_ids)
        self.embed_decoder = init_network(decoder)

        if self.isTrain:
            self.loss_names = ['dis']
            self.dis_loss_cri = nn.BCELoss()
            self.dis_loss_weight = opt.dis_weight

            self.lr = opt.lr_dis

            self.optimizer_g = Adam(
                list(self.net_encoder.parameters()) \
                + list(self.embed_decoder.parameters()),
                lr=self.lr)
            self.optimizers.append(self.optimizer_g)

    def set_input(self, input_dict):
        self.s_true = input_dict['s_true']
        self.s_fake = input_dict['s_fake']
        self.s_super = input_dict['s_super'].to(self.device)

    def forward(self):
        embed_true = self.net_encoder(self.s_true)
        embed_fake = self.net_encoder(self.s_fake)
        embed_super = self.net_encoder(self.s_super)

        logits_true = self.embed_decoder(embed_super - embed_true)
        logits_false = self.embed_decoder(embed_super - embed_fake)
        self.logits = torch.cat([logits_true, logits_false], dim=-1)

    def forward_generate(self):
        embed_fake = self.net_encoder(self.s_fake)
        embed_super = self.net_encoder(self.s_super)
        logits = self.embed_decoder(embed_super - embed_fake)
        labels = torch.ones_like(logits)
        return self.dis_loss_weight * self.dis_loss_cri(logits, labels)

    def compute_losses(self):
        labels = torch.ones_like(self.logits)
        labels[:, 1] = 0
        self.loss_dis = self.dis_loss_cri(self.logits, labels)

    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_dis
        loss.backward()
        self.optimizer_g.step()

    def freeze(self):
        for parameter in self.net_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.embed_decoder.parameters():
            parameter.requires_grad = False

    def unfreeze(self):
        for parameter in self.net_encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.embed_decoder.parameters():
            parameter.requires_grad = True


class StyleDecoder(nn.Module):
    def __init__(self, in_shape):
        super(StyleDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_shape, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, embed):
        return self.fc(embed)