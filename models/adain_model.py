from .base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from . import networks
from argparse import ArgumentParser
from . import loss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import functools


class AdaInModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parent_parser, isTrain=True):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # AdaIn config
        parser.add_argument('--alpha', type=float, default=1.0)
        if isTrain:
            # Losses
            # mm = Moment Matching, gram = Gram matrix based, cmd = Central Moment Discrepancy
            parser.add_argument('--style_loss', type=str, default='mm', choices=['mm', 'gram', 'cmd'])
            parser.add_argument('--style_weight', type=float, default=10.0)
            parser.add_argument('--content_loss', type=str, default='mse', choices=['mse'])
            parser.add_argument('--content_weight', type=float, default=1.0)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        encoder = networks.VGGEncoder()
        decoder = networks.Decoder()
        adain = AdaInstanceNorm2d()
        init_network = functools.partial(
            networks.init_net, 
            init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids
        )
        self.net_encoder = init_network(encoder)
        self.net_decoder = init_network(decoder)
        self.net_adain = init_network(adain)

        self.alpha = opt.alpha

        self.visual_names = ['c', 'cs', 's']
        self.model_names = ['decoder', 'encoder', "adain"]

        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None

        if self.isTrain:
            self.loss_names = ['content', 'style']
            self.style_loss_cri = loss._get_style_loss(opt.style_loss)
            self.style_loss_weight = opt.style_weight
            self.content_loss_cri = loss._get_content_loss(opt.content_loss)
            self.content_loss_weight = opt.content_weight

            self.lr = opt.lr

            self.optimizer_g = Adam(
                list(self.net_decoder.parameters()) \
                    + list(self.net_adain.parameters()),
                lr=self.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_content = torch.tensor(0., device=self.device)
            self.loss_style = torch.tensor(0., device=self.device)

    def set_input(self, input_dict):
        self.c = input_dict['c'].to(self.device)
        self.s = input_dict['s'].to(self.device)
        self.image_paths = input_dict['name']


    def forward(self):
        self.content_embeddings, self.style_embeddings = self._encode(self.c, self.s)
        self.cs = self.net_decoder(self.content_embeddings[-1])

        self.output_embeddings = self.net_encoder(self.cs)
        

    def _encode(self, content, style):
        content_embeddings = self.net_encoder(content)
        style_embeddings = self.net_encoder(style)

        t = self.net_adain(content_embeddings[-1], style_embeddings[-1])
        t = self.alpha * t + (1 - self.alpha) * content_embeddings[-1]

        content_embeddings[-1] = t
        return content_embeddings, style_embeddings

    def compute_losses(self):
        self.loss_content = self.content_loss_weight \
            * self.content_loss_cri(self.content_embeddings[-1], 
                                    self.output_embeddings[-1])
        
        self.loss_style = 0
        for style_feats, output_feats in zip(self.style_embeddings, 
                                             self.output_embeddings):
            self.loss_style += self.style_loss_cri(style_feats, output_feats)
        
        self.loss_style *= self.style_loss_weight
    
    def optimize_parameters(self):
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_style
        loss.backward()
        self.optimizer_g.step()
    

class AdaInstanceNorm2d(nn.Module):
    def __init__(self, mlp_features=None):
        super().__init__()

        # If mlp_features is specified, the bias and scale are estimated by transforming a code vector,
        #   as in MUNIT (https://arxiv.org/pdf/1804.04732.pdf).
        if mlp_features is not None:
            in_features = mlp_features[0]
            out_features = mlp_features[1]

            self._scale = nn.Linear(in_features, out_features)
            self._bias = nn.Linear(in_features, out_features)
        # If mlp_features is not specified, the bias and scale are the mean and std of 2d feature maps,
        #   as in standard AdaIN (https://arxiv.org/pdf/1703.06868.pdf).
        else:
            self._scale = self._std
            self._bias = self._mean

    def forward(self, x, y):
        y_scale = self._scale(y).unsqueeze(-1).unsqueeze(-1)
        y_bias = self._bias(y).unsqueeze(-1).unsqueeze(-1)

        x = F.instance_norm(x)
        x = (x * y_scale) + y_bias
        return x

    def _std(self, x):
        return torch.std(x, dim=[2, 3])

    def _mean(self, x):
        return torch.mean(x, dim=[2, 3])