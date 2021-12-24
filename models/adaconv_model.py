import functools
from argparse import ArgumentParser
from math import ceil, floor

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from . import loss
from . import networks
from .base_model import BaseModel


# TODO(ziyu): split modules into several files

class AdaConvModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parent_parser, isTrain=True):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # AdaIn config
        parser.add_argument('--style_size', type=int, default=256, help='Size of the input style image.')
        parser.add_argument('--style_channels', type=int, default=512,
                            help='Number of channels for the style descriptor.')
        parser.add_argument('--kernel_size', type=int, default=3, help='The size of the predicted kernels.')
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
        self.alpha = 1.0
        encoder = networks.VGGEncoder()
        style_size, style_channels, kernel_size = opt.style_size, opt.style_channels, opt.kernel_size
        style_in_shape = (
            encoder.out_channels,
            style_size // encoder.scale_factor,
            style_size // encoder.scale_factor
        )
        style_out_shape = (style_channels, kernel_size, kernel_size)
        style_encoder = GlobalStyleEncoder(
            in_shape=style_in_shape, out_shape=style_out_shape)
        decoder = AdaConvDecoder(
            style_channels=style_channels, kernel_size=kernel_size)

        init_network = functools.partial(
            networks.init_net,
            init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids
        )
        self.net_encoder = nn.DataParallel(encoder, opt.gpu_ids)
        self.net_decoder = init_network(decoder)
        self.net_style_encoder = init_network(style_encoder)
        self.visual_names = ['c', 'cs', 's']
        self.model_names = ['encoder', 'style_encoder', 'decoder']

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
                + list(self.net_style_encoder.parameters()),
                lr=self.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_content = torch.tensor(0., device=self.device)
            self.loss_style = torch.tensor(0., device=self.device)

            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(iter):
                return 1 / (1 + 0.0002 * iter)

            self.lr_scheduler = LambdaLR(self.optimizer_g, lr_lambda=lr_lambda)

    def setup(self, opt):
        super().setup(opt)
        self.schedulers = []

    def _encode(self, content, style):
        content_embeddings = self.net_encoder(content)
        style_embeddings = self.net_encoder(style)
        return content_embeddings, style_embeddings

    def _decode(self, content_embedding, style_embedding):
        style_embedding = self.net_style_encoder(style_embedding)
        output = self.net_decoder(content_embedding, style_embedding)
        return output

    def forward(self):
        self.content_embeddings, self.style_embeddings = self._encode(self.c, self.s)
        self.cs = self._decode(self.content_embeddings[-1], self.style_embeddings[-1])

        self.output_embeddings = self.net_encoder(self.cs)

    def compute_losses(self):
        self.loss_content = self.content_loss_weight \
                            * self.content_loss_cri(self.content_embeddings[-1],
                                                    self.output_embeddings[-1])

        self.loss_style = 0
        for style_feats, output_feats in zip(self.style_embeddings, self.output_embeddings):
            self.loss_style += self.style_loss_cri(style_feats, output_feats)

        self.loss_style *= self.style_loss_weight
        return self.loss_content + self.loss_style

    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        self.forward()
        loss = self.compute_losses()
        loss.backward()
        self.optimizer_g.step()
        self.lr_scheduler.step()


class AdaConv2d(nn.Module):
    """
    Implementation of the Adaptive Convolution block. Performs a depthwise seperable adaptive convolution on its input X.
    The weights for the adaptive convolutions are generated by a KernelPredictor module based on the style embedding W.
    The adaptive convolution is followed by a normal convolution.

    References:
        https://openaccess.thecvf.com/content/CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.pdf


    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by final convolution.
        kernel_size: The kernel size of the final convolution.
        n_groups: The number of groups for the adaptive convolutions.
            Defaults to 1 group per channel if None.

    Input shape:
        x: Input tensor.
        w_spatial: Weights for the spatial adaptive convolution.
        w_pointwise: Weights for the pointwise adaptive convolution.
        bias: Bias for the pointwise adaptive convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super().__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(ceil(padding), floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)

        # F.conv2d does not work with batched filters (as far as I can tell)...
        # Hack for inputs with > 1 sample
        ys = []
        for i in range(len(x)):
            y = self._forward_single(x[i:i + 1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def _forward_single(self, x, w_spatial, w_pointwise, bias):
        # Only square kernels
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x


class AdaConvDecoder(nn.Module):
    def __init__(self, style_channels, kernel_size):
        super().__init__()
        self.style_channels = style_channels
        self.kernel_size = kernel_size

        # Inverted VGG with first conv in each scale replaced with AdaConv
        group_div = [1, 2, 4, 8]
        n_convs = [1, 4, 2, 2]
        self.layers = nn.ModuleList([
            *self._make_layers(512, 256, group_div=group_div[0], n_convs=n_convs[0]),
            *self._make_layers(256, 128, group_div=group_div[1], n_convs=n_convs[1]),
            *self._make_layers(128, 64, group_div=group_div[2], n_convs=n_convs[2]),
            *self._make_layers(64, 3, group_div=group_div[3], n_convs=n_convs[3], final_act=False, upsample=False)])

    def forward(self, content, w_style):
        # Checking types is a bit hacky, but it works well.
        for module in self.layers:
            if isinstance(module, KernelPredictor):
                w_spatial, w_pointwise, bias = module(w_style)
            elif isinstance(module, AdaConv2d):
                content = module(content, w_spatial, w_pointwise, bias)
            else:
                content = module(content)
        return content

    def _make_layers(self, in_channels, out_channels, group_div, n_convs, final_act=True, upsample=True):
        n_groups = in_channels // group_div

        layers = []
        for i in range(n_convs):
            last = i == n_convs - 1
            out_channels_ = out_channels if last else in_channels
            if i == 0:
                layers += [
                    KernelPredictor(in_channels, in_channels,
                                    n_groups=n_groups,
                                    style_channels=self.style_channels,
                                    kernel_size=self.kernel_size),
                    AdaConv2d(in_channels, out_channels_, n_groups=n_groups)]
            else:
                layers.append(nn.Conv2d(in_channels, out_channels_, 3,
                                        padding=1, padding_mode='reflect'))

            if not last or final_act:
                layers.append(nn.ReLU())

        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        return layers


class GlobalStyleEncoder(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        channels = in_shape[0]

        self.downscale = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            #
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )

        in_features = self.in_shape[0] * (self.in_shape[1] // 8) * self.in_shape[2] // 8
        out_features = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, xs):
        ys = self.downscale(xs)
        ys = ys.reshape(len(xs), -1)

        w = self.fc(ys)
        w = w.reshape(len(xs), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return w


class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_channels = style_channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size=kernel_size,
                                 padding=(ceil(padding), ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)
        bias = self.bias(w)
        bias = bias.reshape(len(w), self.out_channels)
        return w_spatial, w_pointwise, bias
