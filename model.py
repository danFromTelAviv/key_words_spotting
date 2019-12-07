import torch
from torch import nn


class WaveNetConv(nn.Module):
    def __init__(self, num_features_in, num_features_out, res_features, filter_len, dilation):
        self.filter_len = filter_len
        super(WaveNetConv, self).__init__()
        self.conv_tanh = nn.Sequential(*[nn.Conv1d(num_features_in, num_features_out, filter_len, dilation=dilation),
                                         nn.Tanh()])
        self.conv_sig = nn.Sequential(*[nn.Conv1d(num_features_in, num_features_out, filter_len, dilation=dilation),
                                        nn.Sigmoid()])
        self.conv_lin = nn.Conv1d(num_features_out, num_features_out, 1, dilation=dilation)
        self.conv_res = nn.Conv1d(num_features_out, res_features, 1, dilation=dilation)
        self.norm = nn.BatchNorm1d(num_features_in)

    def forward(self, x, res):
        '''
        :param x: [batch,  features, timesteps,]
        '''
        x = self.norm(x)
        x_ = self.conv_tanh(x) * self.conv_sig(x)
        x_res = self.conv_res(x_)
        x_ = self.conv_lin(x_)
        if x_.shape[-1] != x.shape[-1]:
            padding = int((x.shape[-1] - x_.shape[-1]) // 2)
            x_ = x[:, :, padding:-padding] + x_
            res = res[:, :, padding:-padding] + x_res
        else:
            x_ = x + x_
            res = res + x_res
        return x_, res


class WaveNetBlock(nn.Module):
    def __init__(self, num_features_in, num_features_out, res_features, filter_len=3, dilations=[1, 2, 4, 8]):
        self.filter_len = filter_len
        super(WaveNetBlock, self).__init__()
        self.convs = nn.ModuleList([WaveNetConv(num_features_in, num_features_out, res_features, filter_len, dilation)
                                    for dilation in dilations])

    def forward(self, x, res):
        '''
        :param x: [batch, timesteps, features]
        '''
        for idx, conv in enumerate(self.convs):
            x, res = conv(x, res)
        return x, res


class KeyWordSpotter(nn.Module):
    def __init__(self, features_in, features_per_layer=16, res_features=32, num_blocks=6, filter_len=3,
                 dilations=[1, 2, 4, 8]):
        super(KeyWordSpotter, self).__init__()
        self.res_features = res_features
        self.mfcc_to_features = nn.Conv1d(features_in, features_per_layer, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(features_per_layer, features_per_layer, res_features, filter_len, dilations)
            for block_idx in range(num_blocks)])
        self.classifer = nn.Sequential(
            *[nn.ReLU(), nn.Conv1d(res_features, res_features, 1), nn.ReLU(), nn.Conv1d(res_features, 1, 1),
              nn.Sigmoid()])

    def forward(self, x: torch.tensor):
        '''

        :param x: [batch, timesteps, mfcc features]
        :return:
        '''
        x = x.transpose(1, 2).float()  # [batch, mfcc features,  timesteps,]
        x = self.mfcc_to_features(x)
        res = torch.zeros((x.shape[0], int(self.res_features), x.shape[-1])).to(x.device)
        for idx, block in enumerate(self.blocks):
            x, res = block(x, res)
        return self.classifer(res).transpose(1, 2).max(dim=-1)[0].max(dim=-1)[0]
