import logging
import pdb
from collections import Counter
from functools import partial

# from .transformer_base import MultiheadAttention, SimpleTransformer  # Not available
from typing import Any, Optional, Union

import einops
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_


# Placeholder classes for missing transformer_base module
class MultiheadAttention(nn.Module):
    """Placeholder for MultiheadAttention."""

    def __init__(self, embed_dim, num_heads, bias=True, add_bias_kv=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Placeholder implementation
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleTransformer(nn.Module):
    """Placeholder for SimpleTransformer."""

    def __init__(
        self,
        embed_dim,
        num_blocks,
        ffn_dropout_rate=0.0,
        drop_path_rate=0.0,
        attn_target=None,
        pre_transformer_layer=None,
        post_transformer_layer=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Placeholder implementation
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


def IMU_instantiate_trunk(
    embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
):
    return SimpleTransformer(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        ffn_dropout_rate=0.0,
        drop_path_rate=drop_path,
        attn_target=partial(
            MultiheadAttention,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=True,
            add_bias_kv=add_bias_kv,
        ),
        pre_transformer_layer=nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
            EinOpsRearrange("b l d -> l b d"),
        ),
        post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
    )


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                planes,
                planes * self.expansion,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.clone() + identity
        out = self.relu(out)

        return out


class FcBlock(nn.Module):
    # FcBlock(self.lstm_size, output_dim, drop_ratio)
    def __init__(
        self, in_dim, out_dim, mid_dim=256, dropout=0.5, cfg=None
    ):  # original: mid_dim=256, dropout=0.2 could modify to 512
        super(FcBlock, self).__init__()

        self.mid_dim = mid_dim  # 256
        self.in_dim = in_dim
        self.out_dim = out_dim

        # fc layers
        self.fcs = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, self.out_dim),
        )

    def forward(self, x):
        # Handle case where x might be a tuple (from LSTM output)
        if isinstance(x, tuple):
            x = x[0]  # Take the first element (output tensor)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x


class ResNetLSTMSeqNet(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(ResNetLSTMSeqNet, self).__init__()
        data_window_config = dict(
            [
                (
                    "past_data_size",
                    int(cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"]),
                ),  # 0*100
                (
                    "window_size",
                    int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"]),
                ),  # 1.0*100.0
                (
                    "future_data_size",
                    int(cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"]),
                ),  # 0.0*100.0
                (
                    "step_size",
                    int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"]),
                ),
            ]
        )  # 100.0/20
        input_dim = cfg["model_param"]["input_dim"]  # 6
        output_dim = cfg["model_param"]["output_dim"]  # 3
        layer_sizes = cfg["model_param"]["layer_sizes"]  # [2, 2, 2, 2]
        drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.num_layers = cfg["model_param"]["lstm_layers"]  # 1
        self.win_size = (
            data_window_config["window_size"]
            + data_window_config["past_data_size"]
            + data_window_config["future_data_size"]
        )  # 100.0

        self.num_direction = 1
        self.res_net_out_channel = 128
        self.resnet_code = self.res_net_out_channel * int(
            self.win_size / 16 + 1
        )  # 128*7

        self.base_plane = 64
        self.inplanes = self.base_plane  # 64
        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(
                input_dim,
                self.base_plane,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),  # 6 64
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
        )
        # Residual groups
        self.residual_groups = nn.Sequential(
            self.stack_res_layres(ResBlock, 64, layer_sizes[0], stride=1),
            self.stack_res_layres(ResBlock, 128, layer_sizes[1], stride=2),
            self.stack_res_layres(ResBlock, 256, layer_sizes[2], stride=2),
            self.stack_res_layres(ResBlock, 512, layer_sizes[3], stride=2),
        )
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(
                512, self.res_net_out_channel, kernel_size=1, bias=False
            ),  # 512 128
            nn.BatchNorm1d(self.res_net_out_channel),  # 128
        )

        # LSTM
        self.lstm = nn.LSTM(
            self.resnet_code,
            self.lstm_size,
            self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
            bidirectional=False,
        )

        # Output module
        self.output_block1 = FcBlock(
            in_dim=self.lstm_size, out_dim=output_dim, dropout=drop_ratio
        )  # dp mean
        self.output_block2 = FcBlock(
            in_dim=self.lstm_size, out_dim=output_dim, dropout=drop_ratio
        )  # dp cov

        self.initialize()

    def freeze_cov(self):
        for param in self.output_block2.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True

    def stack_res_layres(self, block, planes, layer_sizes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, layer_sizes):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # layer 1
                orthogonal_(m.weight_ih_l0)
                orthogonal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)

                # layer 2
                if self.num_layers > 1:
                    orthogonal_(m.weight_ih_l1)
                    orthogonal_(m.weight_hh_l1)
                    m.bias_ih_l1.data.zero_()
                    m.bias_hh_l1.data.zero_()
                    n = m.bias_hh_l1.size(0)
                    start, end = n // 4, n // 2
                    m.bias_hh_l1.data[start:end].fill_(1.0)

    def init_hidden(self, x, batch_size, first_batch=False):

        weight = next(self.parameters()).data
        if first_batch:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                )
        else:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                )

        return hidden

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, compute_type=None, hn=None, cn=None):
        self.lstm.flatten_parameters()
        batch_size = x.size(0)  # [1, 10, 6, 100]
        seq_len = x.size(1)  # 10
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))  # [10, 6, 100]

        # Resnet_encode
        x = self.input_block(x)  # 10*6*100->10*64*50
        x = self.residual_groups(x)  # 10*64*50->10*512*7
        embed = self.resnet_post_pro(x)  # 10*512*7->10*128*7

        # LSTM
        embed = embed.view(batch_size, seq_len, -1)  # 10*128*7-> 1*10*896
        if hn == None or cn == None:
            (hn, cn) = self.init_hidden(
                x, batch_size
            )  # x:10*512*7  hn:1*1*256  1*1*256
        out, (hn2, cn2) = self.lstm(embed, (hn, cn))  # 1*10*256
        out = out.contiguous().view(
            -1, self.lstm_size * self.num_direction
        )  # out:10*256   lstm_size=256 num_direction=1

        # If used as trunk (no compute_type), return LSTM features only
        if compute_type is None:
            return out

        # If used as standalone model, process through output blocks
        x1 = self.output_block1(out)  # mean  10*3 Linear layer converts 10*256->10*3
        x1 = x1.view(batch_size, seq_len, -1)  # 1*10*3

        if compute_type == "standalone":
            x2 = self.output_block2(out)  # covariance s = log(sigma) out:10*256 -> 10*3
            x2 = x2.view(batch_size, seq_len, -1)  # (1,10,3)
            return x1, x2
        else:
            return x1


class ResNetLSTMSeqNet_Light(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(ResNetLSTMSeqNet_Light, self).__init__()
        data_window_config = dict(
            [
                (
                    "past_data_size",
                    int(cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"]),
                ),  # 0*100
                (
                    "window_size",
                    int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"]),
                ),  # 1.0*100.0
                (
                    "future_data_size",
                    int(cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"]),
                ),  # 0.0*100.0
                (
                    "step_size",
                    int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"]),
                ),
            ]
        )  # 100.0/20
        input_dim = cfg["model_param"]["input_dim"]  # 6
        output_dim = cfg["model_param"]["output_dim"]  # 3
        layer_sizes = cfg["model_param"]["layer_sizes"]  # [2, 2, 2, 2]
        drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.num_layers = cfg["model_param"]["lstm_layers"]  # 1
        self.use_transformer = cfg["model_param"]["use_transformer"]
        self.win_size = (
            data_window_config["window_size"]
            + data_window_config["past_data_size"]
            + data_window_config["future_data_size"]
        )  # 100.0
        self.num_direction = 1
        self.res_net_out_channel = 128
        self.resnet_code = self.res_net_out_channel * int(
            self.win_size / 16 + 1
        )  # 128*13
        self.base_plane = 64
        self.inplanes = self.base_plane  # 64
        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(
                input_dim,
                self.base_plane,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),  # 6 64
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
        )
        # Residual groups
        self.residual_groups = nn.Sequential(
            self.stack_res_layres(ResBlock, 64, layer_sizes[0], stride=1),
            self.stack_res_layres(ResBlock, 128, layer_sizes[1], stride=2),
            self.stack_res_layres(ResBlock, 256, layer_sizes[2], stride=2),
            # self.stack_res_layres(ResBlock, 512, layer_sizes[3], stride=2),
        )
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(
                256, self.res_net_out_channel, kernel_size=1, bias=False
            ),  # 512 128
            nn.BatchNorm1d(self.res_net_out_channel),  # 128
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.res_net_out_channel,
                self.res_net_out_channel,
                kernel_size=1,
                stride=2,
                bias=False,
            ),  # 512 128
            nn.BatchNorm1d(self.res_net_out_channel),  # 128
        )

        # LSTM
        self.lstm = nn.LSTM(
            self.resnet_code,
            self.lstm_size,
            self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
            bidirectional=False,
        )
        if self.use_transformer:
            self.IMU_Trunk = IMU_instantiate_trunk(
                embed_dim=self.lstm_size,
                num_blocks=6,
                num_heads=8,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=0.7,
            )

        # Output module
        self.output_block1 = FcBlock(
            self.lstm_size, output_dim, dropout=drop_ratio
        )  # dp mean
        self.output_block2 = FcBlock(
            self.lstm_size, output_dim, dropout=drop_ratio
        )  # dp cov

        self.initialize()

    def freeze_cov(self):
        for param in self.output_block2.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True

    def stack_res_layres(self, block, planes, layer_sizes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, layer_sizes):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # layer 1
                orthogonal_(m.weight_ih_l0)
                orthogonal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)

                # layer 2
                if self.num_layers > 1:
                    orthogonal_(m.weight_ih_l1)
                    orthogonal_(m.weight_hh_l1)
                    m.bias_ih_l1.data.zero_()
                    m.bias_hh_l1.data.zero_()
                    n = m.bias_hh_l1.size(0)
                    start, end = n // 4, n // 2
                    m.bias_hh_l1.data[start:end].fill_(1.0)

    def init_hidden(self, x, batch_size, first_batch=False):

        weight = next(self.parameters()).data
        if first_batch:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                )
        else:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                )

        return hidden

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, compute_type=None, hn=None, cn=None):

        # x:[32, 10, 6, 200] [bs, win_size, dim, frames]
        self.lstm.flatten_parameters()
        batch_size = x.size(
            0
        )  # [1, 10, 6, 200] B*10*6*200  B * seq_len * 6(acc+gyro) * freq
        seq_len = x.size(1)  # 10
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))  # [BS*10, 6, 200]
        # Resnet_encode
        x = self.input_block(x)  # 10*6*200->10*64*100
        x = self.residual_groups(x)  # 10*64*100->10*256*25
        embed = self.resnet_post_pro(x)  # 10*256*25->10*128*13

        # LSTM
        embed = embed.view(batch_size, seq_len, -1)  # 10*128*25-> 1*10*3200
        if hn == None or cn == None:
            (hn, cn) = self.init_hidden(
                x, batch_size
            )  # x:bs*10*256*25  hn:1*bs*64  1*bs*64
        out, (hn2, cn2) = self.lstm(embed, (hn, cn))  # 1*10*256
        if self.use_transformer:
            out = self.IMU_Trunk(out)
        out = out.contiguous().view(
            -1, self.lstm_size * self.num_direction
        )  # out:10*256   lstm_size=256 num_direction=1

        # FC
        x1 = self.output_block1(out)  # mean  10*3 Linear layer converts 10*256->10*3
        x1 = x1.view(batch_size, seq_len, -1)  # 1*10*3

        if compute_type == None:
            x2 = self.output_block2(out)  # covariance s = log(sigma) out:10*256 -> 10*3
            x2 = x2.view(batch_size, seq_len, -1)  # (1,10,3)
            return x1, x2
        else:
            return x1  # [bs, 10, 3]


class Crossxy_LSTM_Model(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(Crossxy_LSTM_Model, self).__init__()
        data_window_config = dict(
            [
                (
                    "past_data_size",
                    int(cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"]),
                ),  # 0*100
                (
                    "window_size",
                    int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"]),
                ),  # 1.0*100.0
                (
                    "future_data_size",
                    int(cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"]),
                ),  # 0.0*100.0
                (
                    "step_size",
                    int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"]),
                ),
            ]
        )  # 100.0/20
        input_dim = cfg["model_param"]["input_dim"]  # 6
        output_dim = cfg["model_param"]["output_dim"]  # 3
        layer_sizes = cfg["model_param"]["layer_sizes"]  # [2, 2, 2, 2]
        drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.num_layers = cfg["model_param"]["lstm_layers"]  # 1
        self.use_transformer = cfg["model_param"]["use_transformer"]
        self.win_size = (
            data_window_config["window_size"]
            + data_window_config["past_data_size"]
            + data_window_config["future_data_size"]
        )  # 100.0
        self.num_direction = 1
        self.res_net_out_channel = 128
        self.resnet_code = self.res_net_out_channel * int(
            self.win_size / 16 + 1
        )  # 128*13
        self.base_plane = 64
        self.inplanes = self.base_plane  # 64
        self.types = ["car", "drone", "dog", "human"]
        self.output_dim = output_dim
        self.drop_ratio = drop_ratio

        # Input module

        self.accel_xy_conv = nn.Conv2d(
            1, self.base_plane // 8, kernel_size=2, stride=2, padding=(0, 1), bias=False
        )
        self.accel_xz_conv = nn.Conv2d(
            1, self.base_plane // 8, kernel_size=2, stride=2, padding=(0, 1), bias=False
        )
        self.accel_yz_conv = nn.Conv2d(
            1, self.base_plane // 8, kernel_size=2, stride=2, padding=(0, 1), bias=False
        )
        self.gyro_xy_conv = nn.Conv2d(
            1, self.base_plane // 8, kernel_size=2, stride=2, padding=(0, 1), bias=False
        )
        self.gyro_xz_conv = nn.Conv2d(
            1, self.base_plane // 8, kernel_size=2, stride=2, padding=(0, 1), bias=False
        )
        self.gyro_yz_conv = nn.Conv2d(
            1, self.base_plane // 8, kernel_size=2, stride=2, padding=(0, 1), bias=False
        )

        self.kernels2x2 = [
            self.accel_xy_conv,
            self.accel_xz_conv,
            self.accel_yz_conv,
            self.gyro_xy_conv,
            self.gyro_xz_conv,
            self.gyro_yz_conv,
        ]

        self.accel_xyz_conv = nn.Conv2d(
            1,
            self.base_plane // 16,
            kernel_size=3,
            stride=2,
            padding=(0, 1),
            bias=False,
        )
        self.gyro_xyz_conv = nn.Conv2d(
            1,
            self.base_plane // 16,
            kernel_size=3,
            stride=2,
            padding=(0, 1),
            bias=False,
        )

        self.kernels3x3 = [
            self.accel_xyz_conv,
            self.gyro_xyz_conv,
        ]

        self.accelgyroconv = nn.Conv2d(
            1, self.base_plane // 8, kernel_size=6, stride=2, padding=(0, 2), bias=False
        )
        self.input_block = nn.Sequential(
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
        )
        # Residual groups
        self.residual_groups = nn.Sequential(
            self.stack_res_layres(ResBlock, 64, layer_sizes[0], stride=1),
            self.stack_res_layres(ResBlock, 128, layer_sizes[1], stride=2),
            self.stack_res_layres(ResBlock, 256, layer_sizes[2], stride=2),
            # self.stack_res_layres(ResBlock, 512, layer_sizes[3], stride=2),
        )
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(
                256, self.res_net_out_channel, kernel_size=1, bias=False
            ),  # 512 128
            nn.BatchNorm1d(self.res_net_out_channel),  # 128
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.res_net_out_channel,
                self.res_net_out_channel,
                kernel_size=1,
                stride=2,
                bias=False,
            ),  # 512 128
            nn.BatchNorm1d(self.res_net_out_channel),  # 128
        )
        # LSTM
        self.lstm = nn.LSTM(
            self.resnet_code,
            self.lstm_size,
            self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
            bidirectional=False,
        )
        if self.use_transformer:
            self.IMU_Trunk = IMU_instantiate_trunk(
                embed_dim=self.lstm_size,
                num_blocks=6,
                num_heads=8,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=0.7,
            )

        self.initialize()

    def freeze_cov(self):
        for param in self.output_block2.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True

    def stack_res_layres(self, block, planes, layer_sizes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, layer_sizes):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # layer 1
                orthogonal_(m.weight_ih_l0)
                orthogonal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)

                # layer 2
                if self.num_layers > 1:
                    orthogonal_(m.weight_ih_l1)
                    orthogonal_(m.weight_hh_l1)
                    m.bias_ih_l1.data.zero_()
                    m.bias_hh_l1.data.zero_()
                    n = m.bias_hh_l1.size(0)
                    start, end = n // 4, n // 2
                    m.bias_hh_l1.data[start:end].fill_(1.0)

    def init_hidden(self, x, batch_size, first_batch=False):

        weight = next(self.parameters()).data
        if first_batch:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                )
        else:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                )

        return hidden

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, compute_type=None, hn=None, cn=None):
        # x:[32, 10, 6, 200] [bs, win_size, dim, frames]
        self.lstm.flatten_parameters()
        batch_size = x.size(
            0
        )  # [1, 10, 6, 200] B*10*6*200  B * seq_len * 6(acc+gyro) * freq
        seq_len = x.size(1)  # 10
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))  # [BS*10, 6, 200]
        # cross xy:
        gyro_x = x[:, 0:1, ...]
        gyro_y = x[:, 1:2, ...]
        gyro_z = x[:, 2:3, ...]

        accel_x = x[:, 3:4, ...]
        accel_y = x[:, 4:5, ...]
        accel_z = x[:, 5:6, ...]

        view_shape = (gyro_x.shape[0], 1, 2, gyro_x.shape[2])

        kernel2x2data = [
            torch.hstack((accel_x, accel_y)),
            torch.hstack((accel_x, accel_z)),
            torch.hstack((accel_y, accel_z)),
            torch.hstack((gyro_x, gyro_y)),
            torch.hstack((gyro_x, gyro_z)),
            torch.hstack((gyro_y, gyro_z)),
        ]

        feature_2x2 = torch.hstack(
            [
                kernel(data.view((data.shape[0], 1, data.shape[1], data.shape[2])))
                for kernel, data in zip(self.kernels2x2, kernel2x2data)
            ]
        )

        feature_2x2 = feature_2x2[:, :, 0, :-1]

        kernel3x3data = [x[:, 0:3, ...], x[:, 3:6, ...]]

        feature_3x3 = torch.hstack(
            [
                kernel(data.view((data.shape[0], 1, data.shape[1], data.shape[2])))
                for kernel, data in zip(self.kernels3x3, kernel3x3data)
            ]
        )

        feature_3x3 = feature_3x3[:, :, 0, :]

        feature_6x6 = self.accelgyroconv(x.view(x.shape[0], 1, x.shape[1], x.shape[2]))[
            :, :, 0, :
        ]

        x_feature = torch.hstack((feature_2x2, feature_3x3, feature_6x6))
        # Resnet_encode
        x = self.input_block(x_feature)  # 10*6*200->10*64*100
        x = self.residual_groups(x)  # 10*64*100->10*256*25
        embed = self.resnet_post_pro(x)  # 10*256*25->10*128*13
        # LSTM
        embed = embed.view(batch_size, seq_len, -1)  # 10*128*25-> 1*10*3200
        if hn == None or cn == None:
            (hn, cn) = self.init_hidden(
                x, batch_size
            )  # x:bs*10*256*25  hn:1*bs*64  1*bs*64
        out, (hn2, cn2) = self.lstm(embed, (hn, cn))  # 1*10*256
        if self.use_transformer:
            out = self.IMU_Trunk(out)
        out = out.contiguous().view(
            -1, self.lstm_size * self.num_direction
        )  # out:10*256   lstm_size=256 num_direction=1

        return out


class LSTM_Model(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(LSTM_Model, self).__init__()
        data_window_config = dict(
            [
                (
                    "past_data_size",
                    int(cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"]),
                ),  # 0*100
                (
                    "window_size",
                    int(cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"]),
                ),  # 1.0*100.0
                (
                    "future_data_size",
                    int(cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"]),
                ),  # 0.0*100.0
                (
                    "step_size",
                    int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"]),
                ),
            ]
        )  # 100.0/20
        input_dim = cfg["model_param"]["input_dim"]  # 6
        output_dim = cfg["model_param"]["output_dim"]  # 3
        layer_sizes = cfg["model_param"]["layer_sizes"]  # [2, 2, 2, 2]
        drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.num_layers = cfg["model_param"]["lstm_layers"]  # 1
        self.use_transformer = cfg["model_param"]["use_transformer"]
        self.win_size = (
            data_window_config["window_size"]
            + data_window_config["past_data_size"]
            + data_window_config["future_data_size"]
        )  # 100.0
        self.num_direction = 1
        self.res_net_out_channel = 128
        self.resnet_code = self.res_net_out_channel * int(
            self.win_size / 16 + 1
        )  # 128*13
        self.base_plane = 64
        self.inplanes = self.base_plane  # 64
        self.types = ["car", "drone", "dog", "human"]
        self.output_dim = output_dim
        self.drop_ratio = drop_ratio

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(
                input_dim,
                self.base_plane,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),  # 6 64
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
        )
        # Residual groups
        self.residual_groups = nn.Sequential(
            self.stack_res_layres(ResBlock, 64, layer_sizes[0], stride=1),
            self.stack_res_layres(ResBlock, 128, layer_sizes[1], stride=2),
            self.stack_res_layres(ResBlock, 256, layer_sizes[2], stride=2),
            # self.stack_res_layres(ResBlock, 512, layer_sizes[3], stride=2),
        )
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(
                256, self.res_net_out_channel, kernel_size=1, bias=False
            ),  # 512 128
            nn.BatchNorm1d(self.res_net_out_channel),  # 128
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.res_net_out_channel,
                self.res_net_out_channel,
                kernel_size=1,
                stride=2,
                bias=False,
            ),  # 512 128
            nn.BatchNorm1d(self.res_net_out_channel),  # 128
        )
        # LSTM
        self.lstm = nn.LSTM(
            self.resnet_code,
            self.lstm_size,
            self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
            bidirectional=False,
        )
        if self.use_transformer:
            self.IMU_Trunk = IMU_instantiate_trunk(
                embed_dim=self.lstm_size,
                num_blocks=6,
                num_heads=8,
                pre_transformer_ln=False,
                add_bias_kv=True,
                drop_path=0.7,
            )

        self.initialize()

    def freeze_cov(self):
        for param in self.output_block2.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True

    def stack_res_layres(self, block, planes, layer_sizes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, layer_sizes):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # layer 1
                orthogonal_(m.weight_ih_l0)
                orthogonal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)

                # layer 2
                if self.num_layers > 1:
                    orthogonal_(m.weight_ih_l1)
                    orthogonal_(m.weight_hh_l1)
                    m.bias_ih_l1.data.zero_()
                    m.bias_hh_l1.data.zero_()
                    n = m.bias_hh_l1.size(0)
                    start, end = n // 4, n // 2
                    m.bias_hh_l1.data[start:end].fill_(1.0)

    def init_hidden(self, x, batch_size, first_batch=False):

        weight = next(self.parameters()).data
        if first_batch:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                    weight.new(
                        batch_size, self.num_layers * self.num_direction, self.lstm_size
                    ).zero_(),
                )
        else:
            if torch.cuda.is_available():
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    )
                    .zero_()
                    .to(x.device),
                )
            else:
                hidden = (
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                    weight.new(
                        self.num_layers * self.num_direction, batch_size, self.lstm_size
                    ).zero_(),
                )

        return hidden

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, compute_type=None, hn=None, cn=None):
        # x:[32, 10, 6, 200] [bs, win_size, dim, frames]
        self.lstm.flatten_parameters()
        batch_size = x.size(
            0
        )  # [1, 10, 6, 200] B*10*6*200  B * seq_len * 6(acc+gyro) * freq
        seq_len = x.size(1)  # 10
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))  # [BS*10, 6, 200]
        # Resnet_encode
        x = self.input_block(x)  # 10*6*200->10*64*100
        x = self.residual_groups(x)  # 10*64*100->10*256*25
        embed = self.resnet_post_pro(x)  # 10*256*25->10*128*13
        # LSTM
        embed = embed.view(batch_size, seq_len, -1)  # 10*128*25-> 1*10*3200
        if hn == None or cn == None:
            (hn, cn) = self.init_hidden(
                x, batch_size
            )  # x:bs*10*256*25  hn:1*bs*64  1*bs*64
        out, (hn2, cn2) = self.lstm(embed, (hn, cn))  # 1*10*256
        if self.use_transformer:
            out = self.IMU_Trunk(out)
        out = out.contiguous().view(
            -1, self.lstm_size * self.num_direction
        )  # out:10*256   lstm_size=256 num_direction=1

        return out


class OutputHead(nn.Module):
    """
    General purpose output head for a neural network model.

    Attributes:
        lstm_size (int): Size of the LSTM layer.
        output_dim (int): Dimension of the output.
        drop_ratio (float): Dropout ratio.
        compute_type (str): Type of computation to perform.
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
    """

    def __init__(self, cfg, motion_type: str):
        super(OutputHead, self).__init__()

        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.output_dim = cfg["model_param"]["output_dim"]  # 3
        self.drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2
        self.split_z = cfg["model_param"]["split_z"]  # True/False
        # Output module
        self.output_block1 = FcBlock(
            self.lstm_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
        )
        self.output_block2 = FcBlock(
            self.lstm_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
        )
        if self.split_z:
            self.output_block1 = FcBlock(
                self.lstm_size, self.output_dim - 1, dropout=self.drop_ratio, cfg=cfg
            )  # dp mean
            self.output_block2 = FcBlock(
                self.lstm_size, self.output_dim, dropout=self.drop_ratio, cfg=cfg
            )  # dp cov
            self.output_block1_z = FcBlock(
                self.lstm_size, 1, dropout=self.drop_ratio, cfg=cfg
            )  # dp cov
        self.motion_type = motion_type

        # Add learnable scale factor for velocity prediction
        self.velocity_scale = nn.Parameter(torch.ones(1, self.output_dim))
        self.use_velocity_scale = cfg.get("model", {}).get("pred_velocity", False)

    def forward(self, out, batch_size, seq_len, predict_cov=False):
        """
        Forward pass of the module.

        Args:
            out: Input tensor.

        Returns:
            Tensor: Output after forward pass.
        """
        # logging.info(f"Inside current head runing: {self.motion_type}")
        if self.split_z:
            x = self.output_block1(out)  # mean  10*3 Linear layer converts 10*256->10*3
            z = self.output_block1_z(out)
            x1 = torch.cat((x, z), dim=1)
            x1 = x1.view(batch_size, seq_len, -1)
        else:
            x1 = self.output_block1(out)
            x1 = x1.view(batch_size, seq_len, -1)

        # Apply learnable scale factor if velocity prediction is enabled
        if self.use_velocity_scale:
            x1 = x1 * self.velocity_scale

        if predict_cov:
            x2 = self.output_block2(out)
            x2 = x2.view(batch_size, seq_len, -1)
            return x1, x2
        else:
            return x1


class FoundationModel(nn.Module):
    def __init__(self, cfg, trunk: nn.Module, heads: nn.ModuleDict):
        super(FoundationModel, self).__init__()
        self.model = trunk
        self.heads = heads
        self.types = ["car", "dog", "drone", "human"]
        self.lstm_size = cfg["model_param"]["lstm_size"]  # 256
        self.lstm_dropout = cfg["model_param"]["lstm_dropout"]  # 0.0
        self.output_dim = cfg["model_param"]["output_dim"]  # 3
        self.drop_ratio = cfg["model_param"]["drop_ratio"]  # 0.5 original:0.2

    def forward(self, x, motion_type=None, predict_cov=False, compute_all_heads=True):
        """
        Forward pass with optional efficient computation.

        Args:
            x: Input tensor
            motion_type: Motion type tensor or None
            predict_cov: Whether to predict covariance
            compute_all_heads: If True, compute all heads. If False, compute only needed heads.
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Forward pass through shared backbone
        model_output = self.model(x)

        # Handle case where trunk model returns a tuple (output, hidden_states)
        if isinstance(model_output, tuple):
            model_output = model_output[0]  # Take the first element (output tensor)

        if compute_all_heads:
            # Original behavior - compute all heads
            return self._compute_all_heads(
                model_output, batch_size, seq_len, predict_cov
            )
        else:
            # Efficient behavior - compute only needed heads
            return self._compute_needed_heads(
                model_output, batch_size, seq_len, predict_cov, motion_type
            )

    def _compute_all_heads(self, model_output, batch_size, seq_len, predict_cov):
        """Original implementation - compute all heads."""
        multi_head = {}
        multi_head_cov = {}

        if predict_cov:
            car_head, car_cov = self.heads["car"](
                model_output, batch_size, seq_len, predict_cov
            )
            dog_head, dog_cov = self.heads["dog"](
                model_output, batch_size, seq_len, predict_cov
            )
            drone_head, drone_cov = self.heads["drone"](
                model_output, batch_size, seq_len, predict_cov
            )
            human_head, human_cov = self.heads["human"](
                model_output, batch_size, seq_len, predict_cov
            )

            multi_head["car"] = car_head
            multi_head["dog"] = dog_head
            multi_head["drone"] = drone_head
            multi_head["human"] = human_head

            multi_head_cov["car"] = car_cov
            multi_head_cov["dog"] = dog_cov
            multi_head_cov["drone"] = drone_cov
            multi_head_cov["human"] = human_cov

            return multi_head, multi_head_cov
        else:
            car_head = self.heads["car"](model_output, batch_size, seq_len)
            dog_head = self.heads["dog"](model_output, batch_size, seq_len)
            drone_head = self.heads["drone"](model_output, batch_size, seq_len)
            human_head = self.heads["human"](model_output, batch_size, seq_len)

            multi_head["car"] = car_head
            multi_head["dog"] = dog_head
            multi_head["drone"] = drone_head
            multi_head["human"] = human_head

            return multi_head

    def _compute_needed_heads(
        self, model_output, batch_size, seq_len, predict_cov, motion_type
    ):
        """Efficient computation - only compute heads that have data."""
        if motion_type is None:
            # Fallback to all heads if no motion type specified
            return self._compute_all_heads(
                model_output, batch_size, seq_len, predict_cov
            )

        # Determine which heads are needed based on motion type
        needed_heads = self._get_needed_heads(motion_type)

        multi_head = {}
        multi_head_cov = {}

        for motion_type_name in needed_heads:
            if predict_cov:
                pred, cov = self.heads[motion_type_name](
                    model_output, batch_size, seq_len, predict_cov
                )
                multi_head[motion_type_name] = pred
                multi_head_cov[motion_type_name] = cov
            else:
                pred = self.heads[motion_type_name](model_output, batch_size, seq_len)
                multi_head[motion_type_name] = pred

        if predict_cov:
            return multi_head, multi_head_cov
        else:
            return multi_head

    def _get_needed_heads(self, motion_type):
        """Determine which heads are needed based on motion type in batch."""
        if isinstance(motion_type, torch.Tensor):
            # Get unique motion types in batch
            unique_types = torch.unique(motion_type).tolist()
        else:
            unique_types = [motion_type]

        # Map motion type IDs to head names
        motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}
        needed_heads = []

        for motion_id in unique_types:
            if motion_id in motion_types:
                needed_heads.append(motion_types[motion_id])

        return needed_heads

    def get_num_params(self):
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
