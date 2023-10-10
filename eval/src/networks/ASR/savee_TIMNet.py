import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from base import BaseNet


# Implementation of Keras conv1d with padding mode 'causal'
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class SAVEE_TIMNET(BaseNet):
    r"""
        Reimplementation for TIMNET.
        Remember for Keras conv1d takes in (batch size, num#, num of channel / filters num),
        however for Pytorch conv1d takes in (batch size, num of channel / filters num, num#)
    """

    def __init__(self,
                 input_shape=None,
                 nb_filters=39,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=8,
                 activation="relu",
                 dropout_rate=0.1,
                 return_sequences=True,
                 name='TIMNET'):
        super(SAVEE_TIMNET, self).__init__()
        # store model size parameter
        if input_shape is None:
            input_shape = [39, 188]
        self.name = name
        self.input_shape = input_shape  # features * MFCCs
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.supports_masking = True
        self.mask_value = 0.

        # create left most 1x1 conv layer
        self.forward_convd_net = CausalConv1d(in_channels=self.input_shape[0], out_channels=self.nb_filters,
                                              kernel_size=1, dilation=1)
        self.backward_convd_net = CausalConv1d(in_channels=self.input_shape[0], out_channels=self.nb_filters,
                                               kernel_size=1, dilation=1)

        # create tabs
        self.forward_tabs = nn.ModuleList()
        self.backward_tabs = nn.ModuleList()

        for s in range(self.nb_stacks):
            for i in [2 ** i for i in range(self.dilations)]:
                tab_f = nn.Sequential(OrderedDict([
                    ('conv_1', CausalConv1d(in_channels=self.input_shape[0], out_channels=self.nb_filters,
                                            kernel_size=kernel_size, dilation=i)),
                    ('bn_1', nn.BatchNorm1d(num_features=self.nb_filters, affine=True)),
                    ('relu_1', nn.ReLU()),
                    ('drop_1', nn.Dropout1d(p=dropout_rate)),

                    ('conv_2', CausalConv1d(in_channels=self.input_shape[0], out_channels=self.nb_filters,
                                            kernel_size=kernel_size, dilation=i)),
                    ('bn_2', nn.BatchNorm1d(num_features=self.nb_filters, affine=True)),
                    ('relu_2', nn.ReLU()),
                    ('drop_2', nn.Dropout1d(p=dropout_rate))
                ]))
                tab_b = nn.Sequential(OrderedDict([
                    ('conv_1', CausalConv1d(in_channels=self.input_shape[0], out_channels=self.nb_filters,
                                            kernel_size=kernel_size, dilation=i)),
                    ('bn_1', nn.BatchNorm1d(num_features=self.nb_filters, affine=True)),
                    ('relu_1', nn.ReLU()),
                    ('drop_1', nn.Dropout1d(p=dropout_rate)),

                    ('conv_2', CausalConv1d(in_channels=self.input_shape[0], out_channels=self.nb_filters,
                                            kernel_size=kernel_size, dilation=i)),
                    ('bn_2', nn.BatchNorm1d(num_features=self.nb_filters, affine=True)),
                    ('relu_2', nn.ReLU()),
                    ('drop_2', nn.Dropout1d(p=dropout_rate))
                ]))
                self.forward_tabs.append(tab_f)
                self.backward_tabs.append(tab_b)

        self.global_avg_pool = nn.AvgPool2d((1, self.input_shape[1]))
        if not isinstance(nb_filters, int):
            raise Exception()

    def forward(self, inputs, mask=None):
        if self.dilations is None:
            self.dilations = 8
        forward = inputs
        backward = torch.flip(inputs, dims=[1])
        # print(f"Initial forward shape: {forward.shape}")
        # print(f"Initial backward shape: {backward.shape}")

        # left most 1x1 conv layer
        forward_convd = self.forward_convd_net(forward)
        backward_convd = self.backward_convd_net(backward)
        skip_out_forward = forward_convd
        skip_out_backward = backward_convd
        # print(f"forward_convd shape: {forward_convd.shape}")
        # print(f"backward_convd shape: {backward_convd.shape}")

        # begin tabs iteration
        final_skip_connection = []
        tab_idx = 0
        for s in range(self.nb_stacks):
            for i in [2 ** i for i in range(self.dilations)]:
                # forward tab process
                tab_f = self.forward_tabs[tab_idx]
                output_f = tab_f(skip_out_forward)
                output_f = torch.sigmoid(output_f)
                skip_out_forward = skip_out_forward * output_f

                # backward tab process
                tab_b = self.backward_tabs[tab_idx]
                output_b = tab_b(skip_out_backward)
                output_b = torch.sigmoid(output_b)
                skip_out_backward = skip_out_backward * output_b

                # print(f"skip_out_forward shape in tab {tab_idx}: {skip_out_forward.shape}")
                # print(f"skip_out_backward shape in tab {tab_idx}: {skip_out_backward.shape}")

                # adding and produce gn
                temp_skip = skip_out_backward + skip_out_forward
                # global avg pooling, turning each feature to scalar
                temp_skip = self.global_avg_pool(temp_skip)
                # append the gn for final dynammic fusion
                final_skip_connection.append(temp_skip)

                # print(f"temp_skip shape in tab {tab_idx}: {temp_skip.shape}")

                tab_idx += 1

        # Concatenate the each batch_size * MFCCs * 1 ==> batch_size * MFCCs * Dilation_number
        output_2 = final_skip_connection[0]
        for i, item in enumerate(final_skip_connection):
            if i == 0:
                continue
            output_2 = torch.cat((output_2, item), -1)
        x = output_2

        # print(f"Final output shape: {x.shape}")
        return x


