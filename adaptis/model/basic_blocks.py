import torch.nn as nn

import adaptis.model.ops as ops


class ConvHead(nn.Module):
    def __init__(self, out_channels, in_channels=32, num_layers=1, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvHead, self).__init__()
        convhead = []

        for i in range(num_layers):
            convhead.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding),
                nn.ReLU(),
                norm_layer(in_channels) if norm_layer is not None else nn.Identity()
            ])
        convhead.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))

        self.convhead = nn.Sequential(*convhead)

    def forward(self, *inputs):
        return self.convhead(inputs[0])


class FCController(nn.Module):
    def __init__(self, input_size, layers_sizes,  activation='relu', norm_layer=nn.BatchNorm2d):
        super(FCController, self).__init__()

        # flag that indicates whether we use fully convolutional controller or not
        self.return_map = False

        # select activation function
        _activation = ops.select_activation_function(activation)

        controller = []
        for hidden_size in layers_sizes:
            controller.extend([
                nn.Conv2d(input_size, hidden_size, kernel_size=1),
                _activation(),
                norm_layer(hidden_size) if norm_layer is not None else nn.Identity()
            ])
            input_size = hidden_size
        self.controller = nn.Sequential(*controller)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x[:, :, None, None]
        x = self.controller(x)
        return x.squeeze(-1).squeeze(-1)


class SimpleConvController(nn.Module):
    def __init__(self, num_layers, in_channels, latent_channels,
                 kernel_size=1, activation='relu', norm_layer=nn.BatchNorm2d):
        super(SimpleConvController, self).__init__()

        # flag that indicates whether we use fully convolutional controller or not
        self.return_map = True

        # select activation function
        _activation = ops.select_activation_function(activation)

        controller = []
        for i in range(num_layers):
            controller.extend([
                nn.Conv2d(in_channels, latent_channels, kernel_size),
                _activation(),
                norm_layer(latent_channels) if norm_layer is not None else nn.Identity()
            ])
        self.controller = nn.Sequential(*controller)

    def forward(self, x):
        x = self.controller(x)
        return x


class SepConvHead(nn.Module):
    def __init__(self, num_outputs, in_channels, out_channels, num_layers=1,
                 kernel_size=3, padding=1, dropout_ratio=0.0, dropout_indx=0,
                 norm_layer=nn.BatchNorm2d):
        super(SepConvHead, self).__init__()

        layers = []

        for i in range(num_layers):
            layers.append(
                SeparableConv2D(in_channels if i == 0 else out_channels,
                                out_channels,
                                dw_kernel=kernel_size, dw_padding=padding,
                                norm_layer=norm_layer, activation='relu')
            )
            if dropout_ratio > 0 and dropout_indx == i:
                layers.append(nn.Dropout(dropout_ratio))

        layers.append(
            nn.Conv2d(out_channels, num_outputs, kernel_size=1, padding=0)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, *inputs):
        x = inputs[0]

        return self.layers(x)


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_padding, dw_stride=1,
                 activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2D, self).__init__()
        body = []
        _activation = ops.select_activation_function(activation)
        body.append(nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel,
                                stride=dw_stride, padding=dw_padding,
                                bias=use_bias,
                                groups=in_channels))
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias))

        if norm_layer:
            body.append(norm_layer(out_channels))
        if activation:
            body.append(_activation())

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)