import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, input_size, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(input_size, channels, 3, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        xc = x.clone()
        x = F.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        x = x + xc
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type="instance"):
        super(ConvLayer, self).__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm_type = norm_type
        if norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == "none":
            self.norm_layer = None

    def forward(self, x):
        x = self.reflection_pad(x)
        if self.norm_type == "none":
            return self.conv_layer(x)
        else:
            return self.norm_layer(self.conv_layer(x))

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = F.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.norm_layer(self.conv2d(out))
        return out


class FastNeuralStyle(nn.Module):
    def __init__(self):
        super(FastNeuralStyle, self).__init__()

        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.conv2 = ConvLayer(32, 64, 3, stride=2)
        self.conv3 = ConvLayer(64, 128, 3, stride=2)
        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)
        self.conv4 = UpsampleConvLayer(128, 64, 3, 1, 2)
        self.conv5 = UpsampleConvLayer(64, 32, 3, 1, 2)
        self.conv6 = ConvLayer(32, 3, 9, stride=1, norm_type="none")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

# LOSS NETWORK

def content_loss(input, target):
    # b, c, h, w = input.shape
    target = target.detach()
    # target = torch.cat(b*[target.detach()])
    return F.mse_loss(input, target)

def gram_matrix(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x, x_t) / (C*H*W)

def style_loss(input, target):
    target = gram_matrix(target).detach()
    # b, c, h, w = target.shape
    # G = torch.cat(b *[gram_matrix(input)])
    G = gram_matrix(input)

    return F.mse_loss(G, target)

class LossNetwork(nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        feats = models.vgg19(pretrained=True).features
        self.features = feats

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_4', '22': 'relu4_2', '26': 'relu4_4', '35': 'relu5_4'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features
