import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, input_size, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(input_size, channels, 3, 1)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        xc = x.clone()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x + xc
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type="batch"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
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
        self.conv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 3, 9, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, input, target):
        target = target.detach()
        self.loss = F.mse_loss(input, target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, input, target):
        target = gram_matrix(target_feature).detach()
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, target)
        return input

class LossNetwork(nn.Module):
    def __init__(self, style_img):
        super(LossNetwork, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.norm = Normalization(norm_mean, norm_std)

        feats = models.vgg19(pretrained=True).features[:11]

        self.conv1 = feats[0]
        self.relu1 = feats[1]
        self.conv2 = feats[2]
        self.relu2 = feats[3]
        self.pool2 = feats[4]
        self.conv3 = feats[5]
        self.relu3 = feats[6]
        self.conv4 = feats[7]
        self.relu4 = feats[8]
        self.pool4 = feats[9]
        self.conv4 = feats[10]

        self.style_img = self.norm(style_img.detach())
        self.style_loss = StyleLoss()
        self.content_loss = ContentLoss()

    def forward(self, x):
        content_layers = ['conv4']
        style_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

        s_loss = 0
        c_loss = 0

        x = self.norm(x)
        xc = x.clone()

        x = self.relu1(self.conv1(x))
        s_loss += self.style_loss(x, self.style_img)
        x = self.relu2(self.conv2(x))
        s_loss += self.style_loss(x, self.style_img)
        x = self.relu3(self.conv3(self.pool2(x)))
        s_loss += self.style_loss(x, self.style_img)
        x = self.relu4(self.conv4(x))
        s_loss += self.style_loss(x, self.style_img)
        c_loss += self.content_loss(x, xc)
        x = self.conv5(self.pool4(x))
        s_loss += self.style_loss(x, self.style_img)
        return x, c_loss, s_loss
