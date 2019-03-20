import torch.nn as nn
import torch.nn.functional as F
from unet.unet_parts import *
from unet.unet_model import *
from torch import autograd
import torch


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""

    def __init__(self, z_dim=256, image_size=227, conv_dim=64):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim * 8, int(image_size / 16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # If image_size is 64, output shape is as below.
        out = self.fc(z)  # (?, 512, 4, 4)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 64, 32, 32)
        out = F.tanh(self.deconv4(out))  # (?, 3, 64, 64)
        return out


class U_Generator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(U_Generator, self).__init__()
        self.inc = inconv(n_channels, 64)
        # ------------            backup part
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        # self.outc = outconv(64, n_classes)
        # ------------------------ backup part

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.down4 = down(256, 64)
        # self.down4 = down(256, 256)
        # self.up1 = up(512, 256)
        self.up1 = up(320, 256)
        self.up2 = up(512, 64)
        self.up3 = up(192, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""

    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.fc = conv(conv_dim * 8, 1, int(image_size / 16), 1, 0, False)

    def forward(self, x):  # If image_size is 64, output shape is as below.
        out = F.leaky_relu(self.conv1(x), 0.05)  # (?, 64, 32, 32)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 512, 4, 4)
        out = self.fc(out).squeeze()
        return out


class our_Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
        super(our_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 4, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 1, 1, bias=False),
            nn.Conv2d(conv_dim * 8, conv_dim * 8, 1, 1, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(conv_dim * 8, 1, 1, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):  # If image_size is 64, output shape is as below.
        output = self.main(x)
        m = nn.Softmax()
        for idx in range(output.size()[0]):
            temp = m(output[idx, 0, :, :])
            output.clone()[idx, 0, :, :] = temp
        return output.view(-1, 1).squeeze(1)


class U_Discriminator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(U_Discriminator, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Alex_disc(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
        super(Alex_disc, self).__init__()
        self.conv1=nn.Conv2d(1,96,11,stride=4)
        self.conv11 = nn.Conv2d(96,96,1,stride=1)
        self.conv12 = nn.Conv2d(96,96,1,stride=1)
        self.conv2=nn.Conv2d(96, 96, 3, stride=2)
        self.conv21 = nn.Conv2d(96,96,1,stride=1)
        self.conv22 = nn.Conv2d(96,96,1,stride=1)
        self.conv3=nn.Conv2d(96, 256, 5, stride=2)
        self.conv31 = nn.Conv2d(256,512,1,stride=1)
        self.conv32 = nn.Conv2d(512,1024,1,stride=1)
        self.conv33 = nn.Conv2d(1024,1024,1,stride=1)
        self.conv4=nn.Conv2d(1024, 1, 1, stride=1,bias=False)


    def forward(self, x):
        output=F.relu(self.conv1(x))
        output=F.relu(self.conv11(output))
        output=F.relu(self.conv12(output))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv21(output))
        output=F.relu(self.conv22(output))
        output=F.relu(self.conv3(output))
        output=F.relu(self.conv31(output))
        output=F.relu(self.conv32(output))
        output=F.relu(self.conv33(output))
        output=self.conv4(output)
        #print(output.size())

        outputvalue = output.view(-1, 1).squeeze(1)
        return outputvalue
