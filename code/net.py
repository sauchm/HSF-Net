# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual_dense_block import RDB
from utils import *



BN_MOMENTUM = 0.1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out

class better_upsampling(nn.Module):
      def __init__(self, in_ch, out_ch):
          super(better_upsampling, self).__init__()
          self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)

      def forward(self, x,y):
          x = nn.functional.interpolate(x,size= y.size()[2:], mode='nearest', align_corners=None)
          x = F.pad(x, (3 // 2, int(3 / 2), 3 // 2, int(3 / 2)))
          x = self.conv(x)
          return x

class down_Block(nn.Module):
    def __init__(self, in_channels, stride=2):
        kernel_size=3
        super(down_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


class RDBG(nn.Module):
    def __init__(self,CH=16):
        super(RDBG, self).__init__()
        self.conv1= RDB(CH,4,16)
        self.conv2= RDB(CH,4,16)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2

class feature_fusion(nn.Module):
    def __init__(self):
        super(feature_fusion, self).__init__()
        self.conv1 = RDBG(32)
        self.conv2 = RDBG(64)
        self.conv3 = RDBG(64)
        self.down = down_Block(32)
        self.up = better_upsampling(64,32)
    def forward(self,x,y):
        x = self.conv1(x)
        y = self.conv2(y)
        y = self.conv2(y)
        x = x + self.up(y,x)
        y = y + self.down(x)
        return x,y

class final_Net(nn.Module):
    def __init__(self):
        super(final_Net, self).__init__()
        self.conv01 = BasicBlock(3,16)
        self.conv02 = RDBG(16)
        self.down1 = down_Block(16)
        self.conv03= RDBG(32)
        self.down2 = down_Block(32)
        self.block_1 = feature_fusion()
        self.block_2 = feature_fusion()
        self.conv04= RDBG(32)
        self.up = better_upsampling(32,16)
        self.conv05= RDBG(16)
        self.conv06= BasicBlock(16,16)
        self.conv07= BasicBlock(16,3)

    def forward(self, x):
        x = self.conv01(x)
        x = self.conv02(x)
        y = self.down1(x)
        y = self.conv03(y)
        z = self.down2(y)
        y,z = self.block_1(y,z)
        y,z = self.block_2(y,z)
        y = self.conv04(y)
        y = self.up(y,x)
        x = x+y
        x = self.conv05(x)
        x = self.conv06(x)
        x = self.conv07(x)
        return (x)
