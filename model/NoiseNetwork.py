import torch.nn as nn
import torch
import torch.nn.utils.spectral_norm as spectral_norm
import model.BasicBlocks as BasicBlocks
import torch.nn.functional as F
import numpy as np

class NoiseNetwork(nn.Module):
    def __init__(self):
        super(NoiseNetwork, self).__init__()
        self.feature_num = 64
        self.res_input_conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1)  # 第一层卷积，输入通道6  输出通道64  卷积核3×3
        )
        # 膨胀卷积层
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=4),  # 使用膨胀卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=4),  # 使用膨胀卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        #第二个卷积层，缩小通道恢复原状
        self.conv_block = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3),  # 减小通道数
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3,kernel_size=15, stride=1, padding=0, output_padding=0),
            nn.ReLU(inplace=True)
        )


    def forward(self, x, attention):
        res_input = self.res_input_conv(torch.cat([x, attention], 1))
        #print("经过第一个卷积层的尺寸")
        #print(res_input.shape)

        y = self.conv_blocks(res_input)
        #print("y的shape")
        #print(y.shape)
        
        y1=self.conv_block(y)
        #print("y1的shape")
        #print(y1.shape)
        output = y1 + attention

        return output
