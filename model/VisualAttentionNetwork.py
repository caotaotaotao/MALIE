import torch.nn as nn
import torch
import model.BasicBlocks2 as BasicBlocks2
import torch.nn.functional as F
import model.Swimmodel as Swim
import model.swin_transformer as swin
#包含三个子网络，注意力机制用的GAM注意力，UNet中使用了非局部神经网络，用注意力来引导噪声网络。
class VisualAttentionNetwork(nn.Module):
    def __init__(self):
        super(VisualAttentionNetwork, self).__init__()
        self.feature_num = 64

        self.res_input_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1)  # 第一层卷积，输入通道3  输出通道64  卷积核3×3
        )
        #编码部分
        self.res_encoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 1),#卷积，输入通道64  输出通道64
            BasicBlocks2.Residual_Block_New_GAM(64, 64, 3),#残差块处理
        )
        self.down1 = DownSample(64)#下采样操作，将特征图尺寸降低
        self.res_encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 1),  # 卷积，输入通道64  输出通道128
            BasicBlocks2.Residual_Block_New_GAM(128, 128, 2),  # 残差处理
        )
        self.down2 = DownSample(128)#下采样操作，将特征图尺寸降低
        self.res_encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 1),  # 卷积  输入通道128  输出通道256
            BasicBlocks2.Residual_Block_New_GAM(256, 256, 1),  # 残差处理
        )
       #瓶颈注意力
        self.bottleneck = nn.Sequential(
            # 瓶颈部分的卷积层定义
            NonLocalBlock2(in_channels=256), # 在瓶颈处添加Non-local Neural Network模块
        )
       #解码部分
        self.res_decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, 1),#卷积层  输入256  输出通道256
            BasicBlocks2.Residual_Block_New_GAM(256, 256, 1),#残差处理

        )
        self.up2 = UpSample(256)#上采样操作，将特征图尺寸扩大

        self.res_decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),#卷积层  输入256  输入通道128
            BasicBlocks2.Residual_Block_New_GAM(128, 128, 2),#残差处理
        )
        self.up1 = UpSample(128)#上采样操作，将特征图尺寸扩大

        self.res_decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),#卷积层  输入128  输出通道64
            BasicBlocks2.Residual_Block_New_GAM(64, 64, 3),#残差处理
        )

        self.res_final = nn.Conv2d(64, 3, 3, 1, 1)#卷积层，输入通道64  输出通道3，卷积核3×3  变成和原始图像一样大小



    def forward(self, x, only_attention_output=False):
        #3层次内容
        res_input = self.res_input_conv(x)
        encoder1 = self.res_encoder1(res_input)
        encoder1_down = self.down1(encoder1)
        encoder2 = self.res_encoder2(encoder1_down)
        encoder2_down = self.down2(encoder2)
        encoder3 = self.res_encoder3(encoder2_down)
        bottleneck=self.bottleneck(encoder3)#瓶颈层处理
        decoder3 = self.res_decoder3(bottleneck) + encoder3
        decoder3 = self.up2(decoder3, output_size=encoder2.size())
        decoder2 = self.res_decoder2(decoder3) + encoder2
        decoder2 = self.up1(decoder2, output_size=encoder1.size())
        decoder1 = self.res_decoder1(decoder2) + encoder1
        output = self.res_final(decoder1)


        return output

#下采样
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

#上采样
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)#将输入张量的尺寸放大两倍
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out
#自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 计算query、key和value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        # 计算注意力分数
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)

        # 计算加权的value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # gamma系数控制残差连接
        out = self.gamma * out + x

        return out
#非局部神经块
class NonLocalBlock2(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock2, self).__init__()
        self.in_channels = in_channels

        self.theta = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.o = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        theta = self.theta(x).view(batch_size, channels // 8, height * width)
        phi = self.phi(x).view(batch_size, channels // 8, height * width)
        g = self.g(x).view(batch_size, channels // 2, height * width)

        attention = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=1)
        o = self.o(torch.bmm(g, attention.transpose(1, 2)).view(batch_size, channels // 2, height, width))
        out = self.gamma * o + x
        return out
