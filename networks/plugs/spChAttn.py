import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 来自知乎：

# 空间注意力模块
class SpGate(nn.Module):
    def __init__(self, channel = 256):
        super(SpGate, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # self.sum = nn.Parameter(torch.tensor([1],dtype=torch.float32), requires_grad=False)

        # self.highpass_kernel = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32).unsqueeze(0).to(device)

    def forward(self, x):

        # 这个高通滤波器，提取的轮廓，大部分应该是主要特征对应的轮廓吧？所以应该用来增强主要还是增强次要呢？
        # print("self.highpass_kernel.shape",self.highpass_kernel.shape)
        

        z = self.squeeze(x)
        z = self.sigmoid(z)

        # filtered_feature_map = F.conv2d(z, weight=self.highpass_kernel, padding=1)
        # single_feature_map2 = filtered_feature_map[0, 0, :, :]
        # # 创建一个新的图像来展示单个特征图
        # plt.imshow(single_feature_map2.cpu().detach().numpy(), cmap='gray')
        # plt.axis('off')  # 关闭坐标轴显示
        # plt.show()

        # z = self.sigmoid(z + filtered_feature_map)

        # first_feature_map = x[0]

        # reshaped_feature_map = first_feature_map.unsqueeze(1)  # 在通道维度上增加一个维度

        # # 创建一个新的图像来展示第一个特征图
        # plt.imshow(make_grid(reshaped_feature_map).permute(1, 2, 0).cpu().detach().numpy())
        # plt.axis('off')  # 关闭坐标轴显示
        # plt.show()

        # single_feature_map = z[0, 0, :, :]
        # # 创建一个新的图像来展示单个特征图
        # plt.imshow(single_feature_map.cpu().detach().numpy(), cmap='gray')
        # plt.axis('off')  # 关闭坐标轴显示
        # plt.show()
        # return xxx

        # z2 = 1-z
        # single_feature_map2 = z2[0, 0, :, :]
        # # 创建一个新的图像来展示单个特征图
        # plt.imshow(single_feature_map2.cpu().detach().numpy(), cmap='gray')
        # plt.axis('off')  # 关闭坐标轴显示
        # plt.show()
        # return xxx

        # print("self.sum = ",self.sum.data)

        # return x * z, x * (self.sum-z)
        return x * z, x * (1-z)


class SpGate2(nn.Module):
    def __init__(self, in_channels = 256,rate=4):
        super(SpGate2, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        z = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        return x * z, x * (1-z)

class SpatialAttention2d2(nn.Module):
    def __init__(self, channel = 256):
        super(SpatialAttention2d2, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.squeeze.weight)  # 使用 Xavier 均匀分布初始化权重

        # self.highpass_kernel = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32).unsqueeze(0).to(device)

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)

        return x * z

# 通道注意力模块，和SE类似
class GABGate(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GABGate, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z, x * (1-z)

# 通道注意力模块，和SE类似
class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

# 空间+通道，双管齐下
# class SC(nn.Module):
#     def __init__(self, dim = 256):
#         super(SC, self).__init__()
#         self.satt = SpatialAttention2d2(dim)
#         self.catt = GAB(dim)

#     def forward(self, x):
#         z1 = self.satt(x)
#         z2 = self.catt(x)
#         # z = z1 * z2
#         return 


# 空间+通道，双管齐下
class SCse(nn.Module):
    def __init__(self, dim = 256):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d2(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)