from torch import nn
import torch.nn.functional as F
import torch
from networks.plugs.SimAM import Simam_module as SimAM
from networks.plugs.LKA import SpatialAttention as lka
from networks.plugs.GCT import GCT
from .DWConv import DwConv

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = torch.sigmoid(outputs) * 0.2 + 0.5
        return inputs * outputs

class SELayer(nn.Module):
    def __init__(self, channel = 256, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False, device='cuda:0'),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False, device='cuda:0'),
            nn.Sigmoid()
        )
        self.apply(init_weights)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        # y.expand_as(x):将y[b,c,1,1]扩展成和x一样的[b,c,h,w]
        ecaScale = y.expand_as(x)
        return x * ecaScale

# from cbam
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) # [N, in_planes, 1, 1]
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class RSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            # dilation=1,
            bias=False)
        # self.se_block = nn.Sequential(  
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2),
        #     )
            
        # self.se_block = SELayer(channel = self.out_channels)
        # self.se_block = SimAM()
        # self.se_block = lka(out_channels)
        self.se_block = ChannelAttention(self.out_channels)
        self.shortcut = shortcut
        self.apply(init_weights)

    def forward(self, ins):
        def checkpoint(model, input):
            if torch.cuda.is_available():
                return torch.utils.checkpoint.checkpoint(model, input)
            else:
                return model(input)

        x = self.in_conv(ins)
        if self.shortcut:
            out = x + checkpoint(self.se_block,x)
            # out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        
        return out
        # return x

class RSEFPN(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            self.inp_conv.append(
                # nn.Conv2d(
                #     in_channels=out_channels,
                #     out_channels=out_channels // 4,
                #     kernel_size=1,
                #     bias=False)
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=1,
                    shortcut=shortcut)
                # DwConv(
                #     out_channels,
                #     out_channels // 4,
                #     kernel_size=3,
                # )
            )

    def forward(self, x):
        def checkpoint(model, input):
            if torch.cuda.is_available():
                return torch.utils.checkpoint.checkpoint(model, input)
            else:
                return model(input)
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(
            in5, scale_factor=2, mode="bilinear", align_corners=None)  # 1/16
        out3 = in3 + F.interpolate(
            out4, scale_factor=2, mode="bilinear", align_corners=None)  # 1/8
        out2 = in2 + F.interpolate(
            out3, scale_factor=2, mode="bilinear", align_corners=None)  # 1/4

        # p5 = self.inp_conv[3](in5)
        # p4 = self.inp_conv[2](out4)
        # p3 = self.inp_conv[1](out3)
        # p2 = self.inp_conv[0](out2)

        p5 = checkpoint(self.inp_conv[3],in5)
        p4 = checkpoint(self.inp_conv[2],out4)
        p3 = checkpoint(self.inp_conv[1],out3)
        p2 = checkpoint(self.inp_conv[0],out2)

        p5 = F.interpolate(p5, scale_factor=8, mode="bilinear", align_corners=None)
        p4 = F.interpolate(p4, scale_factor=4, mode="bilinear", align_corners=None)
        p3 = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=None)

        # fuse = torch.cat([p5, p4, p3, p2], dim=1)
        # return fuse
        # shuffle
        b,_,h,w = p5.shape
        fuse = torch.cat([p5.unsqueeze(1), p4.unsqueeze(1), p3.unsqueeze(1), p2.unsqueeze(1)], dim=1) # [b,4,c//4,h,w]
        fuse = fuse.permute(0, 2, 1, 3, 4)
        fuse = fuse.reshape(b,-1,h,w)
        return fuse
        