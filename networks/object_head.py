import torch
import torch.nn as nn


# obj decoder
class obj_regHead(nn.Module):
    def __init__(self, channels=256, inter_channels=None, joint_nb=21):
        super(obj_regHead, self).__init__()
        if inter_channels is None:
            inter_channels = channels // 2
            # inter_channels = channels
        # conv1
        self.conv1_1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.constant_(self.conv1_1.bias, 0)
        self.bn1_1 = nn.BatchNorm2d(inter_channels)
        self.conv1_2 = nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.constant_(self.conv1_2.bias, 0)
        self.bn1_2 = nn.BatchNorm2d(channels)
        # conv2
        self.conv2_1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.constant_(self.conv2_1.bias, 0)
        self.bn2_1 = nn.BatchNorm2d(inter_channels)
        self.conv2_2 = nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.constant_(self.conv2_2.bias, 0)
        self.bn2_2 = nn.BatchNorm2d(channels)
        # conv3
        self.conv3_1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.constant_(self.conv3_1.bias, 0)
        self.bn3_1 = nn.BatchNorm2d(inter_channels)
        self.conv3_2 = nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.constant_(self.conv3_2.bias, 0)
        self.bn3_2 = nn.BatchNorm2d(channels)
        # out conv regression
        self.out_conv = nn.Conv2d(channels, joint_nb*3, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.constant_(self.out_conv.bias, 0)
        # activation funcs
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # conv1
        out = self.leaky_relu(self.bn1_1(self.conv1_1(x)))
        out = self.leaky_relu(self.bn1_2(self.conv1_2(out)))
        # conv2
        out = self.leaky_relu(self.bn2_1(self.conv2_1(out)))
        out = self.leaky_relu(self.bn2_2(self.conv2_2(out)))
        # conv3
        out = self.leaky_relu(self.bn3_1(self.conv3_1(out)))
        out = self.leaky_relu(self.bn3_2(self.conv3_2(out)))
        # out conv regression
        out = self.out_conv(out)
        return out


# Pose2DLayer模型类的主要功能是从特征图中预测出关键点的二维坐标和置信度 ,模型的输入output:[b,21*3,32,32]即为21个关键点的三维坐标信息
class Pose2DLayer(nn.Module):
    def __init__(self, joint_nb=21):
        super(Pose2DLayer, self).__init__()
        self.coord_norm_factor = 10
        self.num_keypoints = joint_nb

    # output:[b,21*3,32,32]
    def forward(self, output, target=None, param=None):

        nB = output.data.size(0) # batch
        nA = 1 #锚点数量
        nV = self.num_keypoints #关键点数量(21)
        nH = output.data.size(2) #Height
        nW = output.data.size(3) #Weight

        # .contiguous()：由于转置操作可能导致内存不连续，使用contiguous函数确保张量的内存是连续的
        # => [3 * nV,nB * nA , nH * nW]
        output = output.view(nB * nA, (3 * nV), nH * nW).transpose(0, 1). \
            contiguous().view((3 * nV), nB * nA * nH * nW)  # => [3 * nV,nB * nA * nH * nW]   (24,S*S)---??这注释是不是错了？nV=21啊???

        # 将输出张量切分为置信度(conf)、x 坐标(x) 和 y 坐标(y)
        conf = torch.sigmoid(output[0:nV].transpose(0, 1).view(nB, nA, nH, nW, nV))  # (1,1,S,S,8)  #---通过sigmoid来计算置信度
        x = output[nV:2 * nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        y = output[2 * nV:3 * nV].transpose(0, 1).view(nB, nA, nH, nW, nV)

        grid_x = ((torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA * nV, 1, 1). \
                   view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nW) * self.coord_norm_factor
        grid_y = ((torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA * nV, 1, 1). \
                   view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nH) * self.coord_norm_factor
        grid_x = grid_x.permute(0, 1, 3, 4, 2).contiguous()  # (1,1,S,S,8)---？？？
        grid_y = grid_y.permute(0, 1, 3, 4, 2).contiguous()

        predx = x + grid_x
        predy = y + grid_y

        if self.training:
            predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
            predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor

            out_preds = [predx, predy, conf.view(nB, nH, nW, nV)]
            return out_preds

        else:
            predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
            predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor

            conf = conf.view(nB, nH, nW, nV).cpu().numpy()
            px = predx.cpu().numpy()
            py = predy.cpu().numpy()

            out_preds = [px, py, conf]
            return out_preds