import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import ops
import torch

# from networks.plugs.deformableConv import DeformConv2d

#from nets.cbam import SpatialGate
from copy import deepcopy

# from networks.plugs.spChAttn import SpatialAttention2d2 as SpatialGate
# from networks.plugs.spChAttn import GAB

# from networks.plugs.dropblock import DropBlock2d

# from networks.plugs.msShuffle import msShuffle
# from networks.plugs.HLMixer import Mixer
from networks.plugs.RSE_FPN import RSELayer,RSEFPN
# from networks.plugs.SimAM import Simam_module as SimAM
# from networks.plugs.se_module import SELayer
# from networks.plugs.LKA import SpatialAttention as lka

# from networks.plugs.non_local_dot_product import *

# from .CR import Transformer

# from networks.plugs.spChAttn import SCse,GABGate,SpGate

# 预训练模型权重（用于初始化网络）
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

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

# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         # 全局特征聚合：通过在H和W维度上使用L2范数，把空间特征聚合成为一个向量，其实也可以使用类似SE里的全局平均池化层，主要用于获取全局性的通道信息。
#         Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
#         # 特征归一化：用于计算当前通道相对于其他通道的相对重要性，其值在0~1之间，该方法类似于SE里的sigmoid输出。
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         # 特征校准：这就是一个特征重标定的过程，特征归一化输出的其实是一个权重值，这个值载荷输入x相乘就能获得每个通道的重要程度，GRN中还加入了两个可学习参数gamma和beta用于优化
#         x = self.gamma * (x * Nx) + self.beta + x
#         x = x.permute(0, 3, 1, 2)
#         return x

# 改进的resNet50(encoder) + FPN(decoder) 
class FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(FPN, self).__init__()
        self.in_planes = 64

        print('basline')

        # 初始化手部网络resnet_hand
        resnet_hand = resnet50(pretrained=pretrained)

        #resnet_obj = resnet50(pretrained=pretrained)
        # 初始化物体网络resnet_obj，与手部采用一样的网络
        resnet_obj = deepcopy(resnet_hand) # 深拷贝，与resnet_hand是相互独立的

        

        # ----------hand-resNet50-stage0----------------
        self.layer0_h = nn.Sequential(resnet_hand.conv1, resnet_hand.bn1, resnet_hand.leakyrelu, resnet_hand.maxpool)
        # ----------hand-resNet50-stage1----------------
        self.layer1_h = nn.Sequential(resnet_hand.layer1)
        # ----------hand-resNet50-stage2----------------
        self.layer2_h = nn.Sequential(resnet_hand.layer2)
        # ----------hand-resNet50-stage3----------------
        self.layer3_h = nn.Sequential(resnet_hand.layer3)
        # ----------hand-resNet50-stage4----------------
        self.layer4_h = nn.Sequential(resnet_hand.layer4)

        # 对于obj，stage0、stage1、stage4阶段都是公用的，stage2、stage3是独立的！
        #self.layer0_o = nn.Sequential(resnet_obj.conv1, resnet_obj.bn1, resnet_obj.leakyrelu, resnet_obj.maxpool)
        #self.layer1_o = nn.Sequential(resnet_obj.layer1)
        # ----------obj-resNet50-stage2----------------
        self.layer2_o = nn.Sequential(resnet_obj.layer2)
        # ----------obj-resNet50-stage3----------------
        self.layer3_o = nn.Sequential(resnet_obj.layer3)
        #self.layer4_o = nn.Sequential(resnet_obj.layer4)


         

        # topLayerReduce
        # self.toplayer_h = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # self.toplayer_o = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # # Lateral layers（横向连接层,作用于下采样时某个stagen结束后得到的特征图）----作用:将来自浅层的高分辨率特征与来自深层的高语义特征进行融合
        # self.latlayer1_h = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2_h = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3_h = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # self.latlayer1_o = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2_o = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3_o = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # RSELayer
        # self.toplayer_h = RSELayer( 2048, 256, kernel_size=1)  # Reduce channels
        # self.toplayer_o = RSELayer( 2048, 256, kernel_size=1)  # Reduce channels

        # self.latlayer1_h = RSELayer( 1024, 256, kernel_size=1)
        # self.latlayer2_h = RSELayer( 512, 256, kernel_size=1)
        # self.latlayer3_h = RSELayer( 256, 256, kernel_size=1)

        # self.latlayer1_o = RSELayer( 1024, 256, kernel_size=1)
        # self.latlayer2_o = RSELayer( 512, 256, kernel_size=1)
        # self.latlayer3_o = RSELayer( 256, 256, kernel_size=1)

        self.RSEFPN_h = RSEFPN([256,512,1024,2048],256)
        self.RSEFPN_o = RSEFPN([256,512,1024,2048],256)

        # Smooth layers（平滑层）
        self.smooth3_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)   

        
        # self.spAttn_layer1 = SpatialGate(256)
        # self.GAB_layer3_h = GAB(1024)
        # self.GAB_layer3_o = GAB(1024)

        # gamma_h1 =  tensor([0.8608], device='cuda:0') ,gamma_h2 =  tensor([1.1867], device='cuda:0') ,gamma_h3 =  tensor([0.9611], device='cuda:0') ,gamma_h4 =  tensor([0.8846], device='cuda:0')
        # gamma_o1 =  tensor([0.7328], device='cuda:0') ,gamma_o2 =  tensor([1.1816], device='cuda:0') ,gamma_o3 =  tensor([0.9060], device='cuda:0') ,gamma_o4 =  tensor([0.9900], device='cuda:0')
        
        # shuffle
        # self.shuffle_layer2_h = msShuffle(512)
        # self.shuffle_layer2_o = msShuffle(512)

        # self.shuffle_layer3_h = msShuffle(1024)
        # self.shuffle_layer3_o = msShuffle(1024)
        
        # self.GRN_layer1_h = GRN(256)
        # self.GRN_layer1_o = GRN(256)

        # self.se_layer2_h = SELayer(512)
        # self.se_layer2_o = SELayer(512)

        # self.se_layer3_h = SELayer(1024)
        # self.se_layer3_o = SELayer(1024)
        
        # self.lka_layer2_h = lka(512)
        # self.lka_layer2_o = lka(512)

        # self.lka_layer3_h = lka(1024)
        # self.lka_layer3_o = lka(1024)

        # self.LKA_layer1_h = LKA(256)
        # self.LKA_layer1_o = LKA(256)

        # self.LKA_layer2_h = LKA(512)
        # self.LKA_layer2_o = LKA(512)

        # self.mixer = Mixer(512)
        # self.mixer.apply(init_weights)

        # self.simAM_layer2_h = SimAM()
        # self.simAM_layer2_o = SimAM()

        # self.simAM_layer3_h = SimAM()
        # self.simAM_layer3_o = SimAM()
        


        # self.non_local_layer1_h = NONLocalBlock2D(256, sub_sample=False, bn_layer=True)
        # self.non_local_layer1_o = NONLocalBlock2D(256, sub_sample=False, bn_layer=True)

    # 将x通过二位线性插值扩张到y一样的大小后每个像素相加并返回
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # F.interpolate 可以将特征图的尺寸调整为指定的大小（size=(H, W)），并使用双线性插值方法进行插值计算 
        # 这里的融合方式与FCN网络相同，都是通过将特征图x、y的各个像素相加。（通道数是不变的！）
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y  
        # 疑问1--？？？？这里是否能用转置卷积?(来自Unet Demo------nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2))
        # 疑问2--？？？？这里能否使用像Unet那样的通过堆叠通道的方式来融合下采样时的特征?---即【return torch.cat([y, x], dim=1)】

    def forward(self, x):
        def checkpoint(model, input):
            if torch.cuda.is_available():
                return torch.utils.checkpoint.checkpoint(model, input)
            else:
                return model(input)
        # def waveblock(x):
        #     import random
        #     if self.training:
        #         h, w = x.size()[-2:]
        #         rh = round(0.1 * h)
        #         sx = random.randint(0, h-rh)
        #         mask = (x.new_ones(x.size())) * 1.5
        #         mask[:, :, sx:sx+rh, :] = 1 
        #         x = x * mask 
        #     return x
    # ==============================以下为resnet50网络的部分（编码器)========================
        # Bottom-up
        # x:(b,3,224,224)
        c1_h = self.layer0_h(x) # (b,64,64,64)
        #c1_o = self.layer0_h(x)

        c2_h = self.layer1_h(c1_h) # (b,256,64,64)
        # c2_h = checkpoint(self.layer1_h,c1_h)

        # c2_h = checkpoint(self.non_local_layer1_h,c2_h)
        # c2_o = checkpoint(self.non_local_layer1_o,c2_h)
        
    # ---------hand-resNet50、obj-resNet50分别进行stage2、stage3-------------------
        # c2_h = waveblock(c2_h)
        # c2_o = waveblock(c2_h)

        # c2_h = checkpoint(self.GRN_layer1_h,c2_h)
        # c2_o = checkpoint(self.GRN_layer1_o,c2_o)

        # c2_h = self.mixer(c2_h)

        c3_h = self.layer2_h(c2_h) # (b,512,32,32)
        c3_o = self.layer2_o(c2_h) #obj共享hand-RestNet50中的stage0、stage1的结果！

        # c3_h = checkpoint(self.lka_layer2_h,c3_h)
        # c3_o = checkpoint(self.lka_layer2_o,c3_o)
        

        # c3_h = self.GRN_layer2_h(c3_h)
        # c3_o = self.GRN_layer2_o(c3_o)

        c4_h = self.layer3_h(c3_h) # (b,1024,16,16)
        c4_o = self.layer3_o(c3_o) # (b,1024,16,16)

        # c4_h = checkpoint(self.lka_layer3_h,c4_h)
        # c4_o = checkpoint(self.lka_layer3_o,c4_o)

        # c4_h = checkpoint(self.ema_layer3_h,c4_h)
        # c4_o = checkpoint(self.ema_layer3_o,c4_o)
        
        # c4_h = checkpoint(self.simAM_layer3_h,c4_h)
        # c4_o = checkpoint(self.simAM_layer3_o,c4_o)

        # c4_h = self.simAM_layer3_h(c4_h)
        # c4_o = self.simAM_layer3_o(c4_o)

        # c4_h = self.shuffle_layer3_h(c4_h)
        # c4_o = self.shuffle_layer3_o(c4_o)

        # dropblock:
        # c4_h = self.dropblock_hand_layer3(c4_h)
        # c4_o = self.dropblock_obj_layer3(c4_o)
    # ---------------------------over---------------------------------------------

    # ----------hand-resNet50、obj-resNet50共享hand-resNet50中的stage4子网络的权重！----------------
        c5_h = self.layer4_h(c4_h) # (b,2048,8,8)
        c5_o = self.layer4_h(c4_o) # (b,2048,8,8)

        # c5_h = self.sc_layer4_h(c5_h)
        # c5_o = self.sc_layer4_o(c5_o)

        # dropblock:
        # c5_h = self.dropblock_hand_layer4(c5_h)
        # c5_o = self.dropblock_obj_layer4(c5_o)
    
    # ===========================以下为FPN网络的部分(解码器)==================================
        # ---------------分别对手、物分别使用FCN式的相加法来融合特征网络的上采样----------------------
        # Top-down#---减少通道数，从stage3的768通道缩减到96，方便下面融合过程时，两个fm的channel相同
        # p5_h = self.toplayer_h(c5_h)
        # p4_h = self._upsample_add(p5_h,self.latlayer1_h(c4_h)) # (b,256,16,16)
        # p3_h = self._upsample_add(p4_h,self.latlayer2_h(c3_h)) # (b,256,32,32)
        # p2_h = self._upsample_add(p3_h,self.latlayer3_h(c2_h)) # (b,256,64,64)
       
        # p5_o = self.toplayer_o(c5_o) # (b,256,7,7)
        # p4_o = self._upsample_add(p5_o,self.latlayer1_o(c4_o)) # (b,256,16,16)
        # p3_o = self._upsample_add(p4_o,self.latlayer2_o(c3_o)) # (b,256,32,32)
        # p2_o = self._upsample_add(p3_o,self.latlayer3_o(c2_h)) # (b,256,64,64)                     

        p2_h = self.RSEFPN_h([c2_h,c3_h,c4_h,c5_h])
        p2_o = self.RSEFPN_o([c2_h,c3_o,c4_o,c5_o])
        # p2_o = checkpoint(self.RSEFPN_o,[c2_h,c3_o,c4_o,c5_o])
        

        # Smooth
        p2_h = self.smooth3_h(p2_h)
        p2_o = self.smooth3_o(p2_o)

        
        
        # 这里的特征图p2_h，p2_0的尺寸(C,H,W)为：(64,64,64)
        return p2_h , p2_o

class FPN_18(nn.Module):

    def __init__(self, pretrained=True):
        super(FPN_18, self).__init__()
        self.in_planes = 64

        resnet_hand = resnet18(pretrained=pretrained)

        #resnet_obj = resnet18(pretrained=pretrained)
        resnet_obj = deepcopy(resnet_hand)

        self.toplayer_h = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.toplayer_o = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.layer0_h = nn.Sequential(resnet_hand.conv1, resnet_hand.bn1, resnet_hand.leakyrelu, resnet_hand.maxpool)
        self.layer1_h = nn.Sequential(resnet_hand.layer1)
        self.layer2_h = nn.Sequential(resnet_hand.layer2)
        self.layer3_h = nn.Sequential(resnet_hand.layer3)
        self.layer4_h = nn.Sequential(resnet_hand.layer4)

       # self.layer0_o = nn.Sequential(resnet_obj.conv1, resnet_obj.bn1, resnet_obj.leakyrelu, resnet_obj.maxpool)
        #self.layer1_o = nn.Sequential(resnet_obj.layer1)
        self.layer2_o = nn.Sequential(resnet_obj.layer2)
        self.layer3_o = nn.Sequential(resnet_obj.layer3)
       # self.layer4_o = nn.Sequential(resnet_obj.layer4)


        # Smooth layers
        #self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_h = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.smooth2_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_o = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1_h = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_h = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_h = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)


        self.latlayer1_o = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_o = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_o = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)


        self.pool_h = nn.AvgPool2d(2, stride=2)
        self.pool_o = nn.AvgPool2d(2, stride=2)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1_h = self.layer0_h(x) # (b,64,56,56)
        #c1_o = self.layer0_o(x)

        c2_h = self.layer1_h(c1_h) # (b,256,56,56)
       # c2_o = self.layer1_o(c1_o)
    
        c3_h = self.layer2_h(c2_h) # (b,512,28,28)
        c3_o = self.layer2_o(c2_h) # (b,512,28,28)
  
        c4_h = self.layer3_h(c3_h) # (b,1024,14,14)
        c4_o = self.layer3_o(c3_o) # (b,1024,14,14)
 
        c5_h = self.layer4_h(c4_h) # (b,2048,7,7)
        c5_o = self.layer4_h(c4_o) # (b,2048,7,7) 
    
        # Top-down
        p5_h = self.toplayer_h(c5_h) # (b,256,7,7) 
        p4_h = self._upsample_add(p5_h, self.latlayer1_h(c4_h)) # (b,256,14,14)
        p3_h = self._upsample_add(p4_h, self.latlayer2_h(c3_h)) # (b,256,28,28)
        p2_h = self._upsample_add(p3_h, self.latlayer3_h(c2_h)) # (b,256,56,56)


        p5_o = self.toplayer_o(c5_o)
        p4_o = self._upsample_add(p5_o, self.latlayer1_o(c4_o))
        p3_o = self._upsample_add(p4_o, self.latlayer2_o(c3_o))
        p2_o = self._upsample_add(p3_o, self.latlayer3_h(c2_h))
        # Smooth

        p2_h = self.smooth3_h(p2_h)
        p2_o = self.smooth3_o(p2_o)
       
        return p2_h , p2_o


class ResNet(nn.Module):
    # block即残差块Bottleneck类（即含有残差块的层),layers:Bottleneck层数，从stage1->stage4分别含有3，4，6，3层
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 3:inputChannels,64:outputChannels即卷积核(滤波器)的个数,7:卷积核为7x7xinputChannels，bias:是否包含偏置项（即y=wx+b中的b）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 这里先BN再LeakyRelu！
        self.bn1 = nn.BatchNorm2d(64) #64:inputChannels
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layers=[3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, layers[0],grn = False) #inputChannels=64,outputChannels=256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,grn = False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,grn = False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,grn = False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 全连接层用于将网络的输出特征转换为预测的类别
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化网络的权重，也可以把预训练模型中训练好的权重拿过来~~~
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 对于卷积层,使用了 Kaiming 初始化方法，会将权重向量按照高斯分布进行初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d): # 对于批量归一,使用了常量初始化方法,将权重参数初始化为1，偏置参数初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,grn = False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #每阶段第一层Bottleneck的residual = self.downsample(x)过程中:outputChannels = planes*4，可以使得out += residual这一步骤的channels相等正常运行
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        # index = 0:每个阶段的第一层Bottleneck都需要进行 residual = self.downsample(x)操作~~~~
        layers.append(block(self.inplanes, planes, stride, downsample,grn=grn))

        # 进行完index=0的Bottleneck层(过程为:inputChannels*=4),所以现在的inputChannels为plans * 4~~~~，即可以保证out += residual这一步操作正常运行
        self.inplanes = planes * block.expansion
        # index[1,blocks)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    # x = [batch_size, channels, height, width]，其中batch_size：以batch_size个样本为一组数据单位进行对网络权值的前向、后向传播的优化
    def forward(self, x):

        # -----------stage0(stem)----------------------
        x = self.conv1(x)
        # 这里先BN再LeakyRelu！
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        # -----------stage1----------------------
        x = self.layer1(x)
        # -----------stage2----------------------
        x = self.layer2(x)
        # -----------stage3----------------------
        x = self.layer3(x)
        # -----------stage4----------------------
        x = self.layer4(x)

        # 这行代码对输入 x 进行了平均池化操作。通过调用 mean(dim) 函数并指定维度参数 3 和 2，在 3 维度和 2 维度上进行求平均值的操作，进行一次mean(dim)代表对第dim维度进行一次降维（将该维度降为1）
        # 即对x = [batch_size, channels, height, width] 的第2维度：height先进行一次mean，得到[batch_size, channels, 1, width]【可以类比m*n的矩阵每列缩为一个平均数后，调整成1*n】，再对第3维度width进行一次mean，得到[batch_size, channels, 1, 1]
        x = x.mean(3).mean(2) 
        # x = [batch_size, channels, 1, 1]，那么经过该行代码后，它的形状将变为 [batch_size, channels * 1 * 1]，即把通道、高度和宽度这三个维度合并为一个维度
        x = x.view(x.size(0), -1) # 合并除batch_size外的其他3个维度成为一个维度，以便于下面一步全连接层的转换操作。
        
        # 全连接层，用于将特征转化成分类
        x = self.fc(x)
        return x

# resnet-xxx不同层数版本之间的组成结构都是类似的，网络的主体都是Bottlenect的不同层数的组合
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model Encoder"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # 预训练模型的权重是指在大规模数据集上进行了预先训练的神经网络的参数值，即将已经训练好的模型权重参数拿过来作为初始化当前模型的权重~
    if pretrained: 
        # model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth"))
        # model.load_state_dict(checkpoint['model'])

        checkpoint = torch.load("E:/localPy/HOFEC/networks/pth/resnet50-19c8e357.pth")
        # print("checkpoint = ",checkpoint)
        model.load_state_dict(checkpoint)

        # dict = model.state_dict()
        # state_dict = {k: v for k, v in checkpoint.items() if k in dict.keys()} # 寻找网络中公共层，并保留预训练参数
        # dict.update(state_dict)  # 将预训练参数更新到新的网络层

        # # strict，该参数默认是True，表示预训练模型的层和自己定义的网络结构层严格对应相等
        # model.load_state_dict(dict)
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model

# resNet的残差连接层
class Bottleneck(nn.Module):
    expansion = 4 #为该类的静态变量，可以直接通过Bottleneck.expansion获取~

    def __init__(self, inplanes, planes, stride=1, downsample=None,grn = False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 当stride=2时，相当于panet过程中的downsample_convs！~~~~
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False) # outChannels*=4
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) 
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.grn = grn
        if self.grn:
        #     self.GRN = GRN(planes * self.expansion)
            # self.se = SELayer(planes * self.expansion)
            self.cbam = CBAM(planes * self.expansion)
        # self.SimAM = SimAM()

    def forward(self, x):
        def checkpoint(model, input):
            if torch.cuda.is_available():
                return torch.utils.checkpoint.checkpoint(model, input)
            else:
                return model(input)
        residual = x #保留原输入x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: #BTNK1需要将residual(原x)先下采样再进行后续操作，而BTNK
            residual = self.downsample(x)
        
        if self.grn:
            out = checkpoint(self.cbam,out)
        # out = self.SimAM(out)
        # out = checkpoint(self.SimAM,out)

        out += residual #BTNK1会在上一步修改x的channels等于out，BTNK2的输入为BTNK1的输出，即x的chennels等于out!
        out = self.leakyrelu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,grn = False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# net = FPN(pretrained = False)

# # 查看网络的层数
# num_layers = len(list(net.children()))
# print("网络的层数:", num_layers)

# def count_parameters(model):
#     num_params = sum(p.numel() for p in model.parameters())
#     return num_params

# model = FPN(pretrained = False)
# # 统计模型参数量
# total_params = count_parameters(model)
# print(f"Total Parameters: {total_params}")
# # 34974272

# model = resnet50(pretrained = False)
# 统计模型参数量
# total_params = count_parameters(model)
# print(f"Total Parameters: {total_params}")
# 25557032