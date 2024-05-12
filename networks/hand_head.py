import torch
from torch import nn
import torch.nn.functional as F

from .plugs.maxPool import maxPool

# from .plugs.SEAttention import SEAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # hand_regHead 是一个用于手部关键点检测的模型，它包含了多个 Hourglass 模块和其他组件，用于生成热图并预测手部关键点的坐标。
class hand_regHead(nn.Module):
    # roi_res：ROI（感兴趣区域）的分辨率，默认为32。
    # joint_nb：手部关键点的数量，默认为21。
    # stacks：堆叠的Hourglass模块数目，默认为1。
    # channels：通道数，默认为256。
    # blocks：为Hourglass中每个阶段包含的基本模块的数量，默认为1。
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1):
        """
        Args:
            inr_res: input image size
            joint_nb: hand joint num
        """
        super(hand_regHead, self).__init__()

        # hand head
        self.out_res = roi_res
        self.joint_nb = joint_nb

        self.channels = channels
        self.blocks = blocks
        self.stacks = stacks

        #一个可学习的参数，用于加权手部关键点的预测。
        self.betas = nn.Parameter(torch.ones((self.joint_nb, 1), dtype=torch.float32)) 

        center_offset = 0.5 # 用于网格坐标点的偏移
        # uu 和 vv：形状都是 (self.out_res, self.out_res)。这两个矩阵表示了坐标系内网格点的 x 和 y 值。uu 和 vv 的取值范围是从 0 到 self.out_res-1
        vv, uu = torch.meshgrid(torch.arange(self.out_res).float(), torch.arange(self.out_res).float())
        # 添加 center_offset 的目的是将网格点的坐标进行偏移，使其位于网格单元格的中心。例如，当 center_offset 为 0.5 时，对于一个网格单元格的左上角坐标 (0, 0) ，通过将 uu 和 vv 偏移 0.5，得到的新坐标就是该单元格的中心坐标 (0.5, 0.5)。 ---即在一个网格中，将一个坐标点从网格线上放到网格的中心点上！
        uu, vv = uu + center_offset, vv + center_offset
        # 将 uu 和 vv 注册为模型的缓冲区。这样做的目的是将这些变量作为模型的固定参数，在模型的前向推理过程中保持不变
        self.register_buffer("uu", uu / self.out_res)
        self.register_buffer("vv", vv / self.out_res)
        
        # 用于生成热图的概率分布,对[b,c,...]中的第2维度即channel进行归一化
        self.softmax = nn.Softmax(dim=2)
        block = Bottleneck
        # features：模型内部特征的数量，用于 Hourglass 模块。
        self.features = self.channels // block.expansion # 这里为256/2=128
        
        # hg、res、fc、score、fc_、score_：这些属性是模型的不同部分，包括 Hourglass 模块、残差块、卷积层等。
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(self.stacks):
            hg.append(Hourglass(block, self.blocks, self.features, 4))
            res.append(self.make_residual(block, self.channels, self.features, self.blocks))
            fc.append(BasicBlock(self.channels, self.channels, kernel_size=1))
            score.append(nn.Conv2d(self.channels, self.joint_nb, kernel_size=1, bias=True)) # score模块，将通道数转化为21层（即关键点的个数）
            if i < self.stacks - 1: # 如果不是最后一个stack的话
                fc_.append(nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(self.joint_nb, self.channels, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    # 用于创建多个残差块的方法，这里的多个残差块结构，为resNet中的中间stage的结构，即为1个BTNK1+多个BTNK2的组合
    def make_residual(self, block, inplanes, planes, blocks, stride=1):
        skip = None
        if stride != 1 or inplanes != planes * block.expansion:
            skip = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True))
        layers = []
        # -----1个BTNK1------
        layers.append(block(inplanes, planes, stride, skip))
        # -----多个BTNK2------
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    # 将模型的输出转化为空间上的 softmax 分布（即预测图像中每个像素位置是否为关键点的概率大小），并生成热图
    def spatial_softmax(self, latents):
        # latents[b,c,h,w] ==>> latents[b,c,h*w]，这里共有21通道，每个通道为某个关键点对应的特征图！
        latents = latents.view((-1, self.joint_nb, self.out_res ** 2))
        # self.betas:这是一个用于缩放latents的因子
        latents = latents * self.betas
        # softmax 归一化：将每个通道上的每一个像素值转变为0-1的概率值，表示某个通道上某个像素对应的（该通道的所对应的）关键点的是否存在的概率（注：这一步并不会修改该维度(channel)的大小！）
        heatmaps = self.softmax(latents)
        # 将heatmaps-dim2（self.out_res ** 2）平铺成二维图像
        heatmaps = heatmaps.view(-1, self.joint_nb, self.out_res, self.out_res)
        return heatmaps
    
    # 根据一张图像的概率热图生成各个手部关键点（共21个）的坐标。heatmaps:[b, self.joint_nb, self.out_res, self.out_res]
    def generate_output(self, heatmaps):
        # prerequisite：
        # torch.stack() 函数通过创建一个新的维度，并在这个新维度上对输入张量进行拼接。因此，拼接后的张量的维度比输入张量的维度高一维。如torch.stack(X(3,4)、Y(3,4),dim=0) =>(2,3,4)
        # torch.cat() 函数沿着指定的维度对输入张量进行拼接。拼接后的张量维度和输入张量在拼接维度上是一致的。如torch.cat(X(3,4)、Y(3,4),dim=0) =>(6,4)
        # torch.sum() 函数用于计算张量沿指定维度的元素之和,求和之后，该维度会被消除
        
        # torch.stack在dim2堆叠2个[b,self.joint_nb] => [b,self.joint_nb,2]
        # predictions = [b,self.joint_nb,2]中的dim2的大小为2，这2个维度分别保存了某个通道对应的关键点的坐标x、y
        predictions = torch.stack((
            # 对heatMaps*self.uu的dim2(self.out_res)进行求和得到[b, self.joint_nb, self.out_res]，在对这个结果的dim2求和，得到[b,self.joint_nb]，即dim1（self.joint_nb）中保存的结果为每个关键点通道对应的二维概率图的所有概率和，这个概率和表示该通道所对应的关键点的坐标x的位置，下面一行的torch.sum所求的应该是坐标y的位置。
            torch.sum(torch.sum(heatmaps * self.uu, dim=2), dim=2), #dim(0-based)
            torch.sum(torch.sum(heatmaps * self.vv, dim=2), dim=2)), dim=2)
        return predictions

    # x:[B,256,32,32] 
    def forward(self, x, gcn_heatmap = None):
        # 在这里，采取了串行的方式来生成hm与encoding，对于hm而言，hourglass是很重要的，但是对于encoding的融合而言（代码在mano_head.py中)，或许可以尝试将hourglass并行
        out, encoding, preds = [], [], []
        for i in range(self.stacks): # stacks：堆叠的Hourglass模块数目，默认为1。
            # -------hourglass网络---------
            y = self.hg[i](x)
            # -------resNet残差网络---------<<<<<这一步是为什么？
            y = self.res[i](y)
            # -------基础的单层网络---------
            y = self.fc[i](y) #[b,256,32,32]
            # -------基础的单层网络，输出通道为21即joint_nb（这里每个通道对应了某个关键点的特征图)---------
            latents = self.score[i](y)
            # ------通过softmax计算图像中每个通道中的每个像素是否为该通道对应的关键点的概率值，并生成热图------
            heatmaps= self.spatial_softmax(latents)
            # if gcn_heatmap is not None:
            #     heatmaps = (heatmaps + gcn_heatmap) / 2
            # 输出out即为21通道的概率热图，该概率图中，每个通道中的每个像素值即为是否为该通道对应的关键点的概率值
            out.append(heatmaps)

            # -------根据一张图像的概率热图生成各个手部关键点（共21个）的坐标，predictions = [b,self.joint_nb,2],dim2=2,对应了2个坐标:每个通道下对应的x、y坐标------------
            predictions = self.generate_output(heatmaps)
            preds.append(predictions)
            
            # 如果不是最后一个stack的话-----这里不执行此处，暂时不看这边
            if i < self.stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](heatmaps)
                x = x + fc_ + score_
                encoding.append(x)
            else:
                encoding.append(y)
        return out, encoding, preds


# 基本块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,groups=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=((kernel_size - 1) // 2),
                      groups=groups,bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# 残差块（下采样时使用，每次进行一次BTNK1传播，输入会扩大expansion倍channels！）
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, skip=None, groups=1):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True, groups=groups)
        self.leakyrelu = nn.LeakyReLU(inplace=True)  # negative_slope=0.01
        self.skip = skip
        self.stride = stride

    # 不改变图像H,W
    def forward(self, x):
        residual = x

        # 这边的顺序与backbone_share.py中的Bottlenect的顺序不同！！！
        out = self.bn1(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.skip is not None:
            residual = self.skip(x)

        out += residual

        return out

# Stacked Hourglass Networks - 堆叠沙漏网络，见https://blog.csdn.net/wangzi371312/article/details/81174452?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-81174452-blog-109130642.235%5Ev38%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-81174452-blog-109130642.235%5Ev38%5Epc_relevant_anti_t3_base&utm_relevant_index=10
# 对于单Stacked的Hourglass Networks而言，其为编码器-解码器结构，就类似UNet一样，先下采样，再上采样，区别只在下采样、上采样的具体过程不一样，并且是通过递归实现的，可以灵活的修改下、上采样的深度
class Hourglass(nn.Module):
    # block：代表Hourglass中的基本模块，在Hourglass网络中，通常会使用Residual Block作为基本模块。这个参数应该是一个可调用对象，可以用来构建基本模块。
    # num_blocks：代表Hourglass中每个阶段（stage）包含的基本模块的数量。每个阶段是Hourglass中的一个分支，用于递归地处理输入数据。
    # planes：代表每个基本模块中的输出通道数（planes）。在这里,planes=256/2=128
    # depth：代表Hourglass网络的深度。Hourglass网络通过递归地构建分支和汇合阶段来处理输入数据。深度表示了Hourglass网络分支的递归深度。
    def __init__(self, block, num_blocks, planes, depth):

        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        
        # self.se = SEAttention(channel=planes * block.expansion, reduction=4)
        # 在这里，hg(二维数组),的第一维度为长度为4的数组,其中,第1个元素为4个残差块数组，后3个元素中，每个元素都为3个的残差块数组
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)
        
        # maxPool_modules = []
        # for i in range(2):
        #     item = maxPool(channel = planes * block.expansion)
        #     maxPool_modules.append(item)
        # self.maxPool_modules = maxPool_modules

        # se_modules = []
        # se_gammas = []
        # for i in range(depth):
        #     se_module = SEAttention(channel=planes * block.expansion, reduction=4)
        #     se_modules.append(se_module)
        #     se_gammas.append(nn.Parameter(torch.Tensor([0.25])).to(device))
        # self.se_modules = se_modules
        # self.se_gammas = se_gammas
        
    # 创建num_blocks个残差块
    def _make_residual(self, block, num_blocks, planes):
        layers = []
        # 在这里，num_blocks=1
        for i in range(0, num_blocks):
            # channel changes: planes*block.expansion->planes->2*planes
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        # depth：代表Hourglass网络的深度，这里为4
        for i in range(depth):
            res = []
            for j in range(3):
                # 3 residual modules composed of a residual unit
                # <2*planes><2*planes>
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                # i=0 in a recursive construction build the basic network path
                # see: low2 = self.hg[n-1][3](low1)
                # <2*planes><2*planes>
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        # 在这里，hg为长度为4的数组，第1个元素为4个残差块数组，后3个元素中，每个元素都为3个的残差块数组
        return nn.ModuleList(hg)

    # 这里构建网络采用递归的方法，是由于hourglass是由基本的操作块拼接而成的，为了方便更加灵活的指定hourglass网络的深度depth，而不需要像Unet、FPN那样手动一层一层的手动编码~~~
    # 此时的x:[B,256,32,32]
    def _hour_glass_forward(self, n, x):

        # ==========================递归前操作(encoder):从高层往低层不断的进行[残差操作+最大池化+残差操作]===================================
        # n=4时,up1为hg[n - 1][0]=hg[3][0]为残差块block,该block即为Bottlenect,该步骤进行的就是Bottlenect的forward(self,x)
        up1 = self.hg[n - 1][0](x)  # skip branches
        # 2*2最大池化,[b,c,h,w]->[b,c,h//2,w//2] 
        # if(n>2):
        #     low1 = self.maxPool_modules[4-n](x)
        # else:
            # low1 = F.max_pool2d(x, 2, stride=2) # --- 这一步居然也是可以改进的。。(对n=4[32x32]和n=3[32x32]可以搞4层，对n=2、n=1就不能用多层maxpool)
        low1 = F.max_pool2d(x, 2, stride=2)
        # n=4时,up1为hg[n - 1][0]=hg[3][1]为残差块block，操作后得到x = [B,256*2*2,32,32]
        low1 = self.hg[n - 1][1](low1) # --- newH = h//(2**(n-3)) ：n=4时,h//2;n=3时,h//4;n=2时,h//8;n=1时,h//16;
        
        # 从n=4 => n=3 => n=2,每次递归,都会不断的重复以上3步:不断的对输入x进行[残差操作+最大池化+残差操作]
        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1) #递归直至n==1
        # n=1时,再一次重复以上3步
        else:
            low2 = self.hg[n - 1][3](low1)  # only for depth=1 basic path of the hourglass network
        
        # ============================递归后操作(decoder):从低层往高层不断的进行[残差操作+插值法上采样+拼接]===================================
        low3 = self.hg[n - 1][2](low2)
        # low3 = self.se_modules[n-1](low3)

        # scale_factor=2：尺寸缩放的因子，此处设为2，表示将输入张量的尺寸放大两倍。
        # interpolate默认使用bilinear双线性插值法，应该比邻近插值法好
        up2 = F.interpolate(low3, scale_factor=2,mode='bilinear')  # scale_factor=2 should be consistent with F.max_pool2d(2,stride=2)
        # 转置卷积法---效果巨差。。
        # up2 = self.up(low3)

        # skip connect
        out = up1 + up2
        # out = out + self.se_gammas[n-1] * self.se_modules[n-1](out)
        return out

    def forward(self, x):
        # depth: order of the hourglass network
        # do network forward recursively
        return self._hour_glass_forward(self.depth, x)



class hand_Encoder(nn.Module):
    # num_heatmap_chan=joint_nb, num_feat_chan=channels=256
    def __init__(self, num_heatmap_chan, num_feat_chan, size_input_feature=(32, 32),
                 nRegBlock=4, nRegModules=2):
        super(hand_Encoder, self).__init__()

        self.num_heatmap_chan = num_heatmap_chan
        self.num_feat_chan = num_feat_chan
        self.size_input_feature = size_input_feature

        self.nRegBlock = nRegBlock
        self.nRegModules = nRegModules

        self.heatmap_conv = nn.Conv2d(self.num_heatmap_chan, self.num_feat_chan,
                                      bias=True, kernel_size=1, stride=1)
        self.encoding_conv = nn.Conv2d(self.num_feat_chan, self.num_feat_chan,
                                       bias=True, kernel_size=1, stride=1)

        reg = []
        # 这里res算是加了4*2=8层Residual块
        for i in range(self.nRegBlock):
            for j in range(self.nRegModules):
                reg.append(Residual(self.num_feat_chan, self.num_feat_chan))

        self.reg = nn.ModuleList(reg)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_scale = 2 ** self.nRegBlock # 16--下采样的

        # fc layers----
        self.num_feat_out = self.num_feat_chan * (size_input_feature[0] * size_input_feature[1] // (self.downsample_scale ** 2))

    # hm_list = out_hm[B,21,32,32], encoding_list = [y] (因为stack=1，y为上面hand_head中append到数组中去的唯一元素)
    def forward(self, hm_list, encoding_list):
        # hm_list[-1]取的是第21个热图（即第21通道对应的32*32的热图）
        # encoding_list[-1]取的是y（特征图）[B,channel,32,32]
        x = self.heatmap_conv(hm_list[-1]) + self.encoding_conv(encoding_list[-1]) # 都转换成256通道
        if len(encoding_list) > 1:
            x = x + encoding_list[-2]

        # ------------------------------下游任务------------------------------------------------
        # x: [B,num_feat_chan,32,32]
        for i in range(self.nRegBlock): # 4
            for j in range(self.nRegModules): # 2
                x = self.reg[i * self.nRegModules + j](x) # 进行了8次不改变chennel的残差卷积操作
            x = self.maxpool(x) #进行nRegBlock即4次2*2最大池化，即W、H会缩小2**4=16倍，即W=H=2！

        # x: [B,num_feat_chan,2,2] ==>> [B,num_feat_chan*2*2]
        # 将特征图展平为一维向量，作为手部姿态估计的输出
        out = x.view(x.size(0), -1)

        return out

# 残差块，会将原图像通道先缩小，后恢复为原来大小，而Bottlenect残差块输出会扩张通道数的倍数，，。。。---why？？？？
class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x

        out = self.bn(x)
        out = self.leakyrelu(out)
        out = self.conv1(out)

        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)

        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)
        return out + residual
