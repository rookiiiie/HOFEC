import torch
from torch import nn
from torchvision import ops
from networks.backbone_share2 import FPN,FPN_18

from networks.hand_head import hand_Encoder, hand_regHead
from networks.object_head import obj_regHead, Pose2DLayer
from networks.mano_head import mano_regHead
from networks.CR import Transformer
from networks.loss import Joint2DLoss, ManoLoss, ObjectLoss

from networks.plugs.spChAttn import SpGate as SpatialGate

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias,0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

class HOFEC(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, 
                #  mano_neurons=[384, 192], 
                 mano_neurons=[1024, 512], 
                 coord_change_mat=None,
                 reg_object=True, pretrained=True):

        super(HOFEC, self).__init__()

        self.channels = channels

        self.out_res = roi_res

        # FPN-Res50 backbone（主干模型)
        self.base_net = FPN(pretrained=pretrained) 
        # self.base_net = FPN_18(pretrained=pretrained) 

        # hand head
        #创建了一个名为hand_head的子模型，用于手部姿态估计。它似乎有一些参数，包括ROI（感兴趣区域）的分辨率、手部关节点数量、堆栈数、通道数和块数。----论文中提到的ROIAlign?
        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb, 
                                      stacks=stacks, channels=channels, blocks=blocks) 
        # hand encoder
        #创建了一个名为hand_encoder的子模型，用于对手部图像进行编码。它似乎有一些参数，包括热图通道数、特征通道数和输入特征的大小。
        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))
        # mano branch
        #创建了一个名为mano_branch的子模型，似乎用于手部建模（Mano模型）的相关任务。它有一些参数，包括Mano模型、特征大小、神经元配置和坐标变换矩阵。
        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)
        # object head
        self.reg_object = reg_object #Default True
        #创建了一个名为obj_head的子模型，似乎用于对象的姿态估计。它有一些参数，包括通道数、中间通道数和关节点数量。
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels//2, joint_nb=joint_nb)
        #创建了一个名为obj_reorgLayer的子模型，可能用于对对象的姿态信息进行整理或处理。
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        # CR blocks（Convolutional Regression blocks卷积回归块）
       
        self.transformer_obj = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth, num_heads=transformer_head, drop=0.6)
        self.transformer_hand = Transformer(inp_res=roi_res, dim=channels,depth=transformer_depth, num_heads=transformer_head, drop=0.6) # 0.2?

        self.gamma1 = nn.Parameter(torch.tensor([0.25],dtype=torch.float32), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.tensor([0.25],dtype=torch.float32), requires_grad=True)
        # self.drop = nn.Dropout(0.)

        self.attention_module_hand = SpatialGate()
        self.attention_module_obj = SpatialGate()


        self.hand_head.apply(init_weights)
        self.hand_encoder.apply(init_weights)
        self.mano_branch.apply(init_weights)
        self.transformer_obj.apply(init_weights)
        self.transformer_hand.apply(init_weights)
        self.obj_head.apply(init_weights)

        # self.hand_head.apply(weights_init_kaiming)
        # self.hand_encoder.apply(weights_init_kaiming)
        # self.mano_branch.apply(weights_init_kaiming)
        # self.transformer_obj.apply(weights_init_kaiming)
        # self.transformer_hand.apply(weights_init_kaiming)
        # self.obj_head.apply(weights_init_kaiming)

        
    

    def net_forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None):
        # 首先，根据输入参数 imgs，调用 self.new_method(imgs) 方法获取输入图片的批处理大小 batch。
        batch = imgs.shape[0]

        # bbox_hand = [xmin, ymin, xmax, ymax] 
        # bbox_obj = [xmin, ymin, xmax, ymax] 
        # 然后，计算手部边界框（bbox_hand）和对象边界框（bbox_obj）的交集，得到交集边界框（bbox_inter）。同时，计算一个二进制掩码 msk_inter，用于指示交集是否存在。
        #bbox_hand[:, :2] 表示对 bbox_hand 张量的所有行和前两列进行切片操作，即取手部边界框的左上角点坐标，如果两个张量中在同一个位置上的值分别为 x 和 y，则返回的结果中对应位置上的值为 max(x, y)。
        inter_topLeft = torch.max(bbox_hand[:, :2], bbox_obj[:, :2])
        inter_bottomRight = torch.min(bbox_hand[:, 2:], bbox_obj[:, 2:])
        #它的形状为 [N, 4]，其中 N 表示边界框的数量，而4表示每个边界框由左上角和右下角两个坐标点组成。
        bbox_inter = torch.cat((inter_topLeft, inter_bottomRight), dim=1) 
        #.sum(dim=1) 对每个边界框的差值结果(inter_bottomRight-inter_topLeft)在维度1上求和。由于每个边界框的差值有两个维度，所以在维度1上求和会得到一个长度为 N 的张量，表示每个边界框的差值大于0的元素个数。【即对[N,4]==sum(dim(1))>>[N,1]】若==2，有N个数，分别表示原边界框的2个维度的差值>0的个数count(x1-x2>0,y1-y2>0),若等于2，说明该第n个边界框存在
        msk_inter = ((inter_bottomRight-inter_topLeft > 0).sum(dim=1)) == 2 

        # P2 from FPN Network
        # 接下来，通过预训练的 FPN（Feature Pyramid Network）网络 self.base_net 分别从输入图像 imgs 中提取特征。P2_h 和 P2_o 是从 FPN 网络中获取的手部、物体特征。
        P2_h,P2_o = self.base_net(imgs)

       
        # 创建一个索引张量 idx_tensor，其中包含从0到batch-1的索引序列，device=imgs.device将该张量移动到与imgs张量相同的设备上，float()：将整数序列转换为浮点数类型。
        # view(-1, 1)：通过调用.view方法对张量进行形状重塑，将一维的浮点数序列转换为二维张量，第二维设置为1，到了一个形状为(batch, 1)的索引张量 idx_tensor，它的每个元素是一个浮点数。
        idx_tensor = torch.arange(batch, device=imgs.device).float().view(-1, 1)

        # get roi boxes（Region of Interest Boxes）
        # bbox_hand是手部区域的边界框，它的形状为(batch, 4)。其中每一行表示一个边界框，包含左上角和右下角的坐标信息。------这个bbox_hand具体还是要看一下utils.epoch.py中从sample中获取的数据到底是啥
        # 使用torch.cat函数将索引张量 idx_tensor 和手部边界框张量 bbox_hand 沿着第1维（列方向）进行拼接，生成一个新的张量(batch, 5)。其中第1维度的信息单位为(1个batch索引号，hand-ROI区域的4个边角坐标)
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)

        x_hand = ops.roi_align(P2_h, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
                          sampling_ratio=-1)  # hand -----(batch,256,56,56)

        # 利用 RoIAlign 操作（ops.roi_align）从特征图P2_o中裁剪出属于手部边界框内的对象的特征x_obj----即不是所有对象，而只是在手部区域内即与手部进行交互的对象！
        # x_obj = ops.roi_align(P2_o, roi_boxes_hand, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
        #                   sampling_ratio=-1)  # obj(which in hand's space) ----intersection ----(batch,256,56,56)

        channels =  self.channels
        # obj forward
        # if self.reg_object:
        # 创建对象的边界框（roi_boxes_obj）和交集的边界框（roi_boxes_inter）。
        roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
        roi_boxes_inter = torch.cat((idx_tensor, bbox_inter), dim=1)
        
        # 利用 RoIAlign 操作从特征图中裁剪出对象的特征（y）以及交集的特征（z_x）。
        # 这边裁剪的y，是整个obj空间的特征，区别于上面的部分obj特征x_obj
        y = ops.roi_align(P2_o, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                        sampling_ratio=-1)  # obj
        # 用于增强hand-to-object-enhance，这玩意好像没什么用
        # y_hand = ops.roi_align(P2_h, roi_boxes_obj, output_size=(self.out_res, self.out_res), spatial_scale=1.0/4.0,
        #                 sampling_ratio=-1)

        # 这边裁剪的z_x，是在hand-obj交界处的空间的手部部分的特征，而不是整个手部的特征----------<<<为什么这里不是手部区域的物体呢？？？
        z_x = ops.roi_align(P2_h, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                        sampling_ratio=-1)  # hand(which in inter's space) ----intersection
            
        z_y = ops.roi_align(P2_o, roi_boxes_inter, output_size=(self.out_res, self.out_res), spatial_scale=1.0 / 4.0,
                        sampling_ratio=-1)

        
        x_hand_p, x_hand_second = self.attention_module_hand(x_hand)
        x_obj_p , x_obj_second = self.attention_module_obj(y)

        # 我这边的x_obj_second为什么不能再+msk_inter[:, None, None, None] * z_y * self.gamma2呢，反正都detach()了
        hand_sub = x_obj_second.detach() + msk_inter[:, None, None, None] * z_y * self.gamma1
        hand = self.transformer_hand(x_hand_p,hand_sub)

        obj_sub = x_hand_second.detach() + msk_inter[:, None, None, None] * z_x * self.gamma2
        y = self.transformer_obj(x_obj_p, obj_sub)
        # obj-decoder
        # out_fm输出为[b,21*3,32,32]，其中21*3中的“21”表示obj的3D边界框的1个中心点、8个角点和12个边缘中点，“3”表示21个关键点的三维坐标(x,y,z)
        # 即out_fm输出为21个关键点的三维坐标信息，对于任意一个关键点的其中一维坐标
        out_fm = self.obj_head(y)  #--->>>outputChannels = joint_nb*3 = 21*3
        
        # 将out_fm中的3D边界框的21个关键点的三维坐标信息转变成 每个关键点的2D位置以及每个关键点对应的预测置信度（置信度是指模型对关键点是否在其正确位置的可信度评分，越高说明模型对该关键点位置为正确的位置的可能性越大）
        preds_obj = self.obj_reorgLayer(out_fm)
        # else:
        #     preds_obj = None

        #hand_obj为拼接了手部区域的手部特征图和手部区域的物体特征图，而这里的hand_obj为256*2通道，前256通道为hand_fm，后256通道为obj_mp(in hand space)
        
        # hand = hand_obj[:,0:channels,:,:] # 提取手部特征图---相比与”引用2“的原代码而言，这里的hand特征图是已经被self.transformer_hand增强过的特征图了！
        #hand forward

        # 此时的hand:[B,256,32,32] 
        # out_hm:输出out_hm：[B, self.joint_nb, self.out_res, self.out_res]即为21通道32x32的2D概率热图，该概率图中，每个通道中的每个像素值即为是否为该通道对应的关键点的概率值
        # preds_joints：根据一张图像的概率热图生成的s各个手部关键点（共21个）的坐标，predictions = [b,self.joint_nb,2],dim2=2,对应了2个坐标:每个通道下对应的x、y坐标
        # encoding：stacks=1时，encoding=[y](y为经过hourglass、resNet、fc网络得到的手部特征图，维度为[b,256,32,32])
        out_hm, encoding, preds_joints = self.hand_head(hand)
        # _out_hm, _encoding, _preds_joints = self.hand_head(hand)
        
        # mano_encoding = [B,256*2*2]
        mano_encoding = self.hand_encoder(out_hm, encoding)
        # _mano_encoding = self.hand_encoder(_out_hm, _encoding)

        # 预测的手部结果pred_mano_results和真实的手部结果gt_mano_results
        # mano_branch整合了ManoLayer，作用为：通过输入批量手部姿势和形状矢量，输出相应的三维手部关节和顶点
        # 输入：mano_encoding
        # 输出：# "verts3d": pred_verts, # 预测的三维手部顶点坐标
                # "joints3d": pred_joints, # 预测的三维手部关节坐标
                # "mano_shape": pred_mano_shape,# 预测的 MANO 模型的形状参数
                # "mano_pose": pred_mano_pose_rotmat, # 预测的 MANO 模型的姿势参数，以旋转矩阵形式表示
                # "mano_pose_aa": pred_mano_pose # 预测的 MANO 模型的姿势参数，以 Rodrigues 向量形式表示
        pred_mano_results, gt_mano_results = self.mano_branch(mano_encoding, mano_params=mano_params, roots3d=roots3d)
        # _pred_mano_results, gt_mano_results = self.mano_branch(_mano_encoding, mano_params=mano_params, roots3d=roots3d)

        return preds_joints, pred_mano_results, gt_mano_results, preds_obj
        # return preds_joints, pred_mano_results, gt_mano_results, preds_obj,_pred_mano_results

    def new_method(self, imgs):
        batch = imgs.shape[0]
        return batch

    def forward(self, imgs, bbox_hand, bbox_obj,mano_params=None, roots3d=None):
        if self.training: # training：传递mano_params
            # print("----------------------------training------------------------------")
            preds_joints, pred_mano_results, gt_mano_results, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                                           mano_params=mano_params)
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj

            # preds_joints, pred_mano_results, gt_mano_results, preds_obj, = self.net_forward(imgs, bbox_hand, bbox_obj,
            #                                                                                mano_params=mano_params)
            # return preds_joints, pred_mano_results, gt_mano_results, preds_obj ,
        else: # not training：传递roots3d
            # print("----------------------------testing------------------------------")
            preds_joints, pred_mano_results, gt_mano_results, preds_obj = self.net_forward(imgs, bbox_hand, bbox_obj,
                                                                             roots3d=roots3d)
            return preds_joints, pred_mano_results, preds_obj

            # preds_joints, pred_mano_results, gt_mano_results, preds_obj, = self.net_forward(imgs, bbox_hand, bbox_obj,
            #                                                                  roots3d=roots3d)
            # return preds_joints, pred_mano_results, preds_obj


class HOModel(nn.Module):

    def __init__(self, hofec, mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None):

        super(HOModel, self).__init__()
        self.hofec = hofec
        # supervise when provide mano params
        self.mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)
        # supervise when provide hand joints
        self.mano_joint_loss = ManoLoss(lambda_joints3d=mano_lambda_joints3d,
                                        lambda_regulshape=mano_lambda_regulshape,
                                        lambda_regulpose=mano_lambda_regulpose)
        # object loss
        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

    def forward(self, imgs, bbox_hand, bbox_obj,
                joints_uv=None, joints_xyz=None, mano_params=None, roots3d=None,
                obj_p2d_gt=None, obj_mask=None, obj_lossmask=None):
        if self.training:
            losses = {}
            total_loss = 0
            # preds_joints2d, pred_mano_results, gt_mano_results, preds_obj,  = self.hofec(imgs, bbox_hand, bbox_obj,mano_params=mano_params)
            preds_joints2d, pred_mano_results, gt_mano_results, preds_obj,  = self.hofec(imgs, bbox_hand, bbox_obj,mano_params=mano_params)

            if mano_params is not None:
                # mano_total_loss, mano_losses = self.mano_loss.compute_loss(pred_mano_results, gt_mano_results)
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(pred_mano_results, gt_mano_results)
                total_loss += mano_total_loss
                for key, val in mano_losses.items():
                    losses[key] = val
            if joints_uv is not None:
                joint2d_loss, joint2d_losses = self.joint2d_loss.compute_loss(preds_joints2d, joints_uv)
                for key, val in joint2d_losses.items():
                    losses[key] = val
                total_loss += joint2d_loss
            if preds_obj is not None:
                obj_total_loss, obj_losses = self.object_loss.compute_loss(obj_p2d_gt, obj_mask, preds_obj, obj_lossmask=obj_lossmask)
                for key, val in obj_losses.items():
                    losses[key] = val
                total_loss += obj_total_loss
            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0
            return total_loss, losses
        else:
            # print("----------------------------testing------------------------------")
            preds_joints, pred_mano_results, preds_obj = self.hofec(imgs, bbox_hand, bbox_obj, roots3d=roots3d)
            return preds_joints, pred_mano_results, preds_obj