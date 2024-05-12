import torch
from torch import nn
from torch.nn import functional as F


def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def quat2aa(quaternion):
    """Convert quaternion vector to angle axis of rotation."""
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def mat2quat(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector"""
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q



# x:[B*16,6]
# 6D三维旋转表示方法：参考https://zhuanlan.zhihu.com/p/103893075
def rot6d2mat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)

    # return = [B*16, 3, 3]
    return torch.stack((b1, b2, b3), dim=-1)


# rotation_matrix = [B*16,3,3]
def mat2aa(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector"""
    # Rodrigues 向量是一种用于表示旋转的向量形式。它由三个分量组成，通常表示为 [r1, r2, r3]，其中 r1, r2, r3 是向量在每个轴上的分量。
    # 在计算机图形学和计算机视觉领域，Rodrigues 向量常用于表示旋转矩阵或姿态。它通过将旋转矩阵转换为一个紧凑且易于处理的向量形式来表示旋转。
    # Rodrigues 向量具有紧凑性和数值稳定性，并且在许多算法中都被广泛使用，如姿态估计、三维重建、物体跟踪等。
    def convert_points_to_homogeneous(points):
        if not torch.is_tensor(points):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(points)))
        if len(points.shape) < 2:
            raise ValueError("Input must be at least a 2D tensor. Got {}".format(
                points.shape))

        return F.pad(points, (0, 1), "constant", 1.0)

    if rotation_matrix.shape[1:] == (3, 3):
        rotation_matrix = convert_points_to_homogeneous(rotation_matrix)
    quaternion = mat2quat(rotation_matrix)
    aa = quat2aa(quaternion)
    aa[torch.isnan(aa)] = 0.0

    # aa = [B*16, 3],dim1=3代表Rodrigues向量中的3个分量[r1, r2, r3]
    return aa


# mano模型：https://blog.csdn.net/g11d111/article/details/115539407
class mano_regHead(nn.Module):
    # 疑问：传入的feature_size=256？？这里咋=1024？ 
    # 这里的mano_neurons[a,b]中的a，即隐藏层的数量，如果过拟合的话，是否需要减小?
    def __init__(self, mano_layer, feature_size=1024, mano_neurons=[1024, 512],
                 coord_change_mat=None):
        super(mano_regHead, self).__init__()

        # 6D representation of rotation matrix
        self.pose6d_size = 16 * 6 # 表示旋转矩阵的6维表示，即16个关节的3D旋转表示，每个旋转由6个参数表示。
        self.mano_pose_size = 16 * 3 # 表示手部的姿态表示，即16个关节的3D旋转表示，每个旋转由3个参数表示。

        # Base Regression layers
        # mano_base_neurons = [1024, 1024, 512]。这意味着基础回归层将有三个线性层，输入特征维度分别为 1024、1024 和 512。
        mano_base_neurons = [feature_size] + mano_neurons # mano_base_layer是一个基础回归层，用于将输入特征转化为手部姿态和形状相关的特征表示。
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
            # mano_base_neurons[:-1] = [256, 1024]
            # mano_base_neurons[1:] = [1024, 512]
            # zip([1024, 1024], [1024, 512])得到的是一个元组迭代器，这个迭代器并不完全等同于元组数组[(1024, 1024),(1024, 512)]
            # 相较于元素组成的列表而言，元素组成的迭代器的优点：节省内存，当需要处理大量数据时，使用迭代器可以避免一次性将所有数据加载到内存中，从而节省内存空间。
                zip(mano_base_neurons[:-1], mano_base_neurons[1:])):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.LeakyReLU(inplace=True))
            # 在这里，base_layers一共添加了2层线性变换+2个激活函数，即base_layers=[Linear(1024,1024),LeakRuLu,Linear(1024,512),LeakRuLu]
        self.mano_base_layer = nn.Sequential(*base_layers)

        # Pose layers
        # pose_reg是一个线性层，用于预测手部姿态的6D表示，即由旋转矩阵表示的手指关节姿态。
        self.pose_reg = nn.Linear(mano_base_neurons[-1], self.pose6d_size) # channels:512=>16*6

        # Shape layers
        self.shape_reg = nn.Linear(mano_base_neurons[-1], 10) # channels:512=>10

        # mano_layer是另一个模型，用于根据手部姿态和形状参数生成手部网格模型的顶点坐标和关节位置。
        self.mano_layer = mano_layer

        # 将 coord_change_mat注册为模型的缓冲区。这样做的目的是将这些变量作为模型的固定参数，在模型的前向推理过程中保持不变
        if coord_change_mat is not None:
            self.register_buffer("coord_change_mat", coord_change_mat)
        else:
            self.coord_change_mat = None

    # features：[B,256*2*2]
    # if Train：如果是训练模式，则会传递mano_params参数；else：如果是预测模式，则会传递roots3d
    def forward(self, features, mano_params=None, roots3d=None):
        # features[B,256*2*2=1024] => mano_base_layer=[Linear(1024,1024),LeakRuLu,Linear(1024,512),LeakRuLu] => features[B,512]
        mano_features = self.mano_base_layer(features)
        # 使用线性层pose_reg将输入转化：mano_features[B,512] =>pred_mano_pose_6d[B,16*6]，16*6表示旋转矩阵的6维表示，即16个关节的3D旋转表示，每个旋转由6个参数表示。
        pred_mano_pose_6d = self.pose_reg(mano_features) #-----------------------------------regress pose-----------------------------------------

        # 将pred_mano_pose_6d转换为旋转矩阵表示的手部姿态pred_mano_pose_rotmat，并且变换其形状为[batch_size, 16, 3, 3]----具体是啥意思，不懂
        # pred_mano_pose_6d.view(-1, 6):[B,16*6] => [B*16,6] 
        # rot6d2mat([B*16,6] ) => [B*16, 3, 3]
        # [B*16, 3, 3].view(-1, 16, 3, 3) => [B,16, 3, 3]
        pred_mano_pose_rotmat = rot6d2mat(pred_mano_pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()

        # 线性层：mano_features[B,512] => pred_mano_shape[B,10]
        pred_mano_shape = self.shape_reg(mano_features) # 形状参数 #-----------------------------------regress shape-----------------------------------------

        # 将pred_mano_pose_rotmat转换为Rodrigues vector表示的手部姿态pred_mano_pose
        # pred_mano_pose_rotmat.view(-1, 3, 3): [B,16, 3, 3] => [B*16,3,3]
        # mat2aa([B*16,3,3]) = [B*16, 3],dim1=3代表Rodrigues向量中的3个分量[r1, r2, r3]
        # [B*16, 3].view(-1, self.mano_pose_size) => [B,16*3]
        pred_mano_pose = mat2aa(pred_mano_pose_rotmat.view(-1, 3, 3)).contiguous().view(-1, self.mano_pose_size) # 姿势参数

        # 使用pred_mano_pose和pred_mano_shape作为参数调用mano_layer模型，生成手部网格模型的顶点坐标pred_verts和关节位置pred_joints
        # mano_layer---为manopth.manopth.manopth.ManoLayer模型
        # ManoLayer是一个可微分的PyTorch层，可以确定地从姿势（pred_mano_pose）和形状参数（pred_mano_shape）映射到手部关节和顶点。 它可以作为可微分层集成到任何架构中以预测手部网格。
        pred_verts, pred_joints = self.mano_layer(th_pose_coeffs=pred_mano_pose, th_betas=pred_mano_shape)

        # if Train：如果是训练模式，则会传递mano_params参数。首先从mano_params中提取真实的手部形状gt_mano_shape和手部姿态gt_mano_pose。
        # 然后将gt_mano_pose减去手部平均值，并将其转换为旋转矩阵表示gt_mano_pose_rotmat。
        # 接下来，使用mano_layer模型根据gt_mano_pose和gt_mano_shape生成真实的顶点坐标gt_verts和关节位置gt_joints
        if mano_params is not None:
            gt_mano_shape = mano_params[:, self.mano_pose_size:]
            #print(gt_mano_shape)
            gt_mano_pose = mano_params[:, :self.mano_pose_size].contiguous()
            gt_mano_pose[:,3:] = gt_mano_pose[:,3:] - self.mano_layer.th_hands_mean
            gt_mano_pose_rotmat = batch_rodrigues(gt_mano_pose.view(-1, 3)).view(-1, 16, 3, 3)
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)
            gt_verts = gt_verts/1000
            gt_joints = gt_joints/1000
            gt_mano_results = {
                "verts3d": gt_verts,
                "joints3d": gt_joints,
                "mano_shape": gt_mano_shape,
                "mano_pose": gt_mano_pose_rotmat}
        else:
            gt_mano_results = None

        # if Not Train：如果是预测模式，则会传递roots3d
        # 如果传入了roots3d参数，则将顶点坐标pred_verts和关节位置pred_joints与roots3d相加，并根据需要进行坐标变换。
        # 最终得到预测的手部网格模型的顶点坐标pred_verts3d和关节位置pred_joints3d。
        if roots3d is not None: # evaluation
            roots3d = roots3d.unsqueeze(dim=1)
            pred_verts3d, pred_joints3d = pred_verts + roots3d, pred_joints + roots3d
            #scale to m 并且不转换为opengl坐标
            if self.coord_change_mat is not None:
                pred_verts3d = pred_verts3d.matmul(self.coord_change_mat)
                pred_joints3d = pred_joints3d.matmul(self.coord_change_mat)
            pred_mano_results = {
                "verts3d": pred_verts3d,
                "joints3d": pred_joints3d}
        else:
            pred_verts = pred_verts/1000
            pred_joints = pred_joints/1000
            pred_mano_results = {
                "verts3d": pred_verts, # 预测的三维手部顶点坐标
                "joints3d": pred_joints, # 预测的三维手部关节坐标
                "mano_shape": pred_mano_shape,# 预测的 MANO 模型的形状参数
                "mano_pose": pred_mano_pose_rotmat, # 预测的 MANO 模型的姿势参数，以旋转矩阵形式表示
                "mano_pose_aa": pred_mano_pose} # 预测的 MANO 模型的姿势参数，以 Rodrigues 向量形式表示

        # 返回预测的手部结果pred_mano_results和真实的手部结果gt_mano_results
        return pred_mano_results, gt_mano_results