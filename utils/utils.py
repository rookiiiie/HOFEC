import os
import sys
import json
import time
import datetime
import warnings
import torch


def progress_bar(msg=None):

    L = []
    if msg:
        L.append(msg)

    msg = ''.join(L)
    sys.stdout.write(msg+'\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeters:
    def __init__(self):
        super().__init__()
        self.average_meters = {}

    def add_loss_value(self, loss_name, loss_val, n=1):
        if loss_name not in self.average_meters:
            self.average_meters[loss_name] = AverageMeter()
        self.average_meters[loss_name].update(loss_val, n=n)


class Monitor:
    def __init__(self, hosting_folder):
        self.hosting_folder = hosting_folder

        self.train_path = os.path.join(hosting_folder, "train.txt")
        create_log_file(self.train_path)

        os.makedirs(self.hosting_folder, exist_ok=True)

    def log_train(self, epoch, errors):
        log_errors(epoch, errors, self.train_path)


def create_log_file(log_path, log_name=""):
    log_folder = os.path.dirname(log_path)
    os.makedirs(log_folder, exist_ok=True)
    with open(log_path, "a") as log_file:
        now = time.strftime("%c")
        log_file.write("==== log {} at {} ====\n".format(log_name, now))


def log_errors(epoch, errors, log_path=None):
    now = time.strftime("%c")
    message = "(epoch: {epoch}, time: {t})".format(epoch=epoch, t=now)
    for k, v in errors.items():
        message = message + ",{name}:{err}".format(name=k, err=v)

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
    return message


def print_args(args):
    opts = vars(args)
    print("======= Options ========")
    for k, v in sorted(opts.items()):
        print("{}: {}".format(k, v))
    print("========================")


def save_args(args, save_folder, opt_prefix="opt", verbose=True):
    opts = vars(args)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    opt_filename = "{}.txt".format(opt_prefix)
    opt_path = os.path.join(save_folder, opt_filename)
    with open(opt_path, "a") as opt_file:
        opt_file.write("====== Options ======\n")
        for k, v in sorted(opts.items()):
            opt_file.write("{option}: {value}\n".format(option=str(k), value=str(v)))
        opt_file.write("=====================\n")
        opt_file.write("launched {} at {}\n".format(str(sys.argv[0]), str(datetime.datetime.now())))
    if verbose:
        print("Saved options to {}".format(opt_path))


def load_checkpoint(model, resume_path, strict=True, device=None):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path)
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = {"module.{}".format(key): item for key, item in checkpoint["state_dict"].items()}
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def get_dataset(args, mode):
    from dataset.ho3d import HO3D
    dataset = HO3D(dataset_root=args.HO3D_root, obj_model_root=args.obj_model_root,
                   train_label_root="E:/localPy/HOFEC/data/ho3d_v2", mode=mode, inp_res=args.inp_res)
    return dataset

def get_dex_ycb_dataset(args,mode):
    from dataset.dex_ycb_object_test import dex_ycb
    dataset = dex_ycb(args.dex_ycb_root,mode,inp_res=args.inp_res)
    return dataset

def get_network(args):
   
    # change coordinates for HO3D dataset to OpenGL coordinates
    from manopth.manopth.manolayer import ManoLayer
    # 下面这两步的mano_layer不是没啥区别么。。。
    if args.use_ho3d:
        #from utils.manolayer_ho3d import ManoLayer
        mano_layer = ManoLayer(ncomps=45, center_idx=0, flat_hand_mean=True,
                          side="right", mano_root=args.mano_root, use_pca=False)
    else:
       # from manopth.manolayer import ManoLayer
        mano_layer = ManoLayer(ncomps=45, center_idx=0, flat_hand_mean=True,
                          side="right", mano_root=args.mano_root, use_pca=False)
    coord_change_mat = torch.tensor([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=torch.float32) #坐标变换矩阵，将数据集中的坐标系转变成OpenGL标准下的坐标

    if args.resume is not None:       # 在test.sh中含有此参数，在train.sh无参数，即表示是否使用预训练模型参数作为网络模型初始值
        pretrained = False
    else:
        pretrained = True

    from model import HOFEC, HOModel

    net = HOFEC(stacks=args.stacks, channels=args.channels, blocks=args.blocks,
                mano_layer=mano_layer, mano_neurons=args.mano_neurons,
                transformer_depth=args.transformer_depth,
                transformer_head=args.transformer_head,
                coord_change_mat=coord_change_mat,
                reg_object=True, pretrained=pretrained)

    net = torch.nn.DataParallel(net) #这个模型还被包装为一个DataParallel模型，这意味着它可以在多个GPU上并行计算

    model = HOModel(net, mano_lambda_verts3d=args.mano_lambda_verts3d,
                    mano_lambda_joints3d=args.mano_lambda_joints3d,
                    mano_lambda_manopose=args.mano_lambda_manopose,
                    mano_lambda_manoshape=args.mano_lambda_manoshape,
                    mano_lambda_regulshape=args.mano_lambda_regulshape,
                    mano_lambda_regulpose=args.mano_lambda_regulpose,
                    lambda_joints2d=args.lambda_joints2d,
                    lambda_objects=args.lambda_objects)

    # withoutGcnNet:81035666
    print_model_params(model)

    # args.pth = 'E:/localPy/HOFEC/host_folder/ho3d/checkpoint_50.pth.tar'
    # if args.pth is not None:
    #     print("-----------------checkpoint--------------------------------")
    #     load_checkpoint(model,args.pth)

    return model

def print_model_params(model):
    """打印模型的参数信息"""
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            # print(f"{name}: {num_params}")
    print(f"============================Total number of parameters: {total_params}")

def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))

    # save_zip_path = pred_out_path.replace(".json", ".zip")
    # subprocess.call(["zip", "-j", save_zip_path, pred_out_path])
    # print(f"Saved results to {save_zip_path}")
