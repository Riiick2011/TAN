import os
import torch
import torch.nn.functional as F
import argparse
from config import build_dataset_config, build_model_config
from models import build_model
from utils.misc import load_weight, CollateFunc, build_dataset, build_dataloader
from dataset.transforms import BaseTransform
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import distributed_utils

# 全局变量
gradient = None
activation = None


def parse_args():
    parser = argparse.ArgumentParser(description='TAN')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # Dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24,jhmdb21, ava_v2.2')
    parser.add_argument('--data_root', default='/media/su/d/datasets/UCF24-YOWO/',
                        help='data root')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')

    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=8, type=int,
                        help='total batch size.')
    parser.add_argument('-tbs', '--test_batch_size', default=8, type=int,
                        help='total test batch size')
    parser.add_argument('-accu', '--accumulate', default=16, type=int,
                        help='gradient accumulate.')  # 当增大单卡bs时候，应该减少accu  默认是 单卡bs8xaccu16=bs128

    # Model
    parser.add_argument('-v', '--version', default='tan_large', type=str,
                        help='build TAN')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')  # 恢复训练机制，模型恢复完整路径
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('--freeze_backbone_2d', action="store_true", default=False,
                        help="freeze 2D backbone.")
    parser.add_argument('--freeze_backbone_3d', action="store_true", default=False,
                        help="freeze 3d backbone.")
    parser.add_argument('--weight_folder', default='weights', type=str,
                        help='path to load and save weight')  # 保存训练结果模型权重的文件夹  与预训练模型的路径无关，预训练模型权重在各自的py文件中指定

    # Epoch
    parser.add_argument('--max_epoch', default=10, type=int,
                        help='max epoch.')
    parser.add_argument('--lr_epoch', nargs='+', default=[2, 3, 4], type=int,
                        help='lr epoch to decay')
    parser.add_argument('-lr', '--base_lr', default=0.0001, type=float,
                        help='base lr.')  # 基础学习率
    parser.add_argument('-ldr', '--lr_decay_ratio', default=0.5, type=float,
                        help='base lr.')

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False,
                        help='do evaluation during training.')  # 是否在训练过程中进行评估，默认为否
    parser.add_argument('--eval_epoch', default=1, type=int,
                        help='after eval epoch, the model is evaluated on val dataset.')  # 默认间隔1个周期进行一次评估
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')  # 只对评估时有用
    parser.add_argument('--infer_dir', default='results',
                        type=str, help='save inference results.')  # 评估的推断结果保存地址
    parser.add_argument('--map_path', default='evaluator/eval_results',
                        type=str, help='path to save mAP results')  # frame mAP结果的保存路径
    parser.add_argument('--link_method', default='viterbi',
                        type=str, help='link method in evaluating video mAP')  # 计算video mAP时采用的关联算法
    parser.add_argument('--det_save_type', default='one_class',
                        type=str, help='')  # 默认one_class表示一个检测框只对应一个类别得分，multi_class表示一个检测框对应所有类别得分

    # DDP train  分布式训练
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')  # 用于设置分布式训练的url
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')  # 分布式进程数
    parser.add_argument('--sybn', action='store_true', default=False,
                        help='use sybn.')  # 是否使用批次归一化分布式同步
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')  # local rank

    # Experiment
    parser.add_argument('--sampling_rate', nargs='+', default=[1], type=int,
                        help='sampling rate')  # 长度必须能被len_clip整除，但是其中元素可以重复，比如[4,2,4,1]
    parser.add_argument('--topk_nms', default=10, type=int,
                        help='topk per cls after nms.')  # 在nms之后总共保留的检测个数
    parser.add_argument('--untrimmed_training', action='store_true', default=False,
                        help='use untrimmed frames to train')  # 是否使用untrimmed frames进行训练，以获得鉴定动作起始点的能力提升
    # video mAP水平:frame mAP的评估使用的是trimmed frames，video mAP的评估使用的是untrimmed frames，
    # 模型默认是采用trimmed frames训练的， 使用untrimmed训练的模型只适合单独评估video mAP
    parser.add_argument('--track_mode', action='store_true', default=False,
                        help='use online track mode to train and test')  # 使用在线跟踪模式进行训练和测试，会自动重置tbs为1

    return parser.parse_args()

# Hook for Grad-CAM
def forward_hook(module, input, output):
    global activation
    activation = output

def backward_hook(module, grad_in, grad_out):
    global gradient
    gradient = grad_out[0]

if __name__ == '__main__':
    args = parse_args()

    # 跟踪模式重置tbs
    if args.track_mode:
        args.test_batch_size = 1
        print("Track Mode on: Reset TBS to 1\n")

    # dataset
    if args.dataset == 'ucf24':
        num_classes = 24

    elif args.dataset == 'jhmdb21':
        num_classes = 21

    elif args.dataset == 'ava_v2.2':
        num_classes = 80

    else:
        print('unknown dataset.')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # 加载预训练模型
    # build model
    if args.resume is not None:
        args.resume = os.path.join(os.path.abspath('.'), args.resume)  # 恢复文件的绝对路径
    model, criterion = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=True,
        resume=args.resume)

    # to eval  将batch normalization层和dropout层调整到评估模式    模型全部挪到指定device上
    model = model.to(device).train()

    model.head.cls_heads[0].register_forward_hook(forward_hook)
    model.head.cls_heads[0].register_backward_hook(backward_hook)

    # transform
    # BaseTransform类的实例，用于进行图像变换，改变尺寸，能返回变换后的视频片段tensor和target tensor
    basetransform = BaseTransform(img_size=d_cfg['img_size'])

    # dataset and evaluator  构建训练集实例、frame mAP评估器实例(包含测试集实例，如果要求进行评估)并返回
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=False)

    # dataloader
    batch_size = args.batch_size // distributed_utils.get_world_size()  # 单卡的batch size是batch size除以分布式进程(使用的显卡数量)个数
    dataloader = build_dataloader(args, dataset, batch_size, CollateFunc())

    for iter_i, (batch_img_name, batch_video_clips, batch_time_difs, batch_key_frame_target,
                 batch_clip_target) in enumerate(dataloader):
        # to device
        batch_video_clips = batch_video_clips.to(device).requires_grad_()
        batch_clip_target = [_.to(device) for _ in batch_clip_target]

        # inference
        batch_output = model(batch_video_clips, batch_time_difs, batch_clip_target)
        batch_output['cls_pred'][0].backward(torch.ones_like(batch_output['cls_pred'][0]))  # 为了简化，我们只对输出做backward

        # 获取特征和梯度
        feature = -activation[0]
        print(feature.shape)
        grad = gradient[0]
        print(grad.shape)

        # 平均梯度
        weights = grad.mean(dim=[1, 2], keepdim=True)
        print(weights.shape)

        # 计算 Grad-CAM
        grad_cam = F.sigmoid((weights * feature)).sum(dim=0)
        print(grad_cam.shape)
        print(grad_cam)

        print(batch_img_name[0])

        import cv2
        import numpy as np

        frame_dir = '/media/su/d/datasets/UCF24-YOWO/rgb-images/Skiing/v_Skiing_g01_c02/00001.jpg'
        save_path = '/home/su/TAN/results/'
        index = int(frame_dir[-9:-4])

        # 1. 使用 OpenCV 读取原始图像
        img = cv2.imread(frame_dir)

        # 2. 调整 Grad-CAM 热图的大小以匹配原始图像的大小
        heatmap = cv2.resize(grad_cam.detach().cpu().numpy(), (img.shape[1], img.shape[0]))

        # 3. 将 Grad-CAM 热图应用到原始图像上
        print(heatmap.max())
        print(heatmap.min())
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化到 [0,1]

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)

        cv2.imwrite(os.path.join(save_path, 'gc_{:0>5}.jpg'.format(index)), overlay)

        # 4. 显示或保存结果
        cv2.imshow('Grad-CAM', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()