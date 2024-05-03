#!/usr/bin/python
# encoding: utf-8
"""
本文件包含数据集类ucf_jhmdb的定义
"""

import os
import random
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat


# Dataset for UCF24 & JHMDB  UCF24 & JHMDB数据集类，用于训练和frame级别的评估
class UCF_JHMDB_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 is_train=False,
                 len_clip=16,
                 sampling_rate=[1],
                 untrimmed=False,  # 默认是使用trimmed数据集进行训练的，一定是使用trimmed数据集进行评估的
                 track_mode=False):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        self.track_mode = track_mode
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate  # 默认
        self.untrimmed = untrimmed if self.dataset == 'ucf24' else False  # JHMDB一定是trimmed

        if self.is_train:  # 根据需要可以切换为训练集和测试集   里面存有对应每张图片的标注文件名
            self.split_list = 'trainlist.txt'
            self.untrimmed_list = 'trainlist_untrimmed.txt'
        else:
            self.split_list = 'testlist.txt'
            self.untrimmed_list = 'testlist_untrimmed.txt'

        # load data
        with open(os.path.join(data_root, self.split_list), 'r') as file:
            self.file_names = file.readlines()
        self.num_samples = len(self.file_names)  # 样本数量=有标注的图片的总数
        # 用于untrimmed数据集的video mAP任务进行训练
        if self.untrimmed:
            with open(os.path.join(data_root, self.untrimmed_list), 'r') as file:
                self.file_names_untrimmed = file.readlines()
            self.num_samples = len(self.file_names_untrimmed)  # 样本数量=所有的图片的总数

        if dataset == 'ucf24':
            self.num_classes = 24
        elif dataset == 'jhmdb21':
            self.num_classes = 21

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # load a data
        img_name, video_clip, time_difs, key_frame_target, clip_target = self.pull_item(index)

        return img_name, video_clip, time_difs, key_frame_target, clip_target

    def pull_item(self, index):  # 返回 该帧图片的完整名称，视频片段，真实标注（字典形式)
        index = 1
        """ load a data """
        assert index <= len(self), 'index range error'
        if self.untrimmed:
            img_name = self.file_names_untrimmed[index].rstrip()
            label_name = img_name.replace('jpg', 'txt')
            label_path = os.path.join(self.data_root, 'labels', label_name)
        else:
            label_name = self.file_names[index].rstrip()  # 该帧图片的标注的完整名称
            if self.dataset == 'ucf24':
                img_name = label_name.replace('txt', 'jpg')  # 该帧图片的完整名称
            elif self.dataset == 'jhmdb21':
                img_name = label_name.replace('txt', 'jpg')  # 该帧图片的完整名称
            # path to label 该帧图片的标注的完整路径
            label_path = os.path.join(self.data_root, 'labels', label_name)

        img_split = img_name.split('/')  # ex. ['Basketball', 'v_Basketball_g08_c01', '00070.jpg']
        video_name = os.path.join(img_split[0], img_split[1])
        # image name 该帧图片的名称ID
        frame_id = int(img_split[-1][:5])

        # image folder 该帧图片的文件夹的完整路径
        img_folder = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1])

        # frame numbers  文件夹中的总帧数，其中jhmdb21需要-1 为什么?
        if self.dataset == 'ucf24':
            max_num = len(os.listdir(img_folder))
        elif self.dataset == 'jhmdb21':
            max_num = len(os.listdir(img_folder)) - 1

        # 采样率翻倍系数
        if self.is_train:  # sampling rate  如果训练中则采样率从1和2两个整数中随机取 是一种数据增强的训练方式
            d = random.randint(1, 2)  # 采样率系数
        else:
            d = 1  # 1

        sampling_rate_list = []  # 最后一项不起作用
        for sampling_rate in self.sampling_rate:
            sampling_rate_list.extend([sampling_rate * d] * (self.len_clip // len(self.sampling_rate)))
        sampling_rate_list[-1] = 0

        video_clip = []
        label_path_list = []
        time_difs = []
        for i in range(self.len_clip):
            # 得到临时帧的帧序号
            frame_id_temp = frame_id - sum(sampling_rate_list[-1-i:])  # i=0时是关键帧，也是当前帧
            if frame_id_temp < 1:
                frame_id_temp = 1
            elif frame_id_temp > max_num:
                frame_id_temp = max_num

            # 得到临时帧的图片完整路径和标注完整路径
            img_path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[-3], img_split[-2], '{:05d}.jpg'.format(frame_id_temp))
            label_path_tmp = os.path.join(os.path.split(label_path)[0], '{:05d}.txt'.format(frame_id_temp))
            time_dif = frame_id - frame_id_temp

            # 载入临时帧的图片
            frame = Image.open(img_path_tmp).convert('RGB')  # 一个Image对象
            ow, oh = frame.width, frame.height  # 图片的原始宽度和原始高度
            video_clip.append(frame)

            # 临时帧标注的完整路径
            label_path_list.append(label_path_tmp)

            # 临时帧到当前帧的时间距离-帧数差距
            time_difs.append(time_dif)

        video_clip = video_clip[::-1]  # 按照时间递增排序  video_clip是一个列表，每一项是一个Image对象
        label_path_list = label_path_list[::-1]  # 按照时间递增排序
        time_difs = time_difs[::-1]  # 按照时间递增排序

        time_difs_ = []  # 临时用
        clip_target = []
        for i in range(self.len_clip):
            label_path = label_path_list[i]
            time_dif = time_difs[i]
            time_difs_.append(time_dif)
            if os.path.exists(label_path):  # load an annotation  如果存在，则载入该帧图像对应的标注文件，数组表示
                target = np.loadtxt(label_path)
                # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]  调整标注的排列方式，数组表示
                label = target[..., :1]
                boxes = target[..., 1:]
                target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)
                clip_target.append(target)
            else:  # 如果不存在，则全0数组
                clip_target.append(np.array([]))

        # transform后尺寸变为方形，video_clip是一个列表，每一项对应一帧图像的tensor，
        # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
        # clip_target是一个列表，长度是len_clip
        video_clip, clip_target = self.transform(video_clip, clip_target)
        # List [T, 3, H, W] -> [3, T, H, W]  将视频片段列表转换维度顺序，成为一个[3, T, H, W]维度的tensor
        video_clip = torch.stack(video_clip, dim=1)
        # target_tensor对应video_clip中的标注（默认是两点小数格式)，也有可能是空tensor

        if len(clip_target[-1]):  # 如果最后一项，也就是关键帧,存在标注
            key_frame_target = {
                'boxes': clip_target[-1][:, :4].float(),  # [N, 4]
                'labels': clip_target[-1][:, -1].long() - 1,  # [N,]  #  训练的时候类别要从0开始
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }
            clip_target = clip_target[:-1]
        else:
            key_frame_target = {
                'boxes': clip_target[-1],  # 没有的时候是空tensor
                'labels': clip_target[-1],  # 没有的时候是空tensor
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }

        # 将tensor构成的列表转换为元组构成的列表，长度不一定是len_clip，而是只保留有标注的帧(不含关键帧)
        clip_target = [torch.cat([torch.ones([clip_target[_].shape[0], 1])*time_difs[_],
                                  (clip_target[_][:, -1]-1).unsqueeze(-1), torch.ones([clip_target[_].shape[0], 1]),
                                  clip_target[_][:, :-1]], -1)
                       for _ in range(len(clip_target)) if (len(clip_target[_]) and (time_difs[_] != 0))]
        clip_target = torch.cat(clip_target, 0) if len(clip_target) else torch.tensor([])

        # img_name是该帧图片的名称带后缀
        # video_clip是一个[3, T, H, W]维度的tensor，表示该帧图片对应的视频片段，是方形的
        # time_difs是一个列表，表示video_clip中每一帧到当前帧的时差,包括了clip_target中的时差
        # key_frame_target是一个字典，表示关键帧图片的标注，边界框是默认为两点式百分比的tensor，类别是tensor
        # clip_target是一个tensor，第一维是clip中具有标注的帧数项(不含关键帧)，第二维是7(距关键帧的时差:int>0, 类别从0开始，置信度，两点百分比边界框)
        return img_name, video_clip, time_difs, key_frame_target, clip_target


# Video Dataset for UCF24 & JHMDB UCF24 & JHMDB数据集类，用于video级别的评估，每个视频作为一个独立的数据集实例
class UCF_JHMDB_VIDEO_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 len_clip=16,
                 sampling_rate=[1]):  # untrimmed测试集
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
            
        if dataset == 'ucf24':
            self.num_classes = 24
        elif dataset == 'jhmdb21':
            self.num_classes = 21

    def set_video_data(self, video_name):  # 输入一个视频的名称
        self.video_name = video_name
        # load a video该视频的完整路径
        self.img_folder = os.path.join(self.data_root, 'rgb-images', self.video_name)
        if self.dataset == 'ucf24':
            self.img_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))  # 一个列表，存放该视频路径下的所有图片的完整路径
        elif self.dataset == 'jhmdb21':
            self.img_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        image_path = self.img_paths[index]
        video_name = self.video_name
        # for windows:
        # img_split = image_path.split('\\')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # for linux
        img_split = image_path.split('/')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']

        # image name
        frame_id = int(img_split[-1][:5])
        file_name = img_split[-2]
        class_name = img_split[-3]
        self.max_num = len(os.listdir(self.img_folder))

        img_name = os.path.join(video_name, '{:05d}.jpg'.format(frame_id))
        label_path = os.path.join(self.data_root, 'labels', class_name, file_name,
                                  '{:05d}.txt'.format(frame_id))  # 起始帧的标注文件

        # load video clip载入该图片对应的视频片段
        # 采样率翻倍系数
        d = 1  # 1  video数据集是只用来测试video mAP的 不存在训练的情况  在不适用多采样率的情况下，d=1代表连续

        sampling_rate_list = []  # 最后一项不起作用
        for sampling_rate in self.sampling_rate:
            sampling_rate_list.extend([sampling_rate * d] * (self.len_clip // len(self.sampling_rate)))
        sampling_rate_list[-1] = 0

        video_clip = []
        label_path_list = []
        time_difs = []
        for i in range(self.len_clip):
            # 得到临时帧的帧序号
            frame_id_temp = frame_id - sum(sampling_rate_list[-1 - i:])  # i=0时是关键帧，也是当前帧
            if frame_id_temp < 1:
                frame_id_temp = 1
            elif frame_id_temp > self.max_num:
                frame_id_temp = self.max_num

            # 得到临时帧的图片完整路径和标注完整路径
            img_path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[-3], img_split[-2],
                                        '{:05d}.jpg'.format(frame_id_temp))
            label_path_tmp = os.path.join(os.path.split(label_path)[0], '{:05d}.txt'.format(frame_id_temp))
            time_dif = frame_id - frame_id_temp

            # 载入临时帧的图片
            frame = Image.open(img_path_tmp).convert('RGB')  # 一个Image对象
            ow, oh = frame.width, frame.height  # 图片的原始宽度和原始高度
            video_clip.append(frame)

            # 临时帧标注的完整路径
            label_path_list.append(label_path_tmp)

            # 临时帧到当前帧的时间距离-帧数差距
            time_difs.append(time_dif)

        video_clip = video_clip[::-1]  # 按照时间递增排序  video_clip是一个列表，每一项是一个Image对象
        label_path_list = label_path_list[::-1]  # 按照时间递增排序
        time_difs = time_difs[::-1]  # 按照时间递增排序

        time_difs_ = []  # 临时用
        clip_target = []
        for i in range(self.len_clip):
            label_path = label_path_list[i]
            time_dif = time_difs[i]
            time_difs_.append(time_dif)
            if os.path.exists(label_path):  # load an annotation  如果存在，则载入该帧图像对应的标注文件，数组表示
                target = np.loadtxt(label_path)
                # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]  调整标注的排列方式，数组表示
                label = target[..., :1]
                boxes = target[..., 1:]
                target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)
                clip_target.append(target)
            else:
                clip_target.append(np.array([]))

        # transform后尺寸变为方形，video_clip是一个列表，每一项对应一帧图像的tensor，
        # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
        # clip_target是一个列表，长度是len_clip
        video_clip, clip_target = self.transform(video_clip, clip_target)
        # List [T, 3, H, W] -> [3, T, H, W]  将视频片段列表转换维度顺序，成为一个[3, T, H, W]维度的tensor
        video_clip = torch.stack(video_clip, dim=1)
        # target_tensor对应video_clip中的标注（默认是两点小数格式)，也有可能是空tensor

        if len(clip_target[-1]):  # 如果最后一项，也就是关键帧,存在标注
            key_frame_target = {
                'boxes': clip_target[-1][:, :4].float(),  # [N, 4]
                'labels': clip_target[-1][:, -1].long() - 1,  # [N,]  #  训练的时候类别要从0开始
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }
            clip_target = clip_target[:-1]
        else:
            key_frame_target = {
                'boxes': clip_target[-1],  # 没有的时候是空tensor
                'labels': clip_target[-1],  # 没有的时候是空tensor
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }

        # 将tensor构成的列表转换为元组构成的列表，长度不再是len_clip，而是只保留有标注的帧(不含关键帧)
        clip_target = [torch.cat([torch.ones([clip_target[_].shape[0], 1]) * time_difs[_],
                                  (clip_target[_][:, -1] - 1).unsqueeze(-1), torch.ones([clip_target[_].shape[0], 1]),
                                  clip_target[_][:, :-1]], -1)
                       for _ in range(len(clip_target)) if (len(clip_target[_]) and (time_difs[_] != 0))]
        clip_target = torch.cat(clip_target, 0) if len(clip_target) else torch.tensor([])

        # img_name是该帧图片的名称带后缀
        # video_clip是一个[3, T, H, W]维度的tensor，表示该帧图片对应的视频片段，是方形的
        # time_difs是一个列表，表示video_clip中每一帧到当前帧的时差,包括了clip_target中的时差
        # key_frame_target是一个字典，表示关键帧图片的标注，边界框是默认为两点式百分比的tensor，类别是tensor
        # clip_target是一个tensor，第一维是clip中具有标注的帧数项(不含关键帧)，第二维是7(距关键帧的时差:int>0，类别从0开始, 置信度，两点百分比边界框)

        return img_name, video_clip, time_difs, key_frame_target, clip_target


if __name__ == '__main__':
    import cv2
    from dataset.transforms import Augmentation, BaseTransform

    data_root = '/media/su/d/datasets/UCF24-YOWO'
    dataset = 'ucf24'
    is_train = True
    img_size = 224
    len_clip = 16
    trans_config = {
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5
    }
    train_transform = Augmentation(
        img_size=img_size,
        jitter=trans_config['jitter'],
        saturation=trans_config['saturation'],
        exposure=trans_config['exposure'])
    val_transform = BaseTransform(img_size=img_size)

    train_dataset = UCF_JHMDB_Dataset(
        data_root=data_root,
        dataset=dataset,
        img_size=img_size,
        transform=train_transform,
        is_train=is_train,
        len_clip=len_clip)

    print(len(train_dataset))
    for i in range(len(train_dataset)):
        frame_id, video_clip, target = train_dataset[i]
        key_frame = video_clip[:, -1, :, :]

        # to numpy
        key_frame = key_frame.permute(1, 2, 0).numpy()
        key_frame = key_frame.astype(np.uint8)

        # to BGR
        key_frame = key_frame[..., (2, 1, 0)]
        H, W, C = key_frame.shape

        key_frame = key_frame.copy()
        bboxes = target['boxes']
        labels = target['labels']

        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            key_frame = cv2.rectangle(key_frame, (x1, y1), (x2, y2), (255, 0, 0))

        # cv2 show
        cv2.imshow('key frame', key_frame)
        cv2.waitKey(5)
        
