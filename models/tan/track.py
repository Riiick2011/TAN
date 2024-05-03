"""
定义了跟踪交叉注意力模块
"""
import numpy as np
import torch
import torch.nn as nn
from utils.utils import bbox_iou
from evaluator.kalman_filter import KalmanFilter
from models.basic.attention import get_track_attention
import cv2
from evaluator import nn_matching


"""Track_Det和Track_Tube只用于 KalmanFilter方法"""

class Track_Det(object):
    def __init__(self, det, score, feat):
        self.bbox = det  # x1y1x2y2
        self.score = score
        if type(feat) is np.ndarray and feat.size != 0:
            self.feat = cv2.resize(np.transpose(feat, (1, 2, 0)), (14, 14)).reshape(1, -1)
        else:
            self.feat = None
        self.xyah = self.to_xyah()

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.bbox.copy()
        xc = (self.bbox[0] + self.bbox[2]) / 2
        yc = (self.bbox[1] + self.bbox[3]) / 2
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        ret = [xc, yc, width/height, height]
        return ret


class Track_Tube(object):
    def __init__(self, det):
        self.det_list = [det.bbox]  # 用列表表示,每一项是一个形状为(4,)数组，对应一个检测框
        self.active = True
        self.score_list = [det.score]
        self.tube_score = sum(self.score_list)/len(self.score_list)
        self.miss_link_times = 0

        # 卡尔曼滤波
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(det.xyah)

        # 外观特征采用EMA更新
        self.ema_alpha = 0.9
        if type(det.feat) is np.ndarray:
            self.feat = det.feat / np.linalg.norm(det.feat)
        else:
            self.feat = None

    def __call__(self, det):  # 当前时刻进行了关联，因此也要用当前时刻的测量值更新一次卡尔曼
        # 更新轨迹
        self.det_list.append(det.bbox)  # 每一个时刻最多有一个检测框
        self.score_list.append(det.score)
        self.tube_score = sum(self.score_list) / len(self.score_list)
        self.miss_link_times = 0  # 重置漏检次数

        # 用当前时刻的测量值更新卡尔曼滤波器
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, det.xyah, det.score)

        # 更新外观特征
        if type(det.feat) is np.ndarray:
            if type(self.feat) is np.ndarray:  # 如果管道原来就有外观特征，则更新
                self.feat = (self.feat * self.ema_alpha + det.feat * (1 - self.ema_alpha)) / np.linalg.norm(self.feat)
            else:  # 如果管道原来没有外观特征，则新建
                self.feat = det.feat / np.linalg.norm(det.feat)

    def predict(self):  # 每个时刻调用一次，预测下一时刻的位置，同时预测下一时刻的内部均值
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)  # 预测该时刻的均值和协方差
        xyah, _ = self.kf.project(self.mean, self.covariance)  # 变换回测量空间
        xc, yc, a, h = xyah
        w = a*h
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        xyxy = [x1, y1, x2, y2]

        return xyxy, self.tube_score  # 按理说不该取这个作为预测框的置信度，因为这样取的话，只要没有观测值更新，那么对未来任何时刻的预测的置信度都是相同的

    def miss_link(self):
        self.miss_link_times += 1
        if self.miss_link_times >= 5:  # 当漏检时长过长时，判定该管道死亡
            self.active = False


class TrackPredict(nn.Module):  # 输入一个样本的先前检测框列表，输出该样本对当前时刻的检测框预测注意力图
    """ Track Block """
    def __init__(self, track_method, appear_3d):  # ch_in, kernels
        super().__init__()
        self.track_method = track_method
        self.appear_3d = appear_3d

        if self.track_method == 'KalmanFilter':
            self.iou_threshold = 0.5  # 关联时的iou阈值

    def forward(self, x, feat_3d=None, time_difs=None):
        """
            inputs :
                x : 一个2维度tensor，第一维长度不定，第二维长度为7，分别是时差、置信度、类别、两点百分比框
                feat_3d : C,T,H,W    C=64 H=W=112
                time_difs : 该样本所有图像帧到当前帧的时差
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        """Applies the forward pass through C1 module."""
        dtype = x.dtype
        device = x.device

        # 得到预测框
        if self.track_method == 'KalmanFilter':
            x = x.detach().cpu().numpy()
            time_dif_max = int(x[:, 0].max())  # 该clip中距离当前帧最早的时刻的时差
            dets_clip = [[] for _ in range(time_dif_max)]
            if self.appear_3d:
                appear_h = appear_w = feat_3d.shape[2]
                time_difs = np.array(time_difs)

            for row in x:
                time_dif = int(row[0])  # 该检测框所处时刻与当前帧的时差   不一定所有的检测框所处时刻都刚好在该片段具有对应的图像帧

                # 根据是否开启appear_3d以及是否具有对应的图像帧，创建该样本片段的检测框对象
                feat = None  # 默认不停
                if self.appear_3d:
                    frame_index = np.where(time_difs == time_dif)[0]
                    frame_index = frame_index[0] if frame_index.size > 0 else -1
                    if frame_index != -1:
                        feat = feat_3d[:, frame_index, int(appear_h * row[4]): int(appear_h * row[6]),
                               int(appear_w * row[3]): int(appear_w * row[5])].detach().cpu().numpy()

                dets_clip[int(row[0])-1].append(Track_Det(row[3:], row[2], feat=feat))
            dets_clip = dets_clip[::-1]  # 最早的时刻排在最前面

            # 关联不同时差的检测框
            tubes_list = []
            for i in range(time_dif_max):  # 对于所有的时刻  关联时先考虑考虑置信度
                dets_frame = dets_clip[i]  # 该时刻的检测框
                active_tubes = (
                    sorted([tube for tube in tubes_list if tube.active],
                           key=lambda x: x.tube_score, reverse=True))  # 该时刻存活的管道，按照得分降序排列
                for active_tube in active_tubes:
                    pred_det, _ = active_tube.predict()  # 该时刻存活的管道对该时刻进行预测
                    dets_frame_num = len(dets_frame)  # 该时刻的检测框数量
                    if dets_frame_num > 0:
                        ious_frame = [0 for _ in range(dets_frame_num)]
                        feat_similarities_frame = [0 for _ in range(dets_frame_num)]
                        for det_index in range(dets_frame_num):
                            ious_frame[det_index] = bbox_iou(pred_det, dets_frame[det_index].bbox)
                            if self.appear_3d and ((type(dets_frame[det_index].feat) is np.ndarray) and
                                                   (type(active_tube.feat) is np.ndarray)):  # 只有在开启外观功能并且双方都有外观时才有数值
                                feat_similarities_frame[det_index] = (1 - 1 / (1 + np.exp(-nn_matching._nn_cosine_distance(
                                    active_tube.feat, dets_frame[det_index].feat)[0])))  # 余弦距离
                        if max(ious_frame) >= self.iou_threshold:  # 想进行关联iou首先要超过阈值
                            if self.appear_3d:  # 如果补充了外观信息，则将外观相似度加到iou得分上
                                ious_frame = [ious_frame[_] + feat_similarities_frame[_] for _ in range(dets_frame_num)]
                            iou_max_index = ious_frame.index(max(ious_frame))  # 找到iou最大的索引
                            active_tube(dets_frame.pop(iou_max_index))  # 该管道关联该检测
                        else:
                            active_tube.miss_link()
                    else:
                        active_tube.miss_link()
                for det in dets_frame:  # 对于该时刻上剩余的检测框 新建管道
                    tubes_list.append(Track_Tube(det))

            # 对关键帧进行预测
            preds = []
            active_tubes = (
                sorted([tube for tube in tubes_list if tube.active],
                       key=lambda x: x.tube_score, reverse=True))  # 关键帧时刻存活的管道，按照得分降序排列
            for active_tube in active_tubes:
                pred_det, pred_score = active_tube.predict()
                pred_det.insert(0, pred_score)  # 长度为5
                preds.append(pred_det)
            preds = torch.tensor(preds, dtype=dtype, device=device)
            return preds


class BboxSemanticAtt(nn.Module):
    def __init__(self, bbox_semantic_att_type, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bbox_semantic_att_type = bbox_semantic_att_type
        self.tam = get_track_attention(self.bbox_semantic_att_type, feat_size=self.feat_size)  # 返回相应的跟踪注意力模块
        self.act1 = nn.Sigmoid()

    def forward(self, preds):
        """
        :param preds:  一个列表，每一项是一个tensor对应一个样本片段在当前帧的预测框，tensor可能为空，也可能包含多个预测框
        :return track_conf_map:  一个列表，每一项是一张二维tensor，对应一个样本片段在当前帧的预测框概率分布
        """
        B = len(preds)
        # 根据对当前帧预测的检测框，生成一张置信度图
        device = preds[0].device
        track_conf_maps = []
        for b in range(B):  # 对每个样本片段独立进行
            pred_num = len(preds[b])
            if pred_num:  # 防止没有预测框的情况
                track_conf_map = torch.zeros((self.feat_size, self.feat_size), device=device)
                if self.bbox_semantic_att_type in ['CSAM']:  # 每个预测框对应一个通道，然后使用注意力
                    pred_maps = [torch.zeros((self.feat_size, self.feat_size), device=device) for i in range(pred_num)]
                    confs = preds[b][:, 0]
                    bboxes = torch.floor(preds[b][:, -4:] * self.feat_size).to(torch.int)
                    for i in range(pred_num):  # 预测框数量
                        x1 = bboxes[i, 0]
                        y1 = bboxes[i, 1]
                        x2 = bboxes[i, 2]
                        y2 = bboxes[i, 3]
                        pred_maps[i][y1:y2, x1:x2] = confs[i]
                    pred_maps = torch.stack(pred_maps).unsqueeze(0)  # 1,pred_num,H,W
                    pred_maps = self.tam(pred_maps).squeeze(0)  # pred_num,H,W
                    track_conf_map = torch.sum(pred_maps, dim=0)  # H,W
                elif self.bbox_semantic_att_type == 'TGAM':  # 直接对预测框使用图注意力
                    preds_att = self.tam(preds[b])  # pred_num, 5
                    confs = preds_att[:, 0]
                    bboxes = torch.floor(preds_att[:, -4:] * self.feat_size).to(torch.int)
                    for i in range(bboxes.shape[0]):
                        x1 = bboxes[i, 0]
                        y1 = bboxes[i, 1]
                        x2 = bboxes[i, 2]
                        y2 = bboxes[i, 3]
                        track_conf_map[y1:y2, x1:x2] += confs[i]  # H,W
                elif self.bbox_semantic_att_type in ['EMA', 'TEAM']:
                    confs = preds[b][:, 0]
                    bboxes = torch.floor(preds[b][:, -4:] * self.feat_size).to(torch.int)
                    for i in range(bboxes.shape[0]):
                        x1 = bboxes[i, 0]
                        y1 = bboxes[i, 1]
                        x2 = bboxes[i, 2]
                        y2 = bboxes[i, 3]
                        track_conf_map[y1:y2, x1:x2] += confs[i]
                    track_conf_map = (self.tam(track_conf_map.view(1, 1, self.feat_size, self.feat_size)).
                                      view(self.feat_size, self.feat_size))  # H,W
                else:  # 直接将所有预测框的置信度加起来
                    confs = preds[b][:, 0]
                    bboxes = torch.floor(preds[b][:, -4:] * self.feat_size).to(torch.int)
                    for i in range(bboxes.shape[0]):
                        x1 = bboxes[i, 0]
                        y1 = bboxes[i, 1]
                        x2 = bboxes[i, 2]
                        y2 = bboxes[i, 3]
                        track_conf_map[y1:y2, x1:x2] += confs[i]
            else:  # 如果当前样本片段在当前帧没有预测框
                track_conf_map = torch.ones((self.feat_size, self.feat_size), device=device)

            track_conf_map = self.act1(track_conf_map)  # 对所有的轨迹置信度图进行一次激活函数 限制置信度处于0~1之间
            track_conf_maps.append(track_conf_map)

        # 得到的注意力图要计算损失，损失的标注是将有真实框的区域标注为1，其余位置标注为0；或者
        track_conf_maps = torch.stack(track_conf_maps)
        return track_conf_maps  # BX224x224


class TrackInject(nn.Module):
    def __init__(self, stride, track_mix_ratio, inject_method):
        super().__init__()
        self.stride = stride
        self.track_mix_ratio = torch.nn.Parameter(torch.full((1, ), track_mix_ratio), requires_grad=True)  # 或者不训练直接赋值
        self.inject_method = inject_method
        self.pool = nn.ModuleList([nn.AvgPool2d(stride, stride) for stride in self.stride])  # 用于将conf图缩小为特征对应的空间尺寸

    def forward(self, feats, track_conf_maps):
        """
        :param feats: 一个列表，每一项是一个BCHW维度的tensor对应一个层级，层级之间HW不一样
        :param track_conf_maps: 一个tensor，维度为Bx224x224
        :return: feats: 一个列表，每一项是一个BCHW维度的tensor对应一个层级，层级之间HW不一样
        """

        # 回归和分类应该不一样
        B, H, W = track_conf_maps.shape
        for level in range(len(feats)):
            C = feats[level].shape[1]
            att_map_for_this_level = self.pool[level](track_conf_maps.view(B, 1, H, W))  # B,1,H//s,W//s
            if self.inject_method == 'mix':  # 1.相乘后按比例混合
                feats[level] = (feats[level] * (1 - self.track_mix_ratio) +
                                feats[level] * att_map_for_this_level.repeat(1, C, 1, 1) * self.track_mix_ratio)
            elif self.inject_method == 'ca':  # 2.计算每个通道的交叉注意力，跟踪信息以通道注意力的方式注入，不是完全增强对应区域，而是增强对应区域明显的通道
                attention = torch.bmm(feats[level].view(B, C, -1), att_map_for_this_level.
                                      permute(0, 2, 3, 1).view(B, -1, 1))  # B,C,1 预测框所在区域的特征值高的通道得以加强
                attention = torch.softmax(attention, dim=1).view(B, -1, 1, 1)  # B,C,1,1
                feats[level] = torch.mul(feats[level], attention)

        return feats


class TrackNet(nn.Module):  # 输出到哪个位置也要实验，224x224的注意力图要通过平均池化转换为3种不同尺寸的注意力图
    """ Channel Fuse Series Attention Block """

    def __init__(self, m_cfg, feat_size, stride):  # ch_in, kernels
        super().__init__()

        # 跟踪模块
        self.track_pre_filter_th = m_cfg['track_pre_filter_th']
        self.track_method = m_cfg['track_method']
        self.appear_3d = m_cfg['appear_3d']
        self.tp = TrackPredict(self.track_method, self.appear_3d)

        # 修正模块
        self.bsa = BboxSemanticAtt(m_cfg['bbox_semantic_att_type'], feat_size)

        # 融合模块
        self.ti = TrackInject(stride, m_cfg['track_mix_ratio'], m_cfg['inject_method'])

    def forward(self, x, feats, feats_3d, time_difs):
        """
            inputs :
                x : 由Tensor构成的列表，每一个Tensor对应一个样本，尺寸为[Nx7]，记录过去时刻的检测框，可能有空tensor表示过去完全没有检测框，
                除此之外，过去时刻的时差不一定与clip中的视频帧时差一一对应
                feats_3d: 一个列表，如果长度为1则无用，如果长度为2，则第2项是未经过时间下采样的3D特征图 B,C,T,H,W    C=2048 64 T=1 T   H=W=7 112
                time_difs: 嵌套列表，每个列表对应一个样本中每一帧图片相对当前帧的时差，需要搭配feats_3d来提取特征
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        """Applies the forward pass through C1 module."""
        B = len(x)
        dtype = x[0].dtype
        device = x[0].device

        # 前置置信度过滤
        if self.track_pre_filter_th:  # 如果开启了前置置信度过滤，则过滤掉置信度不够高的框
            x = [track_tensor[track_tensor[:, 2] >= self.track_pre_filter_th] if len(track_tensor) else track_tensor
                 for track_tensor in x]  # 仍然是一个列表，每一项是一个tensor对应一个按照置信度过滤后的样本，可能有空tensor

        preds = [torch.tensor([], dtype=dtype, device=device) for _ in range(B)]  # 可能有空tensor

        # 跟踪预测模块
        for i, track_tensor in enumerate(x):  # 返回B个（N,5）列表，分别对应每个样本在当前帧的预测框
            if len(track_tensor):  # 如果当前样本存在可以跟踪的信息，则将跟踪信息和3d特征送入跟踪模块
                preds[i] = self.tp(track_tensor, feats_3d[-1][i, :, :, :, :], time_difs[i]) if self.appear_3d \
                    else self.tp(track_tensor)

        # 边界框语义注意力模块
        track_conf_maps = self.bsa(preds)  # Bx224x224

        # 跟踪注入模块
        if len(feats) == 2:
            decoupled = True
            cls_feats = feats[0]
            reg_feats = feats[1]
            # 获得注入
            cls_feats = self.ti(cls_feats, track_conf_maps)
            reg_feats = self.ti(reg_feats, track_conf_maps)
            return [cls_feats, reg_feats], track_conf_maps
        else:
            decoupled = False
            # 获得注入
            feats = self.ti(feats, track_conf_maps)
            return feats, track_conf_maps
