import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone import build_backbone_2d
from ..backbone import build_backbone_3d
from models.tan.encoder import build_encoder
from .head import build_head
from utils.nms import multiclass_nms
from .matcher import dist2bbox
from .track import TrackNet


# You Only Watch Once
class TAN(nn.Module):
    def __init__(self, 
                 m_cfg,
                 device,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 nms_iou_type='iou',
                 multi_hot=False,
                 topk_nms=10,
                 det_save_type='one_class',
                 track_mode=False,
                 img_size=224
                 ):
        super(TAN, self).__init__()
        self.m_cfg = m_cfg
        self.device = device
        self.stride = self.m_cfg['stride']  # stride是个列表，包含3项  表示骨架输出乃至最终输出的网格大小
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.multi_hot = multi_hot

        # track_mode
        self.track_mode = track_mode

        # 推断时用
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.nms_iou_type = nms_iou_type
        self.topk_nms = topk_nms
        self.det_save_type = det_save_type

        # ------------------ Network ---------------------
        # 2D backbone  构建骨架网络，如果有预训练模型并且处于训练模式则还载入预训练模型
        # bk_dim_2d是一个3项的列表，每一项表示一个层级的特征图通道数，[256,256,256]
        self.backbone_2d, bk_dim_2d,  = build_backbone_2d(
            self.m_cfg, pretrained=self.m_cfg['pretrained_2d'] and self.trainable)
            
        # 3D backbone  构建骨架网络，如果有预训练模型并且处于训练模式则还载入预训练模型
        # bk_dim_3d是一个3项的列表，每一项表示一个层级的特征图通道数,是[2048]或者[512,1024,2048]
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            self.m_cfg, pretrained=self.m_cfg['pretrained_3d'] and self.trainable)

        self.level_2d = len(bk_dim_2d)  # 特征图层级数量
        self.feat_size_2d = [self.img_size//stride for stride in self.stride]
        self.decoupled_position = self.m_cfg['decoupled_position']  # 解耦位置，可选2DBackbone、Neck、Head
        self.head_type = self.m_cfg['head_type']
        self.fpn = self.m_cfg['fpn']

        # Neck
        if self.decoupled_position in ['2DBackbone', 'Neck']:  # 骨架或者颈部解耦  即头部的输入是解耦的
            # Neck
            # 2D与3D特征融合并编码
            # cls channel encoder 3个不同尺度的分类通道编码器
            self.cls_encoders = nn.ModuleList(
                [build_encoder(self.m_cfg, bk_dim_2d[0]+bk_dim_3d[0], self.m_cfg['head_dim'],
                               attention_type=self.m_cfg['attention_type'][0], feat_size=self.feat_size_2d[i])
                    for i in range(self.level_2d)])

            # reg channel & spatial encoder  3个不同尺度的回归通道编码器，仍然用的是通道注意力
            self.reg_encoders = nn.ModuleList(
                [build_encoder(self.m_cfg, bk_dim_2d[0]+bk_dim_3d[0], self.m_cfg['head_dim'],
                               attention_type=self.m_cfg['attention_type'][-1], feat_size=self.feat_size_2d[i])
                    for i in range(self.level_2d)])

            # Head
            if self.head_type == 'Headv2':  # 原版解耦头Headv2
                # head 分别对应3个不同层级的头部  每个头部包含两个支路，并行输入，并行输出，内部独立的  均采用多层的3x3卷积  通道数不变
                self.head = build_head(self.m_cfg, self.num_classes, decoupled_in=True,
                                       ch=[self.m_cfg['head_dim'] for _ in range(self.level_2d)])
            elif self.head_type == 'Headv8':  # YOLOv8的头-无conf分支的头
                self.reg_max = 1 if 'reg_max' not in self.m_cfg else self.m_cfg['reg_max']  # pred层的回归相关
                self.use_dfl = self.reg_max > 1  # 预测层是否采用dfl   只有noconf才可选是否使用dfl
                # 预测层，不输出conf_pred  内部不同层级是独立计算的
                self.head = build_head(self.m_cfg, self.num_classes, decoupled_in=True,
                                       ch=[self.m_cfg['head_dim'] for _ in range(self.level_2d)])
            else:
                raise Exception('构建网络时遇到未识别的head_type:{}'.format(self.head_type))

        else:  # 骨架或者颈部不解耦  即头部的输入不是解耦的
            # Neck
            # 2D与3D特征融合并编码
            # channel encoder 3个不同尺度的通道编码器，各自进行融合和注意力编码，编码结果拼接成列表送入头部
            self.channel_encoders = nn.ModuleList(
                [build_encoder(self.m_cfg, bk_dim_2d[0] + bk_dim_3d[0], self.m_cfg['head_dim'],
                               attention_type=self.m_cfg['attention_type'][0], feat_size=self.feat_size_2d[i])
                 for i in range(self.level_2d)])

            # Head
            if self.head_type == 'Headv2':  # 原版解耦头Headv2
                # head 分别对应3个不同层级的头部  每个头部包含两个支路，并行输入，并行输出，内部独立的  均采用多层的3x3卷积  通道数不变
                self.head = build_head(self.m_cfg, self.num_classes, decoupled_in=False,
                                       ch=[self.m_cfg['head_dim'] for _ in range(self.level_2d)])
            elif self.head_type == 'Headv8':  # YOLOv8的头-无conf分支的头
                self.reg_max = 1 if 'reg_max' not in self.m_cfg else self.m_cfg['reg_max']  # pred层的回归相关
                self.use_dfl = self.reg_max > 1  # 预测层是否采用dfl   只有noconf才可选是否使用dfl
                # 预测层，不输出conf_pred  内部不同层级是独立计算的
                self.head = build_head(self.m_cfg, self.num_classes, decoupled_in=False,
                                       ch=[self.m_cfg['head_dim'] for _ in range(self.level_2d)])
            else:
                raise Exception('构建网络时遇到未识别的head_type:{}'.format(self.head_type))

        if self.track_mode:
            self.track_net = TrackNet(self.m_cfg, self.img_size, self.stride)

        # init tan
        self.init_tan()

    def init_tan(self):
        # Init yolo  初始化2维骨架yolo中的2D批次归一化
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    # 生成的锚点框是固定的，中心是网格中心，不是先验锚点框   独立一层的anchors
    def make_anchors(self, feats, grid_cell_offset=0.5):
        """
            Generate anchors from features.
            feats是一个列表，每一项是一个tensor对应一个层级的输出，形状为(B,C,H,W)
            strides是一个列表，每一项是一个int对应一个层级的stride
            返回沿着空间尺寸拼接好的anchor_points(M,2)和stride_tensor(M,1),M是所有层级的锚点框总数
        """
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(self.stride):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def bbox_decode(self, anchor_point, pred, xywh=False):
        """
        :param anchor_point: (Tensor)[M,2]
        :param pred:  (Tensor)[B,M,4 or reg_max*4]  检测框回归预测
        :param xywh:  (Bool) 输入的格式,True代表是相对锚点中心的xywh中心式百分比坐标，False代表是相对锚点中心的ltrb两点式百分比坐标
        :return: decoded_box (Tensor)[B,M,4] 返回两点式百分比坐标，还没有乘以stride
        """

        if xywh:  # 将相对锚点中心的xywh中心式百分比坐标预测结果转换为两点式百分比坐标
            # center of bbox  预测框中心的绝对坐标
            pred_ctr_xy = anchor_point + pred[..., :2]
            # size of bbox 预测框的宽高
            pred_box_wh = pred[..., 2:].exp()  # 可能导致inf
            pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
            pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
            decoded_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)  # [B, M, 4] or [M, 4]  两点式百分比坐标
        else:  # 将相对锚点中心的ltrb距离两点式百分比坐标预测结果转换为两点式百分比坐标
            if self.use_dfl:
                b, m, c = pred.shape  # batch, anchors, channels
                pred = pred.view(b, m, 4, c // 4).softmax(3).matmul(
                    torch.arange(self.reg_max, dtype=torch.float, device=pred.device).type(pred.dtype))  # B,M,4  相对每个锚点的lrwb距离(还没乘以stride)
            decoded_box = dist2bbox(pred, anchor_point, xywh=False)  # xyxy两点式(还没乘以stride)
        return decoded_box

    # one_hot模式的后处理  只对eval和test起作用
    def post_process_one_hot(self, bbox_pred, cls_pred, conf_pred=None):
        """
        Input: 一个样本的预测输出
            conf_pred: (Tensor) [M, 1]
            cls_pred(score_pred): (Tensor) [M, Nc]
            bbox_pred: (Tensor) [M, 4]  两点式绝对坐标
        """
        if self.det_save_type == 'multi_class':  # 如果每个检测保存所有类别得分
            if self.head_type in ['Headv8']:
                raise Exception('multi_class保存模式需要提供conf置信度得分')
            elif self.head_type in ['Headv2']:
                # Keep top k top scoring indices only.  把所有预测框按照目标置信度排序
                conf_pred = conf_pred.sigmoid()
                cls_pred = cls_pred.sigmoid()
                cls_pred = torch.sqrt(conf_pred * cls_pred)  # cls得分经过conf修正

                # filter out the proposals with low confidence score 保留修正分类得分大于置信度阈值的预测框以及对应原始序号，这样一个预测框可以对应多个类别
                keep = conf_pred > self.conf_thresh
                conf_pred = conf_pred[keep]
                cls_pred = cls_pred[keep]
                bbox_pred = bbox_pred[keep]  # 保留的预测框  两点式绝对坐标表示

                # to cpu
                scores = conf_pred.cpu().numpy()
                cls_pred = cls_pred.cpu().numpy()
                bboxes = bbox_pred.cpu().numpy()

                # nms 不关注类别的多类别nms，各个类别互相影响
                scores, cls_pred, bboxes = multiclass_nms(
                    scores, cls_pred, bboxes, self.nms_thresh, self.nms_iou_type, num_classes=self.num_classes,
                    topk=self.topk_nms, class_agnostic=True)
            else:
                raise Exception('后处理中发现未识别的head_type:{}'.format(self.head_type))

            return scores, cls_pred, bboxes  # 输出该样本经过置信度筛选和nms筛选过后的预测框输出(不分层级)，其中预测框坐标是两点式绝对坐标
        else:
            if self.head_type in ['Headv8']:
                scores = cls_pred.sigmoid().flatten()  # [M * Nc,]  先排列同一个预测框的C个类别，再排列不同预测框
            elif self.head_type in ['Headv2']:
                # (H x W x C,)用置信度得分来修正每一个类别的得分，先排列同一个预测框的C个类别，再排列不同预测框
                scores = (torch.sqrt(conf_pred.sigmoid() * cls_pred.sigmoid())).flatten()
            else:
                raise Exception('后处理中发现未识别的head_type:{}'.format(self.head_type))

            # torch.sort is actually faster than .topk (at least on GPUs)
            scores, score_ids = scores.sort(descending=True)

            # filter out the proposals with low confidence score 保留修正分类得分大于置信度阈值的预测框以及对应原始序号，这样一个预测框可以对应多个类别
            keep = scores > self.conf_thresh
            scores = scores[keep]
            score_ids = score_ids[keep]

            bbox_ids = torch.div(score_ids, self.num_classes, rounding_mode='floor')  # 保留的预测框所对应的锚点框序号
            labels = score_ids % self.num_classes  # 保留的预测框所对应的类别(在这里一个预测框和不同的类别算作多个预测)

            bboxes = bbox_pred[bbox_ids]  # 保留的预测框  两点式绝对坐标表示

            # to cpu
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            bboxes = bboxes.cpu().numpy()

            # nms 关注类别的多类别nms，各个类别互不影响
            scores, labels, bboxes = multiclass_nms(
                scores, labels, bboxes, self.nms_thresh, self.nms_iou_type, num_classes=self.num_classes,
                topk=self.topk_nms, class_agnostic=False)

            return scores, labels, bboxes  # 输出该样本经过置信度筛选和nms筛选过后的预测框输出(不分层级)，其中预测框坐标是两点式绝对坐标

    # multi_hot模式的后处理  只对eval和test起作用
    def post_process_multi_hot(self, bbox_pred, cls_pred, conf_pred=None):
        """
        Input: 一个样本的预测输出
            conf_pred: (Tensor) [M, 1]
            cls_pred: (Tensor) [M, Nc]
            bbox_pred: (Tensor) [M, 4] 两点式绝对坐标
        """
        if self.head_type in ['Headv8']:
            # cls_pred
            cls_pred = torch.sigmoid(cls_pred)  # [M, Nc]

            # threshold
            keep = cls_pred.amax(1).gt(self.conf_thresh)
            cls_pred = cls_pred[keep]
            bbox_pred = bbox_pred[keep]

            # to cpu
            scores = cls_pred.amax(1).cpu().numpy()  # 预测框的最大得分类别作为得分
            cls_pred = cls_pred.cpu().numpy()
            bboxes = bbox_pred.cpu().numpy()
        elif self.head_type in ['Headv2']:
            # conf pred
            conf_pred = torch.sigmoid(conf_pred.squeeze(-1))   # [M, ]

            # cls_pred
            cls_pred = torch.sigmoid(cls_pred)                 # [M, Nc]

            # threshold
            keep = conf_pred.gt(self.conf_thresh)
            conf_pred = conf_pred[keep]
            cls_pred = cls_pred[keep]
            bbox_pred = bbox_pred[keep]

            # to cpu
            scores = conf_pred.cpu().numpy()
            cls_pred = cls_pred.cpu().numpy()
            bboxes = bbox_pred.cpu().numpy()
        else:
            raise Exception('后处理中发现未识别的head_type:{}'.format(self.head_type))
        # 无视类别的多类别nms
        scores, cls_pred, bboxes = multiclass_nms(
            scores, cls_pred, bboxes, self.nms_thresh, self.nms_iou_type, num_classes=self.num_classes,
            topk=self.topk_nms, class_agnostic=True)

        # [M, 4 + 1 + Nc]
        out_boxes = np.concatenate([bboxes, scores[..., None], cls_pred], axis=-1)

        return out_boxes

    # 前向推断一次
    def forward_propagation(self, video_clips, batch_time_difs, track=None):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        """
        # key frame
        key_frame = video_clips[:, :, -1, :, :]
        # 3D backbone
        feats_3d = self.backbone_3d(video_clips)  # 列表表示

        # 2D backbone and Neck
        if self.decoupled_position in ['2DBackbone']:
            cls_feats, reg_feats = self.backbone_2d(key_frame)
            feat_3d_ups = [F.interpolate(feats_3d[0], scale_factor=2 ** (2 - level)) for level in
                           range(self.level_2d)]
            cls_feats = [self.cls_encoders[level](cls_feats[level], feat_3d_ups[level])
                         for level in range(self.level_2d)]
            reg_feats = [self.reg_encoders[level](reg_feats[level], feat_3d_ups[level])
                         for level in range(self.level_2d)]
            feats = [cls_feats, reg_feats]  # 嵌套列表，先解耦后等级
        elif self.decoupled_position in ['Neck']:
            feats_2d = self.backbone_2d(key_frame)
            feat_3d_ups = [F.interpolate(feats_3d[0], scale_factor=2 ** (2 - level)) for level in
                           range(self.level_2d)]
            cls_feats = [self.cls_encoders[level](feats_2d[level], feat_3d_ups[level])
                         for level in range(self.level_2d)]
            reg_feats = [self.reg_encoders[level](feats_2d[level], feat_3d_ups[level])
                         for level in range(self.level_2d)]
            feats = [cls_feats, reg_feats]  # 嵌套列表，先解耦后等级
        else:
            feats_2d = self.backbone_2d(key_frame)
            feat_3d_ups = [F.interpolate(feats_3d[0], scale_factor=2 ** (2 - level)) for level in
                           range(len(feats_2d))]
            feats = [self.channel_encoders[level](feats_2d[level], feat_3d_ups[level]) for level in
                     range(len(feats_2d))]

        track_conf_maps = torch.tensor([], device=key_frame.device)
        if self.track_mode and self.m_cfg['inject_position'] == 'neck':
            # track是一个batch_size的列表，每一项是一个tensor，tensor的第一维不一样长，第二维都是7，可能存在空tensor
            # 对于每一个tensor都要进行独立运算，每个都生成一张注意力热图，空tensor 用len判断则生成全1的注意力热图
            feats, track_conf_maps = self.track_net(track, feats, feats_3d, batch_time_difs)


        # Head
        if self.head_type == 'Headv2':
            # pred
            conf_preds, cls_preds, reg_preds = self.head(feats)
            if self.track_mode and self.m_cfg['inject_position'] == 'head':
                conf_preds, track_conf_maps = self.track_net(track, conf_preds, feats_3d, batch_time_difs)
            anchor_point, stride_tensor = self.make_anchors(conf_preds)  # 所有层级拼接在一起   层级拼在一起的anchor

            # 将多个层级拼接在一起
            conf_pred = torch.cat([conf_pred.flatten(2, 3) for conf_pred in conf_preds], dim=-1
                                  ).permute(0, 2, 1).contiguous()  # B,M,C
            cls_pred = torch.cat([cls_pred.flatten(2, 3) for cls_pred in cls_preds], dim=-1
                                 ).permute(0, 2, 1).contiguous()  # B,M,C
            reg_pred = torch.cat([reg_pred.flatten(2, 3) for reg_pred in reg_preds], dim=-1
                                 ).permute(0, 2, 1).contiguous()  # B,M,C

            # decode
            bbox_pred = self.bbox_decode(anchor_point, reg_pred, xywh=True)  # B,M,4  xyxy两点式坐标(还没乘以stride)
            return conf_pred, cls_pred, bbox_pred, anchor_point, stride_tensor, track_conf_maps
        elif self.head_type in ['Headv8']:
            # 列表，含有层级数量项，每一项是一个tensor(由reg和cls在通道维度上拼接而成)
            preds = self.head(feats)  # B,C,H,W
            dist_pred, score_pred = \
                torch.cat([xi.view(xi.shape[0], self.reg_max * 4 + self.num_classes, -1) for xi in preds], 2
                          ).permute(0, 2, 1).contiguous().split(
                    (self.reg_max * 4, self.num_classes), 2)  # 跨层级，在锚点框个数上拼接起来，然后再把回归和分类结果分开  B,M,C

            # decode
            anchor_point, stride_tensor = self.make_anchors(preds)  # 所有层级拼接在一起   层级拼在一起的anchor
            bbox_pred = self.bbox_decode(anchor_point, dist_pred)  # B,M,4  xyxy两点式坐标(还没乘以stride)
            return score_pred, bbox_pred, dist_pred, anchor_point, stride_tensor, track_conf_maps
        else:
            raise Exception('前向传递中发现未识别的head_type:{}'.format(self.head_type))

    # 表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建  只对eval和test起作用 不用于训练
    # 前向推断一次并进行后处理
    @torch.no_grad()
    def inference(self, video_clips, batch_time_difs, track=None):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        return:
        """
        batch_size, _, _, img_h, img_w = video_clips.shape
        if self.head_type == 'Headv2':
            conf_pred, cls_pred, bbox_pred, anchor_point, stride_tensor, _ = self.forward_propagation(
                video_clips, batch_time_difs, track)
        elif self.head_type == 'Headv8':
            score_pred, bbox_pred, dist_pred, anchor_point, stride_tensor, _ = self.forward_propagation(
                video_clips, batch_time_difs, track)
        else:
            raise Exception('推断中发现未识别的head_type:{}'.format(self.head_type))

        # 后处理
        bbox_pred = bbox_pred * stride_tensor
        if self.multi_hot:
            batch_bboxes = []
            for batch_idx in range(batch_size):
                # post-process
                if self.head_type == 'Headv2':
                    out_boxes = self.post_process_multi_hot(bbox_pred[batch_idx], cls_pred[batch_idx],
                                                            conf_pred[batch_idx])
                elif self.head_type in ['Headv8']:
                    out_boxes = self.post_process_multi_hot(bbox_pred[batch_idx], score_pred[batch_idx])
                else:
                    raise Exception('后处理中发现未识别的head_type:{}'.format(self.head_type))

                # normalize bbox 归一化输出
                out_boxes[..., :4] /= max(img_h, img_w)
                out_boxes[..., :4] = out_boxes[..., :4].clip(0., 1.)

                batch_bboxes.append(out_boxes)
            return batch_bboxes  # 适配ava evaluator的格式
        else:
            batch_scores = []  # 共有batch size项，每一项是一个tensor对应一个样本的输出
            batch_labels = []
            batch_bboxes = []  # 是两点式百分比坐标
            for batch_idx in range(batch_size):  # 批次内的第batch_idx个样本，逐个样本进行后处理
                # [B, M, C] -> [M, C]
                # post-process  对该样本的多层级输出进行后处理
                # 输入的坐标是两点式绝对坐标
                # 输出该样本经过置信度筛选和nms筛选过后的预测框输出(不分层级)，其中预测框坐标是两点式绝对坐标
                if self.head_type == 'Headv2':
                    scores, labels, bboxes = self.post_process_one_hot(bbox_pred[batch_idx], cls_pred[batch_idx],
                                                                       conf_pred[batch_idx])
                elif self.head_type in ['Headv8']:
                    scores, labels, bboxes = self.post_process_one_hot(bbox_pred[batch_idx], score_pred[batch_idx])
                else:
                    raise Exception('后处理中发现未识别的head_type:{}'.format(self.head_type))

                # normalize bbox  再将坐标归一化并且钳位
                bboxes /= max(img_h, img_w)
                bboxes = bboxes.clip(0., 1.)

                batch_scores.append(scores)
                batch_labels.append(labels)  # 有效类别从0开始
                batch_bboxes.append(bboxes)
            return batch_scores, batch_labels, batch_bboxes  # 列表，含有batch_size项，每一项是一个tensor对应一个样本，其中预测框坐标是两点式百分比坐标

    # 该方法用于训练
    def forward(self, video_clips, batch_time_difs, track=None):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
            batch_time_difs: (List) # 嵌套列表，每个列表对应一个样本中每一帧相对当前帧的时差，包含当前帧，过去为正数
            track: (List) # 如果存在，则训练时为该clip中前几帧的标注，推断时为该clip中前几帧的检测结果
        return:
            outputs: (Dict) -> {
                ("conf_pred": conf_pred,  # (Tensor) [B, M, 1])
                ("cls_pred": cls_pred,  # (Tensor) [B, M, Nc])
                ("score_pred": score_pred,  # (Tensor) [B, M, Nc])
                "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                "anchor_point": anchor_point,  # (Tensor) [M, 2]
                "stride_tensor": stride_tensor}  # (Tensor) [M, 1]
            }
        """
        if not self.trainable:  # 前向推断一次，并进行后处理
            return self.inference(video_clips, batch_time_difs, track)
        else:  # 前向推断一次，返回计算损失所需的结果
            # Output
            if self.head_type == 'Headv2':
                conf_pred, cls_pred, bbox_pred, anchor_point, stride_tensor, track_conf_maps = (
                    self.forward_propagation(video_clips, batch_time_difs, track))
                # output dict
                outputs = {"conf_pred": conf_pred,  # (Tensor) [B, M, 1]
                           "cls_pred": cls_pred,  # (Tensor) [B, M, Nc]
                           "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                           "anchor_point": anchor_point,  # (Tensor) [M, 2]
                           "stride_tensor": stride_tensor,  # (Tensor) [M, 1]
                           "track_conf_maps": track_conf_maps  # 没开启 TrackMode的时候是None
                           }
                return outputs
            elif self.head_type == 'Headv8':
                score_pred, bbox_pred, dist_pred, anchor_point, stride_tensor, track_conf_maps = (
                    self.forward_propagation(video_clips, batch_time_difs, track))
                # output dict
                outputs = {"score_pred": score_pred,  # (Tensor) [B, M, Nc]
                           "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                           "dist_pred": dist_pred,
                           # (Tensor) [B, M, self.reg_max * 4]dist_pred是分布式表示的ltrb两点式(相对锚点、还没乘以stride)
                           "anchor_point": anchor_point,  # (Tensor) [M, 2]
                           "stride_tensor": stride_tensor,  # (Tensor) [M, 1]
                           "track_conf_maps": track_conf_maps  # 没开启 TrackMode的时候是None
                           }
                return outputs
            else:
                raise Exception('训练中发现未识别的head_type:{}'.format(self.head_type))
