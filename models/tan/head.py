import torch.nn as nn
import torch
import math
from models.basic import Conv2d


class Headv2(nn.Module):  # 修改自YOWOv2所用的DecoupledHead，可选解耦，支持多层级输入
    def __init__(self, m_cfg, nc=24, decoupled_in=False, ch=()):
        super().__init__()
        print('==============================')
        print('Head: Headv2\n')
        print('Head Decoupled Input: {}\n'.format(str(decoupled_in)))
        self.decoupled_in = decoupled_in  # 分类和回归分支是否解耦输入
        self.nc = nc  # number of classes
        self.nl = len(ch)  # 输入的层级数量
        self.num_cls_heads = m_cfg['num_cls_heads']
        self.num_reg_heads = m_cfg['num_reg_heads']
        self.act_type = m_cfg['head_act']
        self.norm_type = m_cfg['head_norm']
        self.head_dim = m_cfg['head_dim']
        self.depthwise = m_cfg['head_depthwise']

        self.cls_heads = nn.ModuleList(nn.Sequential(*[
            Conv2d(self.head_dim,
                   self.head_dim,
                   k=3, p=1, s=1,
                   act_type=self.act_type,
                   norm_type=self.norm_type,
                   depthwise=self.depthwise) for __ in range(self.num_cls_heads)])
                                       for _ in ch)  # 定位分支 层级独立
        self.reg_heads = nn.ModuleList(nn.Sequential(*[
            Conv2d(self.head_dim,
                   self.head_dim,
                   k=3, p=1, s=1,
                   act_type=self.act_type,
                   norm_type=self.norm_type,
                   depthwise=self.depthwise) for __ in range(self.num_reg_heads)])
                                       for _ in ch)  # 回归分支 层级独立

        # pred 3个不同尺度的3种预测层
        self.conf_preds = nn.ModuleList(nn.Conv2d(self.head_dim, 1, kernel_size=1) for _ in ch)
        self.cls_preds = nn.ModuleList(nn.Conv2d(self.head_dim, self.nc, kernel_size=1) for _ in ch)
        self.reg_preds = nn.ModuleList(nn.Conv2d(self.head_dim, 4, kernel_size=1) for _ in ch)

        self.bias_init()

    def forward(self, x):
        assert isinstance(x, list)
        if self.decoupled_in:  # x=[cls_feats,reg_feats]
            cls_feats = [self.cls_heads[i](x[0][i]) for i in range(self.nl)]
            reg_feats = [self.reg_heads[i](x[1][i]) for i in range(self.nl)]
        else:  # x=feats[level1,level2,level3.]
            cls_feats = [self.cls_heads[i](x[i]) for i in range(self.nl)]
            reg_feats = [self.reg_heads[i](x[i]) for i in range(self.nl)]

        conf_preds = [self.conf_preds[i](reg_feats[i]) for i in range(self.nl)]
        cls_preds = [self.cls_preds[i](cls_feats[i]) for i in range(self.nl)]
        reg_preds = [self.reg_preds[i](reg_feats[i]) for i in range(self.nl)]
        return conf_preds, cls_preds, reg_preds

    def bias_init(self):
        # Init bias  单独初始化存在目标的置信度的偏置和分类得分的偏置
        # 回归部分的的偏置 采用默认的随机初始化
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))  # -ln99
        # obj pred  # 存在目标的置信度
        for conf_pred in self.conf_preds:
            b = conf_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            conf_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred  # 分类得分
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class Headv8(nn.Module):  # heads  YOLOv8的头部 不同层级互不影响
    # YOLOv8 Detect head for detection models
    def __init__(self, m_cfg, nc=24, decoupled_in=False, ch=()):  # detection layer
        super().__init__()
        print('==============================')
        print('Head: Headv8\n')
        print('Head Decoupled Input: {}\n'.format(str(decoupled_in)))
        self.decoupled_in = decoupled_in  # 分类和回归分支是否解耦输入
        self.nc = nc  # number of classes
        self.nl = len(ch)  # 输入的层级数量
        self.reg_max = m_cfg['reg_max']  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + 4 * self.reg_max  # number of outputs per anchor
        self.stride = torch.tensor(m_cfg['stride'], dtype=torch.float32)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, 4 * self.reg_max)), max(ch[0], self.nc)  # channels

        self.cv_reg = nn.ModuleList(
            nn.Sequential(Conv2d(x, c2, 3, p=1, norm_type='BN', act_type='silu'),
                          Conv2d(c2, c2, 3, p=1, norm_type='BN', act_type='silu'),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)  # 定位分支 层级独立
        self.cv_cls = nn.ModuleList(
            nn.Sequential(Conv2d(x, c3, 3, p=1, norm_type='BN', act_type='silu'),
                          Conv2d(c3, c3, 3, p=1, norm_type='BN', act_type='silu'),
                          nn.Conv2d(c3, self.nc, 1)) for x in ch)  # 分类分支，没经过sigmoid 层级独立
        self.bias_init()

    def forward(self, x):
        assert isinstance(x, list)
        y = []
        if self.decoupled_in:  # x=[cls_feats,reg_feats]
            for i in range(self.nl):  # 层级数量
                y.append(torch.cat((self.cv_reg[i](x[1][i]), self.cv_cls[i](x[0][i])), 1))
        else:  # x=feats
            for i in range(self.nl):  # 层级数量
                y.append(torch.cat((self.cv_reg[i](x[i]), self.cv_cls[i](x[i])), 1))
        return y  # 返回一个列表，其中每一项对应一层输出，每层的输出是reg和cls在通道维度上拼接起来

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        for a, b, s in zip(self.cv_reg, self.cv_cls, self.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


def build_head(m_cfg, num_classes=24, decoupled_in=False, ch=256):
    if m_cfg['head_type'] == 'Headv2':
        return Headv2(m_cfg, num_classes, decoupled_in, ch)
    elif m_cfg['head_type'] == 'Headv8':
        return Headv8(m_cfg, num_classes, decoupled_in, ch)
