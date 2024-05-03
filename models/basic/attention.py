import torch
import torch.nn as nn
from models.basic import Conv2d
from utils.box_ops import calculate_iou


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


# Channel Self Attetion Module  通道自注意力模块，返回(B，C，H，W)
class CSAM(nn.Module):
    """ Channel Self Attention Module """
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # 控制原始输入的所占比重，默认为0
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)  # B，C，C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # 相乘后越大的权重越小
        attention = self.softmax(energy_new)  # B，C，C

        # attention
        out = torch.bmm(attention, value)   # B，C，HxW
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


class CSAML(nn.Module):
    """ Channel Self Attention Module """
    def __init__(self, feat_size=28):
        super(CSAML, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # 控制原始输入的所占比重，默认为0
        self.softmax = nn.Softmax(dim=-1)
        self.feat_size = feat_size
        self.lnq = nn.Linear(self.feat_size * self.feat_size, self.feat_size * self.feat_size)
        self.lnk = nn.Linear(self.feat_size * self.feat_size, self.feat_size * self.feat_size)
        self.lnv = nn.Linear(self.feat_size * self.feat_size, self.feat_size * self.feat_size)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = self.lnq(x.view(B*C, -1)).view(B, C, -1)
        key = self.lnk(x.view(B*C, -1)).view(B, C, -1).permute(0, 2, 1)
        value = self.lnv(x.view(B*C, -1)).view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)  # B，C，C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)  # B，C，C

        # attention
        out = torch.bmm(attention, value)   # B，C，HxW
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


# Spatial Self Attetion Module
class SSAM(nn.Module):
    """ Spatial Self Attention Module """
    def __init__(self):
        super(SSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1).permute(0, 2, 1)   # [B, N, C]
        key = x.view(B, C, -1)                      # [B, C, N]
        value = x.view(B, C, -1).permute(0, 2, 1)   # [B, N, C]

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


class SSAML(nn.Module):
    """ Spatial Self Attention Module """
    def __init__(self, feat_size):
        super(SSAML, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.feat_size = feat_size
        self.lnq = nn.Linear(256, 256)
        self.lnk = nn.Linear(256, 256)
        self.lnv = nn.Linear(256, 256)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = self.lnq(x.view(B*C, -1)).view(B, C, -1)
        key = self.lnk(x.view(B*C, -1)).view(B, C, -1).permute(0, 2, 1)
        value = self.lnv(x.view(B*C, -1)).view(B, C, -1)

        query = self.lnq(x.view(B, C, -1).permute(0, 2, 1).view(-1, C)).view(B, -1, C)   # [B, N, C]
        key = self.lnk(x.view(B, C, -1).permute(0, 2, 1).view(-1, C)).view(B, -1, C).permute(0, 2, 1)  # [B, C, N]
        value = self.lnv(x.view(B, C, -1).permute(0, 2, 1).view(-1, C)).view(B, -1, C)   # [B, N, C]

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


class MPCA(nn.Module):
    # MultiPath Coordinate Attention
    def __init__(self, channels) -> None:
        super().__init__()

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2d(channels, channels)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv2d(channels, channels, (3, 1), p=(1, 0))  # padding保证卷积前后尺寸不变
        self.conv_pool_hw = Conv2d(channels, channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_hw = self.conv_hw(x_pool_hw)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()


class EMA(nn.Module):  # 已经通过将自适应池化修改为确定性池化，将EMA修改为确定性算法
    def __init__(self, channels, feat_size, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        self.feat_size = feat_size
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        # self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.agp = nn.AvgPool2d((self.feat_size, self.feat_size))
        self.pool_h = nn.AvgPool2d((1, self.feat_size))
        self.pool_w = nn.AvgPool2d((self.feat_size, 1))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)  #  32,32,28,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 32，32，28，28
        x2 = self.conv3x3(group_x)   # 32，32，28，28
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


############# Track Self Attention Module  跟踪自注意力模块，返回(C，H，W)#############
# Track Self Attention Module  跟踪自注意力模块，返回(C，H，W)
class TGAM(nn.Module):  # 图注意力方式
    """ Track Self Attention Module """
    def __init__(self):
        super(TGAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # 控制原始输入的所占比重，默认为0
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6, inplace=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        """
            inputs :
                x : input bboxs( N x 5 )  # 置信度以及xyxy，接下来要进行预测框之间的语义信息交互
                # 图注意力机制，
                # 由于每个预测框的尺寸不完全一致
                # 通道注意力并不合适，因为通道注意力学习到的是每张预测框置信度图的权重
                # 因此应该利用预测框的位置、尺寸、置信度进行图注意力计算
            returns :
                out : attention value + input feature
                attention: C x C
        """

        # 计算置信度加权后的中心距离作为权重? 越近权重越高
        pred_num, _ = x.size()
        iou = calculate_iou(x[:, 1:].view(pred_num, 1, 4), x[:, 1:].view(1, pred_num, 4), iou_type='ciou').reshape(
            pred_num, pred_num)  # ciou可能有负数，                1 - calculate
        conf_matrix = torch.matmul(x[:, 0].view(pred_num, 1), x[:, 0].view(1, pred_num))
        attention = iou*conf_matrix  # 越重合，并且可靠性越高的权重越高
        attention = self.softmax(attention)  # 对最后一个维度计算  pred_num, pred_num
        x_prime = x * self.gamma + torch.matmul(attention, x)  # pred_num, 5
        x_prime = x_prime.sigmoid()  # 防止溢出

        return x_prime


class TEAM(nn.Module):  # 已经通过将自适应池化修改为确定性池化，将TEMA修改为确定性算法 同时采用分步卷积和超大核卷积进行
    def __init__(self, channels=1, feat_size=224):
        super(TEAM, self).__init__()
        self.feat_size = feat_size
        self.ch = channels
        self.softmax = nn.Softmax(-1)
        # self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.ap = nn.AvgPool2d((self.feat_size, self.feat_size))
        self.pool_h = nn.AvgPool2d((1, self.feat_size))
        self.pool_w = nn.AvgPool2d((self.feat_size, 1))
        self.conv15x15 = nn.Sequential(nn.Conv2d(self.ch, self.ch, kernel_size=(1, 15), stride=1, padding=(0, 7)),
                                       nn.Conv2d(self.ch, self.ch, kernel_size=(15, 1), stride=1, padding=(7, 0)))
        self.conv23x23 = nn.Sequential(nn.Conv2d(self.ch, self.ch, kernel_size=(1, 23), stride=1, padding=(0, 11)),
                                       nn.Conv2d(self.ch, self.ch, kernel_size=(23, 1), stride=1, padding=(11, 0)))
        self.conv31x31 = nn.Sequential(nn.Conv2d(self.ch, self.ch, kernel_size=(1, 31), stride=1, padding=(0, 15)),
                                       nn.Conv2d(self.ch, self.ch, kernel_size=(31, 1), stride=1, padding=(15, 0)))

    def forward(self, x):
        b, c, h, w = x.size()  # 1,1,h=224,w=224
        x_h = self.pool_h(x)  # 1,1,224,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 1,1,224,1
        x1 = x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()  # b=1，c=1，h=224，w=224
        x2 = self.conv15x15(x)  # b=1，c=1，h=224，w=224
        x3 = self.conv23x23(x)  # b=1，c=1，h=224，w=224
        x4 = self.conv31x31(x)  # b=1，c=1，h=224，w=224
        x11 = self.ap(x1).sigmoid().reshape(b, -1, 1).permute(0, 2, 1)  # b=1,1,c=1
        x21 = self.ap(x2).sigmoid().reshape(b, -1, 1).permute(0, 2, 1)  # b=1,1,c=1
        x31 = self.ap(x3).sigmoid().reshape(b, -1, 1).permute(0, 2, 1)  # b=1,1,c=1
        x41 = self.ap(x4).sigmoid().reshape(b, -1, 1).permute(0, 2, 1)  # b=1,1,c=1

        x12 = x1.reshape(b, c, -1)  # b=1, c=1, hw
        x22 = x2.reshape(b, c, -1)  # b=1, c=1, hw
        x32 = x3.reshape(b, c, -1)  # b=1, c=1, hw
        x42 = x4.reshape(b, c, -1)  # b=1, c=1, hw

        # 第一种:计算任意组合关系 包含自身关系
        # weights = [torch.matmul(xi, xj) for xi in [x11, x21, x31, x41] for xj in [x12, x22, x32, x42]]  # 16,b,c,hw

        # 第二种:计算第一分支与其他分支之间的组合关系
        weights = [torch.matmul(x11, xi) for xi in [x22, x32, x42]]
        weights.extend([torch.matmul(xj, x12) for xj in [x21, x31, x41]])  # 6,b,c,hw

        weights = torch.stack(weights)  # _,b,c,hw

        weights = torch.sum(weights, dim=0).reshape(b, c, h, w)  # b=1,c=1,h,w
        return (x * weights.sigmoid()).reshape(b, c, h, w)


def get_track_attention(track_attention_type='CSAM', feat_size=224):
    if track_attention_type == 'CSAM':
        return CSAM()
    elif track_attention_type == 'EMA':
        return EMA(channels=1, feat_size=feat_size, factor=1)
    elif track_attention_type == 'TGAM':
        return TGAM()
    elif track_attention_type == 'TEAM':
        return TEAM(channels=1, feat_size=feat_size)

