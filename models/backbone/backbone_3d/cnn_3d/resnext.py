import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

__all__ = ['resnext50', 'resnext101', 'resnext152']  # 指明能被其他文件import*时的变量或函数，引入其他函数要写明from xxx import xxx


weight_pth = {
    "resnext50": '/home/su/YOWOv3/weights/resnext-50-kinetics.pth',
    "resnext101": '/home/su/YOWOv3/weights/resnext-101-kinetics.pth',
    "resnext152": '/home/su/YOWOv3/weights/resnext-152-kinetics.pth',
}


class MaxPool2D1D(nn.Module):
    def __init__(self, kernel_size=(3, 3, 3), stride=2, padding=1):
        super(MaxPool2D1D, self).__init__()
        self.kernel_size_2D = kernel_size[:2]
        self.kernel_size_1D = kernel_size[2:]
        self.maxpool2d = nn.MaxPool2d(kernel_size=self.kernel_size_2D, stride=stride, padding=padding)
        self.maxpool1d = nn.MaxPool1d(kernel_size=self.kernel_size_1D, stride=stride, padding=padding)

    def forward(self, x):
        B, C, L, H, W = x.shape
        x = x.reshape(B, C*L, H, W)  # B, C, L, H, W
        x = self.maxpool2d(x)  # B, CXL, H/2, W/2
        x = x.reshape(B, C, L, H//2, W//2)  # B, C, L, H/2, W/2
        x = x.permute(0, 1, 3, 4, 2)  # B, C, H/2, W/2, L
        x = x.reshape(B, C*H*W//4, L)  # B, C*H/2*W/2, L
        x = self.maxpool1d(x)  # B, C*H/2*W/2, L/2
        x = x.reshape(B, C, H//2, W//2, L//2)  # B, C, H/2 ,W/2, L/2
        x = x.permute(0, 1, 4, 2, 3)  # B, C, L/2, H/2 ,W/2
        return x


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    zero_pads = zero_pads.to(out.data.device)
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32,
                 multilevel=False):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.multilevel = multilevel
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = MaxPool2D1D(kernel_size=(3, 3, 3), stride=2, padding=1)  # 为了实现确定性算法
        # self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)
        c2 = self.layer1(c2)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        if self.multilevel:
            return [c3, c4, c5]   # B,C,T,H,W    C=512 1024 2048 T=/4 /8 /16   H=W=28 14 7
        else:
            if c5.size(2) > 1:
                c5 = torch.mean(c5, dim=2, keepdim=True)

            return [c5.squeeze(2)]


def load_weight(model, arch):
    print('Loading 3D backbone pretrained weight: {}'.format(arch.upper()))
    # checkpoint state dict
    checkpoint = torch.load(weight_pth[arch], map_location=torch.device('cpu'))
    checkpoint_state_dict = checkpoint.pop('state_dict')

    # model state dict
    model_state_dict = model.state_dict()
    # reformat checkpoint_state_dict:
    new_state_dict = {}
    for k in checkpoint_state_dict.keys():
        v = checkpoint_state_dict[k]
        new_state_dict[k[7:]] = v

    # check
    for k in list(new_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(new_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                new_state_dict.pop(k)
                # print(k)
        else:
            new_state_dict.pop(k)
            # print(k)

    model.load_state_dict(new_state_dict)
        
    return model


def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext50')

    return model


def resnext101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext101')

    return model


def resnext152(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model = load_weight(model, 'resnext152')

    return model


# build 3D resnet
def build_resnext_3d(model_name='resnext101', pretrained=True, multilevel=False):
    if model_name == 'resnext50':
        model = resnext50(pretrained=pretrained)
        feats = 2048

    elif model_name == 'resnext101':
        model = resnext101(pretrained=pretrained, multilevel=multilevel)
        if multilevel:
            feats = [512, 1024, 2048]
        else:
            feats = [2048]

    elif model_name == 'resnext152':
        model = resnext152(pretrained=pretrained)
        feats = 2048

    return model, feats


if __name__ == '__main__':
    import time
    model, feats = build_resnext_3d(model_name='resnext50', pretrained=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    x = torch.randn(1, 3, 16, 64, 64).to(device)
    # star time
    t0 = time.time()
    out = model(x)
    print('time', time.time() - t0)
    print(out.shape)
