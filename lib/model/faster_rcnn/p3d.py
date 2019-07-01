from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch._utils
import math
import torch.utils.model_zoo as model_zoo
import pdb
import numpy as np
import time
from collections import OrderedDict

__all__ = ['P3D', 'P3D63', 'P3D131','P3D199']


def conv_S(in_planes, out_planes, stride=1, padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=1,
                     padding=padding, bias=False)


def conv_T(in_planes, out_planes, stride=1, padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=1,
                     padding=padding, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_s=0, depth_3d=47, ST_struc=('A', 'B', 'C')):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.depth_3d = depth_3d
        self.ST_struc = ST_struc
        self.len_ST = len(self.ST_struc)

        stride_p = stride
        if not self.downsample == None:
            stride_p = (1, 2, 2)
        if n_s < self.depth_3d:
            if n_s == 0:
                stride_p = 1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            if n_s == self.depth_3d:
                stride_p = 2
            else:
                stride_p = 1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride_p)
            self.bn1 = nn.BatchNorm2d(planes)

        self.id = n_s
        self.ST = list(self.ST_struc)[self.id % self.len_ST]
        if self.id < self.depth_3d:
            self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(planes)
            #
            self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0))
            self.bn3 = nn.BatchNorm3d(planes)
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)

        if n_s < self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm3d(planes * 4)
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def ST_B(self, x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x + tmp_x

    def ST_C(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x + tmp_x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.id < self.depth_3d:  # C3D parts:

            if self.ST == 'A':
                out = self.ST_A(out)
            elif self.ST == 'B':
                out = self.ST_B(out)
            elif self.ST == 'C':
                out = self.ST_C(out)
        else:
            out = self.conv_normal(out)  # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class P3D(nn.Module):

    def __init__(self, block, layers, modality='RGB',
                 shortcut_type='B', num_classes=600, dropout=0.1, ST_struc=('A', 'B', 'C')):
        self.inplanes = 64
        super(P3D, self).__init__()

        self.input_channel = 3 if modality == 'RGB' else 2  # 2 is for flow
        self.ST_struc = ST_struc

        self.conv1_custom  = nn.Conv3d(self.input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                               padding=(0, 3, 3), bias=False)

        self.depth_3d = sum(layers[:3])  # C3D layers are only (res2,res3,res4),  res5 is C2D

        self.bn1 = nn.BatchNorm3d(64)  # bn1 is followed by conv1
        self.cnt = 0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=0)  # pooling layer for conv1.
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), padding=0,
                                      stride=(2, 1, 1))  # pooling layer for res2, 3, 4.

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=(10, 10), stride=1)  # pooling layer for res5.
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # some private attribute
        self.input_size = (self.input_channel, 16, 160, 160)  # input of the network
        self.input_mean = [0.485, 0.456, 0.406] if modality == 'RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality == 'RGB' else [np.mean([0.229, 0.224, 0.225])]

    @property
    def scale_size(self):
        return self.input_size[2] * 256 // 160  # asume that raw images are resized (340,256).

    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p = stride  # especially for downsample branch.

        if self.cnt < self.depth_3d:
            if self.cnt == 0:
                stride_p = 1
            else:
                stride_p = (1, 2, 2)
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(planes * block.expansion)
                    )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_s=self.cnt, depth_3d=self.depth_3d,
                            ST_struc=self.ST_struc))
        self.cnt += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_s=self.cnt, depth_3d=self.depth_3d, ST_struc=self.ST_struc))
            self.cnt += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom (x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.maxpool_2(self.layer1(x))  # Part Res2
        x = self.maxpool_2(self.layer2(x))  # Part Res3
        # x = self.maxpool_2(self.layer3(x))  # Part Res4
        x = self.layer3(x)  # Part Res4

        sizes = x.size()
        x = x.view(-1, sizes[1], sizes[3], sizes[4])  # Part Res5
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(self.dropout(x))

        return x

def P3D63(pretrained=False, modality='RGB', **kwargs):
    """Construct a P3D63 modelbased on a ResNet-50-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained == True:
        if modality == 'RGB':
            pretrained_file = 'resnet50-19c8e357.pth'
        weights = torch.load(pretrained_file)
        weights = transfer_weights(weights, model, [3, 4, 6], [64, 128, 256])
        model.load_state_dict(weights)
    return model


def P3D131(pretrained=False, modality='RGB', **kwargs):
    """Construct a P3D131 model based on a ResNet-101-3D model.
    """
    model = P3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained == True:
        if modality == 'RGB':
            pretrained_file = 'resnet101-5d3b4d8f.pth'
        weights = torch.load(pretrained_file)
        weights = transfer_weights(weights, model, [3, 4, 23], [64, 128, 256])
        model.load_state_dict(weights)
    return model


def P3D199(pretrained=False, modality='RGB', **kwargs):
    """construct a P3D199 model based on a ResNet-152-3D model.
    """
    model = P3D(Bottleneck, [3, 8, 36, 3], modality=modality, **kwargs)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    if pretrained == True:
        if modality == 'RGB':
            pretrained_file = 'p3d199.pth'
            # pretrained_file = 'resnet152-b121ed2d.pth'
        weights = torch.load(pretrained_file)
        # weights = transfer_weights(weights,model,[3, 8, 36],[64, 128, 256])
        model.load_state_dict(weights['state_dict'])
    return model


# transfer imagenet resnet weights to p3d model
def transfer_weights(weights, model, layers, channels):
    '''
    layers: resnet layers
    :param weights: resnet weights
    :param model: p3d model
    :return: p3d weights
    '''
    weights['conv1.weight'] = weights['conv1.weight'].unsqueeze(2)

    for i in range(len(layers)):
        weights['layer{:d}.0.downsample.0.weight'.format(i + 1)] = weights[
            'layer{:d}.0.downsample.0.weight'.format(i + 1)].unsqueeze(2)

    for i in range(len(layers)):
        for j in range(layers[i]):
            weights['layer{:d}.{:d}.conv4.weight'.format(i + 1, j)] = weights[
                'layer{:d}.{:d}.conv3.weight'.format(i + 1, j)]
            weights['layer{:d}.{:d}.conv3.weight'.format(i + 1, j)] = torch.empty(channels[i], channels[i], 3, 1, 1)
            nn.init.normal_(weights['layer{:d}.{:d}.conv3.weight'.format(i + 1, j)], 0, math.sqrt(2. / channels[i] * 3))

            weights['layer{:d}.{:d}.bn4.weight'.format(i + 1, j)] = weights[
                'layer{:d}.{:d}.bn3.weight'.format(i + 1, j)]
            weights['layer{:d}.{:d}.bn4.bias'.format(i + 1, j)] = weights['layer{:d}.{:d}.bn3.bias'.format(i + 1, j)]
            weights['layer{:d}.{:d}.bn4.running_mean'.format(i + 1, j)] = weights[
                'layer{:d}.{:d}.bn3.running_mean'.format(i + 1, j)]
            weights['layer{:d}.{:d}.bn4.running_var'.format(i + 1, j)] = weights[
                'layer{:d}.{:d}.bn3.running_var'.format(i + 1, j)]

            weights['layer{:d}.{:d}.bn3.weight'.format(i + 1, j)] = weights[
                'layer{:d}.{:d}.bn2.weight'.format(i + 1, j)]
            weights['layer{:d}.{:d}.bn3.bias'.format(i + 1, j)] = weights['layer{:d}.{:d}.bn2.bias'.format(i + 1, j)]
            weights['layer{:d}.{:d}.bn3.running_mean'.format(i + 1, j)] = weights[
                'layer{:d}.{:d}.bn2.running_mean'.format(i + 1, j)]
            weights['layer{:d}.{:d}.bn3.running_var'.format(i + 1, j)] = weights[
                'layer{:d}.{:d}.bn2.running_var'.format(i + 1, j)]

    for i in range(len(layers)):
        for j in range(layers[i]):
            for k in [1, 2, 4]:
                weights['layer{:d}.{:d}.conv{:d}.weight'.format(i + 1, j, k)] = weights[
                    'layer{:d}.{:d}.conv{:d}.weight'.format(i + 1, j, k)].unsqueeze(2)

    for j in range(3):
        weights['layer4.{:d}.conv_normal.weight'.format(j)] = weights['layer4.{:d}.conv2.weight'.format(j)]
        del weights['layer4.{:d}.conv2.weight'.format(j)]

        weights['layer4.{:d}.conv4.weight'.format(j)] = weights['layer4.{:d}.conv3.weight'.format(j)]
        del weights['layer4.{:d}.conv3.weight'.format(j)]

        weights['layer4.{:d}.bn_normal.weight'.format(j)] = weights['layer4.{:d}.bn2.weight'.format(j)]
        del weights['layer4.{:d}.bn2.weight'.format(j)]
        weights['layer4.{:d}.bn_normal.bias'.format(j)] = weights['layer4.{:d}.bn2.bias'.format(j)]
        del weights['layer4.{:d}.bn2.bias'.format(j)]
        weights['layer4.{:d}.bn_normal.running_mean'.format(j)] = weights['layer4.{:d}.bn2.running_mean'.format(j)]
        del weights['layer4.{:d}.bn2.running_mean'.format(j)]
        weights['layer4.{:d}.bn_normal.running_var'.format(j)] = weights['layer4.{:d}.bn2.running_var'.format(j)]
        del weights['layer4.{:d}.bn2.running_var'.format(j)]

        weights['layer4.{:d}.bn4.weight'.format(j)] = weights['layer4.{:d}.bn3.weight'.format(j)]
        del weights['layer4.{:d}.bn3.weight'.format(j)]
        weights['layer4.{:d}.bn4.bias'.format(j)] = weights['layer4.{:d}.bn3.bias'.format(j)]
        del weights['layer4.{:d}.bn3.bias'.format(j)]
        weights['layer4.{:d}.bn4.running_mean'.format(j)] = weights['layer4.{:d}.bn3.running_mean'.format(j)]
        del weights['layer4.{:d}.bn3.running_mean'.format(j)]
        weights['layer4.{:d}.bn4.running_var'.format(j)] = weights['layer4.{:d}.bn3.running_var'.format(j)]
        del weights['layer4.{:d}.bn3.running_var'.format(j)]

    return weights


# custom operation
def get_optim_policies(model=None, modality='RGB', enable_pbn=True):
    '''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn3, and many all bn2.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model == None:
        log.l.info('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate = 0.7
    n_fore = int(len(normal_weight) * slow_rate)
    slow_feat = normal_weight[:n_fore]  # finetune slowly.
    slow_bias = normal_bias[:n_fore]
    normal_feat = normal_weight[n_fore:]
    normal_bias = normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "slow_bias"},
        {'params': normal_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "normal_feat"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
    ]

class p3d(_fasterRCNN):
  def __init__(self, classes, num_layers=199, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/P3D199_rgb_299x299_model_best.pth.tar'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    p3d = P3D199()
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      weights = torch.load(self.model_path)
      keys = [key[7:] for key in weights['state_dict'].keys()]
      values = [value for value in weights['state_dict'].values()]
      new_weights = OrderedDict()
      new_weights = new_weights.fromkeys(keys)
      for i in range(len(keys)):
          new_weights[keys[i]] = values[i]
      p3d.load_state_dict(new_weights)
    # Build resnet.
    self.RCNN_base = nn.Sequential(p3d.conv1_custom, p3d.bn1,p3d.relu,p3d.maxpool,
                                   p3d.layer1,p3d.maxpool_2,p3d.layer2,p3d.maxpool_2,p3d.layer3)

    self.RCNN_top = nn.Sequential(p3d.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[8].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[6].train()
      self.RCNN_base[8].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
