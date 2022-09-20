# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:02:04 2022

@author: labpc
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import logging

import torch
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import MODEL_EXTRAS
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from core.function import synthetic_train
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models

### edited
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from yacs.config import CfgNode as CN

from models.pose_hrnet_se import SEModule
###

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
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


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = MODEL_EXTRAS['pose_resnet']
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        
        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # ---------------------------------------------
        self.layer1_se = SEModule(channel_list=[64])
        self.layer2_se = SEModule(channel_list=[256])
        self.layer3_se = SEModule(channel_list=[512])
        self.layer4_se = SEModule(channel_list=[1024])
        
        # self.layer1_se = None
        # self.layer2_se = None
        # self.layer3_se = None
        # self.layer4_se = None
        self.se_config = cfg.MODEL.SE_MODULES

        # if self.se_config[0]:
        #     self.layer1_se = SEModule(channel_list=[64])
        # if self.se_config[1]:
        #     self.layer2_se = SEModule(channel_list=[256])
        # if self.se_config[2]:
        #     self.layer3_se = SEModule(channel_list=[512])
        # if self.se_config[3]:
        #     self.layer4_se = SEModule(channel_list=[1024]) ## last stage

        # ---------------------------------------------
        return

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):     

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # --------------------------------
        x = self.layer1_se([x])[0]
        x = self.layer1(x) ## 256 x 64 x 48

        # --------------------------------
        x = self.layer2_se([x])[0]
        x = self.layer2(x) ## 512 x 32 x 24
        
        # --------------------------------
        x = self.layer3_se([x])[0]
        x = self.layer3(x) ## 1024 x 16 x 12
        
        # --------------------------------
        x = self.layer4_se([x])[0]
        x = self.layer4(x) ## 2048 x 8 x 6
        
        # --------------------------------
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

#is_train  =True
def get_pose_net(is_train, **kwargs):   
    num_layers = 50
    style = 'pytorch'

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and True:
        model.init_weights('D:/MIPNet-main/models/pytorch/ochuman/checkpoint_103.pth')

    return model

def createFolder(directory):  
    try:        
        if not os.path.exists(directory):            
            os.makedirs(directory)    
    except OSError:        
        print ('Error: Creating directory. ' +  directory)

def main():   
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    model = get_pose_net(True)
    model.eval()    
    
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=True).cuda()
    
    # Data loading code
    normalize = transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]    )
    
    # print("start gather dataset")
    
    # train_dataset = eval('dataset.mpii')(
    #     cfg,
    #     cfg.DATASET.ROOT,
    #     cfg.DATASET.TRAIN_SET,
    #     True,
    #     transforms.Compose([transforms.ToTensor(), normalize])
    # )
    # print("gathered dataset")
    
    # print("start gather valid dataset")
    # valid_dataset = eval('dataset.mpii')(
    #     cfg,
    #     cfg.DATASET.ROOT,
    #     cfg.DATASET.TEST_SET,
    #     False,
    #     transforms.Compose([transforms.ToTensor(), normalize])
    # )
    # print("gathered valid dataset")
        
    # # ----------------------------------------------
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=cfg.TRAIN.SHUFFLE,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY
    # )
    
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=cfg.PIN_MEMORY
    # )
    
    train_dataset = eval('dataset.coco')(
        cfg, 
        #root = cfg.DATASET.ROOT,
        cfg.DATASET.DATASET,
        cfg.DATASET.TRAIN_ANNOTATION_FILE,
        cfg.DATASET.DATA_FORMAT,
        cfg.DATASET.TRAIN_SET, 
        True, 
        transforms.Compose([transforms.ToTensor(),normalize]),
        )
    
    
    # train_dataset = []
    # valid_dataset = []
    
    valid_dataset = eval('dataset.coco')(
        cfg, 
        #root = cfg.DATASET.ROOT,
        cfg.DATASET.DATASET,
        cfg.DATASET.TEST_ANNOTATION_FILE,
        cfg.DATASET.DATA_FORMAT,
        cfg.DATASET.TEST_SET, 
        True, 
        transforms.Compose([transforms.ToTensor(), normalize]),
        )

    
    # ----------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # next(iter(train_loader))[0]
    # # # ----------------------------------------------
    
    #pred 값 저장
    now = datetime.now()
    date = now.date()
    hour = now.hour
    minute = now.minute
    
    now_time = str(date) + "_" + str(hour) + "_" + str(minute)
    
    final_output_dir = 'D:/MIPNet-main/output/%s'%now_time
    
    createFolder(final_output_dir)
    
    f = open(final_output_dir + '/log.txt', 'w')
    f.write('wrtie down log in %s'%now_time)
    f.close()
    
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')
    END_EPOCH = 140

    # if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
    #     logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    #     checkpoint = torch.load(checkpoint_file)
    #     begin_epoch = checkpoint['epoch']
    #     best_perf = checkpoint['perf']
    #     last_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])

    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #         checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)    
    
    
    for epoch in range(0, END_EPOCH):

        # # # train for one epoch
        # print('training on coco')
        # train(cfg, train_loader, model, criterion, optimizer, epoch,
        #       final_output_dir, tb_log_dir, writer_dict)

        lr_scheduler.step()

        # if epoch % cfg.EPOCH_EVAL_FREQ == 0:
        #     ### evaluate on validation set
        #     perf_indicator = validate(
        #         cfg, valid_loader, valid_dataset, model, criterion,
        #         final_output_dir, tb_log_dir, writer_dict, epoch=epoch, print_prefix='baseline'
        #     )
        # else:
        #     perf_indicator = 0.0
        
        print("train for one epoch")
        train(cfg, train_loader, model, criterion, optimizer, epoch, final_output_dir)
        
        #perf_indicator = 0.0
        #perf_indicator = validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir)
        validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir)

        # if perf_indicator >= best_perf:
        #     best_perf = perf_indicator
        #     best_model = True
        # else:
        #     best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'latest_state_dict': model.module.state_dict(),
            'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
        #}, best_model, final_output_dir, filename='checkpoint_{}.pth'.format(epoch + 1))

    # # ----------------------------------------------
    ## validate as ending point
    print('validate on coco')
    

    # # ----------------------------------------------
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar' )
    logger.info('=> saving final model state to {}'.format( final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)        

if __name__ == '__main__':
    main()