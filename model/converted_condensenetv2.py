import torch
import torch.nn as nn
from utils import Conv, CondenseLGC, CondenseSFR, HS, SELayer, ResNet

__all__ = ['ConvertedCondenseNetV2', 'converted_cdnv2_a', 'converted_cdnv2_b', 'converted_cdnv2_c', 'converted_cdnv2_d']


class _SFR_DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args, activation, use_se=False):
        super(_SFR_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.group_trans = args.group_trans
        self.use_se = use_se
        ### 1x1 conv i --> b*k
        self.conv_1 = CondenseLGC(in_channels, args.bottleneck * growth_rate,
                                  kernel_size=1, groups=self.group_1x1,
                                  activation=activation)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3,
                           activation=activation)
        ### 1x1 res conv k(8-16-32)--> i (k*l)
        self.sfr = CondenseSFR(growth_rate, in_channels, kernel_size=1,
                               groups=self.group_trans, activation=activation)
        if self.use_se:
            self.se = SELayer(inplanes=growth_rate, reduction=1)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.use_se:
            x = self.se(x)
        sfr_feature = self.sfr(x)
        y = x_ + sfr_feature
        return torch.cat([y, x], 1)
    
class _SFR_DenseLayerLTDN(nn.Module):
    def __init__(self, in_channels, growth_rate, path, args, activation, use_se=False):
        super(_SFR_DenseLayerLTDN, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        self.path = path
        self.group_trans = args.group_trans
        self.use_se = use_se
        
        for i in range(path):
            ### 1x1 conv i --> b*k
            layer1 = CondenseLGC(int(in_channels/path), int(args.bottleneck * growth_rate/path),
                            kernel_size=1, groups=self.group_1x1,
                            activation=activation)
            self.add_module('path_%d%d' %((i + 1),  1), layer1)
            ### 3x3 conv b*k --> k
            layer2 = Conv(int(args.bottleneck * growth_rate/path), int(growth_rate/path),
                            kernel_size=3, padding=1, groups=self.group_3x3,
                            activation=activation)
            self.add_module('path_%d%d' % ((i + 1), 2), layer2)
            ### 1x1 res conv k(8-16-32)--> i (k*l)
            layer3 = CondenseSFR(int(growth_rate/path), int(in_channels/path), kernel_size=1,
                                     groups=self.group_trans, activation=activation)
            self.add_module('path_%d%d' % ((i + 1), 3), layer3)
            if self.use_se:
                self.se = SELayer(inplanes=int(growth_rate/path), reduction=1)

    def forward(self, x):
        input_channels = int(x.shape[1])
        path = self.path
        num_input_part_channels = int(input_channels/path)
        
        input_part = {}
        output_part = {}
        returnList = []
        
        for i in range(path):
            temp_input_part = x[:,i*num_input_part_channels:(i+1)*num_input_part_channels,:,:]
            input_part['input_part{0}'.format(i+1)] = temp_input_part
        
            output_part['output_part{0}'.format(i+1)] = eval(f'self.path_{i+1}{1}')(input_part['input_part{0}'.format(i+1)])
            output_part['output_part{0}'.format(i+1)] = eval(f'self.path_{i+1}{2}')(output_part['output_part{0}'.format(i+1)])
            
            if self.use_se:
                output_part['output_part{0}'.format(i+1)] = self.se(output_part['output_part{0}'.format(i+1)])
            
            input_part['input_part{0}'.format(i+1)] = input_part['input_part{0}'.format(i+1)] + \
                                                        eval(f'self.path_{i+1}{3}')(output_part['output_part{0}'.format(i+1)])
            
        for i in range(path):
            if i%2==0:
                returnList.append(input_part['input_part{0}'.format(i+1)])
                returnList.append(output_part['output_part{0}'.format(i+2)])
                returnList.append(input_part['input_part{0}'.format(i+2)])
                returnList.append(output_part['output_part{0}'.format(i+1)])
        
        return torch.cat(returnList, 1)     


class _SFR_DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, path, args, activation, use_se):
        super(_SFR_DenseBlock, self).__init__()
        
        if args.ltdn_model:
            for i in range(num_layers):
                layer = _SFR_DenseLayerLTDN(in_channels + i * growth_rate, growth_rate, path, args, activation, use_se)
                self.add_module('denselayer_%d' % (i + 1), layer)
                
        else:
            for i in range(num_layers):
                layer = _SFR_DenseLayer(in_channels + i * growth_rate, growth_rate, args, activation, use_se)
                self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class ConvertedCondenseNetV2(nn.Module):
    def __init__(self, args):

        super(ConvertedCondenseNetV2, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        self.paths = args.paths
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.dataset in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        if args.ltdn_model:
            resnet = ResNet(int(self.num_features/2), int(self.num_features/2),
                           kernel_size=[1,3,1])
            self.features.add_module('resnet', resnet)
        
        for i in range(len(self.stages)):
            activation = 'HS' if i >= args.HS_start_block else 'ReLU'
            use_se = True if i >= args.SE_start_block else False
            ### Dense-block i
            self.add_block(i, activation, use_se)

        self.fc = nn.Linear(self.num_features, args.fc_channel)
        if not args.ltdn_model:
            self.fc_act = HS()

        ### Classifier layer
        self.classifier = nn.Linear(args.fc_channel, args.num_classes)
        self._initialize()

    def add_block(self, i, activation, use_se):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _SFR_DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            path=self.paths[i],
            args=self.args,
            activation=activation,
            use_se=use_se,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        print('DenseBlock {} output channel {}'.format(i, self.num_features))
        if not last:
            trans = _Transition()
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))
            if use_se:
                self.features.add_module('se_last',
                                        SELayer(self.num_features, reduction=self.args.last_se_reduction))

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.fc(out)
        if not self.args.ltdn_model:
            out = self.fc_act(out)
        out = self.classifier(out)
        return out

    def _initialize(self):
        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def converted_cdnv2_a(args):
    args.stages = '1-1-4-6-8'
    args.growth = '8-8-16-32-64'
    print('Stages: {}, Growth: {}'.format(args.stages, args.growth))
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.condense_factor = 8
    args.trans_factor = 8
    args.group_1x1 = 8
    args.group_3x3 = 8
    args.group_trans = 8
    args.bottleneck = 4
    args.last_se_reduction = 16
    args.HS_start_block = 2
    args.SE_start_block = 3
    args.fc_channel = 828
    return ConvertedCondenseNetV2(args)


def converted_cdnv2_b(args):
    args.stages = '2-4-6-8-6'
    args.growth = '6-12-24-48-96'
    print('Stages: {}, Growth: {}'.format(args.stages, args.growth))
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.condense_factor = 6
    args.trans_factor = 6
    args.group_1x1 = 6
    args.group_3x3 = 6
    args.group_trans = 6
    args.bottleneck = 4
    args.last_se_reduction = 16
    args.HS_start_block = 2
    args.SE_start_block = 3
    args.fc_channel = 1024
    return ConvertedCondenseNetV2(args)


def converted_cdnv2_c(args):
    args.stages = '4-6-8-10-8'
    args.growth = '8-16-32-64-128'
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.condense_factor = 8
    args.trans_factor = 8
    args.group_1x1 = 8
    args.group_3x3 = 8
    args.group_trans = 8
    args.bottleneck = 4
    args.last_se_reduction = 16
    args.HS_start_block = 2
    args.SE_start_block = 3
    args.fc_channel = 1024
    return ConvertedCondenseNetV2(args)

def converted_cdnv2_d(args):
    args.stages = '4-5-6'
    args.growth = '8-16-32' # 2/4/
    print('Stages: {}, Growth: {}'.format(args.stages, args.growth))
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))
    args.paths = list(map(int, args.paths.split('-')))
    args.condense_factor = 4
    args.trans_factor = 4
    args.group_1x1 = 4
    args.group_3x3 = 4
    args.group_trans = 4
    args.bottleneck = 4
    # args.last_se_reduction = 16
    args.HS_start_block = 10
    args.SE_start_block = 10
    args.fc_channel = 1024
    return ConvertedCondenseNetV2(args)
