import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torchvision import models
import timm
from swin_transformer import SwinTransformer
from itertools import chain
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
from adaptive_avgmax_pool import SelectAdaptivePool2d

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f = False, use_posture=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self._use_posture = use_posture
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        
        if use_posture:
            cls_direction = nn.Linear(linear, 2)
            cls_direction.apply(weights_init_classifier)
            self.cls_direction = cls_direction

        self.add_block = add_block
        self.classifier = classifier
        
    def forward(self, x):
        f = self.add_block(x)
        glogit = self.classifier(f)
        dlogit = None
        if self._use_posture:
            dlogit = self.cls_direction(f)
        
        if self.return_f:
            return [glogit, f, dlogit]
        else:
            return [glogit, dlogit]


def load_pretrained(model, default_cfg, num_classes=1000, in_chans=3, filter_fn=None):

    state_dict = model_zoo.load_url(default_cfg['url'])

    if in_chans == 1:
        conv1_name = default_cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        assert False, "Invalid in_chans for pretrained weights"

    strict = True
    classifier_name = default_cfg['classifier']
    if num_classes == 1000 and default_cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != default_cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = 0.1
        m.inplace = True
      
def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        # x = self.avg_pool(x)
        x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
    
class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out
    

class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

def seresnet50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet50-ce0d4300.pth')
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, drop_rate=0.2,
                 in_chans=3, inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000, global_pool='avg'):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(
                    in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.drop_rate = drop_rate
        self.num_features = 512 * block.expansion
        self.last_linear = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(
            self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        del self.last_linear
        if num_classes:
            self.last_linear = nn.Linear(self.num_features, num_classes)
        else:
            self.last_linear = None

    def forward_features(self, x, pool=True):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def logits(self, x):
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.logits(x)
        return x


class tiger_cnn5_512(nn.Module):
    def __init__(self, stride=1,smallscale=True):
        super(tiger_cnn5_512, self).__init__()

        model = seresnet50(pretrained=True)
        if stride == 1:
            if smallscale:
                model.layer2[0].downsample[0].stride = (1, 1)
                model.layer2[0].conv2.stride = (1, 1)
                   
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
            
        self.backbone = model

    def fix_params(self, is_training=True):
        for p in self.backbone.parameters():
            p.requires_grad = is_training


    def forward(self, x):
        # backbone
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)#B,512,56,56
         
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        
        return  [x] 


class tiger_cnn5_64(tiger_cnn5_512): 
    def __init__(self, stride=1,smallscale=True):
        super().__init__(stride,smallscale)
        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        _weight_init(self.last_conv)
        
    def  forward(self, x):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)#B,512,56,56
        
        x = self.last_conv(x)
        return [x] #B,64,56,56


##################################
# when used as: backbone original-smallscale=False
# backbone for comparison: default
# joint/joint_all: dve=True
# as side dve: dve=True, model_path=PRETRAINED_MODEL_PATH,use always dve_forward
# loading pretrained model on dve: dve=True, model_path=PRETRAINED_MODEL_PATH
##################################
class tiger_cnn5_v1(nn.Module):
    def __init__(self, class_num,stride=1,droprate=0.5,linear_num=512,circle=True,use_posture=True,dve=False,stackeddve=False,smallscale=True):
        # dve means dve feature is needed whether for method1 or joint training
        super(tiger_cnn5_v1, self).__init__()
        self.dve = dve
        self.stacked = stackeddve
        
        self.model = tiger_cnn5_512(stride,smallscale=smallscale)
        self.model.backbone.last_linear = nn.Sequential()
        
        if dve:
            num_dim = 3584 if stackeddve else 512
            self.last_conv = nn.Sequential(
            nn.Conv2d(num_dim, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
            _weight_init(self.last_conv)
        
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle, use_posture=use_posture)
        
    def forward(self, x, dve = False):
        #dve means output dve branch feature
        # backbone
        x = self.model.backbone.layer0(x)
        x = self.model.backbone.layer1(x)
        x_l2 = self.model.backbone.layer2(x)
        
        if dve and not self.stacked:
            assert self.dve == True
            dve_f = self.last_conv(x_l2)
            
        x_l3 = self.model.backbone.layer3(x_l2)
        x_l4 = self.model.backbone.layer4(x_l3)
        
        if dve and self.stacked:
            up_xl3 = F.interpolate(x_l3, size=x_l2.size()[2:], mode='bilinear', align_corners=True)
            up_x14 = F.interpolate(x_l4, size=x_l2.size()[2:], mode='bilinear', align_corners=True)
            x_stacked = torch.cat([x_l2,up_xl3,up_x14],dim=1)
            dve_f = self.last_conv(x_stacked)

        x = torch.mean(torch.mean(x_l4, dim=2), dim=2)
        x = self.classifier(x)

        if dve:
            return x, dve_f
        
        return x
    
    def dve_forward(self, x):
        
        assert self.dve==True
        x = self.model.backbone.layer0(x)
        x = self.model.backbone.layer1(x)
        x_l2 = self.model.backbone.layer2(x)
        if not self.stacked:
            dve_f = self.last_conv(x_l2)
        else:
            x_l3 = self.model.backbone.layer3(x_l2)
            x_l4 = self.model.backbone.layer4(x_l3)
            up_xl3 = F.interpolate(x_l3, size=x_l2.size()[2:], mode='bilinear', align_corners=True)
            up_x14 = F.interpolate(x_l4, size=x_l2.size()[2:], mode='bilinear', align_corners=True)
            x_stacked = torch.cat([x_l2,up_xl3,up_x14],dim=1)
            dve_f = self.last_conv(x_stacked)
        
        return dve_f

    def eval_forward(self,x):
        x = self.model.backbone.layer0(x)
        x = self.model.backbone.layer1(x)
        x = self.model.backbone.layer2(x)
        x = self.model.backbone.layer3(x)
        x = self.model.backbone.layer4(x) #[B,2048,28,28]
        return x
        
    
    def fix_params(self, is_training=True):
        for p in self.model.backbone.parameters():
            p.requires_grad = is_training



# Define the swin_base_patch4_window7_224 Model
# pytorch > 1.6
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, return_feature=True, linear_num=512, pretrained=True,dim=3,img_size=[224,224],use_posture=False):
        super(ft_net_swin, self).__init__()
        self.pretrained = pretrained
        if pretrained:
            model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        else:
            model_ft = SwinTransformer(img_size=img_size,patch_size=4,in_chans=dim,drop_path_rate=0.2,embed_dim=128,
                                       depths=[2, 2, 18, 2], num_heads=[4,8,16,32], window_size=7)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = return_feature, use_posture=use_posture)
        print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.model.forward_features(x)
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        if self.pretrained:
            x = self.avgpool(x.permute((0,2,1)))
            x = x.view(x.size(0), x.size(1))
        
        x = self.classifier(x)
        return x
    def fix_params(self, is_training=True):
        for p in self.model.parameters():
            p.requires_grad = is_training
    
    
class seresnet_dve_1(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=1, circle=True, linear_num=512,dve_dim=64,use_posture=True):
        super(seresnet_dve_1, self).__init__()
        model = seresnet50(pretrained=True)
        if stride == 1:
            model.layer2[0].downsample[0].stride = (1, 1)
            model.layer2[0].conv2.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        
        self.model = model
        self.model.last_linear = nn.Sequential()
        
        self.circle = circle
        self.merge = nn.Sequential(
            nn.Conv2d(
                        in_channels=512+dve_dim,
                        out_channels=512,
                        kernel_size=3,
                        stride=2,
                        padding=1
            ),#B,512,28,28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.merge.apply(weights_init_kaiming)
           
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle, use_posture=use_posture)
   
        
    def forward(self, x, f_dev):
        
        x = self.model.layer0(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)#B,512,56,56
        
        x = torch.concat([x,f_dev],dim=1) #B,512+64,56,56
        x = self.merge(x) #B,512,28,28
        
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.classifier(x)
     
        return x
    
    def get_base_params(self):
        return self.model.parameters()

    def fix_params(self, is_training=True):
        for p in self.get_base_params():
            p.requires_grad = is_training

#working better than 1 & 1_2
class seresnet_dve_1_5(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=1, circle=True, linear_num=512,dve_dim=64,use_posture=True):
        super(seresnet_dve_1_5, self).__init__()
        model = seresnet50(pretrained=True)
        if stride == 1:
            model.layer2[0].downsample[0].stride = (1, 1)
            model.layer2[0].conv2.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        
        self.model = model
        self.model.last_linear = nn.Sequential()
        
        self.circle = circle
        self.merge = nn.Sequential(
            nn.Conv2d(
                        in_channels=512+dve_dim,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        padding=0
            ),#B,512,56,56
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.merge.apply(weights_init_kaiming)
           
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle, use_posture=use_posture)
   
        
    def forward(self, x, f_dev):
        
        x = self.model.layer0(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)#B,512,56,56
        
        x = torch.concat([x,f_dev],dim=1) #B,512+64,56,56

        x = self.merge(x) #B,512,56,56
        
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.classifier(x)
     
        return x
    
    def get_base_params(self):
        return self.model.parameters()

    def fix_params(self, is_training=True):
        for p in self.get_base_params():
            p.requires_grad = is_training


class seresnet_dve_2(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=1, circle=True, linear_num=512,dve_dim=64,use_posture=True):
        super(seresnet_dve_2, self).__init__()
        model = seresnet50(pretrained=True)
        if stride == 1:
            model.layer2[0].downsample[0].stride = (1, 1)
            model.layer2[0].conv2.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        
        self.model = model
        self.model.last_linear = nn.Sequential()
        
        self.circle = circle
        self.pre_merge = nn.Sequential(
            nn.Conv2d(
                        in_channels=512,
                        out_channels=2*dve_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0
            ),#B,128,56,56
            nn.BatchNorm2d(2*dve_dim),
            nn.ReLU(inplace=True)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(
                        in_channels=3*dve_dim,
                        out_channels=512,
                        kernel_size=1,
                        stride=1,
                        padding=0,
            ),#B,64*3,56,56
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pre_merge.apply(weights_init_kaiming)
        self.merge.apply(weights_init_kaiming)
           
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle, use_posture=use_posture)
   
        
    def forward(self, x, f_dev):
        
        x = self.model.layer0(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)#B,512,56,56
        
        x = self.pre_merge(x) #B,128,56,56
        x = torch.concat([x,f_dev],dim=1) #B,128+64,56,56
        x = self.merge(x) #B,512,56,56
        
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.classifier(x)
     
        return x
    
    def get_base_params(self):
        return self.model.parameters()
    
    def fix_params(self, is_training=True):
        for p in self.get_base_params():
            p.requires_grad = is_training




class ft_net_64(nn.Module):

    def __init__(self, stride=2, ibn=False):
        super(ft_net_64, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer2[0].downsample[0].stride = (1,1)
            model_ft.layer2[0].conv2.stride = (1,1)
        self.model = model_ft
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        _weight_init(self.last_conv)
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x) #([8, 512, 56, 56])
        x = self.last_conv(x) #([8, 64, 56, 56])
        
        return [x]
    
if __name__ == '__main__':
    x = Variable(torch.randn(2, 3, 224, 224))
    f_dve = Variable(torch.randn(2, 64, 56, 56))
    model = tiger_cnn5_v1(101,dve=True,stackeddve=True)
    #model = seresnet_dve_2(101)
    out = model(x,True)
    print(out[0])