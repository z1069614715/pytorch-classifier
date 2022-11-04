import torch
import torch.nn as nn
import model as models
from thop import clever_format, profile

def select_model(name, num_classes, input_shape, channels, pretrained=False):
    if 'shufflenet_v2' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    elif 'mobilenetv3' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith('resnet'):
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif name == 'wide_resnet50':
        model = models.wide_resnet50_2(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif name == 'wide_resnet101':
        model = models.wide_resnet101_2(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif name == 'resnext50':
        model = models.resnext50_32x4d(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif name == 'resnext101':
        model = models.resnext101_32x8d(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif 'resnest' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif 'densenet' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier.in_features, num_classes)
        )
    elif 'vgg' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)
    elif 'efficientnet' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)
    elif name.startswith('mnasnet'):
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'vovnet' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
        )
    elif 'convnext' in name:
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=model.head.in_features, out_features=num_classes)
        )
    elif name == 'ghostnet':
        model = models.ghostnet(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)
    elif 'RepVGG' in name:
        model = models.get_RepVGG_func_by_name(name, pretrained)
        model.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.linear.in_features, num_classes)
        )
    elif 'sequencer2d' in name:
        model = eval('models.{}(pretrained={}, in_chans={}, img_size={})'.format(name, pretrained, channels, input_shape[0]))
        model.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.head.in_features, num_classes)
        )
    elif name.startswith('csp') or name.startswith('darknet') or name.startswith('cs3'):
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.head.in_features, num_classes)
        )
    elif name.startswith('dpn'):
        model = eval('models.{}(pretrained={})'.format(name, pretrained))
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
        )
    else:
        raise 'Unsupported Model Name.'

    # 计算参数量和flops
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, channels, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    # --------------------------------------------------------#
    # flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Select Model: {}'.format(name))
    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    model.name = name

    return model

if __name__ == '__main__':
    model = select_model(name='shufflenet_v2_x0_5', num_classes=5, channels=3, input_shape=(224, 224))
    model = select_model(name='shufflenet_v2_x1_0', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='mobilenetv2', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='mobilenetv3_large', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='mobilenetv3_small', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnet18', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnet34', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnet50', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnet101', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='wide_resnet50', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='wide_resnet101', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnext50', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnext101', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnest50', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnest101', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnest200', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='resnest269', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='densenet121', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='densenet161', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='densenet169', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='densenet201', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg11', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg11_bn', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg13', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg13_bn', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg16', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg16_bn', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg19', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vgg19_bn', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b0', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b1', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b2', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b3', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b4', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b5', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b6', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_b7', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_v2_s', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_v2_m', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='efficientnet_v2_l', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='mnasnet', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vovnet39', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='vovnet57', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='convnext_tiny', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='convnext_small', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='convnext_base', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='convnext_large', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='convnext_xlarge', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='ghostnet', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='RepVGG-A0', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='sequencer2d_s', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='cspresnet50', num_classes=5, channels=3, input_shape=(224, 224))
    # model = select_model(name='dpn98', num_classes=5, channels=3, input_shape=(224, 224), pretrained=True)
    pass