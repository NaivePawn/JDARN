import torch
import torch.nn as nn
import torchvision
import math
from torch.autograd import Variable
import torch.nn.functional as F

class ResNet_Conv(nn.Module):
    def __init__(self):
        super(ResNet_Conv, self).__init__()

        pre_model = torchvision.models.resnet50(pretrained=True)
        self.conv1 = pre_model.conv1
        self.bn1 = pre_model.bn1
        self.relu = pre_model.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        self.layer1 = pre_model.layer1
        self.layer2 = pre_model.layer2
        self.layer3 = pre_model.layer3
        self.layer4 = pre_model.layer4
        self.avgpool = pre_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, indice = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x, indice

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        if stride == 1:
            outputpadding = 0
        else:
            outputpadding = 1

        self.deconv1 = nn.ConvTranspose2d(planes * 4, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.deconv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, padding=1, output_padding=outputpadding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.deconv3 = nn.ConvTranspose2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.deconv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Deconv(nn.Module):

    def __init__(self, block, layers):
        super(ResNet_Deconv, self).__init__()

        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.maxunpool = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        if stride == 1:
            outputpadding = 0
        else:
            outputpadding = 1

        downsample = nn.Sequential(
            nn.ConvTranspose2d(planes * block.expansion, inplanes, kernel_size=1, stride=stride, output_padding=outputpadding, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        layers.append(block(inplanes, planes, stride, downsample))

        return nn.Sequential(*layers)

    def forward(self, x, indice):

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.maxunpool(x, indice)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.deconv1(x)

        return x

class ResNet_VAE(nn.Module):
    def __init__(self):
        super(ResNet_VAE, self).__init__()
        self.restored = False

        self.encoder = ResNet_Conv()

        self.mean = nn.Linear(2048, 256)
        self.std = nn.Linear(2048, 256)

        self.fc_decoder = nn.Linear(256, 2048 * 7 * 7)
        self.decoder = ResNet_Deconv(Bottleneck, [3, 4, 6, 3])

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        epsilon = Variable(torch.FloatTensor(var.size()).normal_())
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return epsilon.mul(var).add_(mean)

    def forward(self, x):
        x, indice = self.encoder(x)
        x = x.view(x.size(0), -1)

        mean = self.mean(x)
        std = self.std(x)
        code = self.sampler(mean, std)

        out = self.fc_decoder(code)
        out = out.view(out.size(0), 2048, 7, 7)
        out = self.decoder(out, indice)

        return out, code, mean, std

class ResNet_VAE_Classifier(nn.Module):
    def __init__(self):
        super(ResNet_VAE_Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(256, 31)
        )

    def forward(self, x):
        x = F.dropout(F.relu(x), training=self.training)
        x = self.classifier(x)
        return x



