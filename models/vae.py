import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F


class ResNet_Conv(nn.Module):
    def __init__(self):
        super(ResNet_Conv, self).__init__()

        model = models.resnet50(pretrained=True)
        self.conv = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet_Deconv(nn.Module):
    def __init__(self):
        super(ResNet_Deconv, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(3, momentum=0.01)
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.deconv(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.restored = False

        self.encoder = ResNet_Conv()

        self.fc_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.ReLU(True)
        )

        self.mean = nn.Linear(512, 256)
        self.std = nn.Linear(512, 256)

        self.fc_decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.01),
            nn.ReLU(True)
        )

        self.decoder = ResNet_Deconv()


    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        epsilon = Variable(torch.FloatTensor(var.size()).normal_())
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return epsilon.mul(var).add_(mean)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)

        mean = self.mean(x)
        std = self.std(x)

        code = self.sampler(mean, std)
        out = self.fc_decoder(code)
        out = out.view(out.size(0), 64, 4, 4)
        out = self.decoder(out)

        return out, code, mean, std

class VAE_Classifier(nn.Module):
    def __init__(self):
        super(VAE_Classifier, self).__init__()

        self.restored = False

        self.fc = nn.Sequential(
            nn.Linear(256, 31)
        )

    def forward(self, x):
        x = F.dropout(F.relu(x), training=self.training)
        x = self.fc(x)
        return x