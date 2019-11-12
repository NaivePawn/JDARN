import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from utils import store_feat_maps

class Self_Attn(nn.Module):

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        B, C, W, H = x.size()
        Q = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # [B, N, C]
        K = self.key_conv(x).view(B, -1, W * H)  # [B, C, N]
        energy = torch.bmm(Q, K)
        attention = self.softmax(energy)  # [B, N, N]
        V = self.value_conv(x).view(B, -1, W * H)  # [B, C, N]

        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        return out

class LeNet_Conv(nn.Module):
    def __init__(self):
        super(LeNet_Conv, self).__init__()
        self.restored = False

        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(True),
            Self_Attn(20),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(True),
            Self_Attn(50),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

        self.pool_locs = OrderedDict()

        self.init_weights()


    def init_weights(self):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, locations = layer(x)
            else:
                x = layer(x)
        return x

class LeNet_Deconv(nn.Module):
    def __init__(self):
        super(LeNet_Deconv, self).__init__()

        self.features = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(50, 20, kernel_size=5),
            nn.ReLU(True),
            Self_Attn(20),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(20, 1, kernel_size=5)
        )

        self.unpool2pool_indices = {4: 3, 0: 7}

        self.init_weights()

    def init_weights(self):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.ConvTranspose2d):
                self.features[idx].weight.data.normal_(0.0, 0.02)

    def forward(self, x, pool_locations):
        for idx in range(len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx](x, pool_locations[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x

class LeNet_VAE(nn.Module):
    def __init__(self):
        super(LeNet_VAE, self).__init__()

        self.restored = False

        self.encoder = LeNet_Conv()
        store_feat_maps(self.encoder)
        self.mean = nn.Linear(50 * 4 * 4, 500)
        self.std = nn.Linear(50 * 4 * 4, 500)
        self.fc_decoder = nn.Linear(500, 50 * 4 * 4)
        self.decoder = LeNet_Deconv()

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        epsilon = Variable(torch.FloatTensor(var.size()).normal_())
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return epsilon.mul(var).add_(mean)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        mean = self.mean(x)
        std = self.std(x)
        code = self.sampler(mean, std)

        out = self.fc_decoder(code)
        out = out.view(out.size(0), 50, 4, 4)
        out = self.decoder(out, self.encoder.pool_locs)
        return out, code, mean, std

class LeNet_VAE_Classifier(nn.Module):
    def __init__(self):
        super(LeNet_VAE_Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x