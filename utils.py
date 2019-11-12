import os
import math
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from functools import partial
import contextlib
import torch.nn.functional as F
import config
from preprocess import get_mnist, get_usps, get_svhn, load_images, load


def make_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, train=True):
    print("use dataset: {}".format(name))
    if name == "MNIST":
        return get_mnist(train)
    elif name == "USPS":
        return get_usps(train)
    elif name == "SVHN":
        return get_svhn(train)
    elif name == "A":
        return load_images('data/office/', 'amazon', batch_size=config.batch_size, is_train=train)
    elif name == "W":
        return load_images('data/office/', 'webcam', batch_size=config.batch_size, is_train=train)
    elif name == "D":
        return load_images('data/office/', 'dslr', batch_size=config.batch_size, is_train=train)
    elif name == "B":
        return load('data/image-clef/b_list.txt', batch_size=config.batch_size, is_train=train)
    elif name == "C":
        return load('data/image-clef/c_list.txt', batch_size=config.batch_size, is_train=train)
    elif name == "I":
        return load('data/image-clef/i_list.txt', batch_size=config.batch_size, is_train=train)
    elif name == "P":
        return load('data/image-clef/p_list.txt', batch_size=config.batch_size, is_train=train)


def init_model(net, restore):
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net = net.cuda()

    return net


def store_feat_maps(model):
    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool2d):
            model.pool_locs[key] = output[1]

    for idx, layer in enumerate(model._modules.get('features')):
        layer.register_forward_hook(partial(hook, key=idx))


def adaptation_factor(x):
    if x >= 1.0:
        return 1.0
    den = 1.0 + math.exp(-10 * x)
    lamb = 2.0 / den - 1.0
    return lamb


def inv_lr_scheduler(param_lr, optimizer, p, gamma=10, power=0.75, init_lr=0.0001):
    lr = init_lr * (1 + gamma * p) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


def VAELoss(out, target, mean, std):
    MSE = nn.MSELoss()
    rec_loss = MSE(out, target)
    latent_loss = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp())
    return rec_loss + 0.0001 * latent_loss


def EntropyLoss(input):
    entropy = -(torch.sum(input * torch.log(input)))
    return entropy / float(input.size(0))


def WeightedEntropyLoss(input, input_domain):
    entropy = -(torch.sum(input * torch.log(input), dim=1)).view(input.size(0), -1)
    weights = 1 - (torch.sum(input_domain * torch.log(input_domain), dim=1)).view(input_domain.size(0), -1)
    weights = weights.detach()
    entropy = torch.sum(weights * entropy)
    return entropy / float(input.size(0))


def evaluate(encoder, classifier, data_loader):
    encoder.eval()
    classifier.eval()
    CEL = nn.CrossEntropyLoss()
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for (images, labels) in data_loader:
            images = make_variable(images)
            labels = make_variable(labels)
            _, code, _, _ = encoder(images)
            preds = classifier(code)
            loss += CEL(preds, labels).item()
            pred_cls = torch.argmax(preds, dim=1)
            accuracy += (pred_cls == labels).sum().item()
    loss /= len(data_loader)
    accuracy /= len(data_loader.dataset)
    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, accuracy))


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def cross_entropy(trg_pred, pred):
    return torch.sum(- trg_pred * torch.log(pred)) / trg_pred.size(0)

class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, encoder, classifier, x):
        encoder.zero_grad()
        classifier.zero_grad()
        with torch.no_grad():
            _, code, _, _ = encoder(x)
            pred = F.softmax(classifier(code), dim=1)

        # prepare random unit tensor
        # d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = torch.normal(mean=torch.zeros_like(x), std=1).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(encoder):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                _, code_hat, _, _ = encoder(x + self.xi * d)
                pred_hat = classifier(code_hat)
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = cross_entropy(pred, p_hat)
                adv_distance.backward()
                d = _l2_normalize(d.grad)

                encoder.zero_grad()
                classifier.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            _, code_hat, _, _ = encoder(x + r_adv)
            pred_hat = classifier(code_hat)
            p_hat = F.softmax(pred_hat, dim=1)
            lds = cross_entropy(pred, p_hat)

        return lds