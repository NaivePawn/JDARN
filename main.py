import os
import config
import torch
from train import train_src, train_tgt
from models import Discriminator, VAE, VAE_Classifier
from utils import get_data_loader, init_model, init_random_seed, evaluate

def main():

    init_random_seed(config.manual_seed)

    # load data
    src_data_loader = get_data_loader(config.src_dataset)
    src_data_loader_eval = get_data_loader(config.src_dataset, train=False)
    tgt_data_loader = get_data_loader(config.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(config.tgt_dataset, train=False)

    # init models
    src_encoder = init_model(net=VAE(), restore=config.src_encoder_restore)
    src_classifier = init_model(net=VAE_Classifier(), restore=config.src_classifier_restore)
    tgt_encoder = init_model(net=VAE(), restore=config.tgt_encoder_restore)
    tgt_classifier = init_model(net=VAE_Classifier(), restore=config.tgt_classifier_restore)
    discriminator_one = init_model(net=Discriminator(), restore=config.discriminator_one_restore)
    discriminator_two = init_model(net=Discriminator(), restore=config.discriminator_two_restore)

    # init optimizers
    src_ignored_params = list(map(id, src_encoder.encoder.parameters()))
    src_base_params = filter(lambda p: id(p) not in src_ignored_params, src_encoder.parameters())
    optimizer_src = torch.optim.SGD([
        {'params': src_encoder.encoder.parameters(), 'lr': 1e-4},
        {'params': src_base_params, 'lr': 1e-3},
        {'params': src_classifier.parameters(), 'lr': 1e-3},
    ], momentum=0.9)
    optimizer_tgt = torch.optim.SGD([
        {'params': tgt_encoder.parameters(), 'lr': 1e-5},
        {'params': tgt_classifier.parameters(), 'lr': 1e-5},
    ], momentum=0.9)
    optimizer_dis_one = torch.optim.SGD(discriminator_one.parameters(), lr=1e-3, momentum=0.9)
    optimizer_dis_two = torch.optim.SGD(discriminator_two.parameters(), lr=1e-3, momentum=0.9)

    if not (src_encoder.restored and src_classifier.restored):
        print("=== Training classifier for source domain ===")
        src_encoder, src_classifier = train_src(src_encoder, src_classifier, optimizer_src,
                                                src_data_loader, src_data_loader_eval, tgt_data_loader_eval)

    print("=== Evaluating classifier for source domain ===")
    evaluate(src_encoder, src_classifier, src_data_loader_eval)
    evaluate(src_encoder, src_classifier, tgt_data_loader_eval)

    if not (tgt_encoder.restored and tgt_classifier.restored):
        print("=== init weights of target encoder and target classifier===")
        tgt_encoder.load_state_dict(src_encoder.state_dict())
        tgt_classifier.load_state_dict(src_classifier.state_dict())

    if not (tgt_encoder.restored and discriminator_one.restored):
        print("=== Training encoder for target domain ===")
        tgt_encoder, tgt_classifier = train_tgt(src_encoder, tgt_encoder, tgt_classifier, discriminator_one, discriminator_two,
                                                optimizer_tgt, optimizer_dis_one, optimizer_dis_two,
                                                src_data_loader, tgt_data_loader, tgt_data_loader_eval)

    print("=== Evaluating classifier for target domain ===")
    print("--- source only ---")
    evaluate(src_encoder, src_classifier, tgt_data_loader_eval)
    print("--- domain adaption ---")
    evaluate(tgt_encoder, tgt_classifier, tgt_data_loader_eval)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()