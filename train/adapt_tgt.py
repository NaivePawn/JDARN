import os
import time
import math
import torch
import torch.nn.functional as F
from torch import nn
import config
from utils import make_variable, VAELoss, EntropyLoss, VATLoss, evaluate


def train_tgt(src_encoder, tgt_encoder, tgt_classifier, discriminator_one, discriminator_two,
              optimizer_tgt, optimizer_dis_one, optimizer_dis_two,
              src_dataloader, tgt_dataloader, tgt_data_loader_eval):

    since = time.time()

    CEL = nn.CrossEntropyLoss()
    VATL = VATLoss(eps=4)

    len_data_loader = min(len(src_dataloader), len(tgt_dataloader))
    for epoch in range(config.num_epochs_apt):
        tgt_encoder.train()
        tgt_classifier.train()
        discriminator_one.train()
        discriminator_two.train()

        for step, ((src_images, src_labels), (tgt_images, _)) in enumerate(zip(src_dataloader, tgt_dataloader)):
            #################### train the discriminator ####################
            src_images = make_variable(src_images)
            tgt_images = make_variable(tgt_images)

            src_out, src_code, _, _ = src_encoder(src_images)
            tgt_out, tgt_code, tgt_mean, tgt_std = tgt_encoder(tgt_images)

            src_labels = make_variable(src_labels).detach()
            tgt_labels = torch.argmax(tgt_classifier(tgt_code), dim=1).detach()

            src_domain_pred = discriminator_one(src_code.detach())
            tgt_domain_pred = discriminator_one(tgt_code.detach())
            domain_loss_one = CEL(src_domain_pred, src_labels) + CEL(tgt_domain_pred, make_variable(torch.ones(tgt_images.size(0)).long() * 31))
            optimizer_dis_one.zero_grad()
            domain_loss_one.backward()
            optimizer_dis_one.step()

            src_domain_pred = discriminator_two(src_code.detach())
            tgt_domain_pred = discriminator_two(tgt_code.detach())
            domain_loss_two = CEL(src_domain_pred, make_variable(torch.ones(src_images.size(0)).long() * 31)) + CEL(tgt_domain_pred, tgt_labels)
            optimizer_dis_two.zero_grad()
            domain_loss_two.backward()
            optimizer_dis_two.step()

            ##################### train the target vae ####################
            # VAE Loss
            vae_loss = VAELoss(tgt_out, tgt_images, tgt_mean, tgt_std)

            # tgt_encoder Loss
            tgt_domain_pred_one = discriminator_one(tgt_code)
            tgt_domain_pred_two = discriminator_two(tgt_code)
            tgt_loss = CEL(tgt_domain_pred_one, tgt_labels) + CEL(tgt_domain_pred_two, make_variable(torch.ones(tgt_images.size(0)).long() * 31))

            # Entropy Loss
            tgt_pred_softmax = F.softmax(tgt_classifier(tgt_code), dim=1)
            entropy_loss = EntropyLoss(tgt_pred_softmax)

            # VAT Loss
            vat_loss = VATL(tgt_encoder, tgt_classifier, tgt_images)

            total_loss = tgt_loss + 0.1 * vae_loss + 0.1 * (entropy_loss + vat_loss)
            optimizer_dis_one.zero_grad()
            optimizer_dis_two.zero_grad()
            optimizer_tgt.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer_tgt.step()

            if ((step + 1) % config.log_step_apt == 0):
                print("Epoch [{}/{}] Step [{}/{}]: domain_loss_one={:.4f} domain_loss_two={:.4f} "
                      "vae_loss={:.4f} tgt_loss={:.4f} entropy_loss={:.4f} vat_loss={:.4f} total_loss={:.4f}".format(
                    epoch + 1, config.num_epochs_apt, step + 1, len_data_loader,
                    domain_loss_one.item(), domain_loss_two.item(),
                    vae_loss.item(), tgt_loss.item(), entropy_loss.item(), vat_loss.item(), total_loss.item()))

        if ((epoch + 1) % config.log_eval == 0):
            print("Epoch {}, evaluate...".format(epoch + 1))
            evaluate(tgt_encoder, tgt_classifier, tgt_data_loader_eval)

        if ((epoch + 1) % 2000 == 0):
            torch.save(discriminator_one.state_dict(), os.path.join(config.model_root, "discriminator_one.pt"))
            torch.save(discriminator_two.state_dict(), os.path.join(config.model_root, "discriminator_two.pt"))
            torch.save(tgt_encoder.state_dict(), os.path.join(config.model_root, "target_encoder.pt"))
            torch.save(tgt_classifier.state_dict(), os.path.join(config.model_root, "target_classifier.pt"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(discriminator_one.state_dict(), os.path.join(config.model_root, "discriminator_one.pt"))
    torch.save(discriminator_two.state_dict(), os.path.join(config.model_root, "discriminator_two.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(config.model_root, "target_encoder.pt"))
    torch.save(tgt_classifier.state_dict(), os.path.join(config.model_root, "target_classifier.pt"))

    return tgt_encoder, tgt_classifier