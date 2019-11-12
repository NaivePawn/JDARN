import os
import time
import torch
import torch.nn as nn
import config
from utils import make_variable, VAELoss, evaluate

def train_src(src_encoder, src_classifier, optimizer_src, src_data_loader, src_data_loader_eval, tgt_data_loader_eval):

    since = time.time()

    CEL = nn.CrossEntropyLoss()
    for epoch in range(config.num_epochs_pre):
        src_encoder.train()
        src_classifier.train()

        for step, (images, labels) in enumerate(src_data_loader):
            images = make_variable(images)
            labels = make_variable(labels)

            out, code, mean, std = src_encoder(images)
            preds = src_classifier(code)

            vae_loss = VAELoss(out, images, mean, std)
            classifier_loss = CEL(preds, labels)
            total_loss = classifier_loss + 0.1 * vae_loss
            optimizer_src.zero_grad()
            total_loss.backward()
            optimizer_src.step()

            if ((step + 1) % config.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: vae_loss={:.4f} classifier_loss={:.4f} total_loss={:.4f}".format(
                    epoch + 1, config.num_epochs_pre, step + 1, len(src_data_loader),
                    vae_loss.item(), classifier_loss.item(), total_loss.item()))

        if ((epoch + 1) % config.log_eval == 0):
            print("Epoch {}, evaluate...".format(epoch + 1))
            evaluate(src_encoder, src_classifier, src_data_loader_eval)
            evaluate(src_encoder, src_classifier, tgt_data_loader_eval)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(src_encoder.state_dict(), os.path.join(config.model_root, "source_encoder.pt"))
    torch.save(src_classifier.state_dict(), os.path.join(config.model_root, "source_classifier.pt"))

    return src_encoder, src_classifier