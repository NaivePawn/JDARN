# params for dataset and data loader
data_root = "data"
src_dataset = "D"
tgt_dataset = "W"
batch_size = 16

# params for models
model_root = "snapshots"
src_encoder_restore = "snapshots/source_encoder.pt"
src_classifier_restore = "snapshots/source_classifier.pt"
tgt_encoder_restore = "snapshots/target_encoder.pt"
tgt_classifier_restore = "snapshots/target_classifier.pt"
discriminator_one_restore = "snapshots/discriminator_one.pt"
discriminator_two_restore = "snapshots/discriminator_two.pt"

# params for training
num_epochs_pre = 200
log_step_pre = 10
num_epochs_apt = 50000
log_step_apt = 20
log_eval = 20
manual_seed = 2000