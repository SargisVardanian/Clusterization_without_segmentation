from training_utils import *
import os
import tensorflow as tf
from TransU.TransUnet import *

config = {}
config["image_size"] = 144
config["num_layers"] = 12
config["hidden_dim"] = 64
config["mlp_dim"] = 144
config["num_heads"] = 12
config["dropout_rate"] = 0.2
config["num_patches"] = 144
config["patch_size"] = 12
config["num_channels"] = 3
# class_names = os.listdir(data_dir)
# config["num_classes"] = len(class_names)

checkpoint_path = 'save_checkpoints\\checkpoints'
checkpoint_path_combined = 'checkpoints_mc\\checkpoints_mc'

try:
    gen_mask = tf.keras.models.load_model(checkpoint_path_)
    discriminator = tf.keras.models.load_model(checkpoint_path_)
    # mask_model = tf.keras.models.load_model(checkpoint_path_combined)
    print('Загружены веса')
except:
    gen_mask = TransUnet(config)
    discriminator = Discriminator(config)
    # model.load_weights(checkpoint_path)
    print('Загрузка не вышла')

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
loss_fn_noise = tf.keras.losses.SparseCategoricalCrossentropy()
loss_MAE = tf.keras.losses.MeanAbsoluteError()
metric = ['accuracy']

gen_mask_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_mask_optimizer,
         discriminator_optimizer=discriminator_optimizer,
         generator=gen_mask,
         discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)
