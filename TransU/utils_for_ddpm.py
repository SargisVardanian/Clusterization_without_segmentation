import random
import tensorflow as tf
import os
import typing
import warnings
from urllib import request
from http import client
import numpy as np
import keras.layers as L
from keras.models import Model
from TransU.Under_utils_for_ddpm import *


def Encoder_TransUnet(cf, inputs, CLass):
    """ Patch + Position Embeddings """
    patch = Patches(12)
    print('input', inputs.shape, CLass.shape)
    xx = UnPatches(inputs, cf)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1, dtype=tf.float32)
    positions = tf.expand_dims(positions, axis=0)
    positions = tf.tile(positions, [tf.shape(inputs)[0], 1])
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["num_patches"]*3, trainable=False)(positions)

    inputs = L.Add()([inputs, pos_embed])
    # print('input', x.shape, CLass.shape)
    x = inputs

    for i in range(3):
        x0 = transformer_CrossAT(x, CLass, cf)
        x = L.Add()([x, x0])
        print('x', x.shape)

    skip_connections = []
    """ Transformer Encoder """
    print('Class', CLass.shape, 'xxx', x.shape)
    x = UnPatches(x, cf)
    # inputs = x
    for i in range(1, 4):
        # x = transformer_SelfAt(x, cf)
        # x = conv_block(x, kernel_size=3, num_filters=48, dropout_rate=cf["dropout_rate"])
        x = conv_block(x, kernel_size=12, num_filters=48)
        # x = conv_block(x, kernel_size=6, num_filters=24, dropout_rate=cf["dropout_rate"])
        x = conv_block(x, kernel_size=12, num_filters=24)
        x = conv_block(x, kernel_size=12, num_filters=6)
        print(f'x{i}', x.shape)
        skip_connections.append(x)
    x = patch(x)

    for i in reversed(range(len(skip_connections))):
        print('i', i, len(skip_connections))
        if i == len(skip_connections):
            z0 = L.Concatenate()([skip_connections[i], skip_connections[i]])
            z0 = conv_block(z0, kernel_size=12, num_filters=24)
            # z0 = conv_block(z0, kernel_size=12, num_filters=6)
            skip_connections[i] = conv_block(z0, kernel_size=12, num_filters=3)
            # skip_connections[i] = transformer_CrossAT(skip_connections[i], skip_connections[i], cf)
        else:
            z0 = L.Concatenate()([skip_connections[i-1], skip_connections[i]])
            z0 = conv_block(z0, kernel_size=12, num_filters=24)
            # z0 = conv_block(z0, kernel_size=12, num_filters=6)
            skip_connections[i] = conv_block(z0, kernel_size=12, num_filters=3)
            # skip_connections[i] = transformer_CrossAT(skip_connections[i - 1], skip_connections[i], cf)

    x = skip_connections[0]
    # skip_connections[0] = patch(skip_connections[0])
    # x = transformer_CrossAT(skip_connections[0], inputs, cf)

    x = tf.reshape(x, (-1, 144, 144, 3))
    x0 = x[:, :, :, 0]
    x1 = x[:, :, :, 1]
    x2 = x[:, :, :, 2]

    print(f'x0{x0.shape}, x1{x1.shape}, x2{x2.shape}')

    mask = L.Add(name='mask')([x0, x1, x2])
    print('mask', mask.shape)
    mask_z0 = L.Add(name='mask_z0')([x])

    # expanded_mask = tf.expand_dims(mask, axis=-1)
    # mask_z0 = L.Multiply(name='mask_z0')([expanded_mask, xx])
    print(f'mask {mask.shape}, mask_z0 {mask_z0.shape}')
    return mask, mask_z0


# z = UnPatches(z0, cf)
# for i in range(6):
#     if i == 0:
#         skip_connections[i] = transformer_CrossAT(z0, skip_connections[i], cf)
#         skip_connections[i] = L.GroupNormalization(groups=36)(skip_connections[i])
#         z0 = z
#     else:
#         skip_connections[i] = transformer_CrossAT(skip_connections[i - 1], skip_connections[i], cf)
#         skip_connections[i] = L.GroupNormalization(groups=36)(skip_connections[i])
#     s = UnPatches(skip_connections[i], cf)
#     z0 = L.Concatenate()([z0, s])
#     z0 = conv_block(z0, kernel_size=12, num_filters=12)
#     z0 = conv_block(z0, kernel_size=12, num_filters=6)
#     print('z0', z0.shape)

# x = UnPatches(x, cf)
#         x = deconv_block(x, kernel_size=12, num_filters=18)
#         x = conv_block(x, kernel_size=12, num_filters=6)
#         x = patch(x)

def Decoder_TransUnet(cf, mask, z0, skip_connections):
    patch = Patches(12)
    l = len(skip_connections)
    for i in reversed(range(l)):
        print('i', i , l)
        if i == l:
            skip_connections[i] = transformer_CrossAT(skip_connections[i], skip_connections[i], cf)
        else:
            skip_connections[i] = transformer_CrossAT(skip_connections[i-1], skip_connections[i], cf)

    # for i in range(l):
    #     skip_connections[i] = UnPatches(skip_connections[i], cf)
    #     print(f'z_{i}', skip_connections[i].shape)

    # for i in reversed(range(1, l)):
        # print('i', i , l)
        # if i == l:
        #     skip_connections[i] = transformer_CrossAT(z0, skip_connections[i], cf)
        #     skip_connections[i] = L.GroupNormalization(groups=36)(skip_connections[i])
        # else:
        #     skip_connections[i] = transformer_CrossAT(skip_connections[i-1], skip_connections[i], cf)
        #     skip_connections[i] = L.GroupNormalization(groups=36)(skip_connections[i])
        # s_old = deconv_block(skip_connections[i], kernel_size=6, num_filters=36)
        # s_old = conv_block(s_old, kernel_size=6, num_filters=12)
        #
        # s_new = deconv_block(skip_connections[i-1], kernel_size=6, num_filters=36)
        # s_new = conv_block(s_new, kernel_size=6, num_filters=12)
        #
        # z = L.Concatenate()([s_old, s_new])
        # z = deconv_block(z, kernel_size=6, num_filters=18)
        # z = conv_block(z, kernel_size=12, num_filters=12)
        # skip_connections[i] = z
        # print(f'z_{i - 1}', skip_connections[l - 1 - i].shape)

    z0 = transformer_CrossAT(z0, skip_connections[0], cf)
    z0 = UnPatches(z0, cf)
    # z0 = L.Concatenate()([z0, skip_connections[0]])
    # z0 = deconv_block(z0, kernel_size=12, num_filters=18)
    # z0 = deconv_block(z0, kernel_size=12, num_filters=18)
    # z0 = conv_block(z0, kernel_size=12, num_filters=6)

    print('z00', z0.shape)
    # x = UnPatches(z0, cf, name='noise_output')
    # print('x1', x.shape)
    return z0



if __name__ == "__main__":
    config = {}
    config["image_size"] = 144
    config["num_layers"] = 7
    config["hidden_dim"] = 64
    config["mlp_dim"] = 64
    config["num_heads"] = 3
    config["dropout_rate"] = 0.1
    config["num_patches"] = 144
    config["patch_size"] = 12
    config["num_channels"] = 3

# model = Trans_block(9)
# print(model.summary())

def pre_trained_TransUnet(input_tensor):
    checkpoint_path = 'C:\\Users\\User\\PycharmProjects\\Mythical_Animals\\Diffusion_to_save_checkpoints\\Diffusion_checkpoint'
    model = tf.keras.models.load_model(checkpoint_path)
    # print(model.summary())
    z0_layer_output = model.get_layer('input_1').output
    z12_layer_output = model.get_layer('reshape_4').output
    z9_layer_output = model.get_layer('reshape_3').output
    z6_layer_output = model.get_layer('reshape_2').output
    z3_layer_output = model.get_layer('reshape_1').output

    z0_model = tf.keras.Model(inputs=model.input, outputs=z0_layer_output)
    z3_model = tf.keras.Model(inputs=model.input, outputs=z3_layer_output)
    z6_model = tf.keras.Model(inputs=model.input, outputs=z6_layer_output)
    z9_model = tf.keras.Model(inputs=model.input, outputs=z9_layer_output)
    z12_model = tf.keras.Model(inputs=model.input, outputs=z12_layer_output)

    patch_embed = tf.keras.layers.Dense(64)(input_tensor)
    positions = tf.range(start=0, limit=16, delta=1)  ## (256,)
    pos_embed = tf.keras.layers.Embedding(input_dim=16, output_dim=64)(positions)  ## (256, 768)
    x = patch_embed + pos_embed

    z0_output = z0_model(x)
    z3_output = z3_model(x)
    z6_output = z6_model(x)
    z9_output = z9_model(x)
    z12_output = z12_model(x)
    # print('z0_output, z3_output, z6_output, z9_output, z12_output', z0_output, z3_output, z6_output, z9_output, z12_output)
    return z0_output, z3_output, z6_output, z9_output, z12_output

def pre_trained_TransUnet_ddpm(input_tensor, t, CLass):
    checkpoint_path = 'C:\\Users\\User\\PycharmProjects\\Mythical_Animals\\Diffusion_ddpm_to_save_checkpoints\\Diffusion_checkpoint'
    model = tf.keras.models.load_model(checkpoint_path)
    # print(model.summary())

    z0_layer_output = model.get_layer('input_1').output
    z12_layer_output = model.get_layer('reshape_4').output
    z9_layer_output = model.get_layer('reshape_3').output
    z6_layer_output = model.get_layer('reshape_2').output
    z3_layer_output = model.get_layer('reshape_1').output

    z0_model = tf.keras.Model(inputs=model.input, outputs=z0_layer_output)
    z3_model = tf.keras.Model(inputs=model.input, outputs=z3_layer_output)
    z6_model = tf.keras.Model(inputs=model.input, outputs=z6_layer_output)
    z9_model = tf.keras.Model(inputs=model.input, outputs=z9_layer_output)
    z12_model = tf.keras.Model(inputs=model.input, outputs=z12_layer_output)

    z0_output = z0_model([input_tensor, t, CLass])
    z3_output = z3_model([input_tensor, t, CLass])
    z6_output = z6_model([input_tensor, t, CLass])
    z9_output = z9_model([input_tensor, t, CLass])
    z12_output = z12_model([input_tensor, t, CLass])
    # print('z0_output, z3_output, z6_output, z9_output, z12_output', z0_output, z3_output, z6_output, z9_output, z12_output)

    return z0_output, z3_output, z6_output, z9_output, z12_output
