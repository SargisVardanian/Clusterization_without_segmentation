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
from TransU.Under_utils import *


def Trans_block(num):
    cf = {}
    cf["image_size"] = 256
    cf["num_layers"] = 12
    cf["hidden_dim"] = 64
    cf["mlp_dim"] = 64
    cf["num_heads"] = 6
    cf["dropout_rate"] = 0.1
    cf["num_patches"] = 256
    cf["patch_size"] = 16
    cf["num_channels"] = 3
    input_shape = (cf["image_size"], cf["image_size"], cf["num_channels"])
    inputs0 = L.Input(input_shape)
    inputs1 = tf.image.extract_patches(
        images=inputs0,
        sizes=[1, cf["patch_size"], cf["patch_size"], 1],
        strides=[1, cf["patch_size"], cf["patch_size"], 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    inputs = tf.reshape(inputs1, [-1, cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])

    """ Patch + Position Embeddings """
    patch_embed = L.Dense(cf["hidden_dim"])(inputs)  ## (None, 256, 768)
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)  ## (256,)
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)  ## (256, 768)
    x = patch_embed + pos_embed
    skip_connection_index = [8, 12]
    skip_connections = []

    for i in range(1, cf["num_layers"] + 1, 1):
        x = transformer_encoder(x, cf)
        if i in skip_connection_index:
            skip_connections.append(x)

    z8, z12 = skip_connections

    z8 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z8)
    z12 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z12)
    print('z88', z8.shape)
    print('z1212', z12.shape)
    '''Part for pre+training classification'''
    concat = L.Concatenate()([z8, z12])
    output = L.GlobalAveragePooling2D()(concat)
    output = L.Dense(num, activation='softmax')(output)
    return Model(inputs0, output, name='trained_block')


def Encoder_Block(cf, inputs):
    """ Patch + Position Embeddings """
    patch_embed = L.Dense(cf["hidden_dim"])(inputs)  ## (None, 256, 768)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)  ## (256,)
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)  ## (256, 768)
    x = patch_embed + pos_embed  ## (None, 256, 768)

    """ Transformer Encoder """
    skip_connection_index = [3, 6, 9, 12]
    skip_connections = []

    for i in range(1, cf["num_layers"] + 1, 1):
        x = transformer_encoder(x, cf)

        if i in skip_connection_index:
            skip_connections.append(x)

    z3, z6, z9, z12 = skip_connections
    z0 = L.Reshape((cf["image_size"], cf["image_size"], cf["num_channels"]))(inputs)
    return z0, z3, z6, z9, z12


def Decoder_Block(cf, z0, z3, z6, z9, z12):
    """ CNN Decoder """
    z3 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z3)
    z6 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z6)
    z9 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z9)
    z12 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z12)

    ## Decoder 1
    x = deconv_block(z12, 512)

    s = deconv_block(z9, 512)
    s = conv_block(s, 512)
    x = L.Concatenate()([x, s])

    x = conv_block(x, 512)
    x = conv_block(x, 512)

    ## Decoder 2
    x = deconv_block(x, 256)

    s = deconv_block(z6, 256)
    s = conv_block(s, 256)
    s = deconv_block(s, 256)
    s = conv_block(s, 256)

    x = L.Concatenate()([x, s])
    x = conv_block(x, 256)
    x = conv_block(x, 256)

    ## Decoder 3
    x = deconv_block(x, 128)

    s = deconv_block(z3, 128)
    s = conv_block(s, 128)
    s = deconv_block(s, 128)
    s = conv_block(s, 128)
    s = deconv_block(s, 128)
    s = conv_block(s, 128)

    x = L.Concatenate()([x, s])
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    ## Decoder 4
    x = deconv_block(x, 64)

    s = conv_block(z0, 64)
    s = conv_block(s, 64)

    x = L.Concatenate()([x, s])
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    return x


if __name__ == "__main__":
    config = {}
    config["image_size"] = 256
    config["num_layers"] = 7
    config["hidden_dim"] = 64
    config["mlp_dim"] = 64
    config["num_heads"] = 3
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 16
    config["num_channels"] = 3

# model = Trans_block(9)
# print(model.summary())

def pre_trained_TransBlocl(inputs01, inputs02):
    checkpoint_path = 'C:\\Users\\User\\PycharmProjects\\Mythical_Animals\\path_to_save_checkpoints\\model_checkpoint'
    model = tf.keras.models.load_model(checkpoint_path)
    z8_layer_output = model.layers[15].output
    z12_layer_output = model.layers[23].output

    patch_embed = tf.keras.layers.Dense(64)(inputs01)
    positions = tf.range(start=0, limit=256, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=256, output_dim=64)(positions)  ## (256, 768)
    inputs01 = patch_embed + pos_embed

    patch_embed = tf.keras.layers.Dense(64)(inputs02)
    positions = tf.range(start=0, limit=256, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=256, output_dim=64)(positions)  ## (256, 768)
    inputs02 = patch_embed + pos_embed

    z8_model = tf.keras.Model(inputs=model.input, outputs=z8_layer_output)
    z12_model = tf.keras.Model(inputs=model.input, outputs=z12_layer_output)

    for layer in z8_model.layers:
        layer.trainable = False

    for layer in z12_model.layers:
        layer.trainable = False

    outputs11 = z8_model(inputs01)
    outputs12 = z12_model(inputs01)

    outputs21 = z8_model(inputs02)
    outputs22 = z8_model(inputs02)
    return outputs11, outputs12, outputs21, outputs22

def pre_trained_TransUnet(input_tensor):
    checkpoint_path = 'C:\\Users\\User\\PycharmProjects\\Mythical_Animals\\Diffusion_to_save_checkpoints\\Diffusion_checkpoint'
    model = tf.keras.models.load_model(checkpoint_path)
    # print(model.summary())
    z0_layer_output = model.layers[0].output
    z12_layer_output = model.layers[2].output
    z9_layer_output = model.layers[6].output
    z6_layer_output = model.layers[11].output
    z3_layer_output = model.layers[21].output

    z0_model = tf.keras.Model(inputs=model.input, outputs=z0_layer_output)
    z3_model = tf.keras.Model(inputs=model.input, outputs=z3_layer_output)
    z6_model = tf.keras.Model(inputs=model.input, outputs=z6_layer_output)
    z9_model = tf.keras.Model(inputs=model.input, outputs=z9_layer_output)
    z12_model = tf.keras.Model(inputs=model.input, outputs=z12_layer_output)

    patch_embed = tf.keras.layers.Dense(64)(input_tensor)
    positions = tf.range(start=0, limit=16, delta=1)  ## (256,)
    pos_embed = tf.keras.layers.Embedding(input_dim=16, output_dim=64)(positions)  ## (256, 768)
    input_tensor = patch_embed + pos_embed

    z0_output = z0_model(input_tensor)
    z3_output = z3_model(input_tensor)
    z6_output = z6_model(input_tensor)
    z9_output = z9_model(input_tensor)
    z12_output = z12_model(input_tensor)
    # print('z0_output, z3_output, z6_output, z9_output, z12_output', z0_output, z3_output, z6_output, z9_output, z12_output)
    return z0_output, z3_output, z6_output, z9_output, z12_output

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
from Under_utils import *

def Trans_block(num):
    cf = {}
    cf["image_size"] = 256
    cf["num_layers"] = 12
    cf["hidden_dim"] = 64
    cf["mlp_dim"] = 64
    cf["num_heads"] = 6
    cf["dropout_rate"] = 0.1
    cf["num_patches"] = 256
    cf["patch_size"] = 16
    cf["num_channels"] = 3
    input_shape = (cf["image_size"], cf["image_size"], cf["num_channels"])
    inputs0 = L.Input(input_shape)
    inputs1 = tf.image.extract_patches(
        images=inputs0,
        sizes=[1, cf["patch_size"], cf["patch_size"], 1],
        strides=[1, cf["patch_size"], cf["patch_size"], 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    inputs = tf.reshape(inputs1, [-1, cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])

    """ Patch + Position Embeddings """
    patch_embed = L.Dense(cf["hidden_dim"])(inputs)  ## (None, 256, 768)
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)  ## (256,)
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)  ## (256, 768)
    x = patch_embed + pos_embed
    skip_connection_index = [8, 12]
    skip_connections = []

    for i in range(1, cf["num_layers"] + 1, 1):
        x = transformer_encoder(x, cf)
        if i in skip_connection_index:
            skip_connections.append(x)

    z8, z12 = skip_connections

    z8 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z8)
    z12 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z12)
    print('z88', z8.shape)
    print('z1212', z12.shape)
    '''Part for pre+training classification'''
    concat = L.Concatenate()([z8, z12])
    output = L.GlobalAveragePooling2D()(concat)
    output = L.Dense(num, activation='softmax')(output)
    return Model(inputs0, output, name='trained_block')


def Encoder_TransUnet(cf, inputs):
    """ Patch + Position Embeddings """
    patch_embed = L.Dense(cf["hidden_dim"])(inputs)  ## (None, 256, 768)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)  ## (256,)
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)  ## (256, 768)
    x = patch_embed + pos_embed  ## (None, 256, 768)

    """ Transformer Encoder """
    skip_connection_index = [3, 6, 9, 12]
    skip_connections = []

    for i in range(1, cf["num_layers"] + 1, 1):
        x = transformer_encoder(x, cf)

        if i in skip_connection_index:
            skip_connections.append(x)

    z3, z6, z9, z12 = skip_connections
    z0 = L.Reshape((cf["image_size"], cf["image_size"], cf["num_channels"]))(inputs)
    return z0, z3, z6, z9, z12


def Decoder_TransUnet(cf, z0, z3, z6, z9, z12):
    """ CNN Decoder """
    z3 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z3)
    z6 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z6)
    z9 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z9)
    z12 = L.Reshape((cf["patch_size"], cf["patch_size"], cf["hidden_dim"]))(z12)

    ## Decoder 1
    x = deconv_block(z12, 512)

    s = deconv_block(z9, 512)
    # s = conv_block(s, 512)
    x = L.Concatenate()([x, s])

    # x = conv_block(x, 512)
    x = conv_block(x, 512)

    ## Decoder 2
    x = deconv_block(x, 256)

    s = deconv_block(z6, 256)
    # s = conv_block(s, 256)
    s = deconv_block(s, 256)
    s = conv_block(s, 256)

    x = L.Concatenate()([x, s])
    # x = conv_block(x, 256)
    x = conv_block(x, 256)

    ## Decoder 3
    x = deconv_block(x, 128)

    s = deconv_block(z3, 128)
    s = conv_block(s, 128)
    s = deconv_block(s, 128)
    s = conv_block(s, 128)
    s = deconv_block(s, 128)
    s = conv_block(s, 128)

    x = L.Concatenate()([x, s])
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    ## Decoder 4
    x = deconv_block(x, 64)

    s = conv_block(z0, 64)
    s = conv_block(s, 64)

    x = L.Concatenate()([x, s])
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    return x


if __name__ == "__main__":
    config = {}
    config["image_size"] = 256
    config["num_layers"] = 7
    config["hidden_dim"] = 64
    config["mlp_dim"] = 64
    config["num_heads"] = 3
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 16
    config["num_channels"] = 3

# model = Trans_block(9)
# print(model.summary())

def pre_trained_TransBlocl(inputs01, inputs02):
    checkpoint_path = 'C:\\Users\\User\\PycharmProjects\\Mythical_Animals\\path_to_save_checkpoints\\model_checkpoint'
    model = tf.keras.models.load_model(checkpoint_path)
    z8_layer_output = model.layers[15].output
    z12_layer_output = model.layers[23].output

    z8_model = tf.keras.Model(inputs=model.input, outputs=z8_layer_output)
    z12_model = tf.keras.Model(inputs=model.input, outputs=z12_layer_output)

    for layer in z8_model.layers:
        layer.trainable = False

    for layer in z12_model.layers:
        layer.trainable = False

    outputs11 = z8_model(inputs01)
    outputs12 = z12_model(inputs01)

    outputs21 = z8_model(inputs02)
    outputs22 = z8_model(inputs02)
    return outputs11, outputs12, outputs21, outputs22
