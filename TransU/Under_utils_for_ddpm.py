import tensorflow as tf
import keras.layers as L
import math
import keras
import os
import matplotlib.pyplot as plt

def transformer_SelfAt(x, cf, inputs=None):
    skip_1 = x
    x = L.LayerNormalization()(x)
    if inputs:
        x = L.MultiHeadAttention(
            num_heads=cf["num_heads"],
            key_dim=inputs,
            dropout=cf["dropout_rate"],
            kernel_initializer='glorot_normal'
            )(x, x)
    else:
        x = L.MultiHeadAttention(
            num_heads=cf["num_heads"],
            key_dim=cf["hidden_dim"],
            dropout=cf["dropout_rate"])(x, x)
    x = L.Add()([x, skip_1])
    # skip_2 = x
    # x = L.LayerNormalization()(x)
    # if inputs:
    #     x = MLP(units=inputs, activation_fn=keras.activations.swish)(x)
    # else:
    #     x = MLP(units=cf["num_patches"]*3, activation_fn=keras.activations.swish)(x)
    # x = L.Add()([x, skip_2])
    return x

def transformer_class(x, cf, dim=432):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(
            num_heads=cf["num_heads"],
            key_dim=cf["hidden_dim"],
            dropout=cf["dropout_rate"])(x, x, x)
    x = L.Add()([x, skip_1])
    x = L.LayerNormalization()(x)
    x = MLP(units=dim, activation_fn=keras.activations.swish)(x)
    return x

def transformer_012(x1, x2, x3, cf):
    xx = x1
    x = L.MultiHeadAttention(
        num_heads=cf["num_heads"],
        key_dim=cf["hidden_dim"])(x1, x2, x3)
    x = L.Add()([x, xx])
    return x

def transformer_CrossAT(x0, x1, cf):
    skip_1 = x0
    x0 = L.LayerNormalization()(x0)
    x = L.Add()([x0, x1])
    x = L.MultiHeadAttention(
            num_heads=cf["num_heads"],
            key_dim=cf["hidden_dim"],
            dropout=cf["dropout_rate"])(x0, x, x)
    x = L.Add()([x, skip_1, x0])
    # skip_2 = x
    # x = L.LayerNormalization()(x)
    # x = MLP(units=c, activation_fn=keras.activations.swish)(x)
    # x = L.Add()([x, skip_2])
    return x


def conv_block(x, num_filters=6, kernel_size=6, st=1, padding="same", dropout_rate=None):
    if padding:
        x = L.Conv2D(num_filters, kernel_size=kernel_size, strides=st, padding="same")(x)
    else:
        x = L.Conv2D(num_filters, kernel_size=kernel_size, strides=st)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout_rate is not None:
        x = L.Dropout(dropout_rate)(x)
    return x


def deconv_block(x, num_filters, kernel_size=6, st=1):
    x = L.Conv2DTranspose(num_filters//2, kernel_size=kernel_size, padding="same", strides=st)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class Embedding_my(L.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.dim - 1)
        self.emb = tf.exp(tf.range(self.dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

def MLP(units, activation_fn=tf.keras.activations.swish):
    def apply(inputs):
        x = L.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        x = L.Dropout(0.2)(x)
        x = L.Dense(units, kernel_initializer=kernel_init(1.0))(x)
        x = L.Dropout(0.1)(x)
        return x
    return apply


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


def UnPatches(hidd, cf, name=None):
    hidd = L.Reshape((cf["patch_size"], cf["patch_size"], cf["patch_size"], cf["patch_size"], cf['num_channels']))(hidd)
    hidd = tf.transpose(hidd, perm=[0, 1, 3, 2, 4, 5])
    hidd = L.Reshape((cf["image_size"], cf["image_size"], cf["num_channels"]), name=name)(hidd)
    return hidd

def UnPatches_One_dim(hidd, cf, name=None):
    hidd = L.Reshape((cf["patch_size"], cf["patch_size"], cf["patch_size"], cf["patch_size"]))(hidd)
    hidd = tf.transpose(hidd, perm=[0, 1, 3, 2, 4])
    hidd = L.Reshape((cf["image_size"], cf["image_size"]), name=name)(hidd)
    return hidd

def ResidualBlock(cf, groups=4, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, temb= inputs
        x = L.Add()([x, temb])
        x = L.Dropout(cf["dropout_rate"])(x)
        x = L.Dense(cf['num_patches']*3)(x)
        x = L.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        return x
    return apply


def DownSample(width=1, activation=None):
    def apply(x):
        x = tf.expand_dims(x, axis=-1)
        x = L.Conv2D(
            width,
            kernel_size=8,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0), activation=activation
        )(x)
        x = tf.squeeze(x, axis=-1)
        return x
    return apply


def Upsample(dim):
    return L.Conv2DTranspose(filters=dim,
                             kernel_size=4,
                             strides=2,
                                 padding='SAME')



# def sample_probas(Discriminator, bsize, save_dir, batch_size=50):
#     idxs_real = tf.random.choice(tf.arange(data.shape[0]), size=bsize)
#     idxs_gen = tf.random.choice(tf.arange(batch_size), size=bsize)
#
#     preds_real = Discriminator.predict(sample_data_batch(idxs_real))
#     preds_gen = Discriminator.predict(sample_data_batch(idxs_gen))
#
#     if preds_real.ndim == 1:
#         preds_real = tf.expand_dims(preds_real, axis=1)
#     if preds_gen.ndim == 1:
#         preds_gen = tf.expand_dims(preds_gen, axis=1)
#
#     plt.title('Generated vs real data')
#     plt.hist(tf.exp(preds_real)[:, 0], label='D(x)', alpha=0.5, range=[0, 1])
#     plt.hist(tf.exp(preds_gen)[:, 0], label='D(G(x))', alpha=0.5, range=[0, 1])
#     plt.legend(loc='upper center')
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

config = {}
config["image_size"] = 144
config["num_layers"] = 12
config["hidden_dim"] = 144
config["mlp_dim"] = 144
config["num_heads"] = 8
config["dropout_rate"] = 0.2
config["num_patches"] = 144
config["patch_size"] = 12
config["num_channels"] = 3
