import tensorflow as tf
import keras.layers as L
from keras.models import Model
# from Under_utils import *
# from TransU.utils import *
from TransU.utils_for_ddpm import *
# import tensorflow_probability as tfp
# tfd = tfp.distributions
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def TransUnet(cf):
    patch = Patches(12)
    input_shape = (cf["image_size"], cf["image_size"], cf["num_channels"])
    name_shape = ()

    inputs0 = L.Input(shape=input_shape, name='image_input')

    name1 = L.Input(shape=name_shape, name='ind_input')
    name = L.Embedding(input_dim=32, output_dim=cf["num_patches"] * 3)(name1)

    Patch = Patches(12)
    inputs1 = Patch(inputs0)
    inputs = tf.reshape(inputs1, [-1, cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])

    mask, mask_z0 = Encoder_TransUnet(cf, inputs, name)
    # mask_z0 = mask_multiply_classfication(mask, mask_z0, cf)
    # mask = mask_classfication(mask, cf)
    print('outputs', mask_z0.shape)

    mask = tf.expand_dims(mask, axis=-1)
    mask = L.Multiply()([inputs0, mask])
    mask = patch(mask)
    mask = tf.reshape(mask, [-1, cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])
    print(f'mask {mask.shape}, mask_z0 {mask_z0.shape}')
    mask = transformer_class(mask, cf, dim=432)
    mask = transformer_class(mask, cf, dim=16)
    mask = transformer_class(mask, cf, dim=4)
    mask = transformer_class(mask, cf, dim=4)
    mask = L.Flatten()(mask)
    mask = L.Dense(32, activation='relu')(mask)
    mask = L.Dense(32, activation='softmax', name='mask_classfication')(mask)
    # mask = conv_block(mask, num_filters=144, kernel_size=6, st=6, padding=None)
    # mask = conv_block(mask, num_filters=64, kernel_size=6, st=1)
    # mask = conv_block(mask, num_filters=16, kernel_size=6, st=4)

    # mask_z0 = patch(mask_z0)
    # mask_z0 = UnPatches(mask_z0, cf,  name='mask_multiply_classfication')
    # input = tf.reshape(input, [-1, cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])
    # input = transformer_class(input, cf, dim=432)
    # input = transformer_class(input, cf, dim=432)
    # input = transformer_class(input, cf, dim=432)
    # input = transformer_class(input, cf, dim=64)
    # input = transformer_class(input, cf, dim=16)
    # input = transformer_class(input, cf, dim=4)
    # input = transformer_class(input, cf, dim=4)
    # input = L.Flatten()(input)
    # input = L.Dense(512, activation='relu')(input)
    # mask_z0 = L.Dense(512, activation='softmax', name='mask_multiply_classfication')(input)

    return Model([inputs0, name1], [mask, mask_z0], name="TransUnet")

def Discriminator(cf):
    input_shape = (cf["image_size"], cf["image_size"], cf["num_channels"])
    input = L.Input(shape=input_shape, name='image_input_discrim')
    patch = Patches(12)
    inputs = patch(input)
    inputs = tf.reshape(inputs, [-1, cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])
    inputs = transformer_class(inputs, cf, dim=432)
    inputs = transformer_class(inputs, cf, dim=16)
    inputs = transformer_class(inputs, cf, dim=8)
    inputs = transformer_class(inputs, cf, dim=4)
    inputs = L.Flatten()(inputs)
    inputs = L.Dense(16, activation='relu')(inputs)
    inputs = L.Dense(1, activation='softmax', name='mask_disc')(inputs)
    return Model([input], [inputs], name="Discriminator")

def mask_classfication(cf):
    input_shape = (cf["image_size"], cf["image_size"])
    input = L.Input(shape=input_shape)
    mask = tf.expand_dims(input, axis=-1)
    mask = conv_block(mask, num_filters=144, kernel_size=6, st=6, padding=None)
    mask = conv_block(mask, num_filters=64, kernel_size=6, st=1)
    mask = conv_block(mask, num_filters=16, kernel_size=6, st=4)
    mask = L.Flatten()(mask)
    mask = L.Dense(512, activation='relu')(mask)
    mask = L.Dense(512, activation='softmax')(mask)
    print('mask', mask.shape)
    return Model([input], [mask], name="mask_classfication")

def mask_multiply_classfication(cf):
    input_shape = (cf["image_size"], cf["image_size"], cf["num_channels"])
    inputs = L.Input(shape=input_shape)
    patch = Patches(12)
    input = patch(inputs)
    input = tf.reshape(input, [-1, cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"]])
    input = transformer_class(input, cf, dim=432)
    input = transformer_class(input, cf, dim=64)
    input = transformer_class(input, cf, dim=16)
    input = transformer_class(input, cf, dim=4)
    input = transformer_class(input, cf, dim=4)
    input = L.Flatten()(input)
    input = L.Dense(512, activation='relu')(input)
    mask_z0 = L.Dense(512, activation='softmax')(input)
    print('mask_multiply_classfication', input.shape)
    return Model([inputs], [mask_z0], name="mask_multiply_classfication")

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
config["num_classes"] = 6

# model = TransUnet(config)
#
# print(model.summary())
# #
# data_dir = "C:/Users/User/PycharmProjects/Classification_without_segmentation/animals"
#
# image_size = (144, 144)
#
# datagen_pred = ImageDataGenerator(
#                 rescale=1.0 / 255.0,
#                 width_shift_range=0.1,
#                 height_shift_range=0.1)
#
# batch_size = 1
# filenames = os.listdir(data_dir)
# print('filenames', filenames)
#
# data_generator = datagen_pred.flow_from_directory(
#     data_dir,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode='sparse',
#     shuffle=True)
#
# image, name = data_generator.next()
#
# print('img', image.shape)
#
# plt.imshow(image[0])
# plt.title(f"a: {name}")
# plt.show()
#
# class_to_label = {v: k for k, v in data_generator.class_indices.items()}
# print('class_to_label', class_to_label)
#
# def get_word_vector(word):
#     word2idx_data = np.load("C://Users//User//PycharmProjects//Classification_without_segmentation//word_to_id.npz", allow_pickle=True)
#     word_to_id = dict(word2idx_data["word2idx"].item())
#     if word in word_to_id:
#         word_id = word_to_id[word]
#         return tf.convert_to_tensor([word_id])
#     else:
#         return None
#
# ind = [get_word_vector(class_to_label[label]).numpy()[0] for label in name]
# ind = np.array(ind, dtype=np.int32)
#
# print('image', image.shape, 'ind', ind, ind)
#
# # mask, mask_z0 = model([image, ind])
# mask_output = model.get_layer('mask').output
# mask_z0_output = model.get_layer('mask_z0').output
# image_input = model.get_layer('image_input').output
# ind_input = model.get_layer('ind_input').output
#
# mask_model = Model(inputs=[image_input, ind_input], outputs=[mask_output, mask_z0_output])
# image, ind = image[:1], ind[:1]
# mask, mask_z0 = mask_model([image, ind])
#
# print('mask', mask.shape)
#
# plt.imshow(mask[0], cmap='gray')
# plt.title('mask')
# plt.show()
#
# print('mask_z0', mask_z0.shape)
#
# plt.imshow(mask_z0[0])
# plt.title('mask_z0')
# plt.show()
