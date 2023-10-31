import keras.losses
from losses import *
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
from keras.losses import Loss
from keras.utils import get_custom_objects
from keras.models import Model
from keras.layers import Input
from TransU.TransUnet import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from IPython import display
import PIL
from matplotlib.animation import FuncAnimation

# print(model.summary())

# print(transunet_model.summary())



# checkpoint = ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_freq='epoch',
#     mode='auto')
def get_word_vector(word):
    word2idx_data = np.load("C://Users//User//PycharmProjects//Classification_without_segmentation//word_to_id.npz", allow_pickle=True)
    word_to_id = dict(word2idx_data["word2idx"].item())
    if word in word_to_id:
        word_id = word_to_id[word]
        return tf.convert_to_tensor([word_id])
    else:
        return None

def ind_def(name):
    class_to_label = {k: v for k, v in enumerate(['cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'squirrel'])}
    ind = [get_word_vector(class_to_label[label]) for label in name]
    # ind = [get_word_vector(label) for label in name]
    return np.array(ind, dtype=np.int32)


@tf.function
def train_step(img_mask, img, indd):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        ind, generated_images = gen_mask([img_mask, indd], training=True)
        real_output = discriminator(img, training=True)
        fake_output = discriminator(generated_images, training=True)

        indd = tf.dtypes.cast(indd, tf.int32)
        indd = tf.one_hot(indd, depth=32)
        indd = tf.squeeze(indd, axis=1)

        loss_object = tf.keras.losses.CategoricalCrossentropy()
        gen_loss_class = loss_object(ind, indd)
        gen_loss = generator_loss(fake_output)

        combined_gen_loss = gen_loss + gen_loss_class
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(combined_gen_loss, gen_mask.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_mask_optimizer.apply_gradients(zip(gradients_of_generator, gen_mask.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, gen_loss_class, disc_loss


def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def show_mask(mask, mask_z0):
    # mask_val = mask[0].numpy()
    mask_z0_val = mask_z0[0].numpy()
    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.title('Маска')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_z0_val)
    plt.title('Маска_z0')
    plt.show()
    plt.pause(30)

def Data_generator(data_dir, BATCH_SIZE=640):
    image_size = (144, 144)
    datagen_pred = ImageDataGenerator(
        rescale=1.0 / 255.0)
        # width_shift_range=0.1,
        # height_shift_range=0.1)
    # filenames = os.listdir(data_dir)
    # print('filenames', filenames)
    data_generator = datagen_pred.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True)
    return data_generator
