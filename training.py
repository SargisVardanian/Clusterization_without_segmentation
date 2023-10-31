import os
import time
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
from keras.losses import Loss
from keras.utils import get_custom_objects
from keras.models import Model
from keras.layers import Input
# from TransU.TransUnet import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from IPython import display
import PIL
from matplotlib.animation import FuncAnimation
import random
from keras.preprocessing.image import load_img, img_to_array
from training_utils import *
from losses import *
from tqdm import tqdm

data_dir = "C:/Users/User/PycharmProjects/Classification_without_segmentation/animals"
categories = ['cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'squirrel']

def train(img, ind, epochs, BATCH_SIZE):
    batch_size = 32
    checkpoint.restore(manager._latest_checkpoint)

    if manager._latest_checkpoint:
        print("Загружены веса из", manager._latest_checkpoint)
    else:
        print("Веса не найдены в указанной директории.")
    for epoch in range(epochs):
        shuffled_indices = tf.random.shuffle(tf.range(len(img)))
        img_mask = tf.gather(img, shuffled_indices)
        ind_mask = tf.gather(ind, shuffled_indices)
        img = tf.random.shuffle(image)
        progress_bar = tqdm(total=BATCH_SIZE, position=0, leave=True)
        # for batch_start in range(0, len(img), batch_size):
        #     batch_end = min(batch_start + batch_size, len(img))
        #     batch_img_mask = img_mask[batch_start:batch_end]
        #     batch_img = img[batch_start:batch_end]
        #     batch_ind_mask = ind_mask[batch_start:batch_end]
        #     gen_loss, gen_loss_class, disc_loss = train_step(batch_img_mask, batch_img, batch_ind_mask)
        #     progress_bar.update(batch_end - batch_start)
        #     progress_bar.set_description('Epoch {} - gen_loss {:.2f} - gen_loss_class {:.2f} - disc_loss {:.2f}'
        #                                  .format(epoch + 1, gen_loss.numpy(), gen_loss_class.numpy(),
        #                                          disc_loss.numpy()))
        try:
            # mask_output = gen_mask.get_layer('mask').output
            # mask_z0_output = gen_mask.get_layer('mask_z0').output
            # image_input = gen_mask.get_layer('image_input').output
            # ind_input = gen_mask.get_layer('ind_input').output
            # mask_model = Model(inputs=[image_input, ind_input], outputs=[mask_output, mask_z0_output])
            # mask_model.save(checkpoint_path_combined)
            print(f'mask_model LOAD in the epoch{epoch}')
        except:
            # mask_model = tf.keras.models.load_model(checkpoint_path_combined)
            print(f'mask_model P in the epoch{epoch}')
        # try:
        # checkpoint.save(file_prefix=checkpoint_path)
        im, id = image[:1], ind[:1]
        print(f'im {im.shape}, id {id.shape}')
        # mask_model = tf.keras.models.load_model(checkpoint_path_combined)
        mask, mask_z0 = gen_mask([im, id])
        show_mask(im[0], mask_z0)
        # except:
        #     print("Nothing to show!!!")


BATCH_SIZE = 640
epochs = 10
for i in range(1, epochs + 1):
    print(f'Epoch{epochs}/{i}')
    data_generator = Data_generator(data_dir=data_dir, BATCH_SIZE=BATCH_SIZE)
    image, name = next(data_generator)
    # print('image', image.shape, 'name', name)
    ind = ind_def(name)
    # print('image', image.shape, 'ind', ind.shape)
    # dataset = {idx: img for img, idx in zip(image, ind)}
    # dataset = np.array(dataset)
    # print('train_dataset', dataset)
    train(image, ind, 1, BATCH_SIZE)
    # model.fit(x=[image, ind],
    #     y=[ind, image],
    #     batch_size=batch_size,
    #     epochs=1,
    #     callbacks=[checkpoint])
    # print('sleeping...')
    # time.sleep(60)
    # print('oh shit here we go again...')
    # mask_output = gen_mask.get_layer('mask').output
    # mask_z0_output = gen_mask.get_layer('mask_z0').output
    # image_input = gen_mask.get_layer('image_input').output
    # ind_input = gen_mask.get_layer('ind_input').output
    # mask_model = Model(inputs=[image_input, ind_input], outputs=[mask_output, mask_z0_output])
    # mask_model.save(checkpoint_path_combined)
    # image, ind = image[:1], ind[:1]
    # mask, mask_z0 = mask_model([image, ind])
    # print('mask', mask.shape)

    # plt.imshow(mask[0], cmap='gray')
    # plt.title('mask')
    # plt.show()
    #
    # plt.imshow(mask_z0[0])
    # plt.title('mask_z0')
    # plt.show()

# transunet_output = model.get_layer('TransUnet').output
# print('image', image.shape, 'ind', ind)
# print('mask', mask_clas.shape)

# mask = mask.numpy()
# plt.imshow(mask[0], cmap='gray')
# plt.title('mask')
# plt.show()


# mask_z0 = mask_z0.numpy()
# plt.imshow(mask_z0[0])
# plt.title('mask_z0')
# plt.show()
