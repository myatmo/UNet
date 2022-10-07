import matplotlib.pyplot as plt
import os
import glob
import math
import argparse
import pandas as pd
import numpy as np
import skimage.io as skio
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from unet import unet
from unet import *


def get_train_generator(train_df, batch_size=3, target_size=(256,256), seed=1, image_color_mode="grayscale"):
    # Data augmentation: define rotation angle, width and height shift, shear and zoom range, horizontal flip
    image_gen_args = dict(rotation_range=0.5,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect',
                    rescale=1./255.
                    #validation_split=0.30
                    )

    mask_gen_args = dict(rotation_range=0.5,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='reflect',
                    #validation_split=0.30
                    )
    
    image_datagen = ImageDataGenerator(**image_gen_args)
    mask_datagen = ImageDataGenerator(**mask_gen_args)

    image_generator = image_datagen.flow_from_dataframe(dataframe = train_df,
                                                        x_col = "images",
                                                        y_col = None,
                                                        #subset = "training",
                                                        batch_size = batch_size,
                                                        seed = 1,
                                                        class_mode = None,
                                                        color_mode = image_color_mode,
                                                        target_size = target_size)


    mask_generator = mask_datagen.flow_from_dataframe(dataframe = train_df,
                                                        x_col = "masks",
                                                        y_col = None,
                                                        #subset = "validation",
                                                        batch_size = batch_size,
                                                        seed = 1,
                                                        class_mode = None,
                                                        color_mode = image_color_mode,
                                                        target_size = target_size)

    train_generator = zip(image_generator, mask_generator)
    train_step_size = image_generator.n // image_generator.batch_size

    return train_generator, train_step_size


def get_args():
    # to do
    parser = argparse.ArgumentParser(description='Train the UNet on images and masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    #args = get_args()
    model_path = "../trained_models"
    model_name = "unet_lipid_4"
    # Define paramaters for train generator
    batch_size = 10
    target_size = (256,256)
    seed = 1
    image_folder = "images"
    mask_folder = "masks"
    image_color_mode = "grayscale"
    image_save_prefix  = "image"
    mask_save_prefix  = "mask"
    save_to_dir = "aug"

    # U-net model
    model = unet()

    # Load train_df
    train_df = pd.read_pickle("train_df.pkl")

    # Create data generator for training set with data augmentation
    train_generator, train_step_size = get_train_generator(train_df, batch_size, target_size, seed, image_color_mode)
    
    # Train the model
    model_file = os.path.join(model_path, model_name + ".hdf5")
    model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
    history = model.fit(train_generator, steps_per_epoch=train_step_size, epochs=20, callbacks=[model_checkpoint])

    # Save model history
    history_file_name = os.path.join(model_path, model_name + "_history.npy")
    np.save(history_file_name, history.history)

    # with early stopping
    #es = EarlyStopping(monitor='loss', mode='min', verbose=1)
    #history = model.fit(train_generator,steps_per_epoch=train_step_size,epochs=20,callbacks=[es])
    #model.save(model_file_location)

