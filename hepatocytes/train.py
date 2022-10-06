import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
import math
import pandas as pd
import skimage.io as skio
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from unet import unet
from unet import *
from utils import *


def get_images(ds_path):
    #ds_path = os.getcwd()
    # get all the file path name
    image_file_list = glob.glob(os.path.join(ds_path, 'images') + '/*jpg')
    mask_file_list = glob.glob(os.path.join(ds_path, 'masks') + '/*png')

    # sort the file name
    image_file_list.sort()
    mask_file_list.sort()

    return image_file_list, mask_file_list


def split_data(image_file_list, mask_file_list, image_stacks, split_ratio, validation):
    # need changes: add validation split and 
    test_split = math.floor(len(image_file_list)*split_ratio)
    start, end = image_stacks[0]
    i = 0
  
    while(test_split < start or test_split > end):
        i += 1
        start, end = image_stacks[i]
    
    split1 = 0

    if test_split == end:
        split1 = end
    else:
        split1 = start
    
    training_images = image_file_list[split1:]
    training_masks = mask_file_list[split1:]
    training_set = {'images': training_images,
                    'masks': training_masks}

    test_images = image_file_list[0:split1]
    test_masks = mask_file_list[0:split1]
    test_set = {'images': test_images,
                    'masks': test_masks}

    # convert to dataframe
    train_df = pd.DataFrame(training_set)
    test_df = pd.DataFrame(test_set)

    return train_df, test_df


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
                                                        directory = None,
                                                        x_col = "images",
                                                        y_col = None,
                                                        #subset = "training",
                                                        batch_size = batch_size,
                                                        seed = 1,
                                                        class_mode = None,
                                                        color_mode = image_color_mode,
                                                        target_size = target_size)


    mask_generator = mask_datagen.flow_from_dataframe(dataframe = train_df,
                                                        directory = None,
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


def adjust_mask_data(img, mask):
    if(np.max(img) > 1):
        img = img / 255
    if(np.max(mask) > 1):
        mask = mask / 255
    mask[mask != 0] = 1
    return (img, mask)

def get_generator(train_gen):
    for (img, mask) in train_gen:
        img, mask = adjust_mask_data(img, mask)
        yield (img, mask)

def get_test_generator(batch_size, target_size, test_df, image_color_mode):
    # Define ImageDataGenerator for testing images
    test_data_gen_args = dict(rescale=1./255.)
    test_gen = ImageDataGenerator(**test_data_gen_args)
    test_generator = test_gen.flow_from_dataframe(dataframe = test_df,
                                                directory = None,
                                                x_col = "images",
                                                y_col = None,
                                                batch_size = batch_size,
                                                seed = 1,
                                                class_mode = None,
                                                color_mode = image_color_mode,
                                                target_size = target_size)

    return test_generator

def show_test_image_pairs(results, i):
    mask_test = results[i][:,:,0]
    mask_actual = skio.imread(test_df['masks'][i])
    img_actual = skio.imread(test_df['images'][i])
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(mask_test, cmap='gray')
    ax[1].imshow(mask_actual, cmap='gray')
    ax[2].imshow(img_actual, cmap='gray')
    plt.show()


if __name__ == '__main__':
    ds_path = '../../../../shared/umii-fatchecker-dataset'
    filename = "data/image_stacks"
    split_ratio = 0.3
    validation = False
    image_file_list, mask_file_list = get_images(ds_path)
    print(len(image_file_list), len(mask_file_list))

    # split the data
    image_stacks = load_json_file(filename)
    print(image_stacks)
    train_df, test_df = split_data(image_file_list, mask_file_list, image_stacks, split_ratio, validation)

    # Define paramaters
    batch_size = 10
    target_size = (256,256)
    seed = 1
    image_folder = "images"
    mask_folder = "masks"
    image_color_mode = "grayscale"
    image_save_prefix  = "image"
    mask_save_prefix  = "mask"
    save_to_dir = "aug"

    print(len(train_df), len(test_df))

    model = unet()

    # Create data generator for training set with data augmentation
    train_generator, train_step_size = get_train_generator(train_df[0:30], batch_size, target_size, seed, image_color_mode)
    
    model_file_location = os.path.join(os.getcwd(),"unet_lipid.hdf5")
    #model_file_location = os.path.join("../../../../shared/umii-fatchecker-dataset", "unet_lipid.hdf5")
    print(model_file_location)
    model = unet()

    # Train the model
    model_checkpoint = ModelCheckpoint(model_file_location, monitor='loss',verbose=1, save_best_only=True)
    history = model.fit(train_generator, steps_per_epoch=train_step_size, epochs=5, callbacks=[model_checkpoint])
    print(history.history)

    # with early stopping
    #es = EarlyStopping(monitor='loss', mode='min', verbose=1)
    #history = model.fit(train_generator,steps_per_epoch=train_step_size,epochs=20,callbacks=[es])
    #model.save(model_file_location)

    '''
    # Testing the model
    test_generator = get_test_generator(batch_size, target_size, test_df, image_color_mode)
    test_step_size = test_generator.n // test_generator.batch_size

    # predict and save the results as numpy array
    test_generator.reset()
    results = model.predict(test_generator, steps=test_step_size, verbose=1)

    # show the result test images
    for i in range(20):
        show_test_image_pairs(results, i)

    '''