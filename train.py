import os
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import tensorflow.keras.optimizers as optimizers
from losses import *
from unet import unet


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
    parser = argparse.ArgumentParser(description='Train the model on images and masks')
    parser.add_argument('--num-epochs', '-e', dest='epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')
    #parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--train-data', '-d', dest='train_data', type=str, default='data/train_df.pkl',
                        help='File name of the dataframe that contains train image paths')
    parser.add_argument('--model-path', '-p', dest='model_path', type=str, default='../trained_models',
                        help='Path to save the trained model')
    parser.add_argument('--model-name', '-n', dest='model_name', type=str, required=True,
                        help='Name of the trained model to be saved')
    
    return parser.parse_args()


def get_hyperparameter():
    args = get_args()    
    # Define paramaters for train generator
    target_size = (256,256)
    seed = 1
    image_color_mode = "grayscale"
    #image_folder = "images"
    #mask_folder = "masks"
    #save_to_dir = "aug"
    return args, target_size, seed, image_color_mode

    
def train_model():
    args, target_size, seed, image_color_mode = get_hyperparameter()
    # Load train_df
    train_df = pd.read_pickle(args.train_data)
    # Create data generator for training set with data augmentation
    train_generator, train_step_size = get_train_generator(train_df, args.batch_size, target_size, seed, image_color_mode)
    
    # U-net model
    model = unet()
    #model.summary()

    # Complile the model
    #model.compile(optimizer=tfa.optimizers.Adam(learning_rate=args.lr), loss=tfa.losses.sigmoid_focal_crossentropy, metrics=['accuracy'])
    model.compile(optimizer = optimizers.SGD(lr=0.01, momentum=0.90, decay=1e-6), loss=bce_dice_loss, metrics = [dice_coef])

    # Train the model
    model_file = os.path.join(args.model_path, args.model_name + ".hdf5")
    model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
    history = model.fit(train_generator, steps_per_epoch=train_step_size, epochs=args.epochs, callbacks=[model_checkpoint])

    # Save model history
    history_file_name = os.path.join(args.model_path, args.model_name + "_history.npy")
    np.save(history_file_name, history.history)

    # with early stopping
    #es = EarlyStopping(monitor='loss', mode='min', verbose=1)
    #history = model.fit(train_generator,steps_per_epoch=train_step_size,epochs=20,callbacks=[es])
    #model.save(model_file_location)


if __name__ == '__main__':
    train_model()