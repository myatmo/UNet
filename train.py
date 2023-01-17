import os
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import tensorflow.keras.optimizers as optimizers
from losses import *
from unet import unet
from unet_3plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from dataloaders import get_train_generator
from utils import load_history_file

from neptune_config import config_run


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and masks')
    parser.add_argument('--num-epochs', '-e', dest='epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--unet-type', '-u', dest='unet_type', default='v0',
                        help='Choose UNet type vo:unet, v1:unet3+, v2:unet3+ with deep supervision, v3:unet3+ with cgm)')
    parser.add_argument('--optimizer', '-o', dest='optimizer', default='sgd',
                        help='Optimizer type: adam or sgd')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')
    #parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--train-data', '-d', dest='train_data', type=str, default='../dataset/train_df.pkl',
                        help='File name of the dataframe that contains train image paths')
    parser.add_argument('--val-data', '-f', dest='val_data', type=str, default='../dataset/val_df.pkl',
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
    input_shape = [256, 256, 1]
    output_channels = 1
    
    params = {
        "batch_size": args.batch_size,
        "train_size": target_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "loss": "BCE Dice",
        #"metrics": [dice_coef],
    }

    return args, target_size, seed, params, input_shape, output_channels

    
def train_model():
    args, target_size, seed, params, input_shape, output_channels = get_hyperparameter()

    run = config_run()
    run["parameters"] = params
    run["sys/name"] = args.model_name
    run["sys/unet_type"] = args.unet_type

    # Load train_df
    df1 = pd.read_pickle(args.train_data)
    df2 = pd.read_pickle(args.val_data)
    train_df = pd.concat([df1, df2], ignore_index=True, axis=0)
    # Create data generator for training set with data augmentation
    train_generator, train_step_size = get_train_generator(train_df, args.batch_size, target_size)
    
    # Load U-net model based on version
    if args.unet_type == 'v0':
        model = unet()
    elif args.unet_type == 'v1':
        model = UNet_3Plus(input_shape, output_channels)
    elif args.unet_type == 'v2':
        model = UNet_3Plus_DeepSup(input_shape, output_channels)
    else:
        model = UNet_3Plus_DeepSup_CGM(input_shape, output_channels)
    #model.summary()

    # Complile the model
    if args.optimizer == 'adam':
        model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=bce_dice_loss, metrics = [dice_coef])
        #model.compile(optimizer = optimizers.Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy']) # s16_2
    else:
        model.compile(optimizer = optimizers.SGD(lr=args.lr, momentum=0.90, decay=1e-6), loss=bce_dice_loss, metrics = [dice_coef])

    # Train the model
    model_file = os.path.join(args.model_path, args.model_name + ".hdf5")
    model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
    history = model.fit(train_generator, steps_per_epoch=train_step_size, epochs=args.epochs, callbacks=[model_checkpoint])
    #run["loss"].log(history.history)

    # Save model history
    history_file_name = os.path.join(args.model_path, args.model_name + "_history.npy")
    np.save(history_file_name, history.history)
    load_history_file(history_file_name, args.unet_type, run)

    # with early stopping
    #es = EarlyStopping(monitor='loss', mode='min', verbose=1)
    #history = model.fit(train_generator,steps_per_epoch=train_step_size,epochs=20,callbacks=[es])
    #model.save(model_file_location)

    run.stop()



if __name__ == '__main__':
    train_model()