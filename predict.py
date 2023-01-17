import os
import argparse
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from utils import *
from unet import unet
from unet3 import unet3
from unet_3plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM
from dataloaders import get_test_generator

from neptune.new.types import File
from neptune_config import config_run


def get_args():
    parser = argparse.ArgumentParser(description='Test the trained model on images and masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--unet-type', '-u', dest='unet_type', default='v0',
                        help='Choose UNet type vo:unet, v1:unet3+, v2:unet3+ with deep supervision, v3:unet3+ with cgm)')
    parser.add_argument('--test-data', '-d', dest='test_data', type=str, default='../dataset/test_df.pkl',
                        help='File name of the dataframe that contains test image paths')
    parser.add_argument('--model-path', '-m', dest='model_path', type=str, default='../trained_models',
                        help='Path to the trained model')
    parser.add_argument('--model-name', '-n', dest='model_name', type=str, required=True,
                        help='Name of the trained model to be tested')
    
    return parser.parse_args()


def get_hyperparameter():
    args = get_args()
    target_size = (256,256)
    seed = 1
    input_shape = [256, 256, 1]
    output_channels = 1

    return args, target_size, seed, input_shape, output_channels


def predict_model():
    args, target_size, seed, input_shape, output_channels = get_hyperparameter()
    threshold = 0.5

    run = config_run()
    run["sys/name"] = args.model_name
    run["sys/unet_type"] = args.unet_type
    
    # Load test_df
    test_df = pd.read_pickle(args.test_data)

    # load the pretrained model
    model_file = os.path.join(args.model_path, args.model_name + ".hdf5")

    # Load U-net model based on version
    if args.unet_type == 'v0':
        model = unet(model_file)
    elif args.unet_type == 'v1':
        model = UNet_3Plus(input_shape, output_channels, model_file)
    elif args.unet_type == 'v2':
        model = UNet_3Plus_DeepSup(input_shape, output_channels, model_file)
    else:
        model = UNet_3Plus_DeepSup_CGM(input_shape, output_channels, model_file)

    # Testing the model
    test_generator, test_step_size = get_test_generator(test_df, args.batch_size, target_size, seed)

    # Predict and save the results as numpy array
    test_generator.reset()
    pred = model.predict(test_generator, steps=test_step_size, verbose=1)
    results = pred > threshold

    # Visualize the result test images from each stack
    img_lst = [0, 6, 10, 60, 85, 88, 148, 337, 451, 472, 502]
    save_path = os.path.join('results', args.unet_type, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #print(save_path, len(results), results[0].shape, results[0][:,:,0].shape)
    for i in range(len(results)):
        file_path = os.path.join(save_path, "%d.png"%i)
        plt.imsave(file_path, results[i][:,:,0])
        run["train/predictions"].log(File(file_path))

    run.stop()
    # test one image
    #plt.imsave(os.path.join("results/", model_name+"_test.png"), results[0][0][:,:,0])

    # for i in range(20):
    #     show_test_image_pairs(results, i)


if __name__ == '__main__':
    predict_model()