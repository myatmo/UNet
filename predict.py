import os
import argparse
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from utils import *
from unet import unet
from unet3 import unet3
from unet_3plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM


def get_test_generator(test_df, batch_size=3, target_size=(256,256), seed=1, image_color_mode="grayscale"):
    # Define ImageDataGenerator for testing images
    test_gen_args = dict(rescale=1./255.)
    test_datagen = ImageDataGenerator(**test_gen_args)
    test_generator = test_datagen.flow_from_dataframe(dataframe = test_df,
                                                x_col = "images",
                                                y_col = None,
                                                batch_size = batch_size,
                                                seed = 1,
                                                class_mode = None,
                                                color_mode = image_color_mode,
                                                target_size = target_size,
                                                shuffle=False)

    test_step_size = test_generator.n // test_generator.batch_size

    return test_generator, test_step_size


def get_args():
    parser = argparse.ArgumentParser(description='Test the trained model on images and masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--test-data', '-d', dest='test_data', type=str, default='data/test_df.pkl',
                        help='File name of the dataframe that contains test image paths')
    parser.add_argument('--model-path', '-p', dest='model_path', type=str, default='../trained_models',
                        help='Path to save the trained model')
    parser.add_argument('--model-name', '-n', dest='model_name', type=str, required=True,
                        help='Name of the trained model to be tested')
    
    return parser.parse_args()


def get_hyperparameter():
    args = get_args()
    target_size = (256,256)
    seed = 1
    image_color_mode = "grayscale"

    return args, target_size, seed, image_color_mode
    

def get_unet3_para():
    # U-net 3plus model
    INPUT_SHAPE = [256, 256, 1]
    OUTPUT_CHANNELS = 1


def predict_model():
    args, target_size, seed, image_color_mode = get_hyperparameter()
    # Load test_df
    test_df = pd.read_pickle(args.test_data)

    # load the pretrained model
    model_file = os.path.join(args.model_path, args.model_name + ".hdf5")
    model = unet(model_file)

    #model = UNet_3Plus(INPUT_SHAPE, OUTPUT_CHANNELS, model_file)
    #model = UNet_3Plus_DeepSup(INPUT_SHAPE, OUTPUT_CHANNELS, model_file)

    # Testing the model
    test_generator, test_step_size = get_test_generator(test_df, args.batch_size, target_size, seed, image_color_mode)

    # Predict and save the results as numpy array
    test_generator.reset()
    results = model.predict(test_generator, steps=test_step_size, verbose=1)
    #print(np.max(results), np.min(results), np.mean(results))
    #print(type(results), len(results), results[0].shape, results[1].shape)

    # Visualize the result test images from each stack
    img_lst = [0, 6, 10, 60, 85, 88, 148, 337, 451, 472, 502]
    for i in img_lst:
        plt.imsave(os.path.join("results/", args.model_name+"_%d_predict.png"%i), results[0][i][:,:,0])

    # test one image
    #plt.imsave(os.path.join("results/", model_name+"_test.png"), results[0][0][:,:,0])

    # for i in range(20):
    #     show_test_image_pairs(results, i)


if __name__ == '__main__':
    predict_model()