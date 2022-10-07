import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

from unet import unet
from unet import *
from utils import *


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


if __name__ == '__main__':
    batch_size = 10
    target_size = (256,256)
    seed = 1
    image_color_mode = "grayscale"
    model_path = "../trained_models"
    model_name = "unet_lipid_3"
    model_file = os.path.join(model_path, model_name + ".hdf5")

    # load the pretrained model
    model = unet(model_file)

    # Load test_df
    test_df = pd.read_pickle("test_df.pkl")

    # Testing the model
    test_generator, test_step_size = get_test_generator(test_df, batch_size, target_size, seed, image_color_mode)

    # Predict and save the results as numpy array
    test_generator.reset()
    results = model.predict(test_generator, steps=test_step_size, verbose=1)
    print(np.max(results), np.min(results), results[0].shape)

    # Visualize the result test images from each stack
    img_lst = [0, 6, 10, 60, 85, 88, 148, 337, 451, 472, 502]
    for i in img_lst:
        plt.imsave(os.path.join("results/", "%d_predict.png"%i), results[i][:,:,0])

    # for i in range(20):
    #     show_test_image_pairs(results, i)

