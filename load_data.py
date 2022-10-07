import os
import glob
import math
import pandas as pd
import pickle
from utils import *

def get_image_list(ds_path):
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


def get_train_data():
    # Get list of image and masks
    image_file_list, mask_file_list = get_image_list(ds_path)
    assert len(image_file_list) == len(mask_file_list), "Number of images and masks files must be the same!"
    print(len(image_file_list), " images and masks found.")

    # Split the data
    print("Splitting data into train and test set...")
    image_stacks = load_json_file(filename)
    train_df, test_df = split_data(image_file_list, mask_file_list, image_stacks, split_ratio, validation)

    # save train_df and test_df
    train_df.to_pickle("train_df.pkl")
    test_df.to_pickle("test_df.pkl")
    print("Suffessfully loaded data. See the pickle files for train and test dataframe.")


if __name__ == '__main__':
    ds_path = '../../../shared/umii-fatchecker-dataset'
    filename = "data/image_stacks"
    split_ratio = 0.3
    validation = False

    get_train_data()


