import os
import glob
import math
import argparse
import pandas as pd
from utils import load_json_file

def get_image_list(ds_path):
    # get all the file path name
    image_file_list = glob.glob(os.path.join(ds_path, 'images') + '/*')
    mask_file_list = glob.glob(os.path.join(ds_path, 'masks') + '/*')

    # sort the file name
    image_file_list.sort()
    mask_file_list.sort()

    return image_file_list, mask_file_list


def split_data(image_file_list, mask_file_list, image_stacks, test_ratio, val_ratio):
    test_size = math.floor(len(image_file_list)*test_ratio)
    val_size = math.floor(len(image_file_list)*val_ratio)
    start, end = image_stacks[0]
    i = 0

    # get clean split according to image_stacks
    while(test_size < start or test_size > end):
        i += 1
        start, end = image_stacks[i]
    
    split = 0

    if test_size == end:
        split = end
    else:
        split = start

    # make first split for testing, the rest for training and validation
    test_images = image_file_list[0:split]
    test_masks = mask_file_list[0:split]
    test_set = {'images': test_images,
                    'masks': test_masks}
                        
    training_images = image_file_list[split:]
    training_masks = mask_file_list[split:]
    training_set = {'images': training_images,
                    'masks': training_masks}
                    
    # convert to dataframe
    df = pd.DataFrame(training_set)
    test_df = pd.DataFrame(test_set)

    # shuffle the training set
    df = df.sample(frac=1).reset_index(drop=True)

    # split training set into train and val
    val_df = df.iloc[0:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)

    return train_df, val_df, test_df


def get_train_data(ds_path, filename, test_ratio, val_ratio):
    # Get list of image and masks
    image_file_list, mask_file_list = get_image_list(ds_path)
    assert len(image_file_list) == len(mask_file_list), "Number of images and masks files must be the same!"
    print(len(image_file_list), " images and masks found.")

    # Split the data
    print("Splitting data into train and test set...")
    image_stacks = load_json_file(filename) # to do: add throw exception if file not found
    train_df, val_df, test_df = split_data(image_file_list, mask_file_list, image_stacks, test_ratio, val_ratio)

    # save train_df and test_df
    train_df.to_pickle("train_df.pkl") # to do: add path to save the file
    test_df.to_pickle("test_df.pkl")
    if val_ratio > 0:
        val_df.to_pickle("val_df.pkl")
    print("Suffessfully loaded data. See the pickle files for train and test dataframe.")


def get_args():
    parser = argparse.ArgumentParser(description='Test the trained model on images and masks')
    parser.add_argument('--ds-path', '-p', dest='ds_path', type=str, default='/home/umii/shared/umii-fatchecker-dataset',
                        help='Path of the dataset')
    parser.add_argument('--image-stacks', '-n', dest='filename', type=str, default='data/image_stacks',
                        help='Name of the image stacks file to be loaded')
    parser.add_argument('--test-ratio', '-t', dest='test_ratio', type=float, default=0.15,
                        help='Split ratio for the test set')
    parser.add_argument('--val-ratio', '-v', dest='val_ratio', type=float, default=0.15,
                        help='Split ratio for the validation set')
    
    return parser.parse_args()


def main():
    args = get_args()
    get_train_data(args.ds_path, args.filename, args.test_ratio, args.val_ratio)


if __name__ == '__main__':
    main()