'''
For data preparation: split data into train, val and test set
To avoid data leakage, split data according to the image stacks.
Input: 
        dataset path; file path to images and masks
        image stacks; list of image Ids for which contain different slices of 3D image
Output:
        three pickle files for training, validation and test sets; each file contains dataframe
        that has absolute path to images and masks
To run: python load_data.py --ds-path [dataset path]
                            --image-stacks [file path + name of image image stacks file]
                            --save-path [file path to save the resulting dataframe]

To do: combine the step to creat image_stacks here
'''

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


def get_split_point(image_stacks, size, i):
    # get clean split according to image_stacks
    start, end = image_stacks[i]
    while(size < start or size > end):
        i += 1
        start, end = image_stacks[i]
    
    split = 0

    if size == end:
        split = end
    else:
        split = start
        
    return split, i


def split_data(image_file_list, mask_file_list, image_stacks, test_ratio, val_ratio):
    test_size = math.floor(len(image_file_list)*test_ratio)
    val_size = math.floor(len(image_file_list)*val_ratio)
    columns = ['images', 'masks']
    i = 0

    # make first split for testing
    split_test, i = get_split_point(image_stacks, test_size, i)
    test_images = image_file_list[0:split_test]
    test_masks = mask_file_list[0:split_test]
    test_df = pd.DataFrame(zip(test_images, test_masks), columns=columns)

    # make second split for validation
    split_val, i = get_split_point(image_stacks, val_size+split_test, i)
    val_images = image_file_list[split_test:split_val]
    val_masks = mask_file_list[split_test:split_val]
    val_df = pd.DataFrame(zip(val_images, val_masks), columns=columns)
                        
    # make the rest for training
    train_images = image_file_list[split_val:]
    train_masks = mask_file_list[split_val:]
    train_df = pd.DataFrame(zip(train_images, train_masks), columns=columns)
                    
    # convert to dataframe
    # df = pd.DataFrame(training_set)

    # shuffle the training set
    # df = df.sample(frac=1).reset_index(drop=True)

    # split training set into train and val
    # val_df = df.iloc[0:val_size].reset_index(drop=True)
    # train_df = df.iloc[val_size:].reset_index(drop=True)

    return train_df, val_df, test_df


def get_train_data(ds_path, filename, save_path, test_ratio, val_ratio):
    # Get list of image and masks
    image_file_list, mask_file_list = get_image_list(ds_path)
    assert len(image_file_list) == len(mask_file_list), "Number of images and masks files must be the same!"
    print(len(image_file_list), " images and masks found.")

    # Split the data
    print("Splitting data into train and test set...")
    image_stacks = load_json_file(filename) # to do: add throw exception if file not found
    train_df, val_df, test_df = split_data(image_file_list, mask_file_list, image_stacks, test_ratio, val_ratio)

    # save train_df and test_df
    train_df.to_pickle(os.path.join(save_path, "train_df.pkl"))
    test_df.to_pickle(os.path.join(save_path, "test_df.pkl"))
    if val_ratio > 0:
        val_df.to_pickle(os.path.join(save_path, "val_df.pkl"))
    print("Total images in Train set: ", len(train_df))
    print("Total images in Validation set: ", len(val_df))
    print("Total images in Test set: ", len(test_df))
    print("Suffessfully loaded data. See the pickle files for train, validation and test dataframe.")


def get_args():
    parser = argparse.ArgumentParser(description='Test the trained model on images and masks')
    parser.add_argument('--ds-path', '-p', dest='ds_path', type=str, default='/home/umii/shared/umii-fatchecker-dataset',
                        help='Path of the dataset')
    parser.add_argument('--image-stacks', '-n', dest='filename', type=str, default='../dataset/image_stacks',
                        help='Name of the image stacks json file')
    parser.add_argument('--save-path', '-s', dest='save_path', type=str, default='../dataset',
                        help='Name of the folder that dataframes are to be saved')
    parser.add_argument('--test-ratio', '-t', dest='test_ratio', type=float, default=0.15,
                        help='Split ratio for the test set')
    parser.add_argument('--val-ratio', '-v', dest='val_ratio', type=float, default=0.15,
                        help='Split ratio for the validation set')
    
    return parser.parse_args()


def main():
    args = get_args()
    get_train_data(args.ds_path, args.filename, args.save_path, args.test_ratio, args.val_ratio)


if __name__ == '__main__':
    main()