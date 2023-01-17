import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.transform as trans
import numpy as np
import json
import os


def show_test_image_pairs(mask_test, test_df, i, target_size=(256,256)):
  #mask_test = results[i][:,:,0]
  mask_actual = skio.imread(test_df['masks'][i])
  mask_actual = trans.resize(mask_actual, target_size)
  img_actual = skio.imread(test_df['images'][i])
  fig, ax = plt.subplots(nrows=1, ncols=3)
  ax[0].imshow(mask_test, cmap='gray')
  ax[1].imshow(mask_actual, cmap='gray')
  ax[2].imshow(img_actual, cmap='gray')
  plt.show()


# save values in json file
def save_as_json(file_list, file_name):
  myfile = open( file_name + ".json", "w")
  json.dump(file_list, myfile, indent=6)
  myfile.close()

# load json file
def load_json_file(file_name):
  myfile = open(file_name + ".json")
  loaded_file = json.load(myfile)
  return loaded_file


# load saved history
def load_history_file(file_name, unet_type, run):
  #myfile = os.path.join('../trained_models/', file_name)
  his = np.load(file_name, allow_pickle=True).item()
  keys = list(his.keys())
  if unet_type == 'v0' or unet_type == 'v1':
    losses = his[keys[0]]
    accuracy = his[keys[1]]

  # for v2 and v3, keys[1] is not accuracy values
  else:
    losses = his[keys[1]]
    accuracy = his[keys[6]]

  for i in range(len(losses)):
      print("Epoch: ",i+1, ", loss: ", losses[i], ", ", keys[1] ,": ", accuracy[i])
      run["train/loss"].log(losses[i])
      run["train/accuracy"].log(accuracy[i])