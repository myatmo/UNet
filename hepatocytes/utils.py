import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.transform as trans
import json


def show_test_image_pairs(results, test_df, i, target_size=(256,256)):
  mask_test = results[i][:,:,0]
  mask_actual = skio.imread(test_df['masks'][i])
  img_actual = skio.imread(test_df['images'][i])
  mask_actual = trans.resize(mask_actual, target_size)
  fig, ax = plt.subplots(nrows=1, ncols=3)
  ax[0].imshow(mask_test, cmap='gray')
  ax[1].imshow(mask_actual, cmap='gray')
  ax[2].imshow(img_actual, cmap='gray')
  plt.show()


def get_iou_score(pred, test_df, target_size=(256,256)):
  assert len(pred) == len(test_df['masks']), "Lengths of actual and prediction images should be the same"
  iou = np.zeros(len(pred))
  for i in range(len(pred)):
    mask_test = pred[i][:,:,0]
    img = skio.imread(test_df['masks'][i])
    mask_actual = trans.resize(img, target_size, preserve_range=True, anti_aliasing=False).astype('float32')
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(mask_test, mask_actual)
    iou[i] = m.result().numpy()
  return iou


# save values in json file
def save_as_json(file_list, file_name):
  myfile = open( file_name + ".json", "w")
  json.dump(file_list, myfile, indent=6)
  myfile.close()

# load saved nmi_values list
def load_json_file(file_name):
  myfile = open(file_name + ".json")
  loaded_file = json.load(myfile)
  return loaded_file