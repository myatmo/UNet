from keras import backend as K
from keras.losses import binary_crossentropy
from keras_unet_collection import losses
import tensorflow.keras.metrics as metrics


smooth = 1


def tversky(y_true, y_pred):
    alpha = 0.7
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # True Positives, False Positives & False Negatives
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1-y_pred))
    false_pos = K.sum((1-y_true) * y_pred)
    tversky_idx = (true_pos + smooth) / (true_pos + alpha * false_neg + (1-alpha) * false_pos + smooth)
    return tversky_idx


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    tversky_idx = tversky(y_true, y_pred)
    gamma = 0.75
    focalT = K.pow((1 - tversky_idx), gamma)
    return focalT


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def tp(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 


def tn(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 


# def get_iou_score(pred, test_df, target_size=(256,256)):
#   assert len(pred) == len(test_df['masks']), "Lengths of actual and prediction images should be the same"
#   iou = np.zeros(len(pred))
#   for i in range(len(pred)):
#     mask_test = pred[i][:,:,0]
#     img = skio.imread(test_df['masks'][i])
#     mask_actual = trans.resize(img, target_size, preserve_range=True, anti_aliasing=False).astype('float32')
#     m = tf.keras.metrics.MeanIoU(num_classes=2)
#     m.update_state(mask_test, mask_actual)
#     iou[i] = m.result().numpy()
#   return iou


# hybrid loss for unet3plus; fix it
def hybrid_loss(y_true, y_pred):
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou +loss_ssim