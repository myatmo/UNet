# Training UNet versions on Fatchecker dataset

Training Unet, Unet3+, Unet3+ with deep supervision and Unet3+ with CGM on Fatchecker dataset

## Installation

pip install -r requirements

## Data Preparation

Run load_data.py for preprocessing the dataset.

python load_data.py --ds-path '../umii-fatchecker-dataset' --image-stacks [path to the image_stacks json file] --save-path [path to save the result files]

```python
python load_data.py --ds-path '../umii-fatchecker-dataset' --image-stacks '../image_stacks' --save-path '../data'
```

This creates dataframes for training and testing images paths.

## Training

python train.py --num-epochs [default: 5] --batch-size [default: 1] --unet-type [v0, v1, v2 or v3] --optimizer [adam or sgd] --model-name [name of the model]

For unet-type, v0 is for UNet, v1 is for UNet3+, v2 is for UNet3+ with deep supervision and v3 is for UNet3+ with CGM.

```python
python train.py --num-epochs 200 --batch-size 10 --unet-type 'v0' --optimizer 'adam' --model-name 'unet_adam'
```

## Prediction

python predict.py --batch-size [default: 1] --unet-type [v0, v1, v2 or v3] --model-name [name of the trained model]

```python
python predict.py --unet-type 'v0' --model-name 'unet_adam'
```