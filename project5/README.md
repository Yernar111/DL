# Cards Images Classification

## Description

Pretrained ResNet18 model used as a feature extractor with the final classification layer adapted on the dataset that contains images of 53 different classes of cards.

## Dataset
* Kaggle link: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data
* 8157 images with 53 classes

data/
├── train/
├── test/
├── valid/


### Main Features

- Image classification
- Transfer Learning with ResNet18
- Data preprocessing
- Model evaluation
- Model checkpointing
- Plot with comparising train and validation loss
- Top-K predictions with confidence
- Inference on custom images

## Technologies

* PyTorch
* Torchvision
* PIL
* Matplotlib


