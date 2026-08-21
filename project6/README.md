# Sports Images Classification

## Description

Sports images classification using a pretrained model as a feature extractor with the final classification layer adapted on this dataset

## Dataset
* Kaggle link: https://www.kaggle.com/datasets/gpiosenka/sports-classification/data
* 14.5k images with 100 classes

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
- REST API

## Train

* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Epochs:** 15
* **BatchSize:** 64
* **Learning rate:** 0.001
* **Weight_decay:** 0.001
* **Patience:** 3

## Results

* **Model:** ResNet18(fc)
* **Accuracy:** 0.888
* **F1-macro:** 0.889

## Technologies

* PyTorch
* Torchvision
* PIL
* Matplotlib
* FastAPI
* Uvicorn


