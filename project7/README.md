# Butterfly Species Classification

## Description

Butterfly species classification using a pretrained model as a feature extractor with the final classification layer adapted on this dataset. 

## Project Structure 
 
```
project7/
├── data/
│   ├── check/
│   └── test/
│   └── train/
│   └── Training_set.csv
│   └── Testing_set.csv
├── models/ 
├── src/
│   ├── api.py
│   ├── config.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict1.py
│   └── model.py
├── requirements.txt
└── README.md
```

## Data

- Kaggle link: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/data
- 6499 images and 75 classes


## Train

| Parameter | Value |
|----------|----------|
| Epochs | 30 |
| Patience (early stopping) | 3 |
| Batch Size | 64 |
| Optimizer | Adam (only model.fc) |
| Learning Rate | 0.001 |
| Weight Decay | 1e-3 |
| Image Size | 224x224 |

## Results

| Model | Accuracy | F1-macro |
|--------|------------------|----------|
| ResNet18(fc) | 0.860 | 0.854 |


## Technologies

- PyTorch
- Torchvision
- PIL
- Pandas
- FastAPI
- Uvicorn

### How to use

```bash
git clone https://github.com/Yernar111/DL.git
cd DL/project7

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

![alt text](<loss_plot1.png>)

