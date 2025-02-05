# MGSeg
# Overview
This repository contains an implementation for processing and analyzing mammogram images. The model leverages a transformer-based architecture with a segmentation pipeline to detect abnormalities.

## Model Architecture
![Final Diagram](Final%20Diagram.png)

## Structure
```
HybridSegmentation/
│── models/
│   ├── encoder.py          # Feature extraction
│   ├── decoder.py          # Segmentation process
│   ├── fdb.py              # Discrepancy analysis
│   ├── loss.py             # Loss calculation
│── preprocessing/
│   ├── preprocess.py       # Data preprocessing
│── dataset.py              # Data handling
│── train.py                # Model training
│── test.py                 # Model evaluation
│── utils.py                # Helper functions
│── requirements.txt        # Dependencies
│── README.md               # Documentation
```

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Evaluation
```bash
python test.py
