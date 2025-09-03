# 🚦 Traffic Sign Recognition using CNN & Hyperparameter Tuning

This project implements a **Traffic Sign Recognition System** using deep learning (Convolutional Neural Networks) with **Keras Tuner** for hyperparameter optimization.  
It is based on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.

---

## 📂 Dataset
We use the **GTSRB dataset**, which contains thousands of labeled traffic sign images.  

- **Source**: [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)  
- **Files**:
  - `Train.csv` → Training images and labels  
  - `Test.csv` → Test images and labels  
  - `Meta.csv` → Class metadata  

Each image comes with **Region of Interest (ROI)** coordinates to crop and normalize the traffic sign.

---

## ⚙️ Features
- Load and preprocess GTSRB dataset from Kaggle
- ROI-based cropping of traffic sign images
- Resizing images to **64×64**
- Data normalization and one-hot encoding of labels
- CNN model with **hyperparameter tuning** using `keras-tuner`
- Regularization options: L1, L2, L1_L2
- Optimizers: Adam, RMSProp, SGD
- **K-Fold Cross Validation** for model evaluation
- EarlyStopping to avoid overfitting

---

## 📦 Requirements
The following libraries are required:

```bash
numpy
pandas
matplotlib
seaborn
opencv-python
pillow
scikit-learn
tensorflow
keras-tuner
