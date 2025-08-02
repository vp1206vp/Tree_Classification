# 🌳 Tree Species Classification using CNNs and EfficientNet

This project focuses on building deep learning models to classify different species of trees based on their leaf images. We implement and compare the performance of three models:
1. Basic Convolutional Neural Network (CNN)
2. CNN with Batch Normalization
3. EfficientNetB0 (Transfer Learning)

---

## 📁 Repository Structure

- `Tree_Species_Dataset/` – Image dataset of tree leaves organized in folders by class (Neem, Banyan, Gulmohar, etc.)
- `Tree_Classification.ipynb` – Google Colab notebook containing complete code for:
  - Data preprocessing
  - Model definitions
  - Training and evaluation
- `models/` – [Google Drive folder](https://drive.google.com/drive/folders/1p6ep9_F2eCxBcNoMVOJ503wTHgCdCyam?usp=sharing) containing the `.h5` files for all trained models:
  - `basic_cnn_tree_species.h5`
  - `improved_cnn_model.h5` (CNN with Batch Normalization)
  - `efficientnet_model.h5`

---

## 🧠 Models Overview

### 🔹 Model 1: Basic CNN
A simple 3-layer CNN with max pooling and dropout. Shows overfitting but reaches moderate accuracy.

### 🔹 Model 2: CNN with Batch Normalization
Improved version of the basic CNN with `BatchNormalization` layers after each convolution. **Best performance** on the validation set.

### 🔹 Model 3: EfficientNetB0
Pretrained EfficientNetB0 used as a feature extractor with a custom classification head. Underperforms on this dataset due to mismatch with task size and complexity.

---

## 📊 Results Summary

| Model                   | Validation Accuracy | Remarks                          |
|------------------------|---------------------|----------------------------------|
| Basic CNN              | ~32%                | Moderate overfitting             |
| CNN + BatchNorm        | **~35–40%**         | Best performance overall ✅       |
| EfficientNetB0         | ~9%                 | Underfitted; least effective     |

---

## 📌 Instructions to Run

1. **Clone this repository**
2. Download the [dataset](#) (if not already present)
3. Open [`Tree_Classification.ipynb`](#) in Google Colab
4. Upload the dataset and run all cells to train models
5. Or, download pre-trained models from this [Drive folder](https://drive.google.com/drive/folders/1p6ep9_F2eCxBcNoMVOJ503wTHgCdCyam?usp=sharing) and use them for prediction

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Google Colab
- ImageDataGenerator
- EfficientNetB0 (Transfer Learning)
- Matplotlib for visualization

---

## 📌 Future Improvements

- Improve dataset diversity and size
- Use more advanced augmentation techniques
- Try fine-tuning pretrained models instead of freezing
- Experiment with other lightweight models like MobileNet

---

## 🤝 Acknowledgments

- Dataset prepared and structured by project team
- CNN architecture inspired by best practices from Keras docs
- Pretrained weights sourced from `keras.applications`

---

## 📬 Contact

**Author:** [Your Name]  
**Email:** [your.email@example.com]  
**Institution:** [Your College/University Name]
