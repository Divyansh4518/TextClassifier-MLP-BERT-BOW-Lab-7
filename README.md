# Text Classification using MLP (Bag-of-Words & Embeddings)

This repository contains an implementation of a **text classification model** using a **Multi-Layer Perceptron (MLP)**. Two feature extraction techniques are used in parallel:
- **Bag-of-Words (BoW)** with `CountVectorizer`
- **Pretrained Embeddings** from `bert-base-uncased`

The project follows a **checkpoint-based training** approach and supports continual learning with the **IMDB dataset**.

---
## 📌 **Project Overview**

### **1. Dataset Preparation**
- Splits the dataset into **train (80%)**, **validation (20%)**, and **test** sets.
- Uses **Bag-of-Words** and **BERT embeddings** as parallel text representations.

### **2. MLP Model Architecture**
- Input Layer (Depends on feature size)
- Hidden Layers: `[512 → 256 → 128 → 64]` with ReLU activations
- Output Layer: **2 neurons (binary classification)**

### **3. Training Process**
- **BoW and Embeddings models train separately** using the same MLP architecture.
- Uses **Adam optimizer** (`lr=0.001`) and **CrossEntropyLoss**.
- Implements **checkpointing** to save the best model (`checkpoint_bow.pt` & `checkpoint_embed.pt`).
- **TensorBoard Logging** for loss and accuracy tracking.

### **4. Continual Learning on IMDB Dataset**
- Loads the **best-performing checkpoint** and resumes training.
- Fine-tunes on **IMDB dataset** with `lr=0.0001`.

---
## 🚀 **Setup Instructions**

### **1. Clone the Repository**
```sh
https://github.com/Divyansh4518/TextClassifier-MLP-BERT-BOW-Lab-7.git
```

### **2. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3. Run Training**
```sh
python train.py
```

### **4. Run Evaluation**
```sh
python evaluate.py
```

### **5. Visualize Logs in TensorBoard**
```sh
tensorboard --logdir=runs
```
Then, open [http://localhost:6006/](http://localhost:6006/) in your browser.

---
## 📊 **Results & Metrics**
- **Final Accuracy** on Dataset 1 and IMDB
- **Confusion Matrix** visualization
- **Training & Validation Loss Curves**

---
## 📁 **Project Structure**
```
📂 MLP-Text-Classification
├── 📜 README.md
├── 📜 requirements.txt
├── 📜 train.py
├── 📜 evaluate.py
├── 📜 dataset_loader.py
├── 📜 model.py
├── 📜 utils.py
├── 📂 runs/  # TensorBoard logs
├── 📂 checkpoints/  # Model checkpoints
└── 📂 data/  # Training & testing datasets
```

---
## 🤖 **Future Improvements**
- Implement **Transformer-based classifiers** (e.g., `DistilBERT`, `RoBERTa`).
- Experiment with **Llama-3.1-8B** for embeddings.
- Support **multi-class text classification**.

---
## 📜 **License**
This project is licensed under the **MIT License**.

