# 📧 Smart Email Spam Detection

## 📌 Project Overview

Smart Email Spam Detection is a Machine Learning and Deep Learning based project that classifies messages as **Spam** or **Ham (Not Spam)**.

The project uses **Natural Language Processing (NLP)** techniques to clean and process text data. Two different models are used for classification:

* **Random Forest** – Machine Learning model
* **BiLSTM** – Deep Learning model

The main goal of this project is to compare traditional Machine Learning and Deep Learning approaches for spam message classification.

---

## 🎯 Objectives

* Detect spam messages automatically.
* Clean and preprocess text using NLP techniques.
* Convert text into a suitable numerical representation.
* Train a Random Forest model.
* Train a BiLSTM model.
* Compare the performance of both models.
* Evaluate the models using classification metrics.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* PyTorch
* Matplotlib
* Seaborn
* NLP
* Machine Learning
* Deep Learning

---

## 📂 Dataset

The project uses a `spam.csv` dataset containing text messages and their corresponding labels.

The dataset has two classes:

| Label | Meaning                  |
| ----- | ------------------------ |
| Ham   | Normal message           |
| Spam  | Unwanted or spam message |

The labels are converted into numerical values for model training:

```text
Ham  → 0
Spam → 1
```

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
NLP Preprocessing
   ↓
Train-Test Split
   ↓
Feature / Text Representation
   ↓
 ┌─────────────────┐
 ↓                 ↓
Random Forest     BiLSTM
 ↓                 ↓
Prediction        Prediction
 └────────┬────────┘
          ↓
    Model Evaluation
          ↓
    Model Comparison
```

---

## 🧹 NLP Preprocessing

Since the input data is text, preprocessing is performed before training the models.

The main steps include:

1. Convert text into lowercase.
2. Remove unnecessary characters using Regex.
3. Remove stopwords.
4. Apply lemmatization.
5. Tokenize the text.
6. Convert text into numerical representation.

### Example

**Before preprocessing:**

```text
Congratulations!!! You WON a FREE Prize.
```

**After preprocessing:**

```text
congratulation win free prize
```

---

## 🤖 Random Forest

Random Forest is used as the Machine Learning approach.

It is an ensemble algorithm that combines multiple Decision Trees to make a final prediction.

### Flow

```text
Processed Text
      ↓
Numerical Features
      ↓
Random Forest
      ↓
Spam / Ham
```

Random Forest provides a strong traditional Machine Learning baseline for comparing its performance with the Deep Learning model.

---

## 🧠 BiLSTM

BiLSTM stands for **Bidirectional Long Short-Term Memory**.

It is used to learn patterns and context from sequential text data. Unlike a normal LSTM, BiLSTM processes the sequence in both forward and backward directions.

### BiLSTM Architecture

```text
Input Text
    ↓
Tokenization
    ↓
Numerical Sequence
    ↓
Embedding Layer
    ↓
BiLSTM
    ↓
Linear Layer
    ↓
Spam / Ham
```

The Embedding layer converts tokens into numerical vectors, and the BiLSTM learns useful patterns from the sequence.

---

## 📊 Model Evaluation

The models are evaluated using common classification metrics:

* **Accuracy** – Measures overall correct predictions.
* **Precision** – Measures how many predicted spam messages are actually spam.
* **Recall** – Measures how many actual spam messages are correctly detected.
* **F1-Score** – Provides a balance between precision and recall.
* **Confusion Matrix** – Shows correct and incorrect predictions.

The performance of **Random Forest and BiLSTM** can then be compared to understand which approach works better for the given dataset.

---

## 📁 Project Structure

```text
Smart-Email-Spam-Detection/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── spam_detection.ipynb
│
├── models/
│   ├── random_forest.pkl
│   └── bilstm_model.pth
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── random_forest.py
│   ├── bilstm.py
│   └── train.py
│
├── requirements.txt
├── main.py
└── README.md
```

> The folder structure can be modified according to the actual files in the project.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Smart-Email-Spam-Detection.git
```

Go to the project directory:

```bash
cd Smart-Email-Spam-Detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the Python application:

```bash
python main.py
```

Enter a message as input. The system will process the text and predict whether it is:

```text
Spam
```

or

```text
Ham
```

---

## 💡 Example

### Input

```text
Congratulations! You have won a free prize. Click here to claim it.
```

### Output

```text
Prediction: Spam
```

### Input

```text
Are you coming to college tomorrow?
```

### Output

```text
Prediction: Ham
```

---

## 🚀 Future Improvements

The project can be improved by:

* Using a larger and more diverse dataset.
* Performing hyperparameter tuning.
* Comparing BiLSTM with GRU and Transformer models.
* Using BERT for advanced text classification.
* Creating a Streamlit web application.
* Deploying the model using FastAPI.
* Adding real-time email classification.

---

## 📚 Key Skills

* Python
* Natural Language Processing
* Text Classification
* Data Preprocessing
* Machine Learning
* Random Forest
* Deep Learning
* BiLSTM
* PyTorch
* Scikit-learn
* NLTK
* Model Evaluation

By using **Random Forest and BiLSTM**, the project provides a comparison between traditional Machine Learning and Deep Learning approaches for classifying messages as **Spam or Ham**.
