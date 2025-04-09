# 🧠 MLP (Multilayer Perceptron) Neural Network Using PyTorch

This repository contains an implementation of a **Multilayer Perceptron (MLP)** model in **PyTorch** for predicting GDP based on various country-level features. The dataset used is a cleaned version of the CIA World Factbook dataset (`clean.csv`), which was preprocessed and uploaded for ease of experimentation and learning.

---

## 📜 Background: What is a Neural Network?

A **Neural Network** is a type of machine learning model inspired by the human brain. It consists of layers of connected nodes ("neurons") that can learn complex patterns in data. When structured in layers (input, hidden, and output), this becomes a **Multilayer Perceptron (MLP)** — one of the simplest forms of deep learning models.

### 🧠 Key Concepts:

- **Weights & Bias**: Think of weights as importance multipliers for each input. A bias allows the model to shift its predictions up or down. Together, they shape how the model makes predictions.
  
- **Learning Rate**: Controls how fast the model adjusts weights/biases. If it’s too high, it might overshoot the correct value. Too low and training becomes very slow.

- **Loss Function**: A mathematical formula to measure how far the model's predictions are from the actual results. We try to minimize this during training.

- **Gradient Descent**: A method that adjusts weights and biases to reduce the loss. Like climbing down a hill step-by-step to reach the bottom (lowest error).

---

## 📈 Why MLP?

MLPs are **universal approximators**, meaning they can learn nearly any function given enough data and time. They're particularly good for structured/tabular data, like the one used here.

---

## 📂 Dataset Used

- **Filename**: `clean.csv`
- This dataset contains cleaned and preprocessed country-level indicators.
- The **target column** is `gdp`, which the model tries to predict using other features.
- Cleaned beforehand so you can jump straight into building your neural network in PyTorch.

---

## 🧪 What This Project Does

- Loads the clean dataset
- Prepares a custom PyTorch dataset and DataLoader
- Defines a Multilayer Perceptron (MLP) using `torch.nn`
- Trains the model to predict GDP values
- Evaluates the model using Mean Squared Error and R² Score

---

## 📊 MLP Architecture


---

## 📦 Files in This Repo

| File                                 | Description                                            |
|--------------------------------------|--------------------------------------------------------|
| `MLP Implementation using PyTorch.ipynb` | Jupyter Notebook containing the full PyTorch code     |
| `clean.csv`                          | Cleaned dataset used for training and testing         |
| `README.md`                          | This documentation file                               |

---

## 🛠 How It Works – Step-by-Step

### 1. **Data Preparation**
- Load dataset using `pandas`
- Split into features (`X`) and target (`y`)
- Use `train_test_split` to split into training and test data
- Optionally scale the data using `StandardScaler`

### 2. **Custom Dataset Class**
- `factbook_data` class helps load data in a way PyTorch understands
- Makes it easier to use with DataLoaders for batch training

### 3. **Model Architecture**
- Built using PyTorch's `nn.Sequential`
- 2 hidden layers with ReLU activation functions
- 1 output neuron to predict the GDP

### 4. **Training**
- Loss function: `L1Loss` (also called Mean Absolute Error)
- Optimizer: `Adagrad` (adaptive learning rate)
- Trained for 5 epochs using a batch size of 10

### 5. **Evaluation**
- Uses test set to evaluate how well the model predicts unseen data
- Metrics used:
  - **Mean Squared Error (MSE)** – lower is better
  - **R² Score** – closer to 1 is better

---

## 📉 Example Output


---

## 📚 Requirements

Install the following Python libraries:

```bash
pip install torch pandas scikit-learn numpy
```

## Want to Learn More?
 - This repo is perfect for:

 - Students learning PyTorch

 - Developers wanting to try a simple tabular dataset

 - Anyone curious about how neural networks work under the hood

## 🧠 Fun Fact:

```text
The MLP was one of the first deep learning models invented in the 1980s! It became popular again recently thanks to modern computational power and big data.
```

## 🙌 Credits:

 - Cleaned dataset based on CIA World Factbook

 - MLP implementation inspired by classic deep learning workflows

 - Built using PyTorch

## 🔗 License
 - This project is open-source and free to use under the MIT License.
