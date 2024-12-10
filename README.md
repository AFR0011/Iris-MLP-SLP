## **Overview**

This project contains MATLAB implementations of various neural network architectures for classification tasks using the **Iris dataset**. It focuses on training and evaluating **Single-Layer Perceptrons (SLPs)** and **Multi-Layer Perceptrons (MLPs)** under different configurations and analyzing their performance.

The project is a hands-on exploration of neural network concepts and aims to enhance understanding of data preprocessing, network design, training, and evaluation metrics.

---

## **Acknowledgement**

This project was developed in October 2023 as part of a study on neural network architectures and their applications in classification problems.

---

## **Features**

### **Implemented Architectures**

1. **4x3 SLPs**: 
   - A single-layer perceptron with 4 input features and 3 output classes. 
   - Trained and evaluated on the full Iris dataset.
   
2. **4x2 SLPs**: 
   - A single-layer perceptron with 4 input features and 2 output classes. 
   - Excludes one class from the Iris dataset for binary classification.

3. **4x10x3 MLPs**:
   - A multi-layer perceptron with 4 input features, one hidden layer of 10 neurons, and 3 output classes. 
   - Demonstrates the impact of adding hidden layers.

### **Evaluation Metrics**

- Training and testing accuracy.
- Confusion matrices for detailed classification performance.
- Performance curves, including mean absolute error and best epoch analysis.

### **Hyperparameter Customization**

- Number of training epochs.
- Learning goals.
- Network structure, including the number of hidden layers and their sizes.

### **Graphical Analysis**

- Visualizes confusion matrices for training and testing data.
- Plots network performance metrics over epochs.

---

## **Directory Structure**

```
MLP-SLP/
├── Data/
│   ├── iris inputs.csv # Input features of the Iris dataset
│   └── iris classes.csv # One-hot encoded class labels of the Iris dataset
├── Code/
│   ├── SLP_4_3.m # Code for training and evaluating 4x3 SLPs
│   ├── SLP_4_2.m # Code for training and evaluating 4x2 SLPs
│   └── MLP_4_10_3.m # Code for training and evaluating 4x10x3 MLPs
└── README.md # Project documentation
```

---

## **Getting Started**

### Prerequisites

- MATLAB (R2021a or newer recommended)
- The Iris dataset in CSV format (included in the `Data/` directory)

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/AFR0011/mlp-slp-iris.git
    ```

2. Navigate to the project directory:
    ```
    cd iris-mlp-slp
    ```

3. Open MATLAB and set the current directory to the project directory.

---

## **Usage**

1. **Run any script** (`train_test_4x3_SLP.m`, `train_test_4x2_SLP.m`, or `train_test_4x10x3_MLP.m`) in MATLAB to train and test the respective neural network architecture.

2. Analyze the output metrics:
   - Training and testing accuracies.
   - Confusion matrices.
   - Performance plots.

3. Customize network parameters such as the number of epochs, learning goal, or architecture by editing the respective script.

---

## **Possible Improvements**

- Implement advanced neural network architectures like CNNs or RNNs for more complex datasets.
- Integrate cross-validation for more robust performance evaluation.
- Add hyperparameter optimization techniques such as grid search or random search.
- Use GPU acceleration for faster training of MLPs with larger datasets.

--- 

## **License**

This project is open-source and available under the MIT License. Feel free to use and modify it for educational or research purposes.
