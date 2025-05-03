# WiDS Hackathon 2025: ADHD and Sex Prediction

This repository contains code for the WiDS 2025 Datathon, focusing on predicting ADHD and Sex based on numerical, categorical, and MRI data.

## Overview

The project uses two machine learning models:

1. **Neural Network:** A multi-input neural network with separate branches for metadata and MRI data, merged for final prediction.
2. **LightGBM:** A gradient boosting framework model for both ADHD and Sex prediction using combined features.

## Data

The data is provided by WiDS and consists of:

* **Categorical Metadata:** Demographic and other categorical information.
* **Quantitative Metadata:** Numerical features related to the participants.
* **MRI Data:** Functional connectome matrices representing brain connectivity.
* **Labels:** Ground truth for ADHD and Sex outcomes.

## Preprocessing

* **Categorical Data:** One-hot encoding using `sklearn.preprocessing.OneHotEncoder`.
* **Numerical Data:** Imputation of missing values with the mean.
* **MRI Data:** Scaling using `sklearn.preprocessing.StandardScaler` and dimensionality reduction with PCA (`sklearn.decomposition.PCA`).

## Models

### Neural Network

* Architecture: Two input branches (metadata and MRI), dense layers with ReLU activation, dropout, batch normalization, and a merged layer for prediction.
* Hyperparameters: Optimized through experimentation (see code comments for details).
* Loss Function: Weighted binary cross-entropy for ADHD, binary cross-entropy for Sex.
* Optimizer: Adam.
* Metrics: Accuracy.

### LightGBM

* Model: `lightgbm.LGBMClassifier`.
* Hyperparameters: Optimized through experimentation (see code comments for details).
* Class Weights: Balanced using `sklearn.utils.class_weight.compute_class_weight`.

## Results

* **Neural Network:** Achieved a submission accuracy of 74%.
* **LightGBM:** Achieved a submission accuracy of 76%.

## How to Run

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`. You need to create a requirements.txt file with necessary packages first.
3. Upload the data files to the `data` folder.
4. Run the Jupyter notebook `wids_hackathon.ipynb`.

## Requirements

* Python 3.7+
* Libraries: NumPy, Pandas, Scikit-learn, TensorFlow, Keras, LightGBM, and others (see code for imports).

## Contributing

Feel free to open issues or pull requests for improvements or bug fixes.
