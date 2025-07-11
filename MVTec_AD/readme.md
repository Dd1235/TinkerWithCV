# Anomaly Detection on MVTec AD Dataset

This repository contains a collection of Jupyter notebooks that explore anomaly detection on the MVTec AD (Carpet) dataset. The primary approach demonstrated is using a feature-based method with a pre-trained ResNet model.

## Methodology

The core of the anomaly detection system is built around the idea of a "memory bank" of features from normal (non-anomalous) images.

1.  **Feature Extraction**: A pre-trained ResNet50 model is used as a feature extractor. The final classification layer is removed, and the model is used to generate feature vectors for each image.

2.  **Memory Bank Creation**: A "memory bank" is created by extracting features from all the images in the **training set**, which consists only of "good" (non-anomalous) images. This memory bank represents the normal state of the data.

3.  **Feature Selection**: To reduce dimensionality and focus on the most informative features, the standard deviation of each feature across the memory bank is calculated. The top 500 features with the highest standard deviation are selected for the final memory bank.

4.  **Anomaly Scoring**: For each test image, its features are extracted using the same ResNet model. The anomaly score is then calculated as the mean Euclidean distance between the test image's feature vector and the `k` nearest neighbors in the memory bank (in this case, `k=50`).

5.  **Thresholding**: A threshold for anomaly detection is determined by taking the mean of the reconstruction errors of the training data plus three standard deviations. Any test image with a score above this threshold is classified as an anomaly.

## Notebooks

- `MVTec_AD/resnet_feature.ipynb`: The main notebook implementing the feature-based anomaly detection method described above.
- `MVTec_AD/eda.ipynb`: Performs exploratory data analysis on the MVTec AD dataset.
- `MVTec_AD/make_dataset.ipynb`: Contains code for creating PyTorch datasets and dataloaders.
- `MVTec_AD/train_autoencoder.ipynb`: An approach using an autoencoder for anomaly detection. This uses the raw data, and trains a simple custom autoencoder on it.

## Results

The `resnet_feature.ipynb` notebook includes the following results:

- **t-SNE visualization**: A 2D t-SNE plot of the ResNet features, showing the separation of different defect classes.
- **ROC Curve**: A Receiver Operating Characteristic (ROC) curve is plotted to evaluate the performance of the anomaly detection model, achieving an AUC score of approximately 0.74.
- **Confusion Matrix**: A confusion matrix is generated to visualize the classification performance.
