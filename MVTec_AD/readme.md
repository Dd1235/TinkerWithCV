# Anomaly Detection on MVTec AD Dataset

This repository contains a collection of Jupyter notebooks that explore anomaly detection on the MVTec AD (Carpet) dataset.

Work in Progress, looking into adding model that use Vision Transformer for feature extraction, etc.
Also looking to expand from carpets to other classes as well, and deploy model for small demo.

I also want to look into feature extractor backbone + VAE instead. I might perform better for generalized usecases where we can get image from different angles, and offsets etc.

Will also look into more VAE + FSL possibilities.

## Model 1: autoencoder

- Trained a simple autoencoder model, did not use a feature extractor.
- It performed worse than random classifier, with an AUC score of 0.44.

## Model 2: Resnet and KNN

- Used a pre-trained ResNet50 backbone to extract the feature maps.
- Created a memory bank like in the Patch-core paper to store the features of the training images.
- Used the k-nearest neighbors (KNN) algorithm to compute the anomaly score.
- Achieved an AUC score of approximately 0.74, which is significantly better than the autoencoder approach.

## Model 3: Resnet backbone with autoencoder

- Trained the ResNet50 model as a feature extractor and then used an autoencoder to reconstruct the features.
- This approach gave an AUC of 0.99, but the autoencoder takes training time.

## Model 4: PatchCore

- A simplified implementation of PatchCore, as described [here](https://arxiv.org/abs/2106.08265)
- It is similar to the second method, except that it uses a memory bank of patches instead of the entire image features.
- It gave an AUC of 0.98 and does not have the overhead of training time of autoencoder, simply using pretrained ResNet50.

## Methodology: Model 2

The core of the anomaly detection system is built around the idea of a "memory bank" of features from normal (non-anomalous) images.

1.  **Feature Extraction**: A pre-trained ResNet50 model is used as a feature extractor. The final classification layer is removed, and the model is used to generate feature vectors for each image.

2.  **Memory Bank Creation**: A "memory bank" is created by extracting features from all the images in the **training set**, which consists only of "good" (non-anomalous) images. This memory bank represents the normal state of the data.

3.  **Feature Selection**: To reduce dimensionality and focus on the most informative features, the standard deviation of each feature across the memory bank is calculated. The top 500 features with the highest standard deviation are selected for the final memory bank.

4.  **Anomaly Scoring**: For each test image, its features are extracted using the same ResNet model. The anomaly score is then calculated as the mean Euclidean distance between the test image's feature vector and the `k` nearest neighbors in the memory bank (in this case, `k=50`).

5.  **Thresholding**: A threshold for anomaly detection is determined by taking the mean of the reconstruction errors of the training data plus three standard deviations. Any test image with a score above this threshold is classified as an anomaly.

## Notebooks

- `MVTec_AD/resnet_knn.ipynb`: The main notebook implementing the feature-based anomaly detection method described above.
- `MVTec_AD/eda.ipynb`: Performs exploratory data analysis on the MVTec AD dataset.
- `MVTec_AD/make_dataset.ipynb`: Contains code for creating PyTorch datasets and dataloaders.
- `MVTec_AD/train_autoencoder.ipynb`: An approach using an autoencoder for anomaly detection. This uses the raw data, and trains a simple custom autoencoder on it.
- `MVTec_AD/resnet_backbone.ipynb`: An approach using a ResNet backbone as a feature extractor, followed by an autoencoder for reconstruction.
- `MVTec_AD/patchcore.ipynb`: A simplified implementation of the PatchCore method for anomaly detection.
