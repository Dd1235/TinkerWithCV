# Anomaly Detection on MVTec AD (Carpet)

This repository explores methods for unsupervised anomaly detection on the MVTec AD dataset, focusing on the 'carpet' class. The project demonstrates a progression from simple autoencoders to advanced feature-based and transformer-based approaches, with clear visualizations and benchmarking.

---


## Dataset

We use the [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset, a real-world benchmark for unsupervised anomaly detection. This project focuses on the 'carpet' class, which contains high-resolution images of carpets with various types of defects (color, cut, hole, metal contamination, thread) and normal samples.

**Citation:**

If you use the dataset, please cite:

> Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
> "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
> IEEE Conference on Computer Vision and Pattern Recognition, 2019

**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([details](http://creativecommons.org/licenses/by-nc-sa/4.0/)).

---

## Methodologies

### 1. Autoencoder Baseline

- **Notebook:** [train_autoencoder.ipynb](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/train_autoencoder.ipynb)
- **Summary:** Trains a simple convolutional autoencoder directly on image pixels. The model learns to reconstruct normal images; reconstruction error (L2 loss) is used as the anomaly score. **Reconstruction loss heatmaps** are used to visualize where the model detects anomalies.
- **Result:** AUC ≈ 0.44 (worse than random). Demonstrates the limitations of pixel-space autoencoders for complex textures.

### 2. ResNet50 + KNN (Feature Memory Bank)

- **Notebook:** [resnet_knn.ipynb](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/resnet_knn.ipynb)
- **Summary:** Uses a pre-trained ResNet50 as a feature extractor. Features from normal ("good") training images are stored in a memory bank. For a test image, features are extracted and compared to the memory bank using K-nearest neighbors (KNN); the mean distance to the k closest features is the anomaly score. **Thresholding (mean + 2*std)** is used to classify anomalies. **t-SNE plots** are used to visualize feature separability.
- **Result:** AUC ≈ 0.74. Shows the power of deep features and simple non-parametric scoring.

## Model 3: Resnet backbone with autoencoder

- Trained the ResNet50 model as a feature extractor and then used an autoencoder to reconstruct the features.
- This approach gave an AUC of 0.99, but the autoencoder takes training time.
- This method is also from a paper.

## Model 4: PatchCore

- A simplified implementation of PatchCore, as described [here](https://arxiv.org/abs/2106.08265)
- It is similar to the second method, except that it uses a memory bank of patches instead of the entire image features.
- It gave an AUC of 0.98 and does not have the overhead of training time of autoencoder, simply using pretrained ResNet50.

## Model 5: ViT and KNN

- Used a pre-trained Vision Transformer (ViT) model to extract features.
- Only used the cls label embedding, to reduce training time.
- Created a memory bank of features from the training set.
- Used the k-nearest neighbors (KNN) algorithm to compute the anomaly score.
- Achieved AUC of 0.95, this is only ViT and KNN, did not use patch embeddings, no autoencoder, entirely trained on CPU because i ran out of gpu free credits, yet it performed well.

Plans:

Also looking to expand from carpets to other classes as well, and deploy model for small demo.

I also want to look into feature extractor backbone + VAE instead. I might perform better for generalized usecases where we can get image from different angles, and offsets etc.

Will also look into more VAE + FSL possibilities.

## Notebooks

- [EDA & Dataset Download](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/eda.ipynb): Exploratory data analysis, dataset structure, and citation/license details.
- [Make Dataset](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/make_dataset.ipynb): PyTorch dataset and dataloader creation.
- [Autoencoder Baseline](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/train_autoencoder.ipynb)
- [ResNet50 + KNN](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/resnet_knn.ipynb)
- [ResNet50 + Autoencoder](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/resnet_backbone.ipynb)
- [PatchCore](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/patch_core.ipynb)
- [ViT + KNN](https://github.com/Dd1235/TinkerWithCV/MVTec_AD/vit_knn.ipynb)

---

## Skills & Tools

- **Deep Learning:** PyTorch, torchvision, transformers
- **Computer Vision:** Feature extraction, autoencoders, memory banks, anomaly detection
- **Visualization:** Matplotlib, seaborn, t-SNE, heatmaps
- **Experimentation:** Jupyter Notebooks, Colab, Implementing papers, and also implementing my own ideas using the papers, like using Vision Transformer backbone instead of ResNet50 etc.
- **Reproducibility:** Dataset download, preprocessing, and clear notebook structure

---

## References

- MVTec AD Dataset: Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019. [[project page](https://www.mvtec.com/company/research/datasets/mvtec-ad)]
- PatchCore: Neumann, Lukas, et al. "PatchCore: Towards Total Recall in Industrial Anomaly Detection." CVPR 2021. [[arXiv](https://arxiv.org/abs/2106.08265)]
- [Towards Total Recall in Industrial Anomaly Detection (PatchCore)](https://arxiv.org/abs/2106.08265)

---

## Future Work

- Expand to other MVTec AD classes beyond carpets.
- Deploy a demo for real-time anomaly detection.
- Explore feature extractor backbones with variational autoencoders (VAE) and few-shot learning (FSL) approaches.

<style>#mermaid-1752251928590{font-family:"trebuchet ms",verdana,arial;font-size:16px;fill:#ccc;}#mermaid-1752251928590 .error-icon{fill:#a44141;}#mermaid-1752251928590 .error-text{fill:#ddd;stroke:#ddd;}#mermaid-1752251928590 .edge-thickness-normal{stroke-width:2px;}#mermaid-1752251928590 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1752251928590 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1752251928590 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1752251928590 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1752251928590 .marker{fill:lightgrey;}#mermaid-1752251928590 .marker.cross{stroke:lightgrey;}#mermaid-1752251928590 svg{font-family:"trebuchet ms",verdana,arial;font-size:16px;}#mermaid-1752251928590 .label{font-family:"trebuchet ms",verdana,arial;color:#ccc;}#mermaid-1752251928590 .label text{fill:#ccc;}#mermaid-1752251928590 .node rect,#mermaid-1752251928590 .node circle,#mermaid-1752251928590 .node ellipse,#mermaid-1752251928590 .node polygon,#mermaid-1752251928590 .node path{fill:#1f2020;stroke:#81B1DB;stroke-width:1px;}#mermaid-1752251928590 .node .label{text-align:center;}#mermaid-1752251928590 .node.clickable{cursor:pointer;}#mermaid-1752251928590 .arrowheadPath{fill:lightgrey;}#mermaid-1752251928590 .edgePath .path{stroke:lightgrey;stroke-width:1.5px;}#mermaid-1752251928590 .flowchart-link{stroke:lightgrey;fill:none;}#mermaid-1752251928590 .edgeLabel{background-color:hsl(0,0%,34.4117647059%);text-align:center;}#mermaid-1752251928590 .edgeLabel rect{opacity:0.5;background-color:hsl(0,0%,34.4117647059%);fill:hsl(0,0%,34.4117647059%);}#mermaid-1752251928590 .cluster rect{fill:hsl(180,1.5873015873%,28.3529411765%);stroke:rgba(255,255,255,0.25);stroke-width:1px;}#mermaid-1752251928590 .cluster text{fill:#F9FFFE;}#mermaid-1752251928590 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial;font-size:12px;background:hsl(20,1.5873015873%,12.3529411765%);border:1px solid rgba(255,255,255,0.25);border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1752251928590:root{--mermaid-font-family:sans-serif;}#mermaid-1752251928590:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1752251928590 flowchart{fill:apa;}</style>
