# Autoencoders (VAE)

- Autoencoders are neural networks that learn to compress data into a low-dimensional latent space and then reconstruct the original data from it. This is an unsupervised learning technique based on minimizing reconstruction loss.

- The architecture consists of two parts: the encoder, which maps input data to a latent representation, and the decoder, which attempts to reconstruct the original data from that representation. For images, the encoder is typically built using convolutional layers, and the decoder uses transposed convolutions or upsampling layers.

- The latent representation, often called the bottleneck, acts as a compressed summary of the input. This forces the network to capture only the most essential features of the data.

- Autoencoders are essentially learned compression algorithms. Unlike classical PCA, which is linear, autoencoders can capture complex, nonlinear structures in the data.

- On datasets like MNIST, a 2D latent space can be visualized directly by plotting the embeddings and coloring them by digit class. This helps show how well the model clusters different categories. Higher-dimensional latent spaces (e.g., 5D or more) require dimensionality reduction techniques like t-SNE or PCA for visualization.

- In Variational Autoencoders (VAEs), instead of mapping each input to a single point in latent space, the encoder predicts a distribution (typically a Gaussian with mean and variance). The decoder samples from this distribution to reconstruct the input.

- The VAE loss function consists of two parts: the reconstruction loss and a KL-divergence term. The KL-divergence term encourages the latent space to follow a unit Gaussian distribution, enabling smooth interpolation and sampling of new points.

- A denoising autoencoder takes a corrupted input (e.g., an image with noise) and learns to reconstruct the clean version. This helps the model become robust to small perturbations and generalizes better.

- Neural inpainting is a task where part of the input image (a small patch or region) is removed, and the model is trained to predict the missing portion. This is a direct extension of the denoising setup and can be used for tasks like filling in occluded regions or removing watermarks.

- Autoencoders can be adapted for segmentation tasks. Instead of reconstructing the original image, the decoder is trained to output a segmentation mask. This setup is used in medical imaging, satellite imagery, and more. Architectures like U-Net build on this idea with skip connections to retain spatial details.

- The performance of an autoencoder is highly dependent on the dimensionality of the latent space. Too small a bottleneck might lead to underfitting (loss of key information), while too large a bottleneck may cause the model to simply memorize the data without meaningful compression.

- A well-behaved latent space allows for smooth interpolation between data points. For VAEs, this is one of the key advantages—since the latent space is regularized to follow a known distribution, it's possible to generate new, coherent samples by sampling from it.

- One important question to explore is how the choice of reconstruction loss (e.g., mean squared error vs binary cross entropy) affects the quality of reconstructions, especially for pixel-based image data.

- Another direction worth thinking about is how different priors (other than standard normal) or more structured latent spaces (like discrete latent variables or hierarchical VAEs) impact generative capability and interpretability.

- Finally, while AEs and VAEs focus on reconstruction, they are fundamentally different from generative adversarial networks (GANs), which learn to generate data by training a generator against a discriminator. VAEs trade off reconstruction fidelity for regularized latent space structure, while GANs focus more on sharpness and realism.
  Here’s a refined and expanded version of your notes on **Variational Autoencoders (VAEs)** written as plain bullet points. The content flows with increasing depth, combining fundamentals, technical insights, and things to explore.

### Variational Autoencoders (VAEs)

- Unlike standard autoencoders that map the input to a deterministic latent vector, VAEs learn to map the input to a **distribution** over latent variables. This allows for controlled sampling and generation of new data points.

- Specifically, for each input, the encoder outputs two vectors:

  - a **mean vector** (μ)
  - a **standard deviation (or log-variance)** vector (σ)

- From these, we sample a latent vector **z** from the predicted Gaussian distribution. However, naive sampling breaks the backpropagation chain since random sampling is non-differentiable.

- To enable gradient-based training, the **reparameterization trick** is used:

  - We rewrite the sampling process as:
    `z = μ + σ * ε`
    where `ε` is a noise vector sampled from a standard normal distribution (N(0, I))
  - This separates the stochastic part (ε) from the deterministic parameters (μ and σ), allowing gradients to flow through μ and σ during backpropagation.

- The reconstruction loss remains the same as in a basic autoencoder (e.g., mean squared error or binary cross-entropy), and it encourages the decoder to generate outputs similar to the original input.

- A new term is introduced in the VAE loss: the **KL divergence** between the learned latent distribution `q(z|x)` and the prior `p(z)`, usually chosen to be a standard normal distribution N(0, I).

  - This term regularizes the latent space and ensures that different inputs produce latent vectors that are close to a unit Gaussian.

- During training, the **total VAE loss** becomes:
  `L = Reconstruction Loss + KL Divergence`

- The variable `ε` is treated as a constant during backpropagation, so gradients are not computed through it. This simplifies the implementation and keeps the learning process stable.

- One of the key advantages of VAEs is that the latent space becomes **continuous and smooth**, making interpolation and sampling well-behaved.
- Can use VAE in _Reinforcement Learning_

### Extensions and Explorations

- **Disentangled VAEs** attempt to learn latent spaces where each dimension encodes a distinct and interpretable factor of variation (e.g., digit identity, rotation, thickness in MNIST).

- A well-known variant is **β-VAE**, where the KL divergence term is scaled by a factor `β > 1`.

  - Increasing `β` strengthens the pressure on the latent space to align with the prior distribution, often leading to better disentanglement at the cost of some reconstruction quality.

- In a well-disentangled latent space, tweaking a single latent dimension can produce interpretable changes in the output, making the model more useful for understanding underlying generative factors.

- There’s an active research area around evaluating disentanglement, defining metrics for it, and balancing reconstruction quality with interpretability.

# Resnet

- Say the task is to upscale an image, get higher resolution output.
- A lot of the times, your output starts looking completely bonkers instead of a higher resolution version.
- Think of the difference between the higher resolution vs the input as a residual that is the model better off learning.
- Residual blocks are ofcourse absolutely essential for other usecases too, just need to make sure that the dimensions batch for element wise addition.

# Distance based Anomaly Detection

eg, K = 3, take 3 nearest samples, anomaly score = mean(distance to K nearest neighbors)

- Say normal image (3, 224, 224)
- Use pretrained resnet-50 model, throw away last layer, and use the 2048 dimensional feature vector
- eg (100,3,224,224)
- so we get (100, 2048) feature vectors - memory bank
- Get the std of each feature, sort according to standard deviation in the memory bank, and choose to ignore the columns iwth low predictive power
- so (100, 500)
- In Patch core, we use KNN not on entire image, but based on the patches.
- So patch comparison
- Use all but the last layer of the resnet model
- Visualize the embeddings projecting on to a 2D place using t-SNE
-

# TODO

- [] ViT based encoder
- [] read [this](https://arxiv.org/abs/2103.04257)
