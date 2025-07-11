# VAE

- AutoEncoder has an encoder and decoder, train with reconstruction loss.
- AutoEncoder converts input to lower dimension, representaiton learning. Can use it for Denoising.
- Cannot generate valid image from every point in the latent space. What if each image is instead a distribution of points in the latent space? Such that different samples generate different variation of the input.
- It is not necessary for this distributions to be very wide, collapse intoa point and minimize reconstruction loss.
- Try to make these distributions close to standard guassian.
- Constrain using KL divergence between this distribution and standard guassian.

# Vision Transformers

- Convert into a sequence of tokens, patch embedding does this.
- Patch embedding converts image into sequenceo f patches. Say HxW, make into four patches, H/2xW/2. Number of patches _ D(dimension at which transformer layers will operate).
  Flatten the three channels. So number of patches _ 3 _ h/2 _ w/2. Feed to FC layer to get output of number of patches \* D

So the Patch Embedding converts image into a sequence of patches, add cls token to seqeunce of patches, and add postional information to patches.

(PS WOWWWWWW An image is worth 16x16 words is also google research, I am boggled, Google really is at the forefront of AI research, this gives me so much motivation.)

VIT
