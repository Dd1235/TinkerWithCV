# Rough

- Why?
- For fast-rcnn instead of all the selective search proposals and feeding them one by one through a CNN, you generate features for all proposals in one forward pass, and use an ROI pooling layer to extract the features for each proposal.
- Then fed to classs and box prediciton layers.
- You also incorporate SVD in the FC layers, where the weight matrix was broken down, so two smaller layers instead.
- But fast rcnn is not real time.
- Focus is on reducing proposal generation time
- Train a separate Region Proposal Network (RPN) to generate proposals (replace selective search), and pass these instead to the ROI pooling layer.
- For proposal generation we just need, foreground or background. Instead of sliding window, pass through CNN, and on feature map, get proposals.
- Need to cater to different scales and aspect ratio - so use anchor boxes.
- At each location in the feature map, generate multiple anchors with different scales and aspect ratios.
- About 3 scales, and 3 aspect ratios, so 9 anchors per location.
- Remove those anchor boxes that spill over the image boundary.
- 1000x600 image, generate feature map using cnn, final output 60x40. 9 anchor boxes at each 60x40 locations. 

- RPN: backbone CNN + 3x3 conv layer + 1x1 convolutions, to generate 2k objectness classificaiton scores for each location, and 4k box transformation parameter predictions for each location.

- Training

- Each anchor boxes needs to be assigned a ground truth label, foreground or background. IOU >= 0.7 with ground truth boxes is foreground.
- Possible the ground truth box does not have high overlap with any anchor box, to avoid, we add the one with max iou even if its less than 0.7.
- Loss = classification + localizatino Loss
- For classification, for each anchor, BCE, for localization, smooth L1 loss, and the network predicted transformation parameters.
- One stage - directly predict boxes for multiple categories using single network
-  They can have shared layers
