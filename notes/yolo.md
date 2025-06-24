# V1

One stage: skip the proposal generation step. Directly predits

- Yolo, for each SxS grid cell, it predicts w,h relative to the images, and centres as the offset from the top left corder of the grid cell. Normalized betwee 0 and 1. Also a confidence score for each box. And C class probabilities. 5B+C values for each grid cell. 
- select box with highest IOU with ground truth box

- Localization loss: 
    - For xi, yi, wi, and hi.
- Confidence loss
- Classification loss

continue https://www.youtube.com/watch?v=TPD9AfY7AHo&t=20s