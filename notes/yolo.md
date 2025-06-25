# V1

One stage: skip the proposal generation step. Directly predits

- Yolo, for each SxS grid cell, it predicts w,h relative to the images, and centres as the offset from the top left corder of the grid cell. Normalized betwee 0 and 1. Also a confidence score for each box. And C class probabilities. 5B+C values for each grid cell.
- select box with highest IOU with ground truth box to calculate loss.

## YOLO loss

Localization loss - ensure to learn x,y,width, height closer
Confidence loss - Confidence score - presence of object and how good of a fit. High when high overlap, lower for bad, 0 for prediced object with no object
Classification loss

all are mse losses

continue https://www.youtube.com/watch?v=TPD9AfY7AHo&t=20s
