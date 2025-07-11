# End-to-End Object Detection with Transformers

- this is by Facebook AI (2020)
- architecture is much much simpler

- There can be multiple objects of the same class with overlapping bounding boxes, they can occlude each other etc.
- Image without the labels -> CNN encoder -> set of image features(scale down but make more channels) -> transformer encoder-decoder -> set of box predictions (each box has a class and bounding box)

Bipartite matching loss - lets say you get a set of predictions, (c, b), we always get N number of predictions(maximum number of objects), from database you get the real bounding box predictions, (c,b), just pad with the background class.

Ground truth labels now are also of size N.

The ordering should not be important. Compute the maximum matching. There is a loss function L((c,b), (c', b')) That is takes one prediction, and one ground truth, if they were corresponding to each other what would the loss be?

Essentially wan't to compute a minimum matching, such that the total loss is minimized.

Not dependent on the order, if you output hte same say bird multiple times, only one of them is assigned to the ground truth bird, hence encourage to output diverse bounding boxes. Use Hungarian algorithm to compute such a matching.
Make N large enough to cover all the cases.

Here we factor in IOU loss etc.

The backbone is a CNN, with it we put in some positional encoding.

- The bounding boxes can be pretty large, and the transformer architecture takes care of long range dependencies.

- You get an equal size input from the encoder(more like feature mapping), the decoder has a side input the encoder output, taken in the object queries (different from AIAYN, not auto regressively but one shot), and output another sequence, directly go through classifier ffn, the give the class label bounding box outputs.
- so just give in some N things as object queries

- The object queries are learnt, independent of the input image. So each object query is like interested in say a bounding box of a certain size at a certain position.Here the attention makes sense that these attend to each other.

- can visualize the attention.
