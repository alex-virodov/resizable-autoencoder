This is a small demo of a resizable autoencoder idea. 
That is, we can train an autoencoder on small sections of images (due to translational symmetry), and then run prediction on a whole image of any size.

This avoids resizing the image and losing high-frequency information in the process. My hypothesis is that the high-frequency components contain
key information for good cell instance segmentation.