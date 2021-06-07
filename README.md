# Unet3-Plus
Implementation of Unet3+ and validation on Cityscapes dataset.

The following article describes the architecture of Unet3+ and compares it with previously developed Unet, Unet++.

https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf

According to the article, the Unet3+ architecture has less parameteres than Unet++ and manages to **outperform it both in training time & accuracy**.

![](https://github.com/kochlisGit/Unet3-Plus/blob/main/Unet3%2B/architecture.png)

The original paper describes the detailed architecture of Unet3+ in the following image:

![](https://github.com/kochlisGit/Unet3-Plus/blob/main/Unet3%2B/in_depth.png)

The results of the accuracy of this model seem to be quite impressive. On liver and spleen dataset, it managed to outpeform Deepmind's state of the art DeepLab v3+ with quite higher accuracy:

![](https://github.com/kochlisGit/Unet3-Plus/blob/main/Unet3%2B/results.png)

# My implementation differs from the original one, as I made some improvements to the model:

1. Replaced all MaxPooling layers with **Conv2D + Strides**.
2. Replaceed all UpSampling layers with **Conv2DTranspose + Strides**.
3. Added random **Gaussian Noise** at the inputs of the model, during the training time. This aims to prevent overfitting (Requires extra training time).
4. Added dropout layers to the model (Requires extra training time).

# Data Augmentations
You can also add all sorts of data augmentation (Zooming, Shearing, Rotation, Padding, Brightness, Contrast, Saturation, etc) to images to improve the model's robustness.
