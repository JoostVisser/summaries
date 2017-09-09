# Capita Selecta notes

[TOC]

## Research



## Papers

### Fully Convolutional Networks for Semantic Segmentation

#### Definitions

**Segmentation:** a partition of an image into several "coherent" parts, without understanding what each part is.

**Semantic Segmentation:** partition the image into semantically meaningful parts + classify each part into one of the pre-determined classes.

**End-to-end learning:** Learn the whole neural network in one go, instead of dividing the network into a pipeline of smaller networks.

**Inference:** Understanding an image. (Done by feedfoward)

#### Abstract

Fully convolutional networks trained by themselves, trained end-to-end and pixels-to-pixels, exceed state-in-the-art in *semantic segmentation*.

- Train "fully convolutional" networks of arbitrary size input and gives a similar output.
- Trains very fast.

#### 1. Introduction

Prior work tried to make a prediction at every pixel, but these have shortcomings. 

Fully Convolutional Network (FCN) don't have complicated machinery and work whole-image-at-a-time by dense feedforward computation + backpropagation

- Up-sampling layers enable pixelwise prediction and learning in nets with *subsampled pooling*.

*Global information:* stores the *what* of an object.
*Local information:* stores the *where* of an object.

A **skip achitecture** has been defined to combine deep, coarse semantic info with shallow, fine appearance information. [Segmentation, not important?]

#### 2. Related Work

Skipping this part for now as the next part is more important.

#### 3. Fully convolutional networks

Convolutional network input: $h \times w \times d$.

- $h \times w$ is the pixel size of the image.
- $d$ colour channels. (RBW)

Convolutional layer is $h \times w \times d$.

- $h \times w$ are spatial dimensions, i.e. related to the location of the image.
- $d$ features. Also called **channel dimensions**.
- Locations in these layers correspond to the locations in the *input image* to which this layer is path-connected to.

These are built on *translation invariance*, meaning that their basic components operate on local input regions and only depend on *relative* spatial coordinates.