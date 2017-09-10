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

$\mathbf x_{ij}$ is the data vector at location $(i, j)$. 
$\mathbf y_{ij}$ is the output of this vector, i.e. the data vector for the following layer:
$$
\mathbf y_{ij} = f_{ks}(\{\mathbf x_{s_i + \delta_i,s_j + \delta_j}\}, \forall  0 \leq \delta_i , \delta_j \leq k) 
$$

- $s$ is the stride
- $\delta$ is the index that grabs the current local receptive field of size $k \times k$.
- $f_{ks}$ determines the layer time:
  - Matrix multiplication for standard convolution, i.e. logistic regression
  - Spatial max for max pooling
  - etc

General deep neural network: compute a nonlinear function.
Net with only convolutional layers: compute a nonlinear *filter* (i.e. heatmap).

Still, what loss function to use? How to compute $\ell(\mathbf x; \theta)$?

- Apparently, if we can write the loss function as a sum of the loss of each individual pixel, i.e. $\ell(\mathbf x; \theta) = \sum_{ij} \ell'(\mathbf x_{ij};\theta)$, then its gradient will be a sum over the gradients of each of its spatial components.
  Therefore: SGD on whole image $=$ SGD on sum of each receptive field.
  Thus we can do mini-batch on each of the final layer receptive fields.

When these receptive fields overlap significantly, feedforward and backpropagation are much more efficient layer-by-layer instead of patch-by-patch.

However, there is a problem. 
We want *dense predictions*, a prediction of a probability of a certain class (softmax, One-Hot encoding). Instead, we get *coarse predictions*, a heatmap as output.

So, how to convert these classical nets with dense layers to FCNs with coarse layers? We still want pixelwise prediction, so we need to transform the coarse output back to pixels.

- We can use fast scanning, elaborated in 3.2.
- We can use deconvolution layers for upsampling, elaborated in Section 3.3.
- We can use patchwise sampling, shown in Section 3.4.

##### 3.1. Adaping classifiers for dense networks

**Idea:** We can consider fully connected layer as a convolution with a kernel of the entire image, as all neurons are connected to the whole image creating a massive convolution.

- If we consider the networks this way, then these layers transform into a *fully convolutional layer* that take input of any size and output classification maps.
- We can use convolution loss functions layer-by-layer to speedup the computation considerably.

For example, it's considerly faster when using AlexNet.

This reinterpretation to FCNs yields output maps for inputs of any size, but generally the output dimensions are reduced by subsampling.

- Keep filters small, make computation reasonable.

##### 3.2 Shift-and-stitch is filter rarefaction

*Rarefaction* is similar to *coarsening*, making less dense.

**Goal:** Connect coarse convolutional layer to dense pixels. 

>  I don't fully understand this part.

The idea here is to upsample dense predictions, done as follows:

- Increase the number of pixels by shifting them a few places.
- Process all these $f^2$ inputs ($f =$ downsample factor) and interlace the outputs so that the predictions correspond to the pixels at the *center* of their receptive fields.
- Instead of increasing the cost by $f^2$, we can use an a trous algorithm.

##### 3.3 Upsampling is backwards strided convolution

**Goal:** Connect coarse convolutional layer to dense pixels. 

What about simply upsampling by interpolation? 

For example, simple linear interpolation:

- Each output $y_{ij}$ from the nearest four inputs by a linear map that depends only on *relative positions* of input and output cells.

Upsampling with factor $f$ is like convolution with a *fractional input stride* of $\frac 1 f$.

- Called **backwards convolution** or **deconvolution** with an output stride of $f$.
- Forwardpass is similar to a backward pass in a convo-network and vice versa.
- Note that it doesn't have to be biliniar upsampling, but this can be learned by the network. (How?)

##### 3.4 Patchwise training is loss sampling

Here's an idea: what if instead of upsampling we perform training on patches on the image. These patches consists of all receptive fields of the units below the loss for an image.

Still, full convolutional training is similar to patch-training, as the patches in such FCNs are the local receptive fields of the units below the loss for an image.

But, it has some advantages over FCNs:

- More efficient than uniform sampling of batches.
  - Con: Reduces the number of batches. 


- Can correct class imbalance.
  - Con: Fully FCNs can mitigate this by weighting the loss, however.
- Can mitigate the spatial correlation of dense patches.
  - Con: Fully FCNs can use loss sampling to address spatial correlation.
- If the patches have overlap, then using FCNs can still speedup computation.
  - Con: Fully FCNs have more of a computational speedup.

#### 4. Segmentation Architecture

Here we change succesful ILSVRC (ImageNet) classifiers to FCNs. These are then used for segmentation and are used for the PASCAL VOC 2011 segmentation challenge.

##### 4.1 From classifier to dense FCN

They consider three different classifiers, of which we will treat VGG 16-layer net in depth. The nets are transformed to FCNs as follows:

1. Discard the final classifier layer.
2. Convert all fully connected layers to convoultions.
3. Append $1 \times 1$ convolution layer with 21 features to predict scores for each of the PASCAL classes.

##### 4.2 Combining What and Where

**Problem:** We throw detailed information away if we keep downsampling using pooling layers.

- The detailed information is the *what*, whereas the shallow information is *where*.
- This limits the scale of detail in the upsampled output.

**Solution:** Add skips after certain shallow layers to combine both information!

- Turns the topology from line $\rightarrow$ DAG.
- Combines fine and coarse layers $\implies$ can make local predictions that respect global structure.

We compare three different architecture:

- FCN-32s, with no skip layers. Upsamples stride 32.
- FCN-16s, combine predictions from both final layer and stride 16 layer for finer details.
- FCN-8s, additional predictions from the stride 8 layer for further precision.





## Recap Neural Networks

### Stanford Tutorial

#### Linear Regression

Goal: Predict a target value $y$ starting from a vector of input values $x \in \mathbb R^n$. (*Regression*)

- Example: predict housing prices given the features (e.g. # of bedrooms, garden?) of a house.
- Features of $i$-th example is denoted as $x^{(i)}$. 
- How do we do this? Find a function $h$, where $y=h(x)$, such that $y^{(i)} \approx h(x^{(i)})$.

We use the function $h_\theta(x) = \sum_j\theta_jx_j = \theta^Tx$, which is a linear function; a sum of a parameter for each feature $\theta_j$ times the feature $x_j$ (whether it's in there, for example).

In particular, we want to minimize a certain *cost function*, that measures how close $y^{(i)} \approx h(x^{(i)})$:
$$
J(\theta) = \frac 1 2 \sum_i \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2=\frac 1 2 \sum_i \left( \theta^Tx^{(i)} - y^{(i)} \right )^2
$$

- This is the *mean-squared error* divided by 2 for easy differencing.

We use Gradient Descent to minimize this function, which takes small steps in the opposite direciton of the derivative of the cost with respect to each variable.

Therefore, to minimize the cost function, we need to know the cost $J(\theta)$ as well as its derivative $\nabla_tJ(\theta)$ with respect to each $\theta_j$. Here, the derivative for each $\theta_j$ is:
$$
\frac{\partial J(\theta)}{\partial \theta_j} = \sum_{i} x_j^{(i)}\left( h_\theta(x^{(i)}) - y^{(i)}\right)
$$

#### Logistic regression

Goal: predict a discrete variable instead of continuous. (*Classification*)

A linear function does not make much sense, as the distances between classes, say 0 and 1, don't have the same meaning as a linear distance. Another idea: squash the high and low values from $\theta^T x$ into a probability from $0$ to $1$! This can be done by the sigmoid function:
$$
P(y=1|x) = h_\theta(x) = \frac 1 {1 + e^{-\theta^Tx}} \equiv \sigma(\theta^Tx) \\
P(y=0|x) =1- h_\theta(x) = 1-\sigma(\theta^Tx)
$$
However, this changes the cost we have to use. This is done by maximizing the *log likelihood*.
$$
J(\theta) = - \sum_i\left(y^{(i)}\log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1-h_\theta(x^{(i)}))\right)
$$

- Notice that only one of the two terms in the summation is non-zero.
- We can check to which class an input $x$ belongs by checking whether $h_t(x) > 0.5$, since then $P(y=1 | x) > P(y=0 | x)$.

The partial derivative of the cost with respect to the variable is:
$$
\frac{\partial J(\theta)}{\partial \theta_j} \sum_i x_j^{(i)}(h_\theta(x^{(i)})-y^{(i)})
$$

- This is essentially the same as the gradient for linear regression, except that now $h_\theta(x) = \sigma(\theta^Tx)$.

#### Neural networks

...

#### Convolutional network

**Problem:** Using fully connected neural networks works well on small images, but on large images this becomes computationally unfeasible. Suppose we have a $96 \times 96$ input image and we want to learn $100$ features. Then we need $10^6$ parameters and training would be $\pm100$ times slower!

**Simple solution:** Restrict the connections between input --> hidden units to connect only a small subset of input units. Say only $3 \times 3$. This is similar to how biology does it.

Property of an image: due to spatiality, it has the property of *stationary*. This means that any features that we learn at one part of the image can be applied to other parts as well.

- More concretely, if we learn a feature somewhere in the image, then we can apply this everywhere in the image.


- As an example, suppose we have a $96\times 96$ image and we learn features of patches from $8 \times 8$. This would result in a $89 \times 89$ feature if the *stride* is 1. Suppose we have $100$ features that we want to learn. Now we only have $8 \times 8 \times 100 = 6400$ paramters instead of $1\ 000\ 000$ parameters!

#### Autoencoders

What if you only have unlabeled training examples? Then we can use an **autoencoder** neural network.

**Goal:** Learn from the structure of the data.

- The autoencoder tries to learn a function $h_{W,b}(x) \approx x$. 

  - This is generally very easy to do, but we place some constraints on the network such as constraining the number of hidden nodes. 
    The network is forced to learn a *compressed* representation of the input.

  - Another constraint we can impose on the network is a *sparsity* constraint.

    - We want to constraint the neurons to be *inactive* most of the time.

    - A measure for the sparsity of the network is as follows:
      $$
      \hat p_j = \frac 1 m \sum_{i=1}^m\left[a_j^{(2)}(x^{(i)})\right]
      $$
      We would like to (approximately) enforce that:
      $$
      \hat \rho _j = \rho
      $$
      Where $\rho$ is a sparsitiy parameter, such as $\rho=0.05$. 

    - This can be achieved by adding an extra penalty on the cost, based on the *Kullback-Leibler (KL)* divergence.