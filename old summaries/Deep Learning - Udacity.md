# Deep Learning - Udacity

[TOC]

## Module 1

### Course overview

First session: machine learning.

- Logistic Classification
- Stochastic Optimization
- Parameter Tuning

Second session: Deep neural networks

- Deep networks
- Regularization

Third session: Convolutional neural networks

- Convolutional neural networks

Fourth session: Text and sequences

- Embeddings
- Recurrend models

### Classification

#### Logistic classification

Goal: being able to classify objects in a supervised learning fashion. 

- We're able to perform ranking
- We can classify in images via a slider that checks whether a pedestrian is inside of it.

We will first start to build a *logistic classifier*, also known as a *linear classifier*: 
$$
f(\hat x) = \hat y
$$

- Where $f$ is a linear function, i.e. $f(x) = Wx + b$, with $W$ the *weights* and $b$ the *bias*.

More concretely, suppose we want to classify an image with three classes. We will then have a weight vector for *each class* as well as a bias for each class. We want the *probability* for the correct class be close to 1, while all the other classes should be close to 0.

1. We multiply the training example with the weights and add the bias to get a score for each class.
2. We transform these scores via *softmax* to probabilities.

**Softmax:** Transforms a vector (of classes, for example) into probability values via the formula:
$$
\frac{e^\hat {x_i}}{\sum_j e^\hat {x_j}}
$$


- If you multiply each value with a factor, say 10, then the probabilities will get closer to 0 and 1 because of the exponent factor.
- Dividing it will get it closer to the uniform distribution.

But... how to compare this to the actual result? This is done via **one-hot-encoding**.
**One-hot-encoding:** Have a vector with a single 1 for the correct class and 0 for the others.

This way, we can compare how well we perform by comparing the result of the softmax together with the actual result. How do we do that in a normal manner? The first approach we can think of is *mean squared error*, but this has a problem of learning slow-down when at the edges. Therefore, we will use the **cross-entropy** function. Note that it's not symmetric!
$$
D(S,L) = - \sum_i L_i \log S_i
$$
This whole setting is called **multinomial logistic classification**. $D(S(W\hat x + b), L)$.

#### How to get the parameters?

So, our goal is to get our weights $W$ and bias $b$ such that our classifier predicts $\hat y$ correctly according to the input $\hat x$. 

What we can do is measure the distance for all training data and all available labels. This is called the *training loss* and is one humongeous function:
$$
\ell(\hat x, y) = \frac 1 N \sum_i  D(S(W\hat x + b), y)
$$
The simplest way to solve this problem and try to get a low value is gradient descent. Take the derivate of the loss with respect to the parameters and follow that derivative by taking a step backwards.

The derivative will be done via a *black box* method for now, but the mathematics will be elaborated in the future. We can update the weights and biases accordingly:

$w \leftarrow w - \alpha \Delta _w \ell$

#### Instability

However, a problem that occurs in python is when working with very large and small numbers. In this case, you can get incorrect results because of numerical instability and floating point errors.

Since our loss function is large and complex, with a lot of parameters, we want to avoid this issue. One way to do so is to keep our input value roughly around 0:
$$
\mu (\hat x_i) = 0 \\
\sigma(\hat x_i) = \sigma(\hat x_j)
$$
This is why we **normalize** the input values to 0 mean and standard deviation $\sigma$.

In a similar topic, how do we initialize the weights and biasses? There are many fancy ways to do so, but one way is:

- Sample the weights and biases from a normal distribution $x \sim N(0, \sigma)$.
  - Large sigma means large peaks. Because of softmax, means it's very certain.
  - Small sigma? Because of softmax, it means the distribution is very uncertain.