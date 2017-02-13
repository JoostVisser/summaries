# Neural Networks and Deep Learning

[TOC]

Summary of the information on [this lovely website](http://neuralnetworksanddeeplearning.com).

## Chapter 1 - Using Neural Nets

### Perceptrons

A perceptron is a *binary threshold neuron*, i.e. it a neuron takes several inputs $x_1, x_2, x_3$ and each of these inputs also have *weights:* $w_1, w_2, w_3$ , which are real numbers expressing the importance of the inputs.

The neuron outputs:

- Output = 1 if $\sum_{j}{w_jx_j} > \text{threshold}$ 
- Output = 0 if $\sum_{j}{w_jx_j} \leq \text{threshold}$ 

In this case, the input $x_j$ can only be 1 or 0. So basically, we can assign weights to all inputs to indicate their importance and if it's above a certain threshold we'll output 1, otherwise we'll output a 0.

We can build fancy NAND gates with this and we can even combine multiple layers of *perceptrons*. Since we can make NAND gates, we can make almost anything, e.g. adding two bits to each-other with carry bit.

Furthermore, for making things simpeler, we can use a bias (b) instead of the threshold and a dot product:

- Output = 1 if $w \cdot x + b > 0$
- Output = 0 if $w \cdot x + b \leq 0$

What would happen if we could automatically tune the weights depending on the mistakes that are made? Then we'd device a **learning algorithm**.

### Sigmoid neurons

Idea: when we make a small change in the *weights*, we want to have a small change in the *output* so that we'll come closer to the requirements for our output.
​:arrow_right_hook: Not possible with perceptrons, as a small change can flip the output from e.g. 0 to 1.

**Sigmoid neuron (logistic neurons):** A neuron that outputs using a sigmoid function.

**Sigmoid function:** $\sigma(z)=\frac{1}{1+e^{-z}}$ 

- In this case, we use $z=w\cdot x  + b$, i.e. the sum of all inputs times their weights plus the bias.
- Output = 0,5 if $z=0$
  - Output is close to 0 if $z$ is really small
  - Output is close to 1 if $z$ is really large.
- Notice that inputs of $x$ can take on any value *between* 0 and 1.


- Small changes in weight $\Delta w$ cause a small change in their output $\Delta \text{output}$.

Why use the sigmoid as *activation function* $f(z)$? 

- Easy to differentiate and have lovely properties. :D
- Can and will use other functions.

### The architecture of neural networks

Some terminology:

- *Input neurons* - Leftmost layer, i.e. the inputs layer $x_i$.
- *Output neurons* - Rightmost layer, the neurons in the output layer.
- *Hidden layer* - Middle layer(s), so not an input layer nor an output layer.

Design of input and output layer is generally straightforward, but how to design a hidden layer?

- Several design heuristics for this.

*Feedforward neural networks*: Output of one layer is used as input for next layer. **No loops.**

*Recurrent neural networks*: Neural networks with feedback loops. Neurons fire for a limited duration of time before becoming quiet.

### A simple network to classify handwritten digits

First, we need to segment the handwritten digits into several segments. This is also called a *segmantation problem*, which is a very challenging problem for computers to solve.

Then we need to classify drawings as individual digits. $\leftarrow$ We'll focus on this problem.

- 10 output units for the numbers 0, 1, ..., 9.
  Why not 4 output units? Can encode 16 solutions with that.
  - With 4 output units, then the hidden units before a say the first output unit must try to find representations and evidence for 1, 3, 5, 7 and 9! Not only is this harder, it also makes the output fire with less confidence than if we could have different output neurons for each of these numbers.
  - We could create an additional layer where we could encode the 0-9 to 4 output neurons if we really need this.

### Learning with gradient descent

- Input is $x$, output is $y(x)$.

#### Cost function and gradient descent

For the *cost function* we start with the *Mean Squared Error (MSE)*, therefore:

$C(w,b)=\frac{1}{2n} \sum_{x}{||y(x)-a||^2}$

- $a$ is target output
- n is number of training inputs
- b is bias
- Difference between $y(x)$ and $a$ large? Then large cost. 
- Same holds for small difference between $y(x)$ and $a$.
- Is a smooth function, therefore we can make small changes in the weights and biases to get an improvement in the cost.
- We can use *gradient descent* to minimize this cost function. 
  - Why not use calculus to minimize the cost function? Difficult to do with loads of variables.
  - **Idea**: Take small steps towards the bottom of the cost function valley.
  - Formula for the small step taken: $\Delta C \approx \frac{\partial C}{\partial v_1} \Delta v_1 + \frac{\partial C}{\partial v_2} \Delta v_2$  
    - (Notice that $\frac{\partial C}{\partial v_1}$ stands for the partial derivative of the cost with respect to C, i.e. the slope of $v_1$)
    - In words: How much C changes = how much $v_1$ changes times its slope + how much $v_2$ changes times its slope.
  - To make things easier, we'll vectorize everything $\Delta v = (\Delta v_1, \Delta v_2)^T$ and the gradient of C to be the vector of the partial derivatives. $\nabla C = (\frac{\partial C}{\partial v_1}, \frac{\partial C}{\partial v_2})^T$
    - What does $\nabla C$ mean? A single mathematical object, i.e. a vector. "Hey, $\nabla C$ is a gradient vector!"
  - Therefore, we can rewrite the cost formula as: $\Delta C \approx \nabla C \cdot \Delta v$
  - What to pick as $\Delta v$, i.e. how much v should change? We'll be using $-\eta \nabla C$, therefore the change $v \rightarrow v′ = v − \eta \nabla C$. 
    - So we compute the gradient and then move in the opposite direction, i.e. falling down.

Goal: train a neural network to find weights and biases, $w$ and $b$, which minimize the quadratic cost function $C(w,b)$.

#### Apply gradient descent in neural network

**Idea:** Use gradient descent to find the weights $w_k$ and biases $b_l$ which minimize the MSE cost function.

Update functions:

- $w_k \rightarrow w'_k=w_k - \eta \frac{\partial C}{\partial w_k}$
- $b_l \rightarrow b'_l=b_l - \eta \frac{\partial C}{\partial b_l}$

Challenges: to compute the partial derivatives of the cost function, i.e. calculate $\frac{\partial C}{\partial w_k}$, we need to average over the costs of each individual training example. Therefore, we need to compute the gradient $\nabla C_x$ separately for each training input $x$ and then average over them.

**Speedup idea:**

- Estimate gradient $\nabla C$ by computing $\nabla C_x$ for a small sample of randomly chosen training inputs.
- **Mini-batch:** grab a small number $m$ of randomly chosen training inputs $X_1, X_2, ..., X_m$.
- **Normally:** Use all training inputs $x$ to compute the cost $C$ and try to minimize this cost by using this cost to update the weights.
  **Now:** Use some training inputs to estimate cost $C$. In formula:
  $\frac 1 m \sum_{j=1}^m{\nabla C_{X_j}} \approx \frac 1 n \sum_x{\nabla C_x}=\nabla C$ne
- New update rule function:
  - $w_k \rightarrow w'_k=w_k - \frac \eta m \sum_j{\frac{\partial C_{X_j}}{\partial w_k}}$
  - $b_l \rightarrow b'_l=b_l - \frac \eta m \sum_j{\frac{\partial C_{X_j}}{\partial b_l}}$

After we've exhausted the training inputs, it is said that we have completed an **epoch** of training. After that we'll start with a new training epoch.

**Tip:** sometimes it's better for the cost function to omit the averaging, i.e. the factor $\frac 1 n$. Particularly useful when the total number of training examples isn't known in advance.

- Therefore, the update rule for mini-batch gradient descent can sometimes omit the $\frac 1 m $ term.

### Notes - Implementing NN to classify digits

1. No bias for the input layer, as these are only used to compute the outputs from later layers on.
2. np.weights[1] are the weights connecting the second and third layers of neurons.
   1. We're gonna notate $w_{jk}^3$ as the weights connecting the $k^{th}$ neuron in the second layer to the $j^{th}$ neuron in the third layer. 
      Why this strange ordering of $j$ and $k$? Outputs of $k^{th}$ neuron in the second layer are now stored as column vectors - easy to compute the activations of the next layer of neurons.
3. Generally: More hidden neurons $\rightarrow$ better result
4. Neural network intuition: breaking down a very complicated question into very simple questions answerable at the level of single pixels.
   - **Deep neural networks:** Networks with multiple hidden layers
     *Hard to train* using our current methods, but new set of techniques have been developed that enable learning in deep neural nets.

## Chapter 2 - Backpropagation

How to compute the gradient of the cost function? We'll do that using an algorithm known as *backpropagation*.

Notation: 

- We'll use $w^l_{jk}$ to denote the weight:
  From the $k^\text{th}$ neuron in the $(l-1)^\text{th}$ layer $\rightarrow$ To the $j^\text{th}$ neuron in the $l^\text{th}$ layer.
- We'll use $b_j^l$ to denote the bias of the $j^\text{th}$ neuron in the $l^\text{th}$ layer.
- We'll use $a_j^l$ to denote the activation of the $j^\text{th}$ neuron in the $l^\text{th}$ layer.
- $L=$ number of layers in the network. 

New formula for the activation is:

- $a^l_j = \sigma(\sum_k w^l_{jk} \cdot a^{l-1}_k + b^{l-1}_j)$

Which can be rewritten as:

- $a^l = \sigma(w^l a^{l-1} + b^l)=\sigma(z^l)$
  where $z^l=w^l a^{l-1} + b^l$ which is the *weighted input*.

That's why we write it as $w^l_{jk}$, because otherwise we had to transpose $w$.

### Assumptions about cost function

#### First assumption

**Assumption 1:** cost function can written as an average over cost functions $C_x$ for individual training examples, i.e. $C=\frac 1 n \sum_x C_x$. 

This is the case for e.g. MSE, where the cost for a single training example is $C_x = \frac 1 2 || y - a^L || ^ 2 $ and the total cost is the average of the cost for all training examples.

**Result:** We can compute the partial derivative for a single example, i.e. $\frac{\partial C_x}{\partial w}$. Because of this, we'll drop the subscript $x$ to make it easier notation-wise.

#### Second assumption

**Assumption 2:** The cost can be written as a function of the outputs from the neural network.
I.e. $\text{cost }C = C(a^L)$

### Hadamard product

The hadamard product is an *element-wise* product of two vectors.

$s \odot t$ becomes a single vector where each element equals $s_j t_j$. 

### Four fundamental equations of backpropagation

We introduce the *error* $\delta_j^l$, which gives the error of the $j^\text{th}$ neuron in the $i^\text{th}$ layer.
$\delta_j^l= \frac{\partial C}{\partial z_j^l}$

- If the $error$, i.e. how quickly $C$ changes when we change $z$ (the slope), has a large value, than the cost can be lowered quite a bit by choosing a $z$ value in the opposite direction of this slope, i.e. *down-hill*. 
  - [Note that $a=\sigma(z)$, so if the slope of z is large, then it'll be large for $a$ as well, hence still a long way to go for minimizing the slope. So we can edit $z$ with $\Delta z$ such that it'll go to the right direction.]
  - Why not change $a$ instead? Changing $z$ is mathematically easier.

#### Equation 1: error of the output layer $\delta ^L$

**Error** $\delta_j^L  = \frac{\partial C}{ \partial a^L_j} \sigma ' (z_j^L)$

- So the error of neuron $j$ = how fast the cost is changing as a function of the $j^\text{th}$ output activation * 
  How fast the activation function $\sigma$ is changing at $z^L_j$. 

**Matrix form:** $\delta^L=\nabla_a C \odot \sigma ' (z^L)$, where $\nabla_aC$ is a vector of partial cost derivatives, i.e. rate of change of C w.r.t. output.

#### Equation 2: Equation of error $\delta ^L$ in terms of the error of the next layer, $\delta ^{l+1}$

$\delta^l = ((w^{l+1})^T \delta^{l+1} ) \odot \sigma'(z^l)$

We are basically moving the error *backwards* through the network. So we multiply the weight vector transposed to the error of the next layer, and then multiply it with the Hadamard product.

Using **Equation 1** and **Equation 2**, we can now compute the errors of any layer and neuron!

#### Equation 3: Rate of change of cost w.r.t. any bias

$\frac{\partial C}{\partial b_j^l} = \delta^l_j$

So the error is exactly the rate of change of the bias!

#### Equation 4: Rate of change of cost w.r.t. any weight

$\frac{\partial C}{\partial w_j^{jk}} = a^{l-1}_k \delta^l_j$

In a better understandable way, it's: $\frac{\partial C}{\partial w} = a_{out} \delta_{in}$.

Beautiful, isn't it? We just need to multiply the activation of the output neuron with the error of the input neuron. 

### Insights

The sigmoid function becomes very flat at both ends of the graph. Therefore, the output neuron has *saturated*, so a weight will learn very slowly if there's either high or low saturation.

### Backpropagation algorithm step

1. **Input** - Set $a^1$ to the corresponding input of $x$.
2. **Feedforward** - For each layer $l=2, 3, ..., L$ compute $z^l = w^la^{l-1} + b^l $ and $\sigma(z^l)$.
3. **Output error $\delta^L$**  - Compute the vector $\delta^L = \nabla_a C \odot \sigma'(z^L)$.
4. **Backpropagate the error** - For each $l= L-1, L-2, ..., 2$ compute $\delta^l = ((w^{l+1})^T \delta^{l+1} ) \odot \sigma'(z^l)$.
5. **Output** - Gradient of the cost function is given by $\frac{\partial C}{\partial b_j^l} = \delta^l_j$ and $\frac{\partial C}{\partial w_j^{jk}} = a^{l-1}_k \delta^l_j$.

### Mini-batch gradient descent step

1. **Input a set of training examples**
2. **For each training example $x$** - Perform steps 1-4 in the [previous section](#backpropagation-algorithm-step) to obtain $d^{x,l}$ for each $x$.
3. **Gradient descent** - For each $l = L, L-1, ..., 2$, update the weights according to the rules:
   - $w^l \rightarrow w^l - \frac \eta m \sum_x {\delta^{x,l} (a^{x,l-1})}^T$
   - $b^l \rightarrow b^l - \frac \eta m \sum_x {\delta^{x,l}} $

## Chapter 3 - Improving how NNs learn

Neural networks with the MSE cost function have a problem: when it's badly wrong, it only changes its weights a little compared to when it's a little wrong (towards the right answer).

- For a single neuron with input $x=1$ and desired output $y=0$, we get the following formulas for cost derivate w.t.r. weight, note that $C_{MSE}=\frac{(y-a)^2}2$:
  - $\frac{\partial C}{\partial w} = (a-y)\sigma'(z)x = a \sigma'(z)$ 					[Since $y = 0$ and $x = 1$]
    - $\frac{\partial C}{\partial b} = (a-y)\sigma'(z) = a \sigma'(z)$ 				[Since $y = 0$ and $x = 1$]
- Since $\sigma(x)$ has an S shaped curve, when the output it either close to $a=1$, we get a very flat shape, therefore $\sigma'(z)$ gets really small and doesn't learn that fast.

### Cross-entropy cost function

To address the slow-down, we'll be using the cross-entropy cost function:

$C=-\frac 1 n \sum_x [y \ln a + (1-y) \ln{(1-a)}]$

Or for many output neurons: $C=-\frac 1 n \sum_x \sum_j [y_j \ln a^L_j + (1-y_j) \ln{(1-a^L_j)}]$

It has the two desired properties to be a cost function:

- It's non-negative, since all logarithms will be negative, therefore the sum will be negative. Because there's a minus sign in front of the sum, the total cost will be positive.
- The closer $a$ gets to $y$, the lower the cost function is. 
  - If $y=1$, then $\ln a$ will be added to the cost. An $a$ closer to 1 $\rightarrow \ln a$ closer to 0. 
  - If $y=0$, then $\ln {(1-a)}$ will be added to the cost. As $a$ gets closer to 0, $\ln {(1-a)}$ will get lower.

This results in a new partial derivative as follows:

- $\frac{\partial C}{\partial w_j} = \frac 1 n \sum_x x_j (\sigma(z)-y)$

So, the rate at which the weight learns is controlled by $\sigma(z) - y$, i.e. the error by the output. The larger the error, the faster the neuron will learn.

Cross-entropy is roughly the *measure of surprise*.

### Softmax

Instead of applying the sigmoid function to the output, we'll apply the softmax function to the output:

$a_j^L=\frac{e^{z_j^L}}{\sum_k e^{z_k^L}}$

- Sum of output of all activations of a neuron in the layer is 1, therefore a softmax layer outputs a *probability distribution*.
- Exponentials ensure that all all output activations are positive.
- Output activation of current unit = sum of weighted inputs of current neuron ($z$) / sum of weighted inputs of all neurons. 
- *Feel:* Rescaling $z^L_j$ and squishing them together to form a probability distribution. 
- **Non-locality** - As the output $a_j^L$ depends on all 

New cost: **log-likelihood cost**

$C = - \ln a_y^L$

- Suppose we have a training point of $(x, y)$. Then $a_y^L$ is the activation of the neuron that predicts $y$, e.g. a '7' in case of MNIST. 
- If it does a good job, e.g. $a_y^L = 0.9$, then the log will be quite low hence the cost will be low.
- If it does a bad job, e.g. $a_y^L = 0.2$, then the log will be bigger, thus the cost will increase.

Using *softmax* neurons with the log-likelihood cost is roughly similar as using sigmoid neurons with the cross-entropy function.

*General principle:* Softmax is worth using whenever you want to interpret the output activations as probabilities.

Expression for backpropagation:

- $\delta_j^L = a_j^L - y_j$

### Overfitting and regularization

**Overfitting:** Letting the neural net learn all peculiarities of the training set, while not generalizing well to the test set.

Use a *validation set*! :)

- We compute the classification accuracy on the validation data at the end of each epoch. Once the classification accuracy has saturated (i.e. flattened), we stop training. This technique is called *early stopping*.
- We can also use the validation set to determine a set of good values for the hyper-parameters. Sometimes called *hold-out* method.

How to reduce overfitting?

- Increase the size of the training data.
- **Regularization**

### Regularization

#### Weight decay / L2 regularization

Add an extra term to the cost function, called the *regularization term*. This technique is also known as *weight decay* or *L2 regularization*.

- General formula: $C=C_0 + \frac \lambda {2n} \sum_w w^2$.


- For cross-entropy cost: $C=-\frac 1 n \sum_{xj} [y_j \ln a^L_j + (1-y_j) \ln{(1-a^L_j)}] + \frac \lambda {2n} \sum_w w^2$
- $\lambda$ is known as the *regularization parameter*. 
- This term doesn't include biases.
- **Idea:** The network prefers to learn small weights. Large weights are only allowed if iit considerably improves the first part of the cost function.

New learning rule for the weights:

- $w^l \rightarrow (1- \frac {\eta \lambda} n)w^l - \frac \eta m \sum_x {\delta^{x,l} (a^{x,l-1})}^T$
- The new $(1- \frac {\eta \lambda} n)$ is also known as *weight decay*, since it makes the weights smaller.

Another advantage: 

- Doesn't get stuck as much at *local minima*.
- Why? 
  - If the cost function is unregularized, then the length of the weight vector is likely to grow. Over time this can lead the weight vector to be very large and since changes due to gradient descent only make tiny changes to the direction, it won't change the direction very much.

Why does this help?

- Smaller weights make the model simpler, and a point of view in science is to say that we should go with the simpler explanation unless compelled not to.
- Think of it as a way of making it so single pieces of evidence don't matter too much to the output of the network. These networks respond to types of evidence which are seen often across the training set.
- *However,* not always is the simple solution preferred over complex solutions.

### Other regularization techniques

**L1 regularization:** Similar to *L2 regularization*, but a different regularization term.

$C=C_0 + \frac \lambda n \sum_w |w|$

- In L1 regularization, the weights shrink by a constant amount toward 0.  In L2 regularization, the weights shrink by an amount which is proportional to w.  
- Tends to concentrate the weights of the network in a relatively small number of high-importance connections, whereas the other weights are driven towards 0.

**Dropout:** Randomly & temporarily delete half of the hidden neurons and train on the modified network for the mini-batch. Then randomly / temporarily delete half of the hidden neurons again and do the same.

- **Idea:** Imagine training three or five different neural networks for the same problem. We could then use a voting scheme and if three of them are classifying a digit as a "3", then it's probably a "3". $\rightarrow$ Similar idea for the dropout procedure.
- We can think of dropout as a way of making sure that the model is robust to the loss of any individual piece of evidence.
- Especially useful in training large and deep networks, where the problem of overfitting is acute.

**Artificially expanding the training data:** Obtaining more training data via artificially modifying the training data. 

- In our case, translating / rotating the images.

*General note:* 

- More training data can compensate for differences in the machine learning algorithm used.
- We want both better algorithms *and* better training data.

### Weight initialization

With the weight initialization we used earlier, randomly picking from mean 0 and standard deviation 1, h as some disadvantages:

- An neuron in the hidden layer will have a pretty big distribution for $z$ with a very large standard deviation (with thousand input neurons, then $\mu=0$ and $sd=22.4$) , so either $z \gg 1$ or $z\ll1$. 
- Therefore, the output $\sigma(z)$ will be either 1 or 0 and our hidden neuron will have *saturated*; making small changes in the weights will only affect the activation of the hidden neuron a very tiny bit.

New *initialization*: 

$\mu = 0, sd = \frac 1 {\sqrt {n_{in}}}$ for all input weights of neuron $n$.

- This way, we'll squash the Gaussian distribution which a much smaller distribution.
- Bias can be initialized as the same way as before. Or to 0. It doesn't really matter as long as the weights don't saturate.
- **Result:** We end up much faster to a better score.

Other techniques exist, but this one works good enough.

### Choosing hyperparameters

There are a large number of parameters to choose from. How can we choose these parameters properly?

#### Broad strategy

- First, solve an easier problem that requires less training data. $\rightarrow$ Faster experimentation.
  - Can cut down hidden layers. 
  - Can cut down output layers, e.g. only looking at 0s and 1s.
  - Use less training data / and less test data / monitor and validate more often.
- First figure out a single parameter, say $\eta$, and try to improve your score with it.
  - After it's done, continue with the next parameter, say $\lambda$.
  - Can also increase to more neurons and gradually use more training / test data.

####Specific recommendations

**Learning rate:** Start with a learning rate for which to cost will likely decrease, not increase or oscillate. This can generally be a small learning rate, say $\eta=0.01$.

- If the costs decreases during the first few epochs, you can try to increase $\eta=0.1, 0.3, 1.0, ...$, until you find a value where $\eta$ oscillates / increases, i.e. the maximum.
- If the cost oscillates or increases, try to lower the learning rate, $\eta = 0.001, 0.0001, ...$ until you find a value where $\eta$ decreases the cost.
- You can use a value for $\eta$ that is smaller, say, a factor of two below the threshold.
- Why on the training set and not on the validation? No real reason, personal preference. In practice, it's unlikely to make much difference which criterion to use, as its primary purpose is to control the step size in gradient descent.

**Early stopping to determine number of training epochs:**

- At the end of each epoch, compute the classification accuracy on the validation data.
  ​:arrow_right_hook: Stops improving over some time? $\rightarrow$ Terminate
  - Can use no-improvement-in-ten rule.
  - What if the networks plateaus for a while, after which it starts improving again?
    - Can change it into no-improvement-in-twenty or something.

**Learning rate schedule:** What if you want to vary the learning rate?

1. Hold learning rate constant until validation accuracy starts to get worse. Then decrease by a factor of e.g. 2 or 10, until say ~1024 or ~128.

**Mini-batch size:** How to set the mini-batch size?

- Quite a bit faster than online learning. While the online learning updates faster, the learning rate of mini-batch will be higher. Mini-batch can also make use of a vectorized implementation.
- *Relative independent* to the other hyper-parameters
- Plot the validation accuracy versus the *real time* it takes, see which gives you the most rapid improvement in performance.

**Automated techniques:** You can also use an automated technique for optimizing these hyper-paramters. This can be done using e.g. a *grid search*, which systematically searches through a grid in hyper-parameter space. Another method would be a Bayesian approach to optimize these parameters.

### Other techniques

#### Variations on stochastic gradient descent

**Hessian technique:** Using Taylor's theorem, we can approximate the cost function near a point $w$. Using a hessian matrix $H$, whose $jk^{\text{th}}$ entry is $\frac{\partial ^2 C}{\partial w_j \partial w_k}$. 

- Advantage: Converges faster, can avoid some traps of gradient descent.
- Disadvantage: size of Hessian matrix is big, e.g. #weights x #weights

**Momentum-based gradient descent:** 

- Introduces a notion of "velocity" for the learning rate.
- Introduces a "friction" term which gradually reduces the velocity.
- $\mu$ controls how much friction there is.
- New update rules:
  - $v \rightarrow v' = \mu v - \eta \nabla C$
  - $w \rightarrow w' = w + v'$

**Other techniques:** There are also other promising techniques that show promising results, but there isn't a universally accepted model at the moment.

### Other models of artificial neuron

**Tanh neurons:** Instead of using the sigmoid function, these neurons use the hyperbolic tangent function called *tanh*. Acts very similar as a sigmoid neuron.

**Rectified linear neurons:** 0 if under the threshold, otherwise it'll linearly go up, like in SVMs.

## Chapter 4 - "Proof" that NNs can compute any function

No matter what function, there is guaranteed to be a neural network that forr every possible input, $x$, the value $f(x)$ is the output from the network.

This even holds for many inputs, $f=f(x_1, ..., x_m)$, many outputs and even with a single hidden layer! Neural Networks have a kind of *universality*.

Since almost anything can be thought of as a function (translate Chinese into English, or find the song of a small music tune) NNs can be considered really powerful.

### Two caveats

1. Neural Networks can only get an *approximation*, not an exact function. It generally holds: more hidden neurons $\rightarrow$ better approximation.
2. Only *continuous functions* can be approximated. If there's a sudden sharp jump then it's generally not possible to approximate this using a NN.

### Visual approach - Written in text

If you increase the bias, the output of the neuron goes left, while the shape stays the same.
Decrease bias? $\rightarrow$ Output goes to the right, shape stays the same.

Increase input weight? $\rightarrow$ Curve becomes steeper (and moves a tad to the left)
Decrease input weight? $\rightarrow$ Curve becomes flatter (and moves a tad to the right)

Very high weight value results in a sort of "step function", where if $x$ is greater than some value (depending on the bias) it will output 1, else 0.

This step value happends at $s=\frac{-b} w$.

Using two neurons, with their step values $s$, you can make "bumps" if you set the weight of the lower $s$ (i.e. lower activation) to $+h$ and the weight of the other $s$ to $-h$. So you can make a bump of arbitrary height and width with two neurons.

We can then subdivide an interval (say [0,1]) into $N$ subintervals and use $N$ pairs of hidden neurons to set up peaks of any desired height.

As this is the output of the weighted average $z=\sum_j w_j a_j$, we still don't have a proper output (since the output is $\sigma(z)$, not $z$. $\rightarrow$ Design a hidden layer with weighted output of $\sigma^{-1} \circ f(x)$.

### Many input and output variables

For multiple input neurons, you can design part of the function for each input neuron separately. For all the hidden neurons that are used by the first neurons, put an output weight of 0 for the rest of the neurons so they won't interfere.

This way, we can create N-dimensional "bumps" for each dimension separately. In 3D, we can basically build a "tower" function. This can be used to approximate arbitrary functions just by adding up many towers of different height and different locations.

How to make those "tower" functions? Set the height really high and set the bias such that it'll result in 1 only if both x and y are above the threshold. (Output bias $= (-m + \frac 1 2) h)$.)

The output of all of these towers can be computed using a second hidden layers. These can then be combined to a single output. 

What about multiple output variables? Just create a network for each function separately. I.e. one for approximating $f^1$, another one for $f^2$ and so forth.

This uses two hidden layers, but it's also possible with one hidden layer and "circular towers".

#### Other neurons

The same proof holds for other neurons as well, as long as we get a step function or use a linear rectified neuron.

#### Window of failure

There is a slight window of failure since the sigmoid neuron is not an exact step function, just an approximation. But if we'd create two networks, one of which computes the normal bumps and the other one computes the bumps shifted by half the width of the bump, we get a much better overall approximation. We could even do this with a large number of $M$ overlapping approximations.

#### Why deep neural nets

But if a single hidden layer can compute everything, why use multiple hidden layers?

- The world is pretty hierarchical. If we want to solve these problems, it seems useful to learn the hierarchies of knowledge that are required for solving these problems. (I.e. first learn to recognize what assertions are, then what rumours are. :P)

Deep neural networks do a better job at learning such hierarchies.

## Chapter 5 - Why are deep NNs hard to train?

Consider designing a computer with just two layers of circuits, while you can use an infinite number of AND and NAND gates. It turns out that you can compute *any* function with just two layers. However:

- Normally we design sub-tasks first, which we then split up in even smaller subtasks. This results in more than two layers.
- Some functions require exponentially more circuit elements using a shallow circuit compared to a deep circuit.

A similar idea holds nowadays for Neural Networks. That's why this chapter will go deeper into multiple layers of neural networks; that these are actually much harder because different layers are learning at vastly different speeds.

### The vanishing gradient problem

If we test on the data of MNIST, we see that adding hidden layers does not really help for the classification accuracy. In fact, it can get a tad worse if you add even more layers, which is strange as in the worst case these hidden layers can simply do nothing.

If we plot the gradient $\frac{\partial C}{\partial b}=\delta$ for each neuron in the hidden layer as bars, we see that the bars in the second hidden layer are mostly much larger (i.e. a higher bias / error, so learning more) than the bars in the first hidden layer.

Suppose we denote each bias-gradient / error as $\delta_j^l=\frac{\partial C}{\partial b^l_j}​$ ($j^\text{th}​$ neuron in the $i^\text{th}​$ layer), then we have the vector $\delta^1​$ for all the errors in the first layer, $\delta^2​$ for the second layer and so forth. (Remember that using [gradient descent](#Apply gradient descent in neural network), a part of the error is used for the update, hence it's an indication of the speed of learning.)

The length $\|\delta^1\|$ can then roughly indicate the speed of learning. If we do this for multiple layers, we can see for e.g. 4 layers that: $\|\delta^1\|=0.003, \|\delta^2\|=0.017, \|\delta^3\|=0.070,$ and $\|\delta^4\|=0.285$, thus there is a pattern: early layers learn slower than later layers.

Plotting graphs for these speeds for all the hidden layers shows that it always holds that the early hidden layers learn much more slowly than later hidden layers.

The phenomenon of that the gradient tends to get smaller as we move backward through the hidden layer is called the *vanishing gradient problem*. There is an alternative: sometimes the gradient gets much larger in earlier layers. This is called the *exploding gradient problem* and it's not much better than the former problem.

### Cause of the vanishing gradient problem

To find the cause of this vanishing gradient problem, let us consider a neural network with three hidden layers, each having a single neuron in their layer. The expression for the error of the first hidden layer is:

$\delta^1=\frac{\partial C}{\partial b_1}= \sigma'(z_1)\cdot w_2 \cdot \sigma'(z_2) \cdot w_3 \cdot \sigma'(z_3) \cdot w_4 \cdot \sigma'(z_4) \cdot \frac{\partial C}{\partial a_4}$

So of we make a change to $b_1$, say $\Delta b_1$, it causes a chain reaction by changing the value of $a_1, z_2, a_2, z_3, a_3, z_4, \text{and finally } a_4$. In detail, $\Delta a_2= \frac{\partial a_1}{\partial b_1} \cdot \Delta b_1 = \sigma'(z_1) \cdot \Delta b_1$, which in turn changes $z_2$ to $\sigma'(z_1)w_2 \Delta b_1$. 

Keeping going at this fashion:

- At each neuron we pick up a $\sigma'(z_j)$ term
- Through each weight we pick up a $w_j$ term.

#### Why does the vanishing gradient happen?

Notice that each weight initially satisfy $w_j < 1$. Also note that the derivative $s'(0)= \frac 1 4$ is the maximum value of the derivative of the sigmoid function.

Then we'll get that $|\sigma'(z_j)w_j|<0.25$. When we take a product of many such terms, the product will tend to exponentially decrease ($0.25^3$).

Comparing $\frac{\partial C}{\partial b_1}$ with  $\frac{\partial C}{\partial b_3}$ we'll get:

$\frac{\partial C}{\partial b_1} = \sigma'(z_1) \cdot [<0.25] \cdot [< 0.25] \cdot w_4 \sigma'(z_4) \frac{\partial C}{\partial a_4}$

$\frac{\partial C}{\partial b_3} = \sigma'(z_3) \cdot w_4 \sigma'(z_4) \frac{\partial C}{\partial a_4}$

(We've substituted the parts that are smaller than 0.25 with [<0.25]).

We can see that $\frac{\partial C}{\partial b_1}$ will be a factor of 16 smaller than $\frac{\partial C}{\partial b_3}$, hence we have the vanishing gradient. 

What happens of the weights get larger than 1? Then the first error will grow exponentially quicker, getting the exploding gradient problem.

#### The unstable gradient problem

The fundamental problem here isn't the exploding or vanishing gradient. It's that the gradient in early layers is the product of terms form all the later layers. The only way all layers can learn at close to the same speed is if all those products of terms come close to balancing out. 

Without any mechanism or reason for balancing, it's highly unlikely to happen by chance. So, the neural network suffers from an *unstable gradient problem*. So:

- Standard gradient-based techniques $\rightarrow$ Different layers will tend to learn at different speeds.

#### Sigmoid having a helping hand in the vanishing gradient

If we don't want to have a vanishing gradient, we want to have $|w \sigma'(z)| \geq 1$. However, $\sigma'(z)$ is also dependent on $w$. 

- If we make $w$ really large, we make $\sigma'(z)$ really small because:
  - $\sigma'(z) = \sigma(wa+b)$, so $w$ is in this sigmoid.
  - $\sigma(z)$ looks like an S-curve where it gets really flat at large values of $z$.

In fact, after careful mathematics we get that the range for which this happens of $|w|\approx 6.9$. So it's very likely that the vanishing gradient problem will not happen.

### Other obstacles to deep learning

- Glorot and Bengio found in 2010 evidence that suggests that using sigmoid activation functions can cause the the activations in the final hidden layer to saturate near 0 early in training.
- Making good choices for random weights and momentum schedule make a substantial difference in the ability to train deep neural networks.

## Chapter 6 - Deep learning

### Convolutional neural networks

One of the most used type of deep neural network. 

Note that in the earlier chapters we used a fully connected neural network (*Tabula rasa*). However:

- We treat pixels close to each-other exactly the same as pixels far from each-other.
- It would be better to take advantage of the spatial structure of the data.

Convolutional neural networks use three basic ideas:

1. *Local receptive fields*
2. *Shared weights*
3. *Pooling*

#### Local receptive fields

Instead of connecting each input neuron to each neuron in the neural network, we'll take advantage of the spatial structure of the image. We do this by connecting localized regions to a hidden neuron. 

- In the example of MNIST, we'll connect a 5x5 pixel region to a single hidden neuron.

The region in the input image is called the *local receptive fields* for the hidden neuron.

- **Idea:** "The hidden neuron learns to analyze its particular local receptive field"

We'll slide this local receptive field across the entire input image.

- If we slide per pixel, then every possible local receptive field will have a hidden neuron. 
- Different # of pixels for sliding can be used. The *stride length* can be e.g. two pixels.

#### Shared weights and biases

Each of the 5x5 outgoing weights to a hidden neuron will be *exactly the same* for all hidden neurons! So for the 28x28 neurons to the 24x24 hidden layers, we'll keep track of only 5x5 weights! Why?

- **Idea:** All neurons in the first hidden layer detect exactly the same feature, just at different *locations* in the input of the image. 
- **Example:** Suppose this hidden layer (or more concretely, the *shared weights*) detects a cat, then it doesn't matter where the cat in the picture of a cat in the image is, it will still detect the cat.
- The map from input layer to the hidden layer is called a *feature map*.
  - Weights defining the feature map is called *shared weights*, or sometimes the *kernel* or *filter*.
  - Bias defining the feature map is called the *shared bias*.

However, we can only detect a single kind of localized feature this way. Fret not, as we'll need more than one feature map, therefore:

- A complete convolutional layer consists of several different feature maps.

We'll use convolutional layers with 20 and 40 feature maps later on. Thus, we will have $20 \times 24 \times 24$ neurons in the second layer.

**Advantages:** 

- We're learning things related to the spatial structure.
- Greatly reduces the number of parameters involved in a convolutional network. 
  - MNIST: 20 feature maps requires $20 \cdot (25+1)=520$ parameters vs a fully connected network with 30 hidden neurons of $784\cdot30+30=23550$ parameters. 
    (Not a completely correct example as they're different in essential ways.)

**Disadvantage:** It's difficult to see what these feature detectors are learning.

Why is it called a convolution network?

The formula for the activation of the $j^\text{th}, k^\text{th}$ hidden neuron (where $j$ and $k$ comes from the location in the 24x24 hidden layer) is:

$\sigma(b + \sum_{l=0}^4 \sum_{m=0}^4 w_{l,m} a_{j+l, k+m})$, i.e. the activation of each input times its weight of the shared weight ($w$) of the feature map. This equation is sometimes known as a *convolution*, and rewritten as $a^1 = \sigma(b+w * a^0)$, where:

- $a^1$ it the set of output activations from one feature map.
- $a^0$ is the set of input activations.
- $*$ is called the convolution operator

#### Pooling layers

Pooling layers are usually used immediately after convolutional layers.
**Goal:** Simplify the information in the output from the convolutional layer.

It does so by taking each feature map output, which is the activation of the hidden neurons output from this layer, and maps them to a condensed feature map.

- Example: summarize a region of $2 \times 2$ neurons in the previous layer to a new neuron.

**Max-pooling:** A pooling unit simply outputs the maximum activation in the $2 \times 2$ region.

The idea for this network is to ask whether a given feature is found anywhere in a region of the image. Once it has been found, its exact location isn't as important as its rough location.

**L2 pooling:** The square root of the sum of squares, i.e. $\sqrt{\sum_i act_i^2}$.

- Another way of condensing information from the convolutional layer.

#### Putting it all together - Final layer

The final layer of connections is a fully-connected layer. So, every neuron from the max-pooled layer connects to every one of the 10 output neurons.

In total, we have the following architecture:

1. 28x28 input neurons. $\leftarrow$ Used to encode the pixel intensities for MNIST.
2. 3x24x24 hidden feature neurons. $\leftarrow$ A 5x5 local receptive field for each neuron.
3. 3x12x12 max-pooling layer $\leftarrow$ Max-summary of 2x2 region of convolutional layer.
4. 10 output neurons $\leftarrow$ Output and classifications of the neuron.


*Small note:* one can see the convolutional and pooling layer as a single layer to some extent.

#### Implementation

When using the implementation in the book, we get the following results:

1. Normal neural network with 100 hidden neurons, with softmax at the end results in 97.80% on ` test_data `.
2. Convoluted neural network with single convolution-pooling layer, after which its fully connected to 100 hidden sigmoid neurons and finally 10 softmax output neurons results in 98.78% accuracy. 
3. Adding a second convolution-pooling layer after the first one with as input the 20x12x12 neurons of the activations of the first convolution-pooling layer results in an accuracy of 99.06% accuracy.

Using *tanh* instead of sigmoid neurons results in a similar result, but trains a little faster.

#### Using rectified linear units

**Rectified linear neurons:** Rectified linear neurons have as output 0 if the sum of the input is under the threshold (bias), otherwise it activates linearly w.r.t. the output. $f(z) \equiv \max(0, z)$

4. Using the same model as 3 with two convo-pool layers, while changing the neurons from sigmoid to linear results in an accuracy of 99.23%
5. Expanding the training data from 50 000 images to 250 000 images increases the accuracy to 99.37%.
6. Inserting an extra fully-connected layer results in an accuracy of 99.43%.
7. Adding dropout to the two fully-connected layers results in an accuracy of 99.60%.
8. Using an ensemble of network increases the results to 99.67%.

Why only apply dropout to fully-connected layers? In general, the convulational layerse have considerable inbuilt resistance to overfitting because of the sahred weights.

#### Avoid the (vanishing/exploding) gradient problem

How did we avoid the results of the gradient that either vanished or exploided? 
We didn't, really, but we've done a few things that helps us proceed anyway:

1. Using convolutional layers greatly reduces the number of parameters in those layers. 
   $\rightarrow$ This makes the learning problem easier.
2. Using more powerful regularization techniques (dropout + convolutional layers) to reduce overfitting.
3. Using rectified linear units to speed up training by a factor 3-5.
4. GPU acceleration + training for a long time.
   This, together with (3), it's as though we've training a factor of ~30 times longer than before.

Note that in addition to this, we also used other ideas such as:

- Sufficiently large datasets to reduce overfitting
- Right cost function to avoid a learning slowdown
- Good weight initializations
- Algorithmically expanding the training data.

### Code of convolutional neural network

#### Fully Connected layer

Mine turtle.