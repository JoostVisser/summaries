# 2DI70 - Statistical Learning Theory - Summary

[TOC]

## Lecture 1

### Administration

- Office Hours are for prepared questions and discussions about topics.

**Grades**

- 25% assignments
- 15% report
- 60% written exam

Prerequisites: some probability theory.

Webpage is on Canvas.

### What is ML?

#### Example

**Settings**: From patient $\longrightarrow$ skin sample $\longrightarrow$ Gene sample (~10 000 genes) [Gene expression data]

**Supervised learning:** We have data from $n$ patients. $X_i = $ gene expression, $Y_i$ = Cancer / no cancer

Use $(X_i, Y_i)$ to predict, given only gene ($x_i$) expression, if a patient has cancer $(y_i)$.   

#### Another example

(This example is a graph of a binary classifier, this explains overfitting. Tells about Occam's Razor that a simpler explanation might be better.)

#### Typical architecture of ML

Input $\longrightarrow$ sensor $\xrightarrow{data}$ Feature Selection $\xrightarrow{features}$ Prediction

There are architectures, called deep learning, that combine the feature selection and the prediction. However, nobody exactly knows how they work.

#### Key ingredients

- [Data spaces](#data-spaces)
  - (Input features and output classes)
- [Probability measures](probability-measures)
  - Relation in data.
- [Loss function](#loss-function)
  - How good or bad your predictions are.
- [Statistical risk](#statistical-risk)

### Data spaces

$\mathcal{X} \equiv $ Feature space $\equiv$ Input space

$\mathcal{Y} \equiv$ Label space $\equiv$ Output space

#### Examples

- Gene expression: $\mathcal{X} = [0,1]^{10000}$ and $\mathcal{Y} = \{0,1\}$
- Height and weight (cancer): $\mathcal{X} = R^2_+$ and $\mathcal{Y} = \{ -1, 1\}$
- Regression: $\mathcal{X} = \mathbb{R}$ (e.g. power consumed), $\mathcal{Y} = \mathbb{R}$ (e.g. speed)

### Loss function

Suppose we have a true label $y \in \mathcal{Y}$ and a prediction $\hat{y} \in \mathcal{Y}$.

**Goal:** How different are these?

A *loss function* is a sort of "distance" in the form of $\ell:\mathcal{Y} \times \mathcal{Y} \longrightarrow \mathbb{R}$

**Example:** Binary classification. $\mathcal{Y} = \{0, 1\}$

Let our classifier be defined as followed:

If $y_1 == y_2$:
​    then $\ell(y_1, y_2) = 1$ 
​    else $\ell(y_1,y_2) = 0$

$\ell(\hat y, y) = \mathbb{1} \{\hat y \neq y\}$

This notation means: $1\{A\}$, 1 if $A$ holds, else 0.

**Example:** Spam classification. $\mathcal{Y} = \{0, 1\}$, $0$ for legit and $1$ for spam.

If $y=0$ and $\hat y = 1$
​    then loss is 10.

If $y=1$ and $\hat y = 0$
​    then loss is 1.

Otherwise loss is 0.

(In other words, if a mail is not spam but we classify it as spam, we pay a huge price for it.)

**Example:** $\mathcal{Y} = \mathbb{R}$ for regression, we will use the MSE. $\ell(\hat y, y) = (\hat y - y)^2$

- (Why not use absolute value? Difficult to use for gradient descent.)
- This way, we really penalize large distances.

### Probability Measures

$x \in \mathcal{X}$ and $y \in \mathcal{Y}$ have some sort of relation. 
There is some uncertainty in the way features are collected (with e.g. handwriting recognition.)

**Probability measure:** Define a joint probability measure $\mathbb{P}_{XY}$ over $(\mathcal{X}, \mathcal{Y})$. 
Let $X, Y$ be a pair of random variables (or vector) distributed as $\mathbb{P}_{XY}$

$\mathbb P_X$ - Marginal distribution of features.
$\mathbb P_{Y|X}$ - Distribution of label given feature. 

**Definition:** The expectation operator, given $h:\mathcal X \times \mathcal Y \rightarrow R$ is defined as 
$\mathbb E[h(x,y)]=\int h(x,y) d \mathbb P_{XY}(x,y)$ 

$\mathbb E[h(x,y)]=\int \int h(x,y) f_{XY}(x,y) dx dy$     [If $X,Y$ are continuous random variables]

$\mathbb E[h(x,y)]=\sum_{x\in \mathcal X} \sum _{y \in \mathcal Y} h(x,y) P(X=x,Y=y)$      [If $X, Y$ are discrete.]

### Statistical risk

**Goal:** Construct a map $f:\mathcal X \rightarrow \mathcal Y$, so $x \in \mathbb R$ and $\hat y = f(x)$

**Statistical risk:** Let $(x,y) \sim P_{XY}$ 

$R(f) = \mathbb E[\ell(f(x),y)|f]$, but don't worry too much about the $|f$ conditional part.

**Goal risk:** make it as small as possible.

**Example:** $\mathcal Y = \{0,1\}, f:\mathcal X \rightarrow \{0,1\}$

$R(f) = E[\ell(f(x),y)] = E[1\{f(x)\neq y\}] = P(f(x) \neq y)$

(Since we have a $P(f(x)\neq y)$ chance that it will be 1, else it will be 0.)

Minimum risk: $R^* = \inf _f R(f)$ 

### The learning problem

**Goal:** Construct a prerdiction rule $f$ with small risk $R(f)$ (depends on $\mathbb P_{XY}$

**Problem:** We don't know $\mathbb P_{XY}$ yet.

Data: suppose we have access to examples from $\mathbb P_{XY}$.

$D_n = (X_i, Y_i)^n_{i=1}$ [Training data]

**Strongest assumption:** We assume $(X_i, Y_i)$ are i.i.d. samples for $\mathbb P_{XY}$. 
(Basically, we assume that the training data is representative of all data in the world about this data.)

$\hat f_n: \mathcal X \longrightarrow Y$ and $x, D_n \mapsto \hat f_n(x; D_n) \in \mathcal Y$

Where $\hat f_n(x; D_n) = \hat y$ is the result of the algorithm ($\hat f_n$) we choose.

$R(\hat f_n) = E_{XY}[\ell(\hat f_n(x), y)]$

Our goal is to minimize this expected risk: $E[R(\hat f_n)]$

If I have an algorithm to choose $\hat f_n$, then I can get a guarantee so that $E[R(\hat f_n)]$ is small.

This distribution $P_{XY}$ can be really super weird, but we can still get a small expected risk.

## Lecture 2

### Summary of last lecture

Observe $X \in \mathcal X$ (features)

**Goal:** Want to predict $Y \in \mathcal Y$

- $\mathcal X, \mathcal Y$ are the data spaces
- $\ell : \mathcal Y \times \mathcal Y \rightarrow \mathbb R$ is a loss function (For example: $\ell (\hat y, y) = (\hat y - y)^2$
- $(X, Y) \sim \mathbb P_{XY}$

If I have a prediction rule $f: \mathcal X \rightarrow \mathcal Y$
Define the risk of $f$ as $\mathbb E[(f(x), y]$

Goal of this lecture: What is the best possible prediction rule ($f$) I can make here such that I get the lowest risk.

### Detour

$X, Y$ are random variables where $(X,Y) \sim \mathbb P _{XY}$.

$\mathbb P(\{x \in A\} \cup \{y \in B\}) = \mathbb P(x \in A, y \in B) = \mathbb P(Y \in B | X \in A) \cdot \mathbb P (X \in A)$

-----

If $X, Y$ are discrete, then $f_{Y|X}(y|x) = \mathbb P (Y=y|X=x) = \frac{P(X=x, Y=y)}{P(X=x)} = \frac{f_{XY}(x,y)}{f_x(x)}$.

Capitals: random variables. Non-capital letters: deterministic (e.g. observation).

-----

$\mathbb E[Y | X=x] = \sum_{y\in \mathcal Y} y \cdot f_{y|x}(y|x) = g(x)$

$\mathbb E[Y | X] = g(X)$

Complication:

$\mathbb E [ \mathbb E [Y | X]] = \mathbb E [g(X)] = \mathbb E [\sum_{y\in \mathcal Y} y \cdot f_{y|X}(y|X) ] = \sum_{x \in \mathcal X} g(X)f_X(x) = \sum_{x \in \mathcal X} \sum_{y \in \mathcal Y} y f_{y |x} f_X(x) $

$=\sum_{x \in \mathcal X} \sum_{y \in \mathcal Y} y f_{xy} (x,y) = $ after some math by putting the $y$ in front $ = \sum _{y \in \mathcal Y} y f_y(y)=\mathbb E[y]$

This results in the law of total probability: $\mathbb E[\mathbb E[Y|X]] = \mathbb E [Y]$

### Binary classification

$\mathcal Y = \{0, 1\}$, with loss function $\ell (y_1, y_2) = 1 \{ y_1 \neq y_2 \}$
Suppose we have a prediction rule $f:\mathcal X \rightarrow \mathcal Y$.
Risk: $R(f) = \mathbb E[\ell(f(x), y)] = \mathbb E [1 \{f(x) \neq y\}] = \mathbb P(f(x) \neq y)$

----

**Definition - Bayes' risk:** $R^* = \inf_f R(F)$ 
(Infinum is over all possible $f$ to find the smallest).

----

**Definition - Bayes'  classifier:** Let $\eta(x) = \mathbb P(Y = 1 | X = x)$.
Classifier: $f^*: \mathcal X \rightarrow \mathcal Y$, where $f^*(x) = 1\{\eta(x) \geq \frac 1 2 \}$ 
Property: $R(f^*) = R^* = \inf_f {R(F)}$

-----

Proof: see lecture notes.

$G \Delta G^*$ is similar to the exclusive or, so it's similar to $(G \cup G^*) \setminus (G \cap G^*)$ 

### Regression

$\mathcal Y = \mathbb R$ and $\ell(y_1, y_2) = (y_1-y_2)^2$
$X, Y \sim P_{XY}$, $f:\mathcal X \rightarrow \mathcal Y$
$R(f) = \mathbb E[(f(x)-y)^2]$

-----

**Definition** - $f^*(x) = \mathbb E[Y|X=x]$, which is called the regression function.
$R(f^*) = R^* \geq \inf_f R(F)$

-----

Proof: see lecture notes.

### Learning

We don't know $\mathbb P_{XY}$, thus we cannot compute $f^*$.

What we do know, however, is $\{(X_i, Y_i)\}_{i=1}^n \sim \mathbb P_{XY}$ (Actually i.i.d. w.r.t. $\mathbb P_{XY}$)

Define the *empricial risk*: $\hat R_n(f) = \frac 1 n \sum_{i=1}^n \ell(f(X_i), Y_i) $ 
The idea is that this close to the true risk, e.g. $\approx R(f) = \mathbb E[\ell(f(x), y)]$ 
This is true as $n \rightarrow \infty$

Pick a rule $\hat f_n = \text{argmin}_{f \in \mathcal F} \hat R_n(f)$

###  Questions

1. $\mathbb E[\ell(f(X), Y)] = \mathbb P(f(X) = 1, y = 0) + \mathbb P(y = 1, f(X) = 0)$
2. Why is it $d\mathbb P_X(x)$? Page 16.
3. Why change it to $P(a|X=x)$ at page 16's proof.
4. Why do they change it to the expected value with indicator functions on page 16? 
   I.e. the step where I do my '?'
5. ​

## Lecture 3

### Finding the right estimator

**Goal:** find $f:\mathcal X \rightarrow \mathcal Y$, with $\hat y = f(x)$

- For this, we want to find the distribution $\mathbb P_{XY}$ where $(X, Y) \sim \mathbb P_{XY}$.
- The only ting we know about $\mathbb P_{XY}$ are the training data $D_n = \{(X_i, Y_i)\}_{i=1}^n \stackrel{iid}{\sim} \mathbb P_{XY}$
- To do so, we have the empirical risk.
  - Our goal is to minize this risk, thus
    $\hat f_n = \arg \min_{f \in \mathcal F} \hat R_n(f)$, where $\tilde f$is a class of function $f:\mathcal X \rightarrow \mathcal Y$

Take any $f : \mathcal X \rightarrow \mathcal Y$ fixed.

$\hat R_n(f) = \frac 1 n \sum_{i=1}^n \ell(f(X_i), Y_i)$ and as the $\ell$ is i.i.d. of $\sim Z_i$ we can because of the large number note that this equals $E[\ell(f(X_1), Y_i)] = \hat R_n$

Convergence: $\forall \epsilon > 0: \mathbb P(|\hat R_n - R(f)| > \epsilon) \rightarrow 0$ as $n \rightarrow \infty$. 

Example

- $\mathcal X = [-1, 1]$, $\mathcal Y = \mathbb R$, $\ell(y_1, y_2) = (y_1 - y_2)^2$
- $\mathcal F = \{f:[-1, 1]\rightarrow \mathbb R: f \text{ is a polinomial of degree $d$}\}$
- Example $d=1:$ $\arg \min \frac 1 n \sum_{i=1}^n (a + BX_i - Y_i)^2$ which stands for linear regression.

However, this results in *overfitting* on the training data. However, more data results in more information, which in turn results in better fitting line.

Another example shows a silly example of where the estimated empirical risk equals 0, as it is a function of the input, whereas the actual risk is 1.

### Bias and Variance tradeoff

We will choose $\hat f_n$ from $\mathcal F$ (e.g. $\hat f_n = \arg \min_{\text{all }f} \hat R_n(f)$) .

Benchmark: $R^* = \min _{f \in \mathcal F} R(f)$

$E[R(\hat f_n)]- R^* =$ expected excess risk $=E[R(\hat f_n)] - \min _{f \in \mathcal F} R(f) + \min _{f \in \mathcal F} R(f) - R^*$
$= \underbrace{(E[R(\hat f_n)] - \min _{f \in \mathcal F} R(f))}_{\stackrel{\text{Estimation error (variance for }\ell^2\text{)}}{\text{How well I "pick"  a good model}}} + \underbrace{(\min _{f \in \mathcal F} R(f) - R^*)}_{\stackrel{\text{Approximation error (bias squared }\ell^2\text{)}}{\text{Depends only on $\mathcal F,  \mathbb P_{XY}$ }}}$

The approximation error reduces the more complex $\mathcal F$ is. 
However, the more complex $\mathcal F$ is, the higher the estimation error is.
The expected excess risk $\mathbb E[R(\hat f_n)] - R^*$ is the one that we want to reduce as much as possible.

### Solving the problem

$\hat f_n = \arg \min_{f \in \mathcal F} \hat R_n(f) = \arg \min f_{f \in \mathcal F} \frac 1 n \sum_{i=1}^n \ell(f(X_i), Y_i)$

So at start a complexer $f$ is good and gets us closer to the true min risk $R_n(f)$, however, too complex of an $f$, the empirical risk $\hat R_n(f)$ gets lower but the true risk $R_n(f)$ increases (*overfitting*).

#### Approach 1 - Restricting functions

The first approach is to restrict $\mathcal F$, this might even depend on the amount of data we observe.
So we have $\mathcal F_1, \mathcal F_2, \ldots, \mathcal F_n, \ldots$ where $\mathcal F_1 \subseteq \mathcal F_2 \subseteq \mathcal F_3 \subseteq \ldots$, where we use a different $\mathcal F$ depending on the amount of data. This is called the **Method of Sieves**

#### Approach 2 - Penalize complex models

The idea is to penalize the empricial risk minimalization. 
What we'll do is $\hat f_n = \min_{f \in \mathcal F} \hat R_n(f) + \underbrace{C(F)}_{\text{Penalty}}$.

We want a penalty that is roughly the reversed of the empirical risk, so we hopefully get something that looks more like the true risk.

It's a more general way of writing *approach 1*.

#### Description length methods

> Section 3.1.2

$\hat f_n = \min_{f \in \mathcal F} n \hat R_n(f) + C(F)$

Basic idea: if the fit of the data is very good, then the number of bits to describe the data is little, whereas if it doesn't fit that good , the number of bits to desribe the data is a lot. This is $C(f)$.

What we're trying to do is minimize the number bits possible to describe the data, basically the simplest solution.

Negative log likelihood. Negative version of the entropy.

#### Hold-out methods

$D_n = \{ (X_i, Y_i) \} ^n _{i=1}$, we split this into:

- Training set $D_T = \{ (X_i, Y_i) \} ^n _{i=1}$ 
- Test set $D_V = \{ (X_i, Y_i) \} ^n _{i=1}$

$\hat R_m(f) = \frac 1 m \sum_{i=1}^m \ell(f(X_1), Y_i)​$, define $\hat f _m^{(\lambda)} = \arg \min_{f \in \mathcal F_\lambda} \hat R_m(f)​$

$\hat \lambda = \arg \min_\lambda \hat R_V(\hat f _m(\lambda))$ $\leftarrow$ Expectation / bias estimator of the risk.
Where $ \hat R_V(\hat f _m(\lambda)) = \mathbb E [ \hat R_V(\hat f _m^{(\lambda)}) | \hat f_n^{(\lambda)}] = R(\hat f_m ^{(\lambda)})$

Problem: how to split the data?

- We're wasting a lot of info in $D_V$.
- Solution: **cross-validation**, idea: use different $D_T$ and $D_V$ multiple times.


Example of *cross-validation* will happen later in the course.

## Lecture 4 

### Example

We observe training data $\{(x_i, y_i)\}^n_{i=1}$.

Let $x_i = \frac i n$, $y_i = f^*(x) + W_i$ and $W_i$ are independent random variables with $\mathbb E[W_i] = 0$ and $\mathbb V( W_i) \leq \sigma^2$.
Also, $f : [0,1] \rightarrow \mathbb R$. So we have some function and the actual points are differentiated with distribution $W$ from this function $f$, with $\mu_W = 0$ and $\sigma^2_W = \sigma^2$ 

**Goal**: estimate function $f^*$, that is, construct $\hat f:[0,1] \rightarrow \mathbb R$, where $\hat f_n(x) \equiv \hat f_n(x, D_n)$
Risk: $R(\hat f_n)=\mathbb E[\int_0^1(\hat f_n(x) - f^*(x))^2dx] = \mathbb E[||\hat f_n - f^*||]$

In other words: $\hat R_n(f) = \frac 1 n \sum_{i=1}^n(y_i - f(x_i))^2$

Assumption: $f^*$ is Lipschitz (smooth): $\forall_{s,t\in[0,1]} |f^*(s)-f^*(t)| \leq L |s-t|$ for some $L > 0$. This says that the derivative may not take a value larger than $L$, aka cannot change super fast.