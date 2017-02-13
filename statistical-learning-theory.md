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

$=\sum_{x \in \mathcal X} \sum_{y \in \mathcal Y} y f_{xy} (x,y) = $ after some math by putting the $y$ in front $ = \sum _{y \in \mathcal Y} y f_y(y)=E[y]$

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

