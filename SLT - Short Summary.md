---
typora-copy-images-to: images
---

# SLT - Short Summary

> By Joost Visser

[TOC]

$$
\texttt{LaTeX commands}
\newcommand{\E}{\mathbb E}
\newcommand{\F}{\mathcal F}
\newcommand{\P}{\mathbb P}
\newcommand{\X}{\mathcal X}
\newcommand{\Y}{\mathcal Y}
\newcommand{\1}{\mathbf 1}
\newcommand{\O}{\mathcal O}
$$

## Chapter 1 - Some definitions



## Chapter 2 - Binary classification and Regression



## Chapter 3 - Competing Goals: approximation vs estimation



## Chapter 4 - Estimation of Lipschitz smooth functions



## Chapter 5 - Introduction to PAC learning



## Chapter 6 - Concentration Bounds



## Chapter 7 - General bounds for bounded losses



## Chapter 8 - Countably Infinite Model Spaces

### Complexity d Bounds

-----

**Theorem 6 - Complexity Regularized Model Selection**

Let $\mathcal F$ be a *countable* collection of models and assign a real number $c(f)$ to each $f \in \mathcal F$ such that:
$$
\sum_{f \in \mathcal F} e^{-c(f)} \leq 1
$$
Let us define the minimum complexity regularized model (i.e. the ERM with regularization), this is also called the *minimum penalized empirical risk predictor*:
$$
\hat f_n = \arg \min_{f \in \mathcal F} \left\{  \hat R_n(f)+ \sqrt\frac{c(f) + \frac 1 2 \log n}{2n} \right\}
$$
Then we can bound our expected risk compared to the actual risk:
$$
\E[R(\hat f_n)] \leq \inf_{f \in \F}  \left\{  R(f)+ \sqrt\frac{c(f) + \frac 1 2 \log n}{2n} + \frac 1 {\sqrt n}\right\}
$$

------

This theorem is quite useful, as it lets us bound the expected risk and thererfore gives us an idea of what a good estimator is for $\hat f_n$.

This estimator also tells us that the performance (risk) of $\hat f_n​$ performs almost as well as the best possible prediction rule $\tilde f_n​$.

### Histogram example

The expected value of our empirical risk when choosing the number of bins automatically performs almost also as good as the best $k$ (number of bins) possible! Since:
$$
\mathbb E[R(\hat f _n)] \leq \inf_{k \in \mathbb N} \left\{\min_{f \in \mathcal F_k} R(f) + \sqrt\frac{(k+k^d) \log 2 + \frac 1 2 \log n}{2n} + \frac 1 {\sqrt n}  \right\}
$$

- $\hat f_n = \hat f_n^{(\hat {k}_n)}$ - Our best $k$ we could automatically determine depending on the data using the empricial risk minimizer where the $\hat k_n$ is found by minimizing:
  -  $$\hat k_n = \arg\min_{k\in\mathbb N}$$ ERM of each subclass + penalized term.

## Chapter 9 - The Histogram Classifier revisited

### Complexity reguralization

Here, instead of a coding argument, we use an explicit map for $c(f)$, namely:
$$
c(f) = \log(m_f) + \log(m_f + 1) + m_f\log(2)
$$

- Where $m_f$ is the smallest value of $k$ for which $f \in \F_m$, aka a partitioning of $m$ bins.

Now, via a similar method as the last chapter, we can grab the number of bins $\hat m_n$ via minimizing the (ERM + Reguralization term). Using this $\hat m_n$, we get our estimator $f_n = f_{n, \hat m_n}$.

The theory shows that the estimation error on the $\hat f_n$ is as almost the same (worst-case) as $f_{n, m_{\tilde f}}$, which is the best (im)possible choice of the number of bins.

Even if we do this for $d$ input dimensions instead of 2, we can choose the number of bins in the histogram automatically in such a way that we do almost as well as if we actually knew the right number!

*Note:* The penalization is much more severe than deemed necessary, so we're a bit over-conservative. We can either try to create better bounds or use CV to solve this.

### Leave-one-out Cross Validation

The idea for leave-one-out cross valudation is to consider $n$ splits of the data into training set ($n-1$) and validation set $1$. After some mathematics, we can show that the cross-validation risk $\E[CV_{n,m}] = R(\hat f_{n-1, m})$. If $n$ is not too small, then $R(\hat f_{n,m}) \approx R(\hat f_{n-1,m})$, therefore we get an unbiased estimator of the actual risk $R$ of the classifier, not the estimated risk $\hat R$!

We can even choose the number of bins as follows:
$$
\hat f_n^{(CV)} = \hat f_{n, \hat m_{CV}}
$$

- $$\hat m_{CV} = \arg \min_m CV_{n, m}$$, the best possible number of bins found via CV.
- $CV_{n,m}$ is the CV risk, the formula is shown in chapter 9.

This often works very well in practice, but is not a silver bullet and an answer to all problems.

## Chapter 10 - Decision Trees and Classification

### Excess risk of the penalized empirical risk minimization

Suppose we grab the [Complexity Regularization Bounds](#Complexity-Regularization-Bounds) and rewrite it in terms of the excess risk $\E[R(\hat f_n)] - R^*$, where $R^*$ is the Bayes risk. Then we get the following formula:
$$
\E[R(\hat f_n)] - R^* \leq \inf_{f \in \F}  \left\{  \underbrace{R(f) - R^*}_{\texttt{approximation error}}+ \underbrace{\sqrt\frac{c(f) + \frac 1 2 \log n}{2n} + \frac 1 {\sqrt n}}_{\texttt{bound on estimation error}}\right\}
$$
Thus, we see that complexity regularization automatically optimizes a balance between approximation and estimation error. Therefore it's *adaptive* to this unknown tradeoff.

This result is useful only if the bound in the estimation error is good and the class $\F$ has enough models for the approximation error. To check this, we need to make assumptions of $\P_{XY}$.

### Binary classification

Suppose we have a universe $\X = [0, 1]^2$, which you can idenity as a square. You can consider the best classifier (Bayes' classifier) as a set $G^* = \{ x: \P(Y=1|X=x)\geq\frac 1 2)$, such that our Bayes' classifier $f^*(x) = \1\{x \in G^*\}$. In words, this means that we have a subset of $\X$, namely $G^*$, and if our features $x$ are inside this set/field, then we should predict 1. An illustration of this can be shown below:

<img src="images/Bayes' illustrated.png" height="300px"/> 

The boundary of $G^*$ is called the *Bayes Decision Boundary* and is given by $\{x:\eta(x) = \frac 1 2\}$.

The problem with the histogram classifier is that it tries to estimate $\eta(x)$ everywhere and thesholding the estimate at level $\frac 1 2$ (i.e. it looks for the Bayes Decision Boundary everywhere).

### Binary classification trees

#### Growing

The idea of binary classification trees is similar to the histogram classifier; partition the feature space into separate sets and use a majority vote to choose the predicted label for each set. Where the difference lies is how we partition the set. The histogram partitions it *a priori* with constant sets, whereas the trees learn the partitioning from the data.

The idea of the trees is to:

1. *Grow* a very large, complicated tree. (So subdivide $\X$ a lot.)
2. *Prune* the tree. (Combining partitions to reduce overfitting._

The goal is to have a lot of paritions near the boundary to approximate them well, but to have large partitions far from the boundary to avoid overfitting. For this setting, we will consider *Recursive Dyadic Paritions (RDP)* as these are easy to analyse. These RDPs split the feature space $\X$ in half vertically, then horizontally in one of the two sub-paritions, then vertically and so forth.

So... where are the trees? Well, we can associate any partition of $\X$ with a decision tree. This is even the *most efficient way* to describe an RDP.

- Each *leaf* corresponds to a cell of the partition.
- The *nodes* correspond to the various partition cells that are generated in the *construction* of the tree.

#### Pruning to get our estimator

Let $\F$ be all possible RDPs of $\X$. To make use of our theory and bounds of Chapter 8, we need to find our $c(f)$. We do so by constructing a prefix code:

1. All internal nodes will be 0, all leaves will be 1.
2. Each leaf will have a decision label (either zero or one).
3. Read the code from top-down, left-right.

This results in total of $2k-1$ bits for a tree of $k$ leaves, and $k$ bits for each decision, so a total of $3k-1$ bits. Now, we want to solve the following formula to get our bounded estimator:
$$
\hat f_n = \arg \min_{f \in \mathcal F} \left\{  \hat R_n(f)+ \sqrt\frac{(3k(f)-1)\log 2 + \frac 1 2 \log n}{2n} \right\}
$$
Again, here we have a tradeoff between the *approximation error* and the *estimation error*, which we try to minimize. Using a bottom-up pruning process, we can solve this very efficiently as we balance the reduction of $k(f)$ with an increase with risk.

### Comparing the histogram classifier and classification trees

So, now we have a bound on the *excess risk* of the histogram classifier and the Dyadic Decision trees. How do they compare? 

#### Box-Counting assumption

To compute the excess risk, we first need to compure the approximation error $(R(f)-R^*)$. For this, we need to make an assumption about the distribution $\P_{X}$, namely the **Box-Counting assumption**. 

The *Box-Counting assumption* is essentially stating that the overall length of the Bayes'  decision boundary is finite (i.e. no fractals). More specifically, a length of at most $\sqrt 2 C$. 

Furthermore, we assume that the marginal distribution of $X$ satisfies $\P_X(A) \leq p_{max} \text{vol}(A)$, so we can always bound the volume as well for each subset of $A$.

#### Histogram Risk Bound

After some fancy mathematics to calculate the expected excess risk of the histogram classifier, we can get a bound of:
$$
\E[R(\hat f_n^H)] - R^* = \O(n^{-1/4})
$$
So, as our sample size $n$ grows, the expected excess risk will decrease w.r.t. $n^{-1/4}$.

#### Dyadic Decision Tree

Lemma 10.4.1 in the lecture notes tells us that there exists a DDT with at most $3Ck$ leafs that has the same risk as the best histogram with $\O(k^2)$ bins. Therefore, using a similar calculation but with a different mapping $c(k)$, we can get a better bound on the expected excess risk:
$$
\E[R(\hat f_n^H)] - R^* = \O(n^{-1/3})
$$
This is because our bound on the estimation error is smaller as we only have a factor $k$ instead of $k+k^2$.

### Final remarks

#### Histograms vs Trees

Trees generally work much better than histogram classifiers, because they approximate the Bayes' decision boundary in a much more efficient way:

- They only need a tree of $\O(k)$ bits instead of a histogram that requires $\O(k^2)$ bits.
- Because of this, the expected excess risk will converge faster and we will require less memory.

In fact, the DDTs are very deep histogram classifiers followed by pruning to balance the approximation error and the estimation error.

#### Other remarks

1. There exists an even slightly tighter bounding procedure for the estimation error, shown by Scott and Nowak, by using the depth of the leafs as penalization variable.
2. These bounds will work, but are quite conservative as we are over-estimating the estimation error (as we have an upper bound on the estimation error). We can add an estimator variable for this estimation error and use CV to get a great value for it. 

## Chapter 11 - VC Bounds



## Chapter 12 - Denoising of Piecewise Smooth Functions

