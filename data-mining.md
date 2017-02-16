# 2IMW30 - Foundation of Data Mining - Summary

[TOC]

## Lecture 1 - ML concepts, scikit-learn and kNN

### Assignments

1. Linear models, model selection, ensembles
   - Released: Feb 9. Deadline: Feb 23.
2. ... ...
   - Released: Feb 23, Deadline: Mar 16.
3. Deep Learning
   - Released: Mar 16. Deadline: ??
4. Blergh, something with PCP
   - Released: ??, Deadline: !?

**Todo:** Read prerequisite file and do everything that is in there. #to-do

### ML concepts

**Supervised learning:** Learn from training data and then make predictions.

- Classification: Predict a class label.
  - *Binary* (2 classes) or *multi-class*
- Regression: Predict a continous value.

**Unsupervised Learning:** Explore the structure of data to extract meaningful info.

**Reinforcement learning:** Develop an *agent* that improves its performance on interactions with its environment.

Note, there also exists semi-supervised methods and other types.

**Dimensionality Reduction**

- Data can be very high-dimensional.


- Compress data in fewer dimension.
- Useful for visualisation and for supervised learning.
- Not the same as *feature selection*, which generally is supervised.

Machine learning has multipel components:

- Preprocessing:
  - Feature scaling: Values in the same range.
  - Encoding (categorical $\Rightarrow$numerical)
  - Discretization (numeric $\Rightarrow$ categorical)
  - Feature selection: Remove uninteresting / correlated features.
  - Dimensionality reduction

## Tutorial - Useful Python functions

### 1A - Basic python functions

#### Python functions

### 1B - ML Python libraries

#### Numpy functions

**Iterator**

For loop over a numpy array results in an iteration for each row.

##### ravel() / flat() 

Ravel flattens / squashes an array. Flat returns an iterator to loop over.

##### b, c = hsplit(a, 2)

Splits array a into b and c.

##### a = hstack(b, c)

Puts them back together again.

##### view()

Creates a *new* array object that looks at the same data.

**copy()**

Makes a deep copy of the array and its data.

##### Scipy - Sparse matrices

These are used for large arrays that contain mostly zeros.

```python
from scipy import sparce
eye = np.eye(5)
spare.csr_matrix(eye)
print("{}".format(sparse_matrix))
```

Can also create our own sparse matrix when giving the coordinates where the data should be.

#### Pandas

##### pd.DataFrame(data)

##### Read and write to CSV

## Lecture 2 - Linear models 

### Linear Regression

This is in the form of $\hat y = w_0x_0 + w_1x_1 + \ldots + w_n\cdot x_n+b$

Error $r_i = y_i - \hat y_i$, aka how much I got it wrong.

MSE: $\frac 1 n \sum_{i=1}^b r_i^2$

## Lecture 3 - Model Selection

### Cross-validation

#### The need of cross-validation

Basic form of evaluation: train-test split (75%-25%).

What's wrong with this?

- Overfitting on test data
- We're not using the data optimally, as we're wasting data for training.
- Test data might not be a good representation of the training data.
- What if all examples of 1 class belong to training *xor* test-set?

**k-fold cross-validation (CV):**  Split (randomized) data into $k$ equal-sized parts, called *folds*.

- First fold 1 is the test-set, and folds 2-5 comprise the training set.
- Then fold 2 is the test set, and 1, 3, 4, 5 comprise the training set.
- Compute $k$ evaluation scores, aggregate afterwards.

In sci-kit learn it can be done with `cross_val_score`.

- It does as followed: `scores = cross_val_score(clf, iris.data, iris.target)`
- Does 3-fold CV by default, with accuracy (classification) or $R^2$ (regression).

Tradeoff between training set vs test set. General rule: *10-fold cross-validation* or *5-times-2-fold cross-validation*. 

- `print("Avg scores".format(scores.mean()))`

Benefits of Cross-Validation

- More robust, as every training example will be in a test set exactly once.
- Shows how *sensitive* the model is.
- Better estimation of true performance.

(More expensive, though.)

#### Different CV techniques

What is the data is *unbalanced*?

- Perform **stratified k-fold cross-validation**, makes sure that proportions between classes are conserved.
  - Order examples per class.
  - Separate samples of each class in $k$ sets (called *strata*).
  - Combine corresponding *strata* into *folds*.
- In *scikit-lean*, we can use `KFold` with `shuffle=True` to achieve some shuffling.

Special form: **leave-one-out cross-validation**.

- Suppose we have $n$ examples, then we perform $n$ fold cross-validation.
- Very expensive though.

Another technique is **shuffle-split cross-validation** ($\leftarrow$ Quite old, though).

- Sample # of samples as training set, and disjoint # of samples as test-set. 
  We will leave some data out to be faster.
- Stratified variant: `StratifiedShuffleSplit`.

What is we have a data-set that contains inherent groups? (E.g. patient)

- Use *grouping* of *blocking*: we group all the data-points for specific groups and distribute each group either to the training or test-set.
- Extreme case: **Leave-one-subject-out cross-validation:** create test set for each user individually.
- In *scikit-learn*, we need to make an array with group membership (e.g. an integer), and then give it to the `cross_val_score`.

#### Guidelines for picking performance stimation

1. Always use stratification for classification.
2. Use holdout for very large datasets.
   - Or for when learners don't always converge, e.g. deep learning.
3. Choose *k* depending on dataset size and resources.
   - Use leave-one-out for small datasets.
   - Use cross-validation otherwise.
     - Most popular: 10-fold CV.
     - Literature suggests: 5$\cdot$2-fold CV.
   - Use grouping or leave-one-subject-out CV for grouped data.

### Bias-Variance decomposition

**Bias:** Systemetic error. The classifier always get certain points wrong. 

- Underfitting

**Variance:** Error due to variability of the model. Classifier predicts some points accurately on some training sets, but inaccurately on others.

- Overfitting

This is generally a trade-off, you can exchange bias vs variance with e.g. (de)regularization.

#### Regression Bias and Variance

How to measure this?

- Take 100+ bootstraps or shuffle-splits.
- For each data point $x$
  - $bias(x) = (x_\text{true} - mean(x_\text{predicted}))^2$
  - $variance(x) = var(x_\text{predicted})$
- Total bias = $\sum_{x\in\mathcal X} bias(x)\cdot r_{x} $
  - Where $r_x$ is the ratio of $x$ occurring in the set.
- Total variance = $\sum_{x\in\mathcal X} variance(x)\cdot r_{x}$

#### Classification Bias and jVariance

General procedure:

- Take 100+ bootstraps or shuffle-splits.
- For each data point $x$
  - $bias(x) = $ ratio of misclassification.
  - $variance(x) = 1-\frac{r_{class1}^2 + r_{class2}^2} 2$
- Formulas for the total, only the bias is different from the regression.:
  - Total bias = $\sum_{x\in\mathcal X} bias(x)^2\cdot r_{x} $
  - Total variance = $\sum_{x\in\mathcal X} variance(x)\cdot r_{x}$



#### Counters

How to fix high bias?

- Less regularization
- More flexible / complex model.
- Bias-reduction technique (boosing)

How to fix high variance?

- More regularization
- More data
- Simpler model
- Use a variance-reduction technique (bagging)

### Hyper-parameter optimization

#### Grid search cross-validation

For each hyperparameter, create a list of interesting values

- E.g. $k \in [1, 23, 5, 7, 9, 11, 33, 55, 77, 99]$

Evaluate all possible combinations of hyperparameter values.

- E.g. using cross-validation

Select the hyperparameter with the best result.

- However, will overfit on the test data.
- Solution: `mglearn.plots.plot_threefold_split()` to split in Training set, CV set and test set.

See `mglearn.plots.plot_grid_search_overview()` for an overview of how the sets and models interact with eachother.

We can use `GridSearchCV` to automatically optimize its hyperparameters internally.

- Input: (untrained) model, parameter grid, CV procedure
- Output: optimized model on given training data
- Should only have access to trinaing data

Suppose we state that `grid_search = GridSearchCV(clf, param_grid, cv=5)`, then this `grid_search` becomes a new classifier which we can use to e.g. fit or score.

- We can use `cv_results_` to see the current parameters.
- Hyperparameters depend on other parameters? Then we can use lists of dictionaries to define the hyperparameter space.

Still has the disadvantage that we only do a single split to create the outer test set.
Solution? Use **Nested Cross-validation**.

- Outer loop: split data in training and test sets.
- Inner loop: run gird search, splitting the training data into train and validation sets.
- There will be multiple optimized models and hyper parameter settings, but these will be not returned. Only do this if we're interested in how good the algorithm is. 
- Optimization, set parameters `n_jobs=-1` in `cross_validation_score` to use all cores.

#### Random search

Downside of grid search

- Optimizing many hyperparameters create combinatirial explosion.

Solution? Random search:

- Pick `n_iter` random parameter values.
- Scales better
- Often works better in practice. 

Similar to GridSearchCV, but instead use `RandomizedSearchCV`. 

- We can actually give a distribution to the random search.

#### Model-based optimization

Idea: after a # of random search, we know more about the performance of hyperparameters settings on the given dataset.

Typical way: **Bayesian optimization**

*Acquisition function:* Trades off maximal expected performance and maximal uncertainty.

### Evaluation metrics and scores

#### Metrics for binary classification

Accuracy is kind of the worst evaluation for highly skewed data (imbalanced). 

- Examples: cancer or credit card fraud.

We can use different types of scores, such as *precision*, *recall* and *F1-score*.

## Lecture 4 

