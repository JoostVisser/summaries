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
from scipy import sparse
eye = np.eye(5)
sparse.csr_matrix(eye)
print("{}".format(sparse_matrix))
```

Can also create our own sparse matrix when giving the coordinates where the data should be.

#### Pandas

Pandas is a Python library for data wrangling and analysis. 
Got many nice methods to apply to tables, such as sorting and importing.

##### pd.series()

A one-dimensional array of data with indexed values.

```python
pd.Series([1, 3, np.nan]) # Default with row indices of numbers.
pd.Series([1, 3, 6], index=['a', 'b', 'c']) # Different row indices.
pd.Series({'a': 1, 'b': 2, 'c', 3:}) # Same as previous, but with a dictionary.
# Will only save the ones with given indices.
pd.Series({'a': 1, 'b': 2, 'c', 3:}, index=['b', 'c', 'd'])
```

##### pd.DataFrame(data)

A DataFrame is a table with both row and column index. Similar to an excel table.

```python
data = {'state': ['Ohio', 'Ohio', 'Nevada', 'Nevada'], 
        'year': [2000, 2001, 2001, 2002],
        'pop': [1.5, 1.7, 2.4, 2.9]}
pd.DataFrame(data)
```

All of numpy's universal functions also work with DataFrames.

###### Slicing

- `df[col]` or `df.col` can be used to get the data.
- `df.iloc[a:b, :]` for a matlab-like indices style.
- With `df.ix[]` you can mix both locations and strings. 

Example - Given a dataframe `df`:

- Want the column of Sky? `df.Sky` or `df["Sky"]`
- Want the first three rows? `df[0:3]`
- Want to first three rows of Sky? `df.Sky[0:3]` or `df[0:3].Sky`
- Want Sky and AirTemp? `df.[["Sky", "AirTemp"]]`
- Similarly,  could use the direct location: ` df.iloc[:, 0:2]`
- Want first three rows of Sky and AirTemp? `df.ix[0:3, ["Sky", "AirTemp"]]`

###### df.index

Retrieves an Index object with a list of the row indices.

###### df.columns

Retrieves an Index object with a list of the column indices.

###### df.values

Retrieves a 2D array with all the values in the DataFrame. 

###### df.head()

Returns the first 5 rows.

###### df.tails()

Returns the last 5 rows.

###### df.describe()

Quick stats about the DataFrame per column, such as the count, mean, min, standard deviation, quantiles, etc.

###### df.T

Transpose of the DataFrame.

###### df.sort_index

Sorts the index w.r.t. the lablels. (`df.sort_index(axis=1, ascending=False)`)

###### df.sort

Can sort by values, e.g. `df.sort(columns='b')`

###### df.query

Retrieve data matching a boolean expression: 

```python
df.query('A > 0.4') # Identical to df[df.A > 0.4]
```

###### Operations

DataFrames offer a wide range of operations, such as max, min, mean, sum, ...

```python
df.mean() # Mean of all values per column
```

###### Custom functions

Other custom functions can be applied with `apply(funct)`

```python
# All these functions are applied per column
df.apply(np.max)
df.apply(lambda x: x.max() - x.min())
```

###### df.groupby()

Useful to aggregate data, when you have e.g. two classes then you can group them by class.

```python
df.groupby(['A','B']).sum() # Groups by the indices in A and B with the sum
```

###### df.append()

Appends one DataFrame to another.

```python
df.append(s, ignore_index=True)
```

###### df.drop_duplicates()

Removes all duplicates, i.e. rows that are exactly the same.

```python
df.drop_duplicates()
```

###### df.replace(a, b)

Replaces all values in the DataFrame from `a` to `b`.

```python
df.replace(-1, np.nan)
```

##### pd.merge(df1, df2)

Merges two DataFrames based on common keys.

```python
df1 = pd.DataFrame({'key': ['b', 'b', 'a'], 'data1': range(3)})
df2 = pd.DataFrame({'key': ['a', 'b'], 'data2': range(2)})
pd.merge(df1, df2)
```

##### pd.value_counts() - binning

What if we have data and we want to aggregate it over bins? Then we can use `pd.cut()` and `pd.value_counts()` to cut the data into bins and count them.

```python
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats.labels
pd.value_counts(cats)
```

##### Read and write to CSV or JSON

Want to read or write to CSV? It's built in, hurray! \o/

```python
dfs = pd.read_csv('data.csv')
dfs.to_csv('data.csv', index=False) # Don't export the row index
dfs.to_json('hello.json') # No need to indicate index=False here.
```

#### Matplotlib

The Matplotlib libary creates

1. Figures, which are the different screens. (`fig = plt.figure()`)
2. Axes, the axes of the figures. (`ax = plt.subplot(111)`)

##### plt.plot()

Very useful function to plot figures. Takes in any number of arguments.

- **plot(y):** Only give single list of data? Then it assumes it is a sequence of $y$ values and automatically generates an $x$.
- **plot(x, y):** Plots w.r.t. the $x$ and $y$ lists given.
- **plot(x, y, type):** The type can be set to various string formats for different effets:
  - 'b-': Stands for blue solid line.
  - Useful colours: ['b', 'g', 'r', 'c', 'm', 'y', 'k' (black), 'w']
  - Useful markers:
    - '-'   Solid line style
    - '--'  Dashed line style
    - ':'    Dotted line
    - '.'    Point marker
    - 'o'   Circle marker
    - '+'   Plus marker
    - 'x'   X marker
    - 's'    Square marker
- **plot(x, y, type, x2, y2, type2):** Two plots in a single diagram.
  You can do this for many different plots.

Various args can be set of `Line2D`, such as the line width. Call `setp()` with a line as argument for a list.

##### Multiple figures - plt.figure()

The `figure()` is an optional command for having multiple figures. 

- Function `plt.figure(n)` can be called to change focus of plots.
- Default is `figure(1)` which is done automatically.

##### Multiple subplots - plt.subplot()

The `subplot()` command is useful for having multiple plots / graphs in a single figure.

- Function `plt.subplot(abc)` or `plt.subplot(a, b, c)` in case either `a`, `b` or `c` $>10$.
  - Specified as followed: `numrows`, `numcols`, `fignum` where `fignum` is the focus of the current plot. 
- Default, called automatically, is `plt.figure(111)`.
- Suppose we want to have a 2x2 plot with four figures in a window, then we have to call it as:
  - `plt.subplot(221)` for the first plot, `222` for the second one and `223`  and `224` for the third + fourth respectively. 

##### Labels - plt.xlabel(), plt.ylabel() and plt.title()

Default labels are done via `plt.xlabel()` and `plt.ylabel()` for the $x$ and $y$ label respectively. The title can be set with `plt.title()`. Example:

```python
plt.title("Relation between cookies and time.")
plt.ylabel("Number of cookies required")
plt.xlabel("Time without cookies")`
```

##### plt.clf()

Clears the current figure.

##### plt.text() - Text in the figure

With `text(x, y, text)` we can place the text anywhere in the figure. 

- The $x$ and $y$ are actually the real $x$ and $y$ points of the Axes itself! 
  If you want to use relative or percentages, then set `transform=ax.transAxes`.
  (In this case, you need to get the correct ax; `ax=plt.subplot(...)`)

> Fun fact, matplotlib has a built-in TeX expression parser! Just include your string as a raw string, such as `r'\sqrt{\frac a b}'` and you've got formulas in titles / text!

##### plt.setp() - Setting properties.

All the commands above return an instance, e.g. a Text instance. These can later be edited via the `setp()` command.

##### plt.annotate() - Annotate something, using e.g. an arrow

There are various annotations in the graph. For this, we need to know the location (`xy`) being annotated and the location of the text (`xytext`). 

Syntax: `plt.annotate(text, xy=(.., ..), xytext=(.., ..))`

##### plot.xscale('log') - Logarithmic axis



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

Many classifiers return a probability per class, which we can retrieve with:

- `decision_function()`, default = 0,  and `predict_proba()`, default = 0.5.
- This way, we can change the threshold if our goals are different, e.g. when FP is much worse than FN.

#### Precision-Recall curve

If we create a precision-recall curve, then we can see the tradeoff for *precision* and *recall*.

- We can change recall for precision depening on the threshold.
- Different classifiers work best on different parts of the curve.

#### ROC

**Receiving Operating Characteristics (ROC curve):** Plotting True Positives vs False positives results.

- Choosing threshold is called *threshold calibration*.
- Can choose how many extra true positives we want vs the extra false positives we get.

**ROC Isometrics:** Some sort of cost curve where you can reduce the cost as much as possible depending on the ratio of FP and FN: this is done via isometrics, i.e. lines of equal cost.

Again, we compare multiple models by looking at the ROC curves to calibrate the threshold on whether we need a high recall or low FPR.

### AUC

**Area under the ROC curve (AUC):** Good summary measure.

- Use `roc_auc_score()` , not `auc()`.

> Remember
>
> - **AUC** is highly recommended, especially on imbalanced data. Much better than accuracy.
> - Calibrate the threshold to your needs.
>
> We can even optimize for AUC when doing grid search.

### Multi-class classification

Can create a *confusion matrix* to see where things go wrong. 

- Plots true label vs prediction label.

#### Classification

**Macro-averaging:** ... 

- Good when you canre about each class equally uch.

**Weighted-averaging:** ...

- Save pick, good when data is imibalanced.

**Micro-averaging:** ...

- Use when you care about each sample equally much.

#### Regression

Mean squared error.

Mean absolute error:

- Less sensitive to outliers and large errors.

R squared: Ratio of variation explained by the model over the total variation.

- Does not measure bias.

## Lecture 4 

### Decision tree

- Split the data in two (or more) parts
- Search over *all possible* splits and choose the one that is most *informative*.
  - Many heuristics, such as accuracy or *information gain*.
- Repeat recursive partitioning.

Classification: Find leaf for new data point, predict majority class.

Regression: Similar, but predict the mean of all values.

**Impurity measures**

- Misclassification error, leads to larger tree: $1-argmax_k \hat p_k$
  -  ​
- Gini-index (probablistic predictions)
  -  ​
- Entropy: How likely will random example have class $k$?
  - $E(X) = - \sum_{k=1}^K \hat p_k log_2 \hat p_k$
- Information gain (Kullback–Leibler divergence) for choosing attribute $X_i$ to split the data.
  - $G(X, X_i) = E(X) - ...$
  - If your node is more pure after splitting, then your information gain goes up.
- Gain Ratio (not available in scikit-learn). 

#### Overfitting

Decision trees can very easily overfit the data. Regularization strategies:

- Pre-pruning: stop creation of new leafs at some point.
  - Limit the depth of the tree. (Setting a low `max_depth`  and `max_leaf_nodes`.)
  - Require a minimal leaf size. (Setting a higher `min_samples_leaf`.)
- Post-pruning: build full tree, then prune leafs. (Not yet supported in scikit-learn.)
  - Evaluate against *held-out* data.
  - Other strategies also exist.

We can also manually check the decision tree.

#### Other useful Decision properties to know

1. The DecisionTreeClassifier also returns the *feature importances*, so we can see which features are useful.
2. Want to use the decision tree for regression, then use the `DecisionTreeRegressor()` instead!
3. Decision trees are **very bad** at extrapolating data.
4. Works well with features on completely different scales, aka does not require normalization.
5. Tends to overfit easily, but we can use an ensemble of trees to avoid that.

### Ensemble Learning

Combines multiple decision trees to create a more powerful classifier:

1. Random Forests: Build randomized trees on random samples of the data.
2. Gradient boosting machines: Build trees iteratively, heigher weights to points misclassified by previous trees.
3. **Stacking** is another technique that builds a (meta)model over the predictions of each member.

Predictions are made by doing a vote over the members of the example.

#### Random forests

1. Take a *bootstrap sample* of your data.
   - Randomly sample with replacement.
   - Build a tree on each bootstrap.
2. Repeat `n_estimators`  times:
   - Higher values $\rightarrow$ more trees + more smoothing
   - Make prediction by aggregating the individual tree predictions.
     - Also known as Bootstrap aggregating or bagging.

- **RandomForest:** Randomize trees by considering only a random subset of features of size `max_features` in each node.
  - Small `max_features` $\rightarrow$ More different trees, more smoothing.
  - Default: $\sqrt{\text{n_features}}$ for classification and $\log_2 \text{n_features}$ for regression. (History, no real reason.)

How to make predictions?

- Classification: **soft voting**
  1. Every member returns probability for each class.
  2. After averaging, the class with the highest probability wins.
- Regression: *mean* of all predictions.

**ExtraTreesClassifier:** Similar to RandomForest, but will grow deeper trees and faster.

Most important parameters:

- `n_estimators` $\leftarrow$ Higher is better, but diminishing returns
- `max_features` $\leftarrow$ Default generally ok, smaller to reduce space/time requirements.
- `max_depth` or `leaf_size` 
- `n_jobs` $\leftarrow$ should be set to $-1$ for multicore processing.

RandomForests can use the **out-of-bag (OOB)** error to evaluate performance.

- For each tree grown, 33-36% of samples are not selected in bootstrap.
  - Called out-of-bootstrap (OOB) samples.
  - Predictions are made as if they were novel test samples.
  - Majority vote is computed for all OOB samples from all trees.
- OOB estimates test error is rather accurate in practice, as good as CV estimates.
- Can be calculated with `oob_error = 1 - clf.oob_score_`

RandomForests provide more reliable *feature importances*, based on many alternative hypotheses (trees).

##### Pros

1. Don't require a lot of tuning
2. Very accurate models
3. Handles heterogeneous features well.
4. Implicitly selects most relevant features.

##### Cons

1. Less interpretable and slower to train. (But parallellizable)
2. Don't work well on high dimensional sparse data, such as test.

#### Gradient Boosted Regression Trees (Gradient boosting machines)

**Gradient boosting machines:** 

1. Use strong pre-pruning to build very shallow trees.
2. Iteratively build new trees by increaing weights of points that were *badly predicted*.
   - Can finetune this with the **learning rate**.
   - Gradient descent finds optimal set of wegiths.
3. Repeats ` n_estimator` times.

Most important parameters:

- ​

Tends to use much simpler trees.

##### Pros

1. Among the most powerful + widely used models.
2. Works well on heterogeneous features & different scales.

##### Cons

1. Requires careful tuning, takes longer to train.
2. Does not work well on high-dimensional sparse data

##### Hyperparameters

1. `n_estimators` $\leftarrow$ Higher is better, but will start to overfit
2. `learning_rate` $\leftarrow$ Lower rate means more trees needed to get complex models.
3. Set `n_estimators` as high as possible, then tune `learning_rate`.
4. `max_depth` $\leftarrow$ Typically low (<5), reduce when overfitting.
5. `loss` (Default is usually good) $\leftarrow$ Cost function used for gradient descent, `deviance` (log-likelihood), `exponential` for classification and `ls` (Least squared error) for regression. 

#### XGBoost

**XGBoost:** Python library for gradient boosting that uses approximation techniques to make it faster.

- Can do 10x or 100x more boosting iterations.
- Is not in SKLearn, but you can install / import it [here](https://xgboost.readthedocs.io/en/latest/).

#### Heterogeneous ensembles

Combine models from different algorithms by letting each algorithm predict.

Can either vote over them or build a meta-classifier that trains on the predictions: **stacking**.