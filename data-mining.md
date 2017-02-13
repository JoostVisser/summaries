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