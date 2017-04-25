# Time series summary

> 2DD23 Time series analysis and forecasting

## Lecture 1

Goal of this course is to predict the future with forecasting. We'll learn some tools to do this, but we don't want this to be a black box:

- We can apply some techniques with some assumption which might not be valid.
- Maybe the implementation is incorrect.

We want to understand the basic models that are used these days in real-time in companies.

Build the bridge between the Theory (model) $\rightarrow$  Realization.

Main emphasis of this course is to understand what time series is.

### Introduction

Example: Age of Death of 42 Successive Kings of England. 
*Time aspect:* Generation of each king.
Not related? $\Longrightarrow$ Can use Statistics
Related? $\Longrightarrow$ Can use time series.

Idea: adapt our model with extra information to include that idea. Thus, don't forget about gathering data of the environment when applying a model to the data.

### Schedule

Schedule for the course:

1. **Preliminary**
   - Exploratory Data Analysis 
     (Understand the data + get fingerprint by looking at the graph.)
   - Component Analysis
   - Seasonal Decomposition


2. **Focus 1 - Exponential Smoothing Models**
   - Simple Exponential Smoothing 
   - Holts Exponential Smoothing 
   - Holt-Winters Exponential Smoothing
3. **Focus 2 - Box-Jenkins ARIMA Models**
   - ARMA Models for Stationary Series 
   - ARIMA Models for Non-Stationary Series 
   - SARIMA Models for Series with Seasonality
4. **Focus 3 - Multiple Time Series**
   - Probably don't have enough time for this, but can spend some time.
   - Regression model $\rightarrow$ Time dependent.
   - Tough stuff.

### Objectives

1. Description
   - Find main properties of the series: trend, seasonality, outliers, sudden changes, ...
2. Modeling & Explanation
   - Understand the mechanism that generates the time series.
3. Forecasting
   - Predict future values of the series.
4. Control
   - Improve control over physical & economic systems.

#### Examination

Probably going to be a hybrid approach, with a small written exam and a small assignment in groups.