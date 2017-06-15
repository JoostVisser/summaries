# Time series summary

> 2DD23 - Time series analysis and forecasting

[TOC]

## Lecture 1

### Goal of this course

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

> *Important tip for throughout the course*
>
> We will be looking at the data in the context, not as just random numbers. In other words, we will be looking at why the data follows this trend and look for any explanation. For example:
>
> 1. Maybe there was a war during that period, causing the kings to die younger than usual.
> 2. December is summer in Australia, so that could explain the jump in souvenir sales each year.
>
> Always remember to look at the context of the data and not just the data itself!

### Schedule

Schedule for the course:

1. **Preliminary**
   - Exploratory Data Analysis 
     (Understand the data + get fingerprint by looking at the graph.)
   - Component Analysis
   - Seasonal Decomposition


2. **Focus 1 - Exponential Smoothing Models**
   - Simple Exponential Smoothing 
   - Holt's Exponential Smoothing 
   - Holt-Winters Exponential Smoothing
3. **Focus 2 - Box-Jenkins ARIMA Models**
   - ARMA Models for Stationary Series 
   - ARIMA Models for Non-Stationary Series 
   - SARIMA Models for Series with Seasonality
4. **Focus 3 - Multiple Time Series**
   - Probably don't have enough time for this, but will spend some time to it.
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

## Lecture 2

### Programming for upcoming weeks

Six datasets we will be discussing:

1. Kings dataset, with the ages of the death of successive kings.
2. Annual rainfall in London.
3. Volcanic dust dataset, where the eruptions are peaks in an otherwise flat graph.
4. Average skirt radius of women over time, showing a hype.
5. Monthly sales for Australian souvenir shop - A multiplicative (+ additive) seasonal model.
6. Births per month in New York - An additive seasonal model.

Basic steps of understanding what is in the data:

- Exploratory Data Analysis
- Component Analysis
- Seasonal Decomposition

Some concepts will look somewhat strange, but it will help us later on with the Box-Jenkins type of models.

### Exploratory Data Analysis

This part consists of looking at the graph. What are the main properties? Any sudden changes? Outliers? Any missing values?

We'll be *fingerprinting* the data with **autocorrelation**.
Fingerprinting looks at the main properties of this data, i.e. the answers of the previous questions.

*Small note:* If one value is missing, that's pretty dangerous, as e.g. our seasonality detection could be ruined.

**Autocorrelation function (time-lag 1):** Check the correlation coefficient of $\{(x_t, x_{t-1})\}$.
E.g. compare the current graph with the same graph shifted one time-step to the left. This results in a 2-D plot on which we can do linear regression. This results in an autocorrelation value $r_{1}$.The calculation is similar to the calculation of the variance.


Why not shift with lag 2 for $r_2$? Can we still see a relation? Or how about lag=3? If these are still correlated, then we can use this information to predict our next values.

- If we plot the `acf` (auto-correlation function) in R, we will get blue lines which indicate a *rough* guide where the correlation is significant.
- No autocorrelation? $\implies$ No history. 
  Therefore, it's very important to have an autocorrelation.
- Autocorrelation with timelag $k$ gives the **overall** correlation between $x_t$ and $x_{t-k}$.
- Notice that if there is a consistent positive trend, then the autocorrelation will be strongly possible for all values. Negative trend $\implies$ negative values.
- Looks at the *trend* and looks at how similar the graph looks. Negative trend $\implies$ looks reversed.

However, what if we have a recursive correlation, where $x_t$ depends on $x_{t-1}$, while in turn $x_{t-1}$ depends on $x_{t-2}$ and so forth, up and until $x_{t-k}$. 
Then we have a **partial autocorrelation**. `pacf(kings.ts)`

- Stepwise correlation between $x_t$ and $x_{t-k}$. 
- Can calculate the autocorrelation function via this partial autocorrelation function.
  (Outside of the scope of this course)

*Another note:* We can apply the log-transformation to the souvenir dataset. A log-transformation changes the multiplicative seasonal model to an additive seasonal model.

**Trend**

- Long term changes in mean/median
- Can be estimated & modelled
- Can be corrected for

**Seasonal Variation**

- Periodic variations over time
- Can be estimated, modelled and corrected for.

If we correct for the trend and the seasonal variation, we might find some other *cyclic variation* or other irregular fluctuations. Correcting for both to these things results in the **stationary time series**.

### R-code

#### Loading and plotting time-series

This weeks R-code consists of how to create a time-series object:

```R
# Here we will load and plot the kings dataset.
load("~/workspace/github/r-projects/time-series/data/kings.RData")
kings.ts <- ts(kings) # Loading the kings dataset as a timeseries
plot(kings.ts) # Plotting this object
```

In R, we can load the data and put it inside a TimeSeries object: `kings.ts <- ts(kings)`. We can plot this if we want to: `plot(kings.ts)`.

#### Autocorrelation and partial autocorrelation

```R
# Plot the first 12 lags of 
lag.plot(kings.ts, lags=12, do.lines=F)

# Calculating the correlation and auto-correlation function of the timeseries.
kings.acf <- acf(kings.ts)

# This plots ACF value vs lag. We can retrieve just a single variable via $ sign.
kings.acf			# Compare these two assignments.
kings.acf$acf		# And see how they differ.

# The partial autocorrelation plots.
kings.pacf <- pacf(kings.ts)
```

 In general, we can use `tsdisplay` to show a combination of all the three plots:

```R
# Show the normal, the autocorrelation and the partial autocorrelation plots.
tsdisplay(kings.ts)
```

#### Other useful plots

With this plot we can plot the effects between the different months, as well as the effects on each month over the years

```R
# Loading data and showing some fancy plots for the births dataset
load("~/workspace/github/r-projects/time-series/data/births.RData")
births.ts <- ts(births, start=c(1946,1), frequency=12)

forecast::tsdisplay(births.ts) # Showing correlation and autocorrelation

# Plot the seasonal stats for the births dataset for each year.
seasonplot(births.ts, season.labels=TRUE, year.labels=TRUE, col=rainbow(12))

# Plotting the difference in each month for all months
monthplot(births.ts)
```

## Lecture 3 - Getting a stationary time series

We will get a stationary time series by correcting the data in this order:

1. Find and subtracting the trend.
2. Correct the seasonality.

### Correcting the trend

**Goal:** Find a trend (long-term change in mean) and subtract the trend-line from the data. In particular, make sure that the trend is not dominant anymore.

There are various kinds of trends models that you can fit the data:

- Most common approach is to start with a straight line (linear regression).
  You can always do piecewise linear trend
- Fit a polynomial function.
- Exponential curve.
- S curve, Gompertz curve, Logistic curve.

#### Finite Differencing

Consider a linear function $f(x) = ax + b$. 
How can we change this into a linear function that doesn't increase/decrease anymore?
Well, we can apply *differentiation* to take care of the slope!

Consider a polynomial function $f(x) = ax^2 + bx + c$. How to get rid of the slope?
Well, we can differentiate twice!

How does differentiation work? We look at the slope between $x$ and $x+\delta$, namely $\frac{f(x) + f(x+\delta)} \delta$. What if we change this to a discrete version and look at the slope with steps of e.g. 12? This is **finite differencing**.

- Aim: remove trend effect.
- Higher order differencing might be needed with nonlinear trend.
- Don't go too far in differencing, as it might be too complicated (5 times is too much).

Fun fact: we can go in the reverse direction via integration! 
So we can get a prediction by integrating. However, the more we integrate, the more noise we have, so we increase the uncertainty.

### Correcting the seasonality (Seasonal decomposition)

#### Additive model

Here the seasonality is a constant difference w.r.t. each other.

$X_t = m_t + S_t + \epsilon_t$

- $m_t$ is the deseasonalized mean level (mean line of data).
- $S_t$ is the Seasonal effect (e.g. for all months of January we add a constant value)
- $\epsilon_t$ is the noise level.

#### Multiplicative model

Here, the seasonality is proportional to the seasonal factor (multiplicative / percentage based).

- $X_t = m_t \cdot S_t + \epsilon _t$
- $X_t = m_t \cdot S_t \cdot \epsilon_t$
  - Can be transformed to the additive model using the log transformation.

#### Principle

Calculate a deseasonalized mean level $m_t$ of the series.

- In math terms, we take the moving average over a timespan of $s$.
- There is a special endpoint weights with even span of periodicity.
- In terms of a year, we take the average of the year.

Estimate the Seasonal effect, $S_t$, of the series:

- Local additive effect: $S_t = X_t - m_t$. (January is the data minus the average.)
- Local multiplicative effect: $S_t = \frac{X_t}{m_t}$.
- Global effect: $S_s = \bar S_{t,s}$.

We assume the global one is the correct one. So in this case, our seasonality model would be:
$$
m_t + \bar S_t
$$
We can compare this value with $X_t$.

#### Residuals

So after the seasonal decomposition we still have some residual unexplained, namely:
$$
\epsilon_t = X_t - (m_t + \bar S_t)
$$
If it looks random then we're done and cannot find anything else. 
However, if we still have a pattern in the residuals, such as very small values, then our job is still not down. It could be that we applied the wrong models, but also

We can look at the autocorrelation again, as well as the partial autocorrelation. There might also be some outliers that in the data that we can see. If we see some significant (partial) autocorrelation in `tsdisplay`, then this model might not be good enough.

> #### Alternative
>
> We can also use an alternative for seasonal decomposition. LOESS uses local polynomial regression instead of the moving average, which is a better estimation.

#### Small trend

Suppose there is a time series that have a very small, smooth trend. Then the additive and the multiplicative model would result in roughly the same result. However, if there is an exponential model, then the differences might be much higher.

### Seasonal Differencing

Can we combine the differencing and the seasonal decomposition? This is roughly the same as seasonal decomposition, but instead of trying to fit a model we try to remove all the seasonal terms in there $(y_t \equiv \nabla_s x_t = x_t - x_{t-s})$.

- We can also apply higher order seasonal differencing for nonlinear patterns.
- It doesn't matter in which order we perform finite differencing and seasonal differencing.

### R code

#### Finite Differencing

Consider the births dataset again. What if we want to perform finite differencing one time?

```R
births.d1 <- diff(births.ts, differences=1)
tsdisplay(births.d1)
```

We see that at time-lag 12 and at time-lag 24 in the ACF plot and at time-lag 12 of the PACF that there still is seasonality.

#### Seasonal decomposition

Now we will remove the seasons. The births dataset looks like an additive dataset.

```R
births.deco <- decompose(births.ts, type="additive")
names(births.deco) # Show the variable names of births.deco
plot(births.deco)
```

The graph shows our seasonal model together with the moving estimate. Finally, we can model the residuals to see if there is any outliers or other thing that happens.

```R
tsdisplay(births.deco$random)
```

We can change the `type="additive"` to `type="multiplicative"` for multiplicative seasonal  decomposition. Showing `souvenir.deco$figure` we can see what the factor is that each month is multiplied with.

#### Seasonal differencing

Now we are going to look at what happens if we perform seasonal differencing.

```R
# Applying differencing with lag=12. 
births.sd1 <- diff(births.ts, lag=12, differences=1)
tsdisplay(births.sd1)
```

Here we can see that there is some trend in the data, so we can apply the **finite differencing** for a resulting stationary time series.

## Lecture 4 - Fourier analysis

### Spectral analysis

Goal: transform the time-domain plot into a frequency domain plot.
Basic idea: seasonality is a pure harmonic time series.
$$
x_t = R \cos (\omega t + \theta)
$$

- R = amplitude 
- $\omega$ = angular frequency = "# of cycles in unit of time"
- $\theta$ = phase = "starting point of time"

We can mix several waves with different frequencies to obtain a new pattern, which is the sum of the wave. This looks like a strange periodic multi-level wave.

- This way, we can model different seasonalities (monthly + weekly) as a single model.

$$
X_t = \sum_{i=1}^k R_i \cos(\omega_i t + \theta_t)
$$

The trick is that given a combined model, using a Fourier transformation we can decompose it into several sine waves with different frequencies. Therefore, any periodic signal can be built with harmonics under certain constraints.

If we then remove all high frequency harmonics, then we will get a version without fast periodic cycles, aka seasonality, which is similar to the moving average. Thus, we correct for the seasonality.

We can also return to the time-domain by doing an inverse Fourier transformation.

Limitations

- We cannot detect frequencies that go faster than the frequency time.
- Harmonics that go very slow can be missed of the time frame is limited.

#### Frequency domain

So now we have a graph with the different frequencies. What is fingerprinting for the frequency domain? Well, the spectrum and phase determine which time-series graph we get, so these are the fingerprints of the graph.

### Data Transformation

Here are some traps to avoid when transforming the data:

- **Data Consistency** - Prevent from combining data series that are very different.
  (Comparing "apples"  with "oranges" with different holidays etc)
- **Data Smoothing** - Discover systematic components in a noisy time series. 
  We can also aggregate and zoom in to find these components.
- **Data Stationarity** - ... :(
- Data Filtering



### R code

#### Signal Analysis

We want to transform the time-series into the frequency domain. There are two parts of 

```R
# Sample Periodogram
spectrum(births.ts)

# Sample Spectrum, which basically is a moving average over 10 frequencies.
spectrum(births.ts, span=10)
```

 We can try to detect harmonic components which can be translated to the seasonality by looking at the high values of the spectrum.

Furthermore, we can look for the spectrograph after removing finite differencing. We suppress the low frequencies because these are very slow waves.

```R
# Fourier transformation after applying finite differencing.
births.d1 <- diff(births.ts)
spectrum(births.d1)
spectrum(births.d1, span=10)
```

#### Moving average

If we want to filter the data, then we can perform the moving average filter over the data to remove the high frequencies and focus on the lower frequencies:

```R
# Filtering with moving average of 3 and 9
rain.ma3 <- filter(rain.ts, filter=rep(1/3,3), sides=2)
rain.ma9 <- filter(rain.ts, filter=rep(1/9,9), sides=2)
plot(rain.ma3)
plot(rain.ma9)
```

## Lecture 5 - Exponential smoothing

### Final exploratory data analysis

#### More smoothing

There are more advanced smoothing settings:

- **Exponentially Weighted Moving Average (EWMA)** - This only has one variable $\alpha$.
- **Running medians** - Take the median from a restricted moving window, which is more robust for outliers. :)
  - You can even apply special variants of this filter, for example: 3RSH
    - R: Iteravely take running medians
    - S: Split series at flat mesas and dales 
    - H: Apply additional Hanning MA-smoother.

*Note:* For time-series for economic or political situations, there are special R-packages that apply several special filters and scales for transforming the data to international standards, such as `x12`, `x12GUI` or `x13binary`.  Preferred within SARIMA-framework.

Again, our goal is to obtain a stationary time series which we use for further modelling.

> Tip
>
> You can have multiple seasonalities in your time-series. You can solve this in two ways:
>
> 1. Apply multiple seasonal differencing to the time-series.
> 2. Subdivide the data in multiple time-series, where you apply the weekly seasonality on the weekly value and another time-series for the yearly data where you can use the yearly time-series. 

#### Variance stabilization

**Goal:** Stabilize variance so that it is independent of the mean.

Suppose that our seasonality is multiplicative, then our variance is not independent but changes value. But then we can apply the natural logarithm to apply multiplicative model into an additive seasonal model.

Then we can apply better models to the new data and we can return to the data to the predicted data $\hat y_t$ by performing $e^x$. Disadvantage: confidence intervals will explode.

**Box-cox Transformation:** Look at transformation where the data $y_t$ will be transformed to $y_t^\lambda$ (roughly) with the exception of $\lambda=0$, which is $\ln y_t$.
Pro: Is popular and easy to use.
Con: Don't use it blindly as it would be a blackbox and it couldn't be good number after all. There needs to be an explanation for applying this transformation. (For example, BoxCox results in $\lambda = 0.0002$ that could indicate in a logarithmic transformation.)

#### Outliers and missing values

Outliers may have impact on the 'fingerprint' of the sample date. 

- If we don't have a clue for it, we can apply for both analysis for with and without the outlier, then if it doesn't result in a different answer, then there is no need to bother with it. However, this doesn't work so well in time-series. 
- We can consider to exclude it to the data, if we know the reason for it.
- *Missing values:* we can estimate a reasonable substitute for the missing values.

There are various different kinds of detection, classification and handling outliers, but this is outside the scope of this course.

#### Final remarks

Ask yourself the following questions:

1. Do you understand the context.
2. Have the right variables been measured?
3. Have all time-series been plotted?

Data cleaning:

1. Any outliers? If so, how to fix?
2. Any missing values? If so, how to fix?
3. Any obvious discontinuities? If so, how to fix it?
4. Does it make sense to apply a transformation?

Stationary Time Series:

1. Is trend present? If so, how to fix it?
2. Is seasonality present? If so, how to fix it?

### Exercise intermezzo

| Time-series | Trend        | Seasonality              |
| :---------- | :----------- | :----------------------- |
| A           | Strong trend | Weak (if any)            |
| B           | No trend     | Not yet clear            |
| C           | Cyclic wave  | ???                      |
| D           |              | Strong, additive pattern |

Because $A$ has a strong trend, we know that $A$ is probably either 1 or 2.

As for the frequency question:
$A \leftrightarrows 3$
$B \leftrightarrows 1$
$C \leftrightarrows 2$
$D \leftrightarrows 4$

**White noise:** Spectrum is flat with some amplitude.

### GoldenGate

So what exactly does the value of the harmonics mean? Well, it indicates the # of cycles a year:

- Frequency of 1: 1 cycle a year. Much larger than the rest.
- Frequency of 2: 2 cycles a year. We see that it's slightly larger than the rest.
- Frequency of 6: 6 cycles year.

Quite some higher harmonics? $\rightarrow$ More complicated pattern, then this could indicate that the seasonality was not perfect, but could e.g. be the squared wave shown in class.

More broadly, it depends on the scale was used, e.g. the Time scale what 1 unit is.

You can actually estimate a confidence bandwidth of the spectrum. This is indicated in the upper right corner.

## Lecture 6 - Exponential smoothing

### Types of exponential Smoothing

Three types of exponential smoothing:

1. Simple Exponential Smoothing
   - No trend, no seasonality.
2. Holt's Exponential Smoothing
   - Might contain trend, no seasonality
3. Holt-Winters Exponential Smoothing
   - Might contain trend, might contain seasonality

There are some various naive models, such as taking the mean or taking some slight form of trend, which can be done with functions like `meanf`. However, these barely take trend or seasonality into account, so these are not adequate to model the various time-series model and have way too much uncertainty.

That's why we're gonna use exponential smoothing.
**Goal:** Extending on the idea of decomposition. Generally, it does not contain an underlying model and as such no underlying stochastic processes.

Advantages over decomposition: more flexible with better prediction.

### Simple Exponential Smoothing

**Simple Exponential Smoothing:**

- No trend, no seasonality.

We can apply finite differencing and seasonal differencing to transform our time-series stationary. However, if you apply this, remember to apply the inverse differencing. This works, but we can do better.

**Idea:** Use all our data to get a *one-step ahead* forecast.
$$
\hat x_T (1) = c_0x_T + c_1 x_{T-1} + c_2 x_{T-2} + \cdots
$$
Note that the naive methods, such as the mean or using the last data point, are specific methods of this exponential smoothing method.

Idea for weights: **Geometric weight**: $a(1-a)^i$. $0 \leq \alpha \leq 1$

- The farther away the value in the past, the less the weight will be, as $i$ is dependent on the time.

#### Meaning of alpha and the exponential smoothing formula

Now, we can rewrite the one-step ahead forecast with some recursive mathematics:
$$
\hat x_T(1) = \alpha x_T + (1-\alpha) \hat x_{T-1}(1)
$$

- $\alpha = 1$: Immediate response, grab the previous value - Immediate update.
- $\alpha=0$: Slow response, grab the value it always was in the history.- No update.

$$
\hat x_T(1) = \hat x_{T-1}(1) + \alpha (x_T - \hat x_{T-1}(1))
$$

Overview: $x_{t-1}(1) = \hat L_{t-1}$ is our forecast for $\hat x_t$. Suppose we get $x_t$, we can update our "level forecast" as follows:
$$
\hat L_t = \alpha x_t + (1-\alpha) \hat L_{t-1}
$$

- The **level** is the value of the stationary time-series.

Simple Exponential Smoothing allows for updates of level estimates.

Using some mathematics, we can find the optimal value for alpha, but this will be for next week. 

#### Applying exponential smoothing to a data-set.

If we apply simple exponential smoothing, we get a prediction. We can use this prediction to calculate the sum of square errors to see how well our model performs on the actual data set. 

- Would be preferred to use training set and validation set, but this is outside of the scope of this course.

### R-code

#### Simple Exponential Smoothing

We use HoltWinters with it, but we ignore the beta and the gamma. Our alpha is the *initial* value for the estimate ($\hat L_1 \approx x_1$).

```R
# Applying simple exponential smoothing on the rain time-series.
rain.ses1 = HoltWinters(rain.ts, alpha=0.2, beta=FALSE, gamma=FALSE)
name(rain.ses1) 		# Available information
rain.ses1 				# All parameters
rain.ses1$fitted 		# Data points, notice that xhat = level.
plot(rain.ses1$fitted) 	# We're looking at a very basic model.
plot(rain.ses1) 		# Both prediction and realization plots
```

## Lecture 7 - Holt's Exponential Smoothing

### Simple Exponential Smoothing 2

#### Goodness-of-Fit Measures

There are various difference measures to check how well our estimation performs with respect to the data. Examples of this would be:

Taking the signs into account for the bias:

- $ME = $ Mean Error
- $MPE = $ Mean Percentage Error

Taking the signs not in account, for the variability:

- $MAE = $ Mean Absolute Error
- $SSE = $ Sum of Squared errors
- $RMSE = $ Root Mean Squared Error
- $MAPE = $ Mean Absolute Percentage Error

There exists libraries in R that automatically finds the optimal $\alpha$ with respect to a certain error measure (SSE + regularization penalty).

#### Holt-Winters Forecasts

So with our Simple Exponential Smoothing model, how do we look into the future. We can derive a forecast using the same mathematics.

The two-step ahead uses the value of the one-step ahead prediction, as this is our best prediction. So does the three-step ahead prediction with the two- and one-step ahead prediction. 

In fact, the values will stay the  same because of this reason. Thus our prediction will be a horizontal line.

#### Checking the residuals of the model

We can actually check all the residuals of the model, i.e. the differences between the model and the actual value. In fact, we can even fingerprint these residuals to check the autocorrelation function and the partial autocorrelation function.

If there is still some significant autocorrelation, then perhaps we didn't use the correct $\alpha$ value and we should try a different model.

We can even check for normality of the residuals, which was part of our assumption.

We can even do an (uncommon) statistical test by combining different time-lags to see for any statistical significant different between - Prediction vs Actual value - :

- This is called the Box-Ljung test.
- Generally: `lag=10` when not having seasonality and `lag=20` if we do have one.

### Exponential smoothing with trend

Now we'll add an extra variable. First, let us recap $\alpha$:

- Level $\alpha$ is the level of the model, which is basically the standard value without trend and seasonality.

Now, the estimate of the trend will use both $\alpha$ and a new parameter $\beta$, which is the trend parameter. For the type of model where there is a clear trend we can use the **Holt's model**.

The principle consists of:

- $x^T_{t-1} = $ The value of the data-point.
- $\hat L_{t-1} = $ The estimate of the level of the current time-step.
- $\hat T_{t-1} = $ The estimate of the trend of the current time-step.

First we estimate our current value:
$$
\hat x^T_{t-1}(1) = \hat L_{t-1} + \hat T_{t-1}
$$
Then we can update for the next values.

- The level update, which is done accordingly:
  - $\hat L_t = \alpha x_t^T + (1-\alpha) (\hat L_{t-1} + \hat T_{t-1})$
- The trend estimate, with the following formula:
  - $\hat T_t = \beta(\hat L_t - \hat L_{t-1}) + (1-\beta)\hat T_{t-1}$
  - $\beta$ close to 1: immediate change to the present, fast changes.
  - $\beta$ close to 0: Slow changes, more to the past. (Does **not** mean that there is no trend.)
- Finally, we can estimate the next value of the datapoint $\hat x_{t-1}^T$.

*Small side-notes:*

- Similar to ARIMA(0,2,1) model
- Two parameter version of exponential smoothing. 
- Brown's Exponential Smoothing is $\alpha = \beta$. (Less prone to overfitting.)
- Influence of *inital values* for the level and the trend doesn't influence the result much. We can just use the value of $\hat L_1 \approx x_1$ and $\hat T_1 \approx x_2 - x_1$.
- We can even add constraint to the score function, such as no negative values for forecast volcano activity.

### R-code

#### Forecasting

With this function, we can see the predictions of our models for the next time-steps, also known as forecasting.

```R
# Forecasting using the Simple Exponential Smoothing model for 10 timesteps.
rain.ses1.fore <- forecast.HoltWinters(rain.ses1, h=10)
```

#### Residuals
##### Residuals of the forecast

How does my residuals look like, which are the differences between the actual value with the forecast of my model. We can actually fingerprint this!

```R
# Performing fingerprinting on the residuals of our model
tsdisplay(rain.ses1.fore$residuals)
```

##### Box-Ljung test

There exists a test that combines different time-lags for the auto-correlations and see if there is a significant auto-correlation. 

```R
Box.test(rain.ses1.fore$residuals, lag=20, type="Ljung-Box")
```

##### Normality of residuals

We can also check the normality of the residuals:

```R
# Needs package "car"
qqPlot(rain.ses1.fore$residuals)
shapiro.test(rain.ses1.fore$residuals)
```

#### Model testing
##### Accuracy of our model

How do we check the In-Sample accuracy for our model? Luckily, there exists a standard `accuracy` function in R:

```R
# Check various errors, such as ME, RMSE, MAPE, etc.
with(rain.ses1, accuracy(fitted, x))
```

##### Optimal alpha estimated

If we want to find an optimal value for $\alpha$, then it can be automatically determined from the 'running' 1-step ahead.

```R
# Value with optimal alpha for minimizing the penalized empirical risk
rain.ses2 <- HoltWinters(rain.ts, beta=FALSE, gamma=FALSE)
rain.ses2		# Check the smoothing parameters
```

This uses a sum of squares plus some regularization penalty.

#### Holt's exponential smoothing

Here is the automatic implementation of the Holt's exponential smoothing.

```R
# Perform Holt's exponential smoothing to obtain a model for the skirts dataset.
skirts.hes <- HoltWinters(x=skirts.ts, gamma=FALSE)
skirts.hes 					# Check the smoothing parameters
plot(skirts.hes$fitted) 	# Plotting the xhat, level and trend.
```

#### Forecasting Holt's exponential smoothing

So, how our model forecast? Let us find the values that predict with the respective confidence interval.

```R
# Plotting the forecast of our Holt's exponential smoothing model.
skirts.hes.fore <- forecast.HoltWinters(skirts.hes, h=19)
plot.forecast(skirts.hes.fore)
```

## Lecture 8 - Holt-Winters Exponential Smoothing

### Exponential smoothing with trend and seasonality

There exists both an additive model and a multiplicative model for modelling the seasonality in the Holt-Winters Exponential Smoothing. We'll just be treating additive, but the multiplicative model is very similar other than just multiplying the seasonality factor instead of adding.

Now we have three components that is part of our prediction, namely the:

- Level - $\hat L_t$
- Trend - $\hat T_t$
- Seasonality - $\hat I_t$

For our prediction we can just add all these three components to each-other:
$$
x_{t-1}^{TS} = \hat L_{t-1} + \hat T_{t-1} + \hat I_{t-s}
$$
(Notice that for seasonality, it is $\hat I_{t-s}$ and not $\hat I_{t-1}$. For the multiplicative, we multiply $\hat I_{t-s}$ to both terms instead of adding. Furthermore, note that $TS$ just stands for Trend+Seasonality.)

New formula for $\hat L_t$ is very similar, but we correct for seasonality by subtracting $\hat I_{t-s}$ from $x_t^{TS}$. (For multiplicative seasonality, we have to divide 
The formula for the trend, $\hat T_t$, is the same as in Holt's ES.
Finally, the new formula for $\hat I_t$ is in a similar sense as the other formulas.

Remember, if we want to exclude e.g. the trend, then we have to put `beta=FALSE` instead of putting it to 0.

There are various ways of initializing the initial values. We're not going to discuss them in detail, as their are some automatic choices which are already implemented.

Finally, there are various optimization options to set constraints on the automatic hyper-parameter optimization, such as that all numbers should be non-negative.

### Automatic Exponential Smoothing Forecasting

**Goal:** Select the best model from a set of candidates.

Principles of automatic smoothing:

- Compare candidate models using a Goodness-of-Fit tests
  - Keep it Simple, aka account for complexity. (Uses $AIC$ as criterion.)
  - Independent validation set preferred
- Perform a residual analysis for the final model selected!

*Warning:* There does not exist a "single best model" but generally many models perform equally well. Some might have preferred properties over other models.

The automatic forecasting uses a more general class of models:

- Trend: None, Additive (+ Damped), and Multiplicative (+ Damaged)
- Seasonality: None, Additive, and Multiplicative.

We just consider every single possible combinations of these two classes ($5 \cdot 3 = 15$ combos).



### R code

#### Holt-Winters Exponential Smoothing

With the following code, we can perform Holt-Winters with automatic parameter optimization with regards to $SSE$.

```R
# Performing Holt-Winters ES on the data, with additive seasonality.
births.hw <- HoltWinters(births.ts, seasonal="additive")
births.hw				# Showing the smoothing parameters.
births.hw$fitted		# Running component estimates: xhat, trend, ...
plot(births.hw$fitted)	# Plotting all of the components.
plot(births.hw) 		# Plotting both the model and the actual values.
births.hw$SSE			# Running SSE

# In-Sample accuracy measurements
with(births.hw, accuracy(fitted, x))

# Plotting the forecast including the statistical intervals.
plot.forecast(births.hw.fore)

# Fingerprinting the residuals
tsdisplay(births.hw.fore$residuals)
Box.test(births.hw.fore$residuals, lag=20, type="Ljung-Box")

# Look for the normality of the data.
qqPlot(births.hw.fore$residuals)
```

Since alpha and gamma are very similar, we can put them equal to each other for reduced overfitting. Furthermore, notice that the residuals still have some statistical significant autocorrelation, perhaps there might be even another seasonality / cycle in the data.

- Normality assumption is used for the *confidence interval* and the forecast.

#### Automatic Exponential Smoothing Forecasting

Here we will perform automatic ES forecasting on the Kings dataset.

```R
# Performing automatic ES on the kings dataset.
kings.ets <- ets(kings.ts)
kings.ets				# Showing the model used + smoothing parameters + criteria.
accuracy(kings.ets)		# In-sample accuracy measurements
tsdiag(kings.ets)		# In-sample diagnostics
plot(kings.ets)			# Plotting all of the components.

# Plotting the forecast including statistical intervals
plot.forecast(kings.ets)

# Analyzing the residuals
tsdisplay(kings.ets.fore$residuals)
Box.test(kings.ets.fore$residuals, lag=10, type="Ljung-Box")
```
## Lecture 8 - Box-Jenkins ARIMA Models

### Box-Jenkins ARIMA models

The goal of these models is to find a theoretical finger print of the data. We want to figure out which Stochastic Processes or *models* we want to use for the data. 

Similar to the Exponential Smoothing, we will consider three different models:

1. ARIMA Models for Stationary Series
   - No tend nor seasonality
2. ARIMA Model for Non-Stationary Series
   - With trend but no seasonality
3. SARIMA Models for Series with Seasonality
   - With both seasonality and trend

### Models

How do we model the data we have and predict? Well, there are various models we can follow, but the general flow goes as follows.

1. Research Question - What do we want to do. Such as, how hot is it in Eindhoven?
2. Experiment - Obtain data via experiments. Examples:
   - Obtain temperatures for $N$ selected months. 
   - Obtain monthly temperatures for $N$ selected years.
   - Obtain monthly temperatures for $N$ consecutive years.
3. Create a stochastic model from the results:
   - Perhaps the temperature follows a normality with certain $\mu$ and $\sigma$.
   - Perhaps each month has its own factors - stochasts - where the data depends on this.
   - Perhaps there exists some individual stochasts where the model depends on these, depending on time.

Right now, we're going to look at the different formal models and their respective fingerprints. In the end, we will compare it to the actual fingerprint and combine them to get the fingerprint of the model that as best looks at the fingerprint of the actual data.

### Stochastic Processes

Consider a *stochastic process*. Stochastic means that the outcome of this process is random, so there can be *multiple realizations* of one stochastic process.

- However, with time-series we often know just **a single** realization.

Consider a stochastic process in the form of $X_{\{t\}} = \{\cdots, X_{t_1}, \dots, X_{t_2}, \cdots\}$. There are various statistics that we can consider of such stochastic process:

- Mean: $\mu_T = E(X_T)$
- Variance: $\sigma_t^2 = Var(X_t)$.
- Autocovariance: $\gamma(t_1, t_2) = E([X_{t_1} - \mu_{t_1}]\cdot[X_{t_2} - \mu_{t_2}])$

We will now consider the most basic stochastic processes.

#### Strictly Stationary Process

In this process, shifting the time origin by any $\tau$ has no effect on the distribution, as it will always remain the same. Consequences for the statistics:

1. $E(X_t) = \mu$
2. $Var(X_t) = \sigma^2$
3. Autocovariance: $\gamma(t_1, t_2) = \gamma(\tau) = cov(X_t, X_{t+\tau})$.
4. Autocorrelation: $\frac{\gamma(\tau)}{\gamma(0)}$ 

#### Weakly stationary process

#### Purely Random Process - White noise

Our building block is that each stochast, the random variable over time, i.e. each  follows a normality and is independent of the rest of the stochasts. 

1. $\mathbb E[Z_t] = 0$, as we assume it's the sum of independent random variables.
2. $\mathbb  V[Z_t] = \sigma^2$.

### R-code

#### Simulating white noise

```R
# Simulating white noise with ARIMA
ts.sim <- arima.sim(list(order=c(0,0,0)), n=100)
plot(ts.sim)
```

## Lecture 9 - ARMA explained

### Moving Average Process - MA(q)

$X_t =Z_t + \beta_1 Z_{t-1} + \cdots + \beta_q Z_{t-q} = (1 + b_1 B + \ldots + B^q) \cdot Z_t = \theta_q(B) \cdot Z_t$.

- B is the backshift operator
- $\theta_q(B)$ is a shorthand of doing the backshift operator once, twice, ..., $q$ times. 

So basically, it's a moving average of $Z_t$

Let us consider the following example, where the current time depends on:
Random noise at time stamp $t$ + Random noise at time stamp $t-1$.

$X_t = Z_t + 0.1 Z_{t-1}$

- $\mathbb E[X_t] = \mathbb E[Z_t + 0.1Z_{t-1}] = \mathbb E[Z_t] + 0.1 \mathbb E[Z_{t-1}] = 0 + 0 = 0$
  $$
  \begin{align*}
  Cov(X_t, X_{t+k}) 
  &= \mathbb E(X_t \cdot X_{t+k}) - \overbrace{\mathbb E(X_t) \cdot E(X_{t+k})}^{0} \\
  &= \mathbb E[(Z_t + 0.1 Z_{t-1})(Z_{t+k} + 0.1 Z_{t+k-1})] \\
  &= \mathbb E [
  \underbrace{Z_t Z_{t+k}}_{\begin{aligned}K \neq 0 &\rightarrow 0 \\ K = 0 &\rightarrow \mathbb V(Z_t)\end{aligned}} 
  + \underbrace{0.1 Z_t Z_{t+k-1}}_{\begin{aligned}K \neq 1 &\rightarrow 0 \\ K = 1 &\rightarrow 0.1 \mathbb V(Z_t)\end{aligned}} 
  + \underbrace{0.1 Z_{t-1}Z_{t-k}}_{\begin{aligned}K \neq -1 &\rightarrow 0 \\ K = -1 &\rightarrow 0.1 \mathbb V(Z_t)\end{aligned}} 
  + \underbrace{0.01 Z_{t-1} Z_{t+k-1}}_{\begin{aligned}K \neq 0 &\rightarrow 0 \\ K = 0 &\rightarrow 0.01\mathbb V(Z_t)\end{aligned}} ]
  \end{align*}
  $$

- If we divide everything by the variance, then we get the correlation $\rho$.



However, the autocorrelation is **not** a unique fingerprint of the data! There are, in fact, two different models with the same fingerprint:
$$
X_t = Z_t + 0.1 Z_{t-1} \qquad \text{and} \qquad X_t = Z_t + 10 Z_{t-1}
$$
The left one is what we want, the right one is exploding. How do we check it?

Well, we can check the polynomial function with the backshift operator that we used to get to $X_t$ from $Z_t$. For example, in this sense, we have on the left part $1 + 0.1x = 0$ and on the right part $1 + 10x = 0$. We can solve for $x$ and look at the unit circle (in real and imaginary). 
If it's outside the unit circle then we're save, but if it's inside the unit circle then we have a case of an exploding pattern.

**Condition for invertibility:** $\phi_p(B) \neq 0$. ($\sim$ Outside of the complex unit circle.)

#### Intermezzo

Witte noise: Elke tijdsunit heeft een random variable, een stochast, achter zich wat de value bepaald. We nemen aan dat deze normality volgt met $\mu = 0$ en variance is $\sigma^2$.

Now with the moving average model is just a model where we think that the current value is dependent on the stochastic value of the previous values.

Correlation: similar to ACF, just change the diagram with $k$ timesteps and see if there still is a correlation.

###  Autoregressive Process - AR(p)

Let's try another type of model! 
We're going to predict our current $X_t$ in relation to $Z_t$, but is more or less resembling the levels which it resembles in the past.
$$
X_t = \alpha_1 X_{t-1} + \cdots + \alpha_p X_{t-p} + Z_t
$$
*Interpretation:* Present = Moving Average of Past Values + Current Disturbance

Notation: $B \cdot X_t = X_{t-1}$

$\phi_p(B) \cdot X_t = Z_t$, with $\phi_p(B) 1 - \alpha_1 B - \cdots - \alpha_p B^p$, where the minus signs comes from moving the $X_{t-1}, \ldots, X_{t-p}$ to the left side.

Here, it actually is quite similar to MA(q), but the other way around!
Considering $AR(1)$, we actually get $X_t = Z_t + \alpha Z_{t-1} + \alpha^2 Z_{t-2} + \cdots$. 
Frankly, it looks like $AR (1) \leftrightarrows MA(\infty)$.

Similarly, we want that this model is not exploding. In this case, it should hold that $-1 < \alpha < 1$.

Statistics:

- $\mathbb E(X_t) = 0$
- $\mathbb V(X_t) = \frac{\sigma^2_Z}{1-\alpha}$
- $\rho(k)=$ some infinite series. We can also use the Yule-Walker equation for an iterative solution.

In fact, the autocorrelation keeps reducing. So the autocorrelation 

#### Example

Consider the following example: $X_t = 0.8 X_{t-1} + Z_t$.

Then the current value is quite related to the previous value. So this means that it won't fluctuate as much but instead will be lower-frequency wave.

If we do the reverse, $X_t = -0.8 X_{t-1} + Z_t$, then it's actually the reversed! So this means that it will take the negative of the previous value, and thus will change between positive and negative really fast.

*Fun tip:* For the auto-regressive model, it has a direct relation to the partial autocorrelation function instead of the ACF, as it has a chaining effect.

**Condition for stationary:** $\phi_p(B) \neq 0$. ($\sim$ Outside of the complex unit circle.)

### Autoregressive Moving Average Process - ARMA(p,q)

How about combining both of the models into a single model?
Then we have a more flexible model that can model both situations.

*Interpretation:* Process is influenced both by *levels* and by *disturbances* from the past.

$X_t = (\alpha_1 X_{t-1} + \cdots + \alpha_p X_{t-p}) + (Z_t + \beta_1 Z_{t-1} + \cdots + \beta_q X_{t-q} )$

Operator notation: $\phi_p (B) \cdot X_t = \theta_q (B) \cdot Z_t$

- $\phi_p(B) = 1-\alpha_1B - \cdots - \alpha_p B^p$
- $\theta_q(B) = 1 + \beta_1  B + \cdots + \beta_qB^q$

#### Seasonality

What if we add trend and seasonality by adding time steps of 12?

### R-code

#### Moving average simulations



#### Theoretical fingerprint

```R
ACF.theor <- ARMAacf(ma=c(0
```



## Lecture 10 - Fitting the model with the realization

The goal is to make a connection with the finger print and the theoretical finger print.

### Stationary models

Our model is a standard ARIMA model: $\phi_p(B) X_t = \theta_q(B)Z_t$.

These are the different stages of the ARIMA modelling procedure:

1. Stage 1 - Model Identification
   - Find the type + order of the adequate stochaastic process, namely the $p$ and $q$.
2. Stage 2 - Model Estimation
   - There are default packages for this in $R$, but it's for estimating the $\alpha$ and $\beta$ values.
3. Stage 3 - Model Verification
4. Stage 4 - Model Validation
   - Use independent data to validate the model.
5. Stage 5 - Model Use

#### Stage 1 - Model Identification

Find the type + order of an adequate stochastic process:

- Moving average model: 
  - Suppose we have $MA(2) \rightarrow X_t = z_t - 0.1 Z_{t-1} + 0.04Z_{t-2}$
    Autocorrelation: 1 high value, 2 high value, rest suddenly insign. - "Sudden drop"
    Partial autocorrelation: "Decaying"
- Autoregressive model:
  - Partial autocorrelation function: "Sudden drop"
    Autocorrelation function: "Decaying"

Remember that an autoregressive model can be changed to a moving average model. Together with noise, it could be that we have to use both models.

- ARMA:
  - Mix of both of them.

#### Stage 2 - Model Estimation

We want to estimate the parameters.

AR(p) model:

- Remove the average to all values.
- We can do a standard regression techniques, such as least squares estimation.

MA(q) model:

- Slightly more advanced to get a correct estimation.There are numerical methods needed.
- Maximum Likelihood Estimator.

We can test if the parameters are *significantly zero*, rule of thumb: $ N > 50$. Testing the $\alpha$ and $\beta$.

### Stationary series without trend

#### Volcanic dust example

##### Identification and estimation

We first look for a sudden drop or a decaying.
We can go for a really high order, say order of 22. But this results in a lot of coefficients, so we are first going to look at the first values of the ACF.

Decaying behaviour in ACF? $\rightarrow$ Perhaps $AR(3)$.
Sudden drop in PACF? $\rightarrow$ We have some idea that it might be an autoregressive model.
The sudden drops happens at timelag 2, so perhaps it is an $AR(2)$ model.

Or maybe we can consider $MA(3)$ as there is some sort of a sudden drop at timelag 3 at the ACF.
Perhaps even $ARMA(1,1)$?

With $MA(3)$ we assume that:
$$
AR(3) \vdash X_t - \mu = (1 + \beta_1 B + \beta_2 B^2+ \beta_3 B^3) \cdot Z_t
$$
Performing it with the `Arima` command in R results in the following coefficients:

- $\mu = 57.5$
- $\beta_1 = 0.7438$
- $\beta_2 = 0.45$
- $\beta_3 = 0.19$

Futhermore, we get the following values: $AIC$ is the Aki iki index that has some in-built penalty for more complicated formulas.

Finally, with the forecast, we notice that the confidence band go into the negative numbers, so we can comment on this model.

Furthermore, the arma gives a resulting standard error (s.e.). If the ratio of $\bar x - \mu$ and the standard error ($\frac s {\sqrt n}$) is greater than 2, then it's probably significant. It's kind of the signal to noise ratio. In this case, you notice that $\frac {57.5} {7.6}$ is much greater than 2, so the third order of the moving average is needed. Want a $p$-value? Then you have to do some more effort.

Moreover, the $AR(2)$ model has the following formula:
$$
\tilde X_t = 0.75 \tilde X_{t-1} - 0.12 \tilde X_{t-2} + Z_t
$$
Finally, we can do the $ARMA(1, 1)$ model:
$$
\tilde X_t = 0.58 \tilde X_{t-1} + Z_t + 0.12 Z_{t-1}
$$

##### Validation

In principle, we're going to predict in-sample values form the model and compare them with the actual values. This will be done with the one-step ahead prediction, i.e. $\hat x_t(1)$.

(Idea behind the prediction: in principle, you can predict the $Z_t$ from the $X_t$. With these $Z_t$ we can generate the $X_t$ again.)

There are various [Goodness-of-Fit](#Goodness-of-Fit-Measures) measures that we can use, although the information criteria can be better, namely AIC and BIC, as they penalize $k$ $\leftarrow$ The number of coefficients in the model.
In general: BIC for long series, $AIC_c$ for the relatively short series. The corrected version is slightly better.

Still some significance in the ACF of residuals? Then we can use the patterns which gives us the signal to be more careful in the order of my model.

In total, we can gather all AIC values and have *multiple* good candidate models that we can use.

> There are automated tests, but it only focusses on a single value, namely AIC for example.
> It doesn't look at the forecast, not at the residuals.
>
> - There is not a "single best model"
> - Don't forget to take into account to context.

### Code

```R
# Performing an MA(3) model on the Volcano timeseries.
volcano.arma.0.3 <- Arima(volcano.ts, order=c(0,0,3))
# c(x, y, z) --> AR(x), finite differencing of y and MA(z).
# Showsing all paarameters and variables.
volcano.arma.0.3

# Forecasts
volcano.arma.0.3.fore <- forecast(volcano.arma.0.3, h=19)
plot.forecast(volcano.arma.0.3.fore)

# Goodness-of-fit tests
accuracy(volcano.arma.0.3)

# Residuals fingerprint - In-sample diagnostics
tsdiag(volcano.arma.0.3)
tsdisplay(volcano.arma.0.3.fore$residuals)

# Performing the AR(2) model.
volcano.arma.2.0 <- Arima(volcano.ts, order=c(2,0,0))

# Performing automatic model selection
volcano.arma <- auto.arima(volcano.ts)
volcano.arma.fore <- forecast(volcano.arma, h=19)
```

## Lecture 11 - Series with trend and seasonality

### Fitting a model with trend

#### skirts.ts

For the skirts, there are two possibilities: 

- There might be just a trend
- There might be a slow cycle

By the way: $AR(1) \leftrightarrows ARIMA(0, 1, 0)$, since they both have one backshift operator and a similar formula, when the parameter of the $AR(1)$ is close to 1.

We'll look at the ACF of `skirts.ts`, `skirts.d1`, and `skirts.d2`. Notice that when we plotted the finite diffencing, we noticed that there still was some trend in `skirts.d1`.

When we look at the `tsdisplay(skirts.d1)`, then we can notice the following thing: 

- There is a drop at time lag 4, so there might be some sort of Moving Average, but this cannot be detected in the partial auto correlation.
- Perhaps some AutoRegressive part is in here.

When we look at `tsdisplay(skirts.d2)`, we notice that there is not really any trend in there anymore.

- Perhaps there is some $AR(1)$ and $MA(1)$, as there is a sudden drop at timelag 1.
- However, there is also a sudden time lag 5 of the ACF and timelag 5, 10 and perhaps even 15.
  Any seasonality? Perhaps extend for cyclic behaviour.

When we look at the results `Arima(1, 2, 1)`, we notice that the moving average is really small and the autocorrelation is quite small as well.

- We'll remove the `MA(1)` part as the coefficients is really slow.
  AR(1) is also quite low, but might be significant after recalculation.
- We want the model to be as simple as it can be.

Then we perform the model forecast.

We could now check the residuals and perform e.g. the Ljung-box test.

*Auto-ARIMA* results in the 2 times finite differencing and auto-regressive order 1. 

> The difference between autoregression and finite differencing.
>
> Autoregressive is actually related quite a lot to finite differencing, as shown in the formulas.
> $$
> (1 - \alpha_1\beta)(1 - \beta)^2 X_t = Z_t \\ \equiv (1-\tilde \alpha_1 B - \tilde \alpha_2 B^2 - \tilde \alpha_3 B^3)X_t = Z_t
> $$
> So we could also do AR(3, 0, 0) and this would also be a nice fit. However, this results in more parameters, and more overfitting.
>
> That's why we perform finite differencing first, because we know that there is some sort of trend and therefore we could already get some information out of the model. 
>
> - A factored polynomial is always preferred over a high-level model that is not factored, as we already know some solutions / coefficients that are fitted.
>
> That's why we always perform finite differencing first before applying the autoregressive model. Less parameters and less complexity.
>
> (Another reason is that we want to find the solutions that should be on or outside the unit circle. However, the finite differencing are an edge case that are *exactly* on the unit circle. Using finite differencing helps with this edge case.)

### Timeseries with trend and seasonality

#### Births.ts

Realize that seasonal differencing fits in the type of behaviour by $(1-B^s)X_t$.
It might also be that with an additive series that the order of $s$ is lower than a multiplicative series.

Dominant trend at seasonal at timelag 12.

Finite differencing once and seasonal differencing one results in a series that's kinda stationary.

Looking at the `tsdisplay(births.d1.sd1)`, then we notice that it could be:

- MA(3) and AR(1), but there are also possibilities.
- In the autocorrelation we have a sudden drop at 12, whereas at 12, 24 and perhaps 36 in the PACF it looks like a decaying behaviour. That's why we can look at the seasonal $MA(1)$.
- This results in $SARIMA(2, 1, 3)(0, 1, 1)_{12}$.

Looking at another SARIMA model, namely $SARIMA(2, 1, 3)(0, 1, 1)_{12}$ results in significant coefficients except for $AR(2)$. Now we can decide for $AR(1)$.

### Code

#### Skirts.ts

```R
## The skirt lengths
tsdisplay(skirts.ts)

# Finite differencing once
skirts.d1 <- diff(skirts.ts, differences=1)
tsdisplay(skirts.d1) # Looks like if there is still some trend.

# Finite differencing twice
skirts.d2 <- diff(skirts.ts, differences=2)
tsdisplay(skirts.d2) # Looks more stationary.

# Applying the ARIMA(1, 2, 1) model.
skirts.arima.1.2.1 <- Arima(skirts.ts, order=c(1, 2, 1))
skirts.arima.1.2.1

# Forecasting
skirts.arima.1.2.1.fore <- forecast(skirts.arima.1.2.1, h=19)
plot(skirts.arima.1.2.1.fore)

# Auto ARIMA
skirts.arima = auto.arima(skirts.ts)
skirts.arima

# Some stats.
accuracy(skirts.arima)
tsdiag(skirts.arima)

# And we should also forecast and check the tsdisplay of the $residuals.
# Can also do a qqplot and shapiro.
```

#### Births.ts

```R
# Performing SARIMA(3, 1, 3)(1, 1, 0)_12
births.sarima.3.1.3.1.1.0 <- Arima(births.ts, order=c(3, 1, 3), seasonal=list(order=c(1, 1, 10), period = 12)               
```

## Lecture 12 - Multivariate analysis

### Validation

Instead of in-sample verification, which can be prone to overfitting, we will validate on an *independent* data. Verification is similar to cross-validation, aka pre-selection and selecting the best model. 

We can do validation via a train-test split, which is done by *data partitioning*. Generally, we do so by taking the last 10% or 20% of the model as a test-set, whereas the others is the training set where we fit the parameters on.











### R-code

#### Data Partitioning

```R
# Data Partitioning
births.ts.fit <- window(births.ts, start=c(1946, 1), end=c(1957, 12))
births.ts.val <- window(births.ts, start=c(1958, 1))
```













## Report tips

- Explicit formulae should be mentioned! Together with the coefficients.
  - We can check if the formula is invertible.
- Try to search on the interwebz for extra information.
- Look at the difference in results of the validation and diagnostics of the model.
  - Forecasting? $\rightarrow$ Narrow band preferred
  - Understanding? $\rightarrow$ Order of the model is preferred.
- Judge the forecast and the residuals.
- Find the context for some outliers for the second time series.

The context for the first series is not really known.

- Report the procedure / intermediate steps while fitting an adequate model.

Idea: compare our best model with the automatic model.

- Generally, 2 times finite differencing is enough, but for some of the series of exercise 1 4 times finite differencing is needed!
- Extend the time-lag further for checking e.g. the seasonality. This can be done with `tsdisplay(...ts, lag.max=40)`.
- Only change 1 variable at a time.

Start with modelling and the conjecture itself, and then perform auto-ARIMA.

- Check if one model has a different behaviour in the confidence band when comparing models.

If you have a multiplicative seasonal pattern, then we can decide to go for a logarithmic transformation. Then we can apply a seasonal differencing of a lower order.

- With drift is similar to one times order finite differencing.

There are two procedures for model checking. 

1. Whole series available. I select a percentage (75%) and we use this for validation.
2. Woop

We should select one model checking and argue why we use them.