# Statistics Cheat Sheet

[TOC]

## Normality

Test for checking normality of the data

```flow
st=>start
cond=>condition: Data contains ties?
da=>end: Anderson-Darling
shap=>end: Shapiro-Wilk

st->cond
cond(yes)->da
cond(no)->shap
```

## Two independent samples

### Numerical - Homogeneity (variance)

Comparing variance of two samples

```flow
st=>start
norm=>condition: Data follows normality?
ftest=>end: F-test
bf=>end: Brown-Forsythe test

st->norm
norm(yes)->ftest
norm(no)->bf
```



### Numerical - Locality

Comparing location of two samples.

```flow
st=>start
norm=>condition: Normal data?
ft=>operation: Perform F-test
var=>condition: SD similar?
bf=>operation: Perform Brown-Forsythe test.
var2=>condition: SD similar?
shap=>condition: Data distrib 
similar?
eqtt=>end: T-test with equal variances
uneqtt=>end: T-test with unequal variances
wrst=>end: Wilcoxon Rank-sum test
ks=>end: Kolmogorov-Smirnov


st->norm
ft->var
bf->var2
norm(yes)->ft
norm(no)->shap
var(yes)->eqtt
var(no)->uneqtt
shap(yes)->bf
shap(no)->ks
var2(yes)->wrst
var2(no)->ks
```

### Binary

This test checks whether the 2 distributions follows the same $p$-value for the Bernoulli distrib.

```flow
st=>start
size=>condition: Sample size >20
ftest=>end: F-test
fe=>end: Fisher exact test
cs=>operation: Perform Chi-square test
e=>condition: All E >= 1?
e2=>condition: For 80%: E >=5?
suc=>end: Success!
fail=>end: Failure!

st->size
cs->e
size(yes)->cs
size(no)->fe
e(yes)->e2
e(no)->fail
e2(yes)->suc
e2(no)->fail
```

## Paired samples

Data is considered paired whenever two measurements are taken on the same unit:

- Two measurements of the same variable, i.e. grade of students on exam and re-exam.
- One measurement from two variables, i.e. height and width of a person.

### Continuous - Locality

Comparing location of two continuous samples.

```flow
st=>start: 
norm=>condition: Data follows 
normality?
sim=>condition: Distribution data 
similar?
ttest=>end: Paired T-test
wsrt=>end: Wilcoxon signed rank test
sign=>end: Sign test

st->norm
norm(yes)->ttest
norm(no)->sim
sim(yes)->wsrt
sim(no)->sign
```

### Binary

Compare whether two paired binary samples follow the same $p$ value (null-hypothesis).s

Generally use McNemar test.

## Correlated samples

A test of correlation that is a measure of **concordance**. Pairs $x_r, y_r$ and $x_s, y_s$ are concordant when $(x_r - x_s)(y_r - y_s) > 0$, discordant when $<0$. Aka measures correlation between two variables.

### Continuous

Can use:

- Kendell's tau
- Spearman's rank
- Pearson correlation

### Binary

Phi-coefficient

## Repeated measurements

There are two different types of ANOVA:

- One-way ANOVA random effects: Useful if the interest is in the full population of groups.
  - Overall view: being in a group has a random effect.
- One-way ANOVA fixed effects: Useful if the interest is in each of the included groups of the data.
  - In-depth view: Being in a particular group has a certain effect.

```flow
st=>start
anova=>operation: Perform any ANOVA
norm=>condition: Residuals follow 
					normality?
bf=>operation: Perform BF test
var=>condition: Variance 
				homogeneity?
int=>condition: Interest in groups?
kw=>end: Kruskall-Wallis
anovafix=>end: One-way ANOVA fixed effects
anovarand=>end: One-way ANOVA random effects

st->anova->norm
norm(yes)->bf
norm(no)->kw
bf->var
var(yes)->int
var(no)->kw
int(yes)->anovafix
int(no)->anovarand
```

## Randomness

### Binary data

- Use Wald-Wolfowitz test, also called Single runs test.

### Continuous data

```flow
st=>start
norm=>condition: Normal data?
ties=>condition: No ties in data?
rsac=>end: Rank Serial correlation
or Normal autocorrelation
bf=>end: Brown-Forsythe test
srt=>end: Single runs test for continuous data.
doom=>end: You're doomed!

st->norm
norm(yes)->rsac
norm(no)->ties
ties(yes)->srt
ties(no)->doom
```

## Outlier tests

### Single outlier tests

These tests generally assume normality. You have a choice of:

- Doornbos test with Bonferroni correction
- Grubbs' test
- Dixon test

### Multiple outlier tests

These are generally distribution free. 

- Hampel's rule
- Tukey's method - Generally implemented in the boxplots.