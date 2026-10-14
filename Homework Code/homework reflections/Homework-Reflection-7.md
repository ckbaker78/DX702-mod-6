# Homework Reflection 7 — Regression

### 1. Create a regression with an omitted confounder; determine the direction of bias and explain it.

I generate independent standard-normal Z, treatment noise, and outcome noise, then set X = Z + treatment noise and Y = 2X + 3Z + outcome noise. Z is a confounder because it increases both X and Y. The true causal X coefficient is 2, but leaving Z out overestimates it: the fitted X-only slope is **3.463721**, compared with **1.974390** when Z is included. In the population, the omitted-variable bias is 3 × Cov(X,Z)/Var(X) = 3/2, so the X-only slope is 3.5. The omitted model attributes some of Z’s positive contribution to X. This is bias in the regression coefficient as an estimate of the causal effect; an observed correlation is not itself the causal coefficient.

### 2. Run a regression with a zero coefficient 1,000 times; report the smallest p-value and explain repeated analysis.

For each of 1,000 independent datasets of 100 observations, I generate independent standard-normal W, X, and outcome noise, set Y = 2X + outcome noise, and regress Y on an intercept, X, and W. W’s true coefficient is exactly zero. With seed 7027, the smallest two-sided W p-value is **0.000153434** (trial **279**), and **46** trials produce p < 0.05. This does not mean W’s coefficient is truly nonzero: under the null, a p-value measures the probability of a test statistic at least as extreme as the observed one, not the probability that the null is true. Selecting the smallest result after repeated testing creates false-positive evidence; with independent tests the probability of at least one p < 0.05 is 1 − 0.95^1000, essentially one. Prespecify the test, report all attempts, and if treating the 1,000 tests as a family use a multiplicity correction: the Bonferroni threshold is 0.05/1000 = **0.00005**, which even this minimum does not meet.

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from IPython.display import display
import matplotlib.pyplot as plt
BLUE, DARK = "#2563EB", "#374151"

def fit_ols(y, predictors):
    # Add an intercept and calculate conventional, homoskedastic OLS SEs.
    y = np.asarray(y, dtype=float)
    predictors = np.asarray(predictors, dtype=float)
    if predictors.ndim == 1:
        predictors = predictors[:, None]
    design = np.column_stack([np.ones(len(y)), predictors])
    assert np.linalg.matrix_rank(design) == design.shape[1]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    df = len(y) - design.shape[1]
    variance = residuals @ residuals / df
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * variance)
    pvalues = 2 * stats.t.sf(np.abs(beta / se), df)
    return beta, se, pvalues


# Reflection 1: positive omitted-variable bias with a known true coefficient.
rng = np.random.default_rng(42)
n = 1000
confounder = rng.normal(size=n)
x = confounder + rng.normal(size=n)
y = 2 * x + 3 * confounder + rng.normal(size=n)
omitted_beta, _, _ = fit_ols(y, x)
adjusted_beta, _, _ = fit_ols(y, np.column_stack([x, confounder]))
reflection_bias = pd.DataFrame({
    'Model': ['True causal coefficient', 'X only: confounder omitted', 'X and Z: confounder included'],
    'X coefficient': [2.0, omitted_beta[1], adjusted_beta[1]],
})
display(reflection_bias)
np.testing.assert_allclose(adjusted_beta[1:], LinearRegression().fit(np.c_[x, confounder], y).coef_)

fig, ax = plt.subplots()
ax.bar(['True', 'Z omitted', 'Z included'], reflection_bias['X coefficient'], color=BLUE)
ax.axhline(2, color=DARK, linestyle='--', label='True causal coefficient = 2')
ax.set(title='Omitting a positive confounder inflates the X coefficient',
       ylabel='X coefficient', ylim=(0, 4))
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# Reflection 2: every trial has a true W coefficient of zero.
rng = np.random.default_rng(7027)
reflection_trials = []
for trial in range(1000):
    w = rng.normal(size=100)
    x = rng.normal(size=100)
    y = 2 * x + rng.normal(size=100)
    beta, se, pvalues = fit_ols(y, np.column_stack([x, w]))
    reflection_trials.append({'Trial': trial + 1, 'W coefficient': beta[2], 'W p-value': pvalues[2]})
reflection_trials = pd.DataFrame(reflection_trials)
best = reflection_trials.loc[reflection_trials['W p-value'].idxmin()]
false_positives = int((reflection_trials['W p-value'] < 0.05).sum())
bonferroni_threshold = 0.05 / len(reflection_trials)
display(pd.DataFrame({
    'Measure': ['Smallest W p-value', 'Trial with smallest p-value',
                'W coefficient in that trial', 'Trials with p < 0.05', 'Bonferroni threshold'],
    'Value': [best['W p-value'], best['Trial'], best['W coefficient'], false_positives, bonferroni_threshold],
}))
assert best['W p-value'] > bonferroni_threshold

fig, ax = plt.subplots()
ax.hist(reflection_trials['W p-value'], bins=np.linspace(0, 1, 21), color=BLUE, edgecolor='white')
ax.axvline(0.05, color=DARK, linestyle='--', label='Nominal significance threshold = 0.05')
ax.set(title='P-values across 1,000 independent trials with a true zero W coefficient',
       xlabel='Two-sided p-value for W', ylabel='Trials')
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
```

Executed outputs and figures are in [the Week 7 notebook](../HW7/homework_7.1_regression.ipynb).
