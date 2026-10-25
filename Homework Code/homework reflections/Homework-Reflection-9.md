# Homework Reflection 9 — Standard errors

**1. Heteroskedasticity.** I generate 2,000 independent datasets of 500 observations using X ~ N(0,1) and Y = 1 + 2X + (0.2 + X²)ε, where ε ~ N(0,1) independently. The empirical SD of the estimated X coefficient is **0.18170**, versus mean conventional OLS SE **0.08202** and mean HC3 SE **0.17624**. A pairs bootstrap with 1,500 resamples of a separate prespecified dataset gives **0.17033**. Conventional OLS substantially understates uncertainty because the outcome noise is largest at influential, extreme X values; HC3 and pairs resampling allow heterogeneous noise. I independently verify the ordinary OLS coefficient and SE against scipy.stats.linregress, a similar regression implementation allowed by the question. The bootstrap is conditional on one observed sample, whereas the full simulation generates fresh samples, so exact equality is not expected.

**2. Correlated errors.** I generate independent groups of 50 observations each, using X = a group predictor + 0.5 times individual noise and Y = 1 + 2X + √0.95 times a group shock + √0.05 times individual noise; all predictor and outcome-shock draws are independent. Within-group error correlation is 0.95. With 8 groups (400 rows), 2,000 full-DGP trials give slope SD **0.29607**, compared with mean OLS SE **0.04340**. Resampling individual rows from a separate dataset gives SE **0.02710**, while resampling whole groups gives **0.19337**. Row resampling destroys the dependence, so it substantially understates uncertainty; group resampling preserves it, but only eight independent groups provide limited information about the distribution of group shocks.

**3. More independent data.** At 200 groups (10,000 rows), the full-DGP SD is **0.05624**, mean OLS SE is **0.00895**, and the separate-sample row and group bootstrap SEs are **0.00985** and **0.06146**. Repeating the group bootstrap on 100 new source datasets shows its finite-sample variability directly. A large dataset helps when it supplies many independent groups and the bootstrap respects the dependence; more rows in the same few groups or more bootstrap resamples do not repair an inappropriate row bootstrap. These results demonstrate both the mismatch with ordinary OLS and the difference between bootstrap uncertainty from limited data and repeated simulation of the full generating process.

```python
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option('display.precision', 5)
plt.rcParams.update({'figure.figsize': (9, 4.8), 'figure.dpi': 110,
                     'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.2, 'axes.axisbelow': True})
BLUE, GOLD, DARK = '#2563EB', '#B7791F', '#374151'
R, B = 2000, 1500

def ols(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xc = x - x.mean()
    sxx = xc @ xc
    slope = xc @ (y - y.mean()) / sxx
    intercept = y.mean() - slope * x.mean()
    residual = y - intercept - slope*x
    conventional = np.sqrt((residual @ residual)/(len(x)-2)/sxx)
    leverage = 1/len(x) + xc**2/sxx
    hc3 = np.sqrt(np.sum((xc*residual/(1-leverage))**2))/sxx
    return slope, conventional, hc3

def pairs_bootstrap(x, y, rng, draws=B):
    slopes = []
    for start in range(0, draws, 100):
        idx = rng.integers(0, len(x), size=(min(100, draws-start), len(x)))
        xb, yb = x[idx], y[idx]
        xc, yc = xb-xb.mean(axis=1, keepdims=True), yb-yb.mean(axis=1, keepdims=True)
        slopes.extend(np.sum(xc*yc,axis=1)/np.sum(xc**2,axis=1))
    return np.std(slopes, ddof=1)


def hetero_data(rng, n=500):
    x = rng.normal(size=n)
    error = (0.2 + x**2)*rng.normal(size=n)
    return x, 1+2*x+error

rng = np.random.default_rng(9001)
hetero_trials = np.array([ols(*hetero_data(rng)) for _ in range(R)])
xh, yh = hetero_data(np.random.default_rng(9002))
check = stats.linregress(xh, yh)
np.testing.assert_allclose(ols(xh, yh)[:2], [check.slope, check.stderr], rtol=1e-10)
hetero_boot = pairs_bootstrap(xh, yh, np.random.default_rng(9003))
hetero_sd = hetero_trials[:,0].std(ddof=1)
hetero_summary = pd.DataFrame({
    'Method': ['Full DGP: empirical slope SD', 'Mean conventional OLS SE',
               'Mean HC3 robust SE', 'Pairs bootstrap SE: one sample'],
    'Uncertainty': [hetero_sd, hetero_trials[:,1].mean(),
                    hetero_trials[:,2].mean(), hetero_boot]})
hetero_summary['Ratio to empirical SD'] = hetero_summary.Uncertainty/hetero_sd
display(hetero_summary)
# E[X²(0.2+X²)²] = .04 + .4*3 + 15 = 16.24.
print('Large-sample theoretical SD:', np.sqrt(16.24/500))
print('Monte Carlo SD approximate SE:', hetero_sd/np.sqrt(2*(R-1)))
print('Independent scipy OLS coefficient and SE verification passed.')


fig, axes = plt.subplots(1,2,figsize=(12,4.8))
residual_h = yh-yh.mean()-ols(xh,yh)[0]*(xh-xh.mean())
axes[0].scatter(xh,residual_h,s=13,alpha=0.55,color=BLUE)
axes[0].set(title='Residual spread grows with |X|',xlabel='X',ylabel='OLS residual (outcome units)')
labels = ['Full DGP SD','OLS SE','HC3 SE','Pairs boot SE']
axes[1].bar(labels,hetero_summary.Uncertainty,color=[DARK,GOLD,BLUE,BLUE])
axes[1].set(title='Slope uncertainty: 2,000 trials, n = 500',ylabel='Slope units',ylim=(0,None))
axes[1].tick_params(axis='x',labelsize=10)
fig.tight_layout()
plt.show()


def cluster_data(rng, groups, size=50, rho=0.95):
    x = rng.normal(size=(groups,1)) + 0.5*rng.normal(size=(groups,size))
    error = np.sqrt(rho)*rng.normal(size=(groups,1)) + np.sqrt(1-rho)*rng.normal(size=(groups,size))
    return x, 1+2*x+error

def group_bootstrap(x, y, rng, draws=B):
    g, m = x.shape
    sx, sy = x.sum(axis=1), y.sum(axis=1)
    sxx, sxy = (x*x).sum(axis=1), (x*y).sum(axis=1)
    idx = rng.integers(0,g,size=(draws,g))
    sumx, sumy = sx[idx].sum(axis=1), sy[idx].sum(axis=1)
    numerator = sxy[idx].sum(axis=1)-sumx*sumy/(g*m)
    denominator = sxx[idx].sum(axis=1)-sumx**2/(g*m)
    return np.std(numerator/denominator,ddof=1)

rng = np.random.default_rng(9010)
cluster_trials, summaries, examples = {}, [], {}
for groups in [8,200]:
    trials = np.empty((R,3))
    for r in range(R):
        x,y = cluster_data(rng,groups)
        trials[r] = ols(x.ravel(),y.ravel())
    cluster_trials[groups] = trials
    x,y = cluster_data(np.random.default_rng(9100+groups),groups)
    examples[groups] = (x,y)
    empirical = trials[:,0].std(ddof=1)
    row_se = pairs_bootstrap(x.ravel(),y.ravel(),np.random.default_rng(9200+groups))
    group_se = group_bootstrap(x,y,np.random.default_rng(9300+groups))
    summaries.append({'Groups':groups,'Rows':x.size,'Mean slope':trials[:,0].mean(),
                      'Full DGP SD':empirical,'Mean OLS SE':trials[:,1].mean(),
                      'Row bootstrap SE':row_se,'Group bootstrap SE':group_se,
                      'Group boot / DGP SD':group_se/empirical})
cluster_summary = pd.DataFrame(summaries)
display(cluster_summary)
# Check an exact whole-group refit against sufficient statistics.
x,y = examples[8]
idx = np.random.default_rng(999).integers(0,8,size=8)
xf,yf = x[idx].ravel(),y[idx].ravel()
slope_sums = ((x[idx]*y[idx]).sum()-xf.sum()*yf.sum()/len(xf))/((x[idx]**2).sum()-xf.sum()**2/len(xf))
np.testing.assert_allclose(slope_sums,stats.linregress(xf,yf).slope)
print('Exact group-resampling refit verified against scipy.')


rng = np.random.default_rng(9400)
bootstrap_ratios = {}
for groups in [8,200]:
    reference_sd = cluster_summary.set_index('Groups').loc[groups,'Full DGP SD']
    ratios = []
    for sample in range(100):
        x,y = cluster_data(rng,groups)
        ratios.append(group_bootstrap(x,y,rng)/reference_sd)
    bootstrap_ratios[groups] = np.array(ratios)
ratio_summary = pd.DataFrame([
    {'Groups':g, 'Median ratio':np.median(v), '10th percentile':np.quantile(v,.1),
     '90th percentile':np.quantile(v,.9)} for g,v in bootstrap_ratios.items()])
display(ratio_summary)
fig,axes = plt.subplots(1,2,figsize=(12,4.8))
for index,row in cluster_summary.iterrows():
    vals = row[['Full DGP SD','Mean OLS SE','Row bootstrap SE','Group bootstrap SE']].to_numpy(dtype=float)
    axes[0].bar(np.arange(4)+(index-0.5)*0.36, vals, width=0.36,
                color=[BLUE,GOLD][index],label=f"{int(row.Groups)} groups")
axes[0].set(xticks=range(4),xticklabels=['Full DGP SD','OLS SE','Row boot SE','Group boot SE'],
            title='Error correlation = 0.95 within each group',ylabel='Slope units',ylim=(0,None))
axes[0].tick_params(axis='x',labelsize=9)
axes[0].legend(frameon=False)
axes[1].boxplot([bootstrap_ratios[8],bootstrap_ratios[200]],tick_labels=['8 groups','200 groups'],
                showfliers=True, medianprops={'color': DARK, 'linewidth': 1.5})
axes[1].axhline(1,color=DARK,linestyle='--',label='Bootstrap SE = full DGP SD')
axes[1].set(title='Group bootstrap accuracy across 100 samples',ylabel='Group bootstrap SE / full DGP SD',ylim=(0,None))
axes[1].legend(frameon=False,loc='upper right',fontsize=9)
fig.tight_layout()
plt.show()


print('Python:', platform.python_version())
for package in ['numpy','pandas','scipy','matplotlib','nbformat','nbclient']:
    print(package,importlib.metadata.version(package))
assert np.isfinite(hetero_trials).all()
assert all(np.isfinite(v).all() for v in cluster_trials.values())
assert all(np.isfinite(v).all() for v in bootstrap_ratios.values())
print('All simulation and bootstrap outputs are finite.')

```

Executed results and figures: [Week 9 notebook](../HW9/homework_9_standard_errors.ipynb).
