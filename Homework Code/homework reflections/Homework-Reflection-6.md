# Homework Reflection 6 — Treatment effects

### 1. What is a potential problem with comparing each untreated item to its counterfactual and taking the maximum difference?

Taking the maximum can select a large random error as well as a large true effect. If each estimated effect contains noise, searching many observations favors those whose errors happen to be positive, making the selected maximum prone to upward selection bias (the winner’s curse). A noisy outcome or poor counterfactual match can therefore determine the entire estimate, and the chosen observation may change in a new sample. The maximum is still the quiz’s requested statistic, but its value alone does not establish the largest true treatment effect.

### 2. Propose a solution and write code that implements it. Explain what the solution will do.

I use the 90th percentile of the untreated matched effects as the suggested upper-tail proxy. The code first calculates treated-counterfactual minus untreated-observed Y for every untreated item, then calls `np.quantile(effects, 0.90, method="linear")`. This summarizes a high point of the distribution without allowing one extreme item to set the result: **1.927962**, compared with the maximum **2.172470**. It reduces sensitivity to an isolated extreme value, but changes the target from a maximum to a percentile; it does not fully remove measurement error, matching bias, or unmeasured confounding. This proxy is for the reflection and does not replace Q4’s answer.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Run from the project root.
data = pd.read_csv(Path("Homework Instructions + Data Files/homework_6.1.csv"))
treated = data.loc[data["X"] == 1].reset_index(drop=True)
untreated = data.loc[data["X"] == 0].reset_index(drop=True)
model = NearestNeighbors(n_neighbors=1).fit(treated[["Z"]])
positions = model.kneighbors(untreated[["Z"]], return_distance=False).ravel()
effects = treated.iloc[positions]["Y"].to_numpy() - untreated["Y"].to_numpy()
print(f"Maximum: {effects.max():.6f}")
print(f"90th percentile: {np.quantile(effects, 0.90, method='linear'):.6f}")
```

Executed code, results, and the comparison figure are included in
[the Week 6 notebook](../HW6/homework_6.1_treatment_effects.ipynb).
