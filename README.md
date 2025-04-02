# Introduction
A rust library for calculating Information Theoretics

# Examples
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install muturin numpy
```

```
python
>>> from information_theory import entropy
import numpy as np
data = np.random.normal(0,1,10000)
# entropy(data, data_type, bin_size)
#  - data: list of data points
#  - data_type: "data" (calculate directly) or "kde" (sample after calculating Kernel Density Estimation)
#  - bin_size: float, width of bins to calculate probabilitys, defaults to Scotts rule
result = entropy(data, "kde", None)
```
