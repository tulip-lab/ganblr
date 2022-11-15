# Introduction

GANBLR is a tabular data generation model...
# Usage Example

In this example we load the [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)* which is a built-in demo dataset. We use `GANBLR` to learn from the real data and then generate some synthetic data.

```python3
from ganblr.utils import get_demo_data
from ganblr.ganblr import GANBLR
from sklearn.preprocessing import OrdinalEncoder 

# this is a discrete version of adult since GANBLR requires discrete data.
df = get_demo_data('adult')
x, y = df.values[:,:-1], df.values[:,-1]

model = GANBLR()
model.fit(x, y, epochs = 10)

#generate synthetic data
synthetic_data = model.sample(1000)
```

The steps to generate synthetic data using `GANBLR++` are similar to `GANBLR`, but require an additional parameter `numerical_columns` to tell the model the index of the numerical columns.

```python3
from ganblr.utils import get_demo_data
from ganblr.ganblr import GANBLRPP
import numpy as np

# raw adult
df = get_demo_data('adult-raw')
x, y = df.values[:,:-1], df.values[:,-1]

def is_numerical(dtype):
    return dtype.kind in 'iuf'

column_is_numerical = df.dtypes.apply(is_numerical).values
numerical_columns = np.argwhere(column_is_numerical).ravel()

model = GANBLRPP()
model.fit(x, y, epochs = 10)

#generate synthetic data
synthetic_data = model.sample(1000)
```
# Install

# Citation
If you use GANBLR, please cite the following work:

*Y. Zhang, N. A. Zaidi, J. Zhou and G. Li*, "GANBLR: A Tabular Data Generation Model," 2021 IEEE International Conference on Data Mining (ICDM), 2021, pp. 181-190, doi: 10.1109/ICDM51629.2021.00103.

```LaTeX
@inproceedings{ganblr,
    author={Zhang, Yishuo and Zaidi, Nayyar A. and Zhou, Jiahui and Li, Gang},  
    booktitle={2021 IEEE International Conference on Data Mining (ICDM)},   
    title={GANBLR: A Tabular Data Generation Model},   
    year={2021},  
    pages={181-190},  
    doi={10.1109/ICDM51629.2021.00103}
}
@inbook{ganblrpp,
    author = {Yishuo Zhang and Nayyar Zaidi and Jiahui Zhou and Gang Li},
    title = {<bold>GANBLR++</bold>: Incorporating Capacity to Generate Numeric Attributes and Leveraging Unrestricted Bayesian Networks},
    booktitle = {Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
    pages = {298-306},
    doi = {10.1137/1.9781611977172.34},
}
```