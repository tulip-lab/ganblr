# Introduction

GANBLR is a tabular data generation model...
# Usage Example

In this example we load the [Car Dataset](https://archive.ics.uci.edu/ml/datasets/car+evaluation)* which is a built-in demo dataset. We use GANBLR to learn from the real data and then generate some synthetic data.

```python3
from ganblr.utils import get_demo_data
from ganblr.ganblr import GANBLR
from sklearn.preprocessing import OrdinalEncoder 

df = get_demo_data('car')
data = OrdinalEncoder(dtype=int).fit_transform(df)
x, y = data[:,:-1], data[:,-1]

model = GANBLR()
model.fit(x, y, epochs = 50)

#generate synthetic data
synthetic_data = model.sample(1000)
```

# Install

# Citation
If you use CTGAN, please cite the following work:

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
```