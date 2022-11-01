from .ganblr import GANBLR
from sklearn.mixture import BayesianGaussianMixture
import numpy as np

class DMMDiscritizer:
    def __init__(self, random_state):
        self.__dmm = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * 5, reg_covar=0, init_params='random',
            max_iter=1500, mean_precision_prior=.1,
            random_state=random_state)
        
        self.__arr_mu = None
        self.__arr_mode = None
        self.__arr_sigma = None
    
    def fit(self):
        return self

    def fit_transform(self) -> np.ndarray:
        pass

    def transform(self) -> np.ndarray:
        pass

    def inverse_transform(self) -> np.ndarray:
        pass

class GANBLRPP:

    def __init__(self):
        self.__discritizer = DMMDiscritizer()
        self.__ganblr = GANBLR()
        pass

    def fit(self, x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=1):
        x = self.__discritizer.fit_transform(x)
        return self.__ganblr.fit(x, y, k, batch_size, epochs, warmup_epochs, verbose)
    
    def sample(self, size=None):
        synthetic_data = self.__ganblr.sample(size)
        return self.__discritizer.inverse_transform(synthetic_data)

    def evaluate(self):
        pass
