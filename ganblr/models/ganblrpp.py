from .ganblr import GANBLR
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import truncnorm
import numpy as np

class DMMDiscritizer:
    def __init__(self, random_state):
        self.__dmm = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * 5, reg_covar=0, init_params='random',
            max_iter=1500, mean_precision_prior=.1,
            random_state=random_state)
        
        self.__scaler = MinMaxScaler()
        self.__arr_lbes = []
        self.__arr_mu = []
        self.__arr_mode = []
        self.__arr_sigma = []
    
    def fit(self, x):
        """
        Do DMM Discritization.

        Parameter:
        ---------
        x (2d np.numpy): data to be discritize. must bu numeric data.

        Return:
        ----------
        self

        """
        assert(isinstance(x, np.ndarray))
        assert(len(x.shape) == 2)

        x_scaled = self.__scaler.fit_transform(x)
        for i in range(x.shape[1]):
            cur_column = x_scaled[:,i:i+1]
            y = self.__dmm.fit_predict(cur_column)
            lbe = LabelEncoder().fit(y)
            mu  = self.__dmm.means_[:len(lbe.classes_)]
            sigma = np.sqrt(self.__dmm.covariances_[:len(lbe.classes_)])

            self.__arr_mode.append(lbe.transform(y))
            self.__arr_lbes.append(lbe)
            self.__arr_mu.append(mu.ravel())
            self.__arr_sigma.append(sigma.ravel())
        return self

    def transform(self, x) -> np.ndarray:
        _and = np.logical_and
        _not = np.logical_not
        
        x = self.__scaler.transform(x)
        discretized_data = []
        for i, (mode, mu, sigma) in enumerate(zip(
            self.__arr_mode,
            self.__arr_mu, 
            self.__arr_sigma)):

            cur_column = x[:,i]
            cur_mu     = mu[mode]
            cur_sigma  = sigma[mode]
            x_std      = cur_column - cur_mu

            less_than_n3sigma = (x_std <= -3*cur_sigma)
            less_than_n2sigma = (x_std <= -2*cur_sigma)
            less_than_n1sigma = (x_std <=   -cur_sigma)
            less_than_0       = (x_std <=            0)
            less_than_1sigma  = (x_std <=    cur_sigma)
            less_than_2sigma  = (x_std <=  2*cur_sigma)
            less_than_3sigma  = (x_std <=  3*cur_sigma)
            
            discretized_x = np.full_like(cur_column, np.nan, dtype=int)
            discretized_x[less_than_n3sigma]                                = 0
            discretized_x[_and(_not(less_than_n3sigma), less_than_n2sigma)] = 1
            discretized_x[_and(_not(less_than_n2sigma), less_than_n1sigma)] = 2
            discretized_x[_and(_not(less_than_n1sigma), less_than_0)]       = 3
            discretized_x[_and(_not(less_than_0)      , less_than_1sigma)]  = 4
            discretized_x[_and(_not(less_than_1sigma) , less_than_2sigma)]  = 5
            discretized_x[_and(_not(less_than_2sigma) , less_than_3sigma)]  = 6
            discretized_x[_not(less_than_3sigma)]                           = 7
            discretized_data.append(discretized_x.reshape(-1,1))
        
        return np.hstack(discretized_data)
    
    def inverse_transform(self, x) -> np.ndarray:
        def __assign(arr, flag, mu, sigma):
            arr[flag] = mu[flag] + sigma[flag]
        _and = np.logical_and
        _not = np.logical_not
        data_size = len(x)

        inversed_data = []
        for i, (mode, mu, sigma) in enumerate(zip(
            self.__arr_mode,
            self.__arr_mu, 
            self.__arr_sigma)):

            cur_column = x[:,i]
            cur_mu     = mu[mode]
            cur_sigma  = sigma[mode]

            range_min = np.zeros_like(cur_column)
            range_max = np.zeros_like(cur_column)
            inversed_x = np.zeros_like(cur_column, dtype=float)

            inversed_x[cur_column == 0] = truncnorm.rvs()
            range_min[x == 0] = np.NINF
            __assign(range_max, x == 0, mu, -3 * cur_sigma)
            
            __assign(range_min, x == 1, mu, -3 * cur_sigma)
            __assign(range_max, x == 1, mu, -2 * cur_sigma)

            __assign(range_min, x == 2, mu, -2 * cur_sigma)
            __assign(range_max, x == 2, mu, -cur_sigma)

            __assign(range_min, x == 3, mu, -cur_sigma)
            __assign(range_max, x == 3, mu, 0)

            __assign(range_min, x == 4, mu, 0)
            __assign(range_max, x == 4, mu, cur_sigma)

            __assign(range_min, x == 5, mu, cur_sigma)
            __assign(range_max, x == 5, mu, 2 * cur_sigma)

            __assign(range_min, x == 6, mu, 2 * cur_sigma)
            __assign(range_max, x == 6, mu, 3 * cur_sigma)

            __assign(range_min, x == 7, mu, 3 * cur_sigma)
            range_max[x == 7] = np.Inf

    def fit_transform(self, x) -> np.ndarray:
        return self.fit(x).transform(x)

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
