from .ganblr import GANBLR
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from scipy.stats import truncnorm
from itertools import product
from copy import copy
import numpy as np

class DMMDiscritizer:
    def __init__(self, random_state):
        self.__dmm = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=2 * 5, reg_covar=0, init_params='random',
            max_iter=1500, mean_precision_prior=.1,
            random_state=random_state)

        self.__scaler = MinMaxScaler()
        self.__ordinal_encoder = OrdinalEncoder(dtype=int)
        #self.__label_encoders = []
        self.__dmms = []
        self.__arr_mu = []
        #self.__arr_mode = []
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
        self.__internal_fit(x_scaled)
        return self

    def transform(self, x) -> np.ndarray:        
        x = self.__scaler.transform(x)
        arr_modes = []
        for i, dmm in enumerate(self.__dmms):
            modes = dmm.predict(x[:,i:i+1])
            modes = LabelEncoder().fit_transform(modes)
            arr_modes.append(modes)
        return self.__internal_transform(x, arr_modes)

    def fit_transform(self, x) -> np.ndarray:
        assert(isinstance(x, np.ndarray))
        assert(len(x.shape) == 2)

        x_scaled = self.__scaler.fit_transform(x)
        arr_modes = self.__internal_fit(x_scaled)
        return self.__internal_transform(x_scaled, arr_modes)

    def __internal_fit(self, x):
        arr_mode = []
        for i in range(x.shape[1]):
            cur_column = x[:,i:i+1]
            dmm = copy(self.__dmm)
            y = dmm.fit_predict(cur_column)
            lbe = LabelEncoder().fit(y)
            mu  = self.__dmm.means_[:len(lbe.classes_)]
            sigma = np.sqrt(self.__dmm.covariances_[:len(lbe.classes_)])

            arr_mode.append(lbe.transform(y))
            #self.__arr_lbes.append(lbe)
            self.__dmms.append(dmm)
            self.__arr_mu.append(mu.ravel())
            self.__arr_sigma.append(sigma.ravel())
        return arr_mode

    def __internal_transform(self, x, arr_modes):
        _and = np.logical_and
        _not = np.logical_not

        discretized_data = []
        for i, (modes, mu, sigma) in enumerate(zip(
            arr_modes,
            self.__arr_mu, 
            self.__arr_sigma)):

            cur_column = x[:,i]
            cur_mu     = mu[modes]
            cur_sigma  = sigma[modes]
            x_std      = cur_column - cur_mu

            less_than_n3sigma = (x_std <= -3*cur_sigma)
            less_than_n2sigma = (x_std <= -2*cur_sigma)
            less_than_n1sigma = (x_std <=   -cur_sigma)
            less_than_0       = (x_std <=            0)
            less_than_1sigma  = (x_std <=    cur_sigma)
            less_than_2sigma  = (x_std <=  2*cur_sigma)
            less_than_3sigma  = (x_std <=  3*cur_sigma)
            
            base = 8 * modes
            discretized_x = np.full_like(cur_column, np.nan, dtype=int)
            discretized_x[less_than_n3sigma]                                = base
            discretized_x[_and(_not(less_than_n3sigma), less_than_n2sigma)] = base + 1
            discretized_x[_and(_not(less_than_n2sigma), less_than_n1sigma)] = base + 2
            discretized_x[_and(_not(less_than_n1sigma), less_than_0)]       = base + 3
            discretized_x[_and(_not(less_than_0)      , less_than_1sigma)]  = base + 4
            discretized_x[_and(_not(less_than_1sigma) , less_than_2sigma)]  = base + 5
            discretized_x[_and(_not(less_than_2sigma) , less_than_3sigma)]  = base + 6
            discretized_x[_not(less_than_3sigma)]                           = base + 7
            discretized_data.append(discretized_x.reshape(-1,1))
        
        return self.__ordinal_encoder.fit_transform(np.hstack(discretized_data))

    def inverse_transform(self, x) -> np.ndarray:
        def __assign(arr, flag, mu, sigma):
            arr[flag] = mu[flag] + sigma[flag]
        x = self.__ordinal_encoder.inverse_transform(x)
        x_modes = x // 8
        x_bins = x % 8
            
        inversed_data = []
        for i, (mu, sigma) in enumerate(zip(
            self.__arr_mu, 
            self.__arr_sigma)):
           
            cur_column_modes = x_modes[:,i]
            cur_column_bins  = x_bins[:,i]
            cur_column_mode_uniques    = np.unique(cur_column_modes)

            inversed_x = np.zeros_like(cur_column_modes, dtype=float)      

            for mode in cur_column_mode_uniques:
                cur_mode_idx = cur_column_modes == mode
                cur_mode_mu = mu[mode]
                cur_mode_sigma = sigma[mode]

                sample_results = self.__sample_from_truncnorm(cur_column_bins[cur_mode_idx], cur_mode_mu, cur_mode_sigma)
                inversed_x[cur_mode_idx] = sample_results
            
            inversed_data.append(inversed_x.reshape(-1, 1))

        return self.__scaler.inverse_transform(np.hstack(inversed_data))

    @staticmethod
    def __sample_from_truncnorm(bins, mu, sigma, random_states): 
        sampled_results = np.zeros_like(bins, dtype=float)
        def __sampling(idx, range_min, range_max):
            sampling_size = np.sum(idx)
            if sampling_size != 0:
                sampled_results[idx] = truncnorm.rvs(range_min, range_max, loc=mu, scale=sigma, size=sampling_size, random_states=random_states)
        
        #shape param (min, max) of scipy.stats.truncnorm.rvs are still defined with respect to the standard normal
        __sampling(bins == 0, np.NINF, -3)
        __sampling(bins == 1, -3, -2)
        __sampling(bins == 2, -2, -1)
        __sampling(bins == 3, -1,  0)
        __sampling(bins == 4, 0,   1)
        __sampling(bins == 5, 1,   2)
        __sampling(bins == 6, 2,   3)
        __sampling(bins == 7, 3, np.inf)
        return sampled_results     

class GANBLRPP:

    def __init__(self, numerical_columns):
        self.__discritizer = DMMDiscritizer()
        self.__ganblr = GANBLR()
        self._numerical_columns = numerical_columns
        pass
    
    def fit(self, x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=1):
        numerical_columns = self._numerical_columns
        x[numerical_columns] = self.__discritizer.fit_transform(x[numerical_columns])
        return self.__ganblr.fit(x, y, k, batch_size, epochs, warmup_epochs, verbose)
    
    def sample(self, size=None):
        synthetic_data = self.__ganblr.sample(size)
        numerical_columns = self._numerical_columns
        numerical_data = self.__discritizer.inverse_transform(synthetic_data[numerical_columns])
        synthetic_data[numerical_columns] = numerical_data
        return synthetic_data

    def evaluate(self):
        pass