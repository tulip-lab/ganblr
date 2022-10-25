"""Data related functionalities."""

DEMO_DATASETS = {
    '': (
        '',
    ),
    '': (
        '',
    )
}

from sklearn.preprocessing import OneHotEncoder
from pandas import read_csv
import numpy as np

class DataUtils:
    """
    useful data utils for the preparation before training.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.data_size = len(x)
        self.num_features = x.shape[1]

        yunique, ycounts = np.unique(y, return_counts=True)
        self.num_classes = yunique
        self.class_counts = ycounts
        self.feature_uniques = [len(np.unique(x[:,i])) for i in range(self.num_features)]
        
        self.__ohe = None
    
    def get_ohe_x(self, use_dense=True) -> np.ndarray:
        if self.__ohe == None:
            self.__ohe = OneHotEncoder()
            self.__ohe.fit(self.x)
        ohex = self.__ohe.transform(ohex)
        if use_dense:
            ohex = ohex.todense()
        return ohex
    
    def set_ohe(self, ohe:OneHotEncoder):
        '''
        ohe: fitted Sklearn OneHotEncoder.
        '''
        self.__ohe = ohe