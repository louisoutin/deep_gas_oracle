import pickle
import pandas as pd
from sklearn import preprocessing
from .utils.RankGauss import GaussRankScaler


class Normalizer:

    def __init__(self, mapping_cols_normalizer: dict):
        self.scalers = {}
        for col, normalizer in mapping_cols_normalizer.items():
            if normalizer == "minmax":
                self.scalers[col] = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            elif normalizer == "rankgauss":
                self.scalers[col] = GaussRankScaler()
            elif normalizer == "std":
                self.scalers[col] = preprocessing.StandardScaler()
            elif normalizer == "robust":
                self.scalers[col] = preprocessing.RobustScaler()
            else:
                raise RuntimeError(f"Normalizer : {normalizer} not supported: currently available are "
                                   f"'minmax', 'rankgauss', 'robust' or 'std'")

    @staticmethod
    def load_from_file(path: str):
        with open(path, 'rb') as input:
            normalizer = pickle.load(input)
        return normalizer

    def fit(self, df: pd.DataFrame):
        for c in df.columns:
            self.scalers[c].fit(df[[c]])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        for c in df.columns:
            res[[c]] = self.scalers[c].transform(df[[c]])
        return res

    def invert(self, df: pd.DataFrame, cols_with_suffix: bool = True):
        res = df.copy()
        for c in df.columns:
            if cols_with_suffix:
                column_name = "_".join(c.split("_")[:-1])
            else:
                column_name = c
            res[[c]] = self.scalers[column_name].inverse_transform(df[[c]])
        return res

    def save(self, path: str):
        with open(path, 'wb') as output:
            pickle.dump(self, output)
