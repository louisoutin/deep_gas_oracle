import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tsmoothie.smoother import SpectralSmoother


class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset: pd.DataFrame,
                 features: list,
                 targets: list = None,
                 window_size: int = 200,
                 predict_size: int = 10,
                 smooth_fraction: float = 0.2):

        self.dataset = dataset
        self.window_size = window_size
        self.predict_size = predict_size
        self.features = features
        self.targets = targets
        if self.targets is None or len(self.targets) == 0:
            self.predict_mode = True
        else:
            self.predict_mode = False
        self.col_to_index = {dataset.columns[i]: i for i in range(len(dataset.columns))}
        self.features_idxs = [self.col_to_index[c] for c in features]
        self.targets_idxs = [self.col_to_index[c] for c in targets]
        self.smooth_fraction = smooth_fraction

    def __len__(self):
        return len(self.dataset) - self.window_size - self.predict_size

    def _fourrier_transform(self, x: np.ndarray):
        smoother = SpectralSmoother(smooth_fraction=self.smooth_fraction, pad_len=1)
        smoother.smooth(np.transpose(x))
        return np.transpose(smoother.smooth_data)

    def __getitem__(self, idx):
        """
        We return the time at which the prediction is done (not the target time we want to predict the gas price)
        :param idx:
        :return:
        """
        t_1 = idx+self.window_size
        # I copy the df to make sure the target will not be smoothed
        x_smoothed = self._fourrier_transform(self.dataset.iloc[idx:t_1, self.features_idxs].copy().values)
        if self.predict_mode:
            return {
                "x": x_smoothed,
                "t": np.datetime64(self.dataset.index[t_1 - 1]).astype(np.int64),
            }
        else:

            y_base = self.dataset.iloc[idx: t_1 + self.predict_size, self.targets_idxs].values
            y = np.empty(shape=(self.window_size, self.predict_size, len(self.targets_idxs)))
            for i in range(self.window_size):
                y[i] = y_base[i:i+self.predict_size]
            return {
                "x": x_smoothed,
                "y": y,
                "t": np.datetime64(self.dataset.index[t_1 - 1]).astype(np.int64),
            }
