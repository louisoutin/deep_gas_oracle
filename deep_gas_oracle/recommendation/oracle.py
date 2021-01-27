import numpy as np
import pandas as pd

from ..preprocessing.norm import Normalizer
from ..modeling.model import GruMultiStep


class EthGasPriceOracle:

    def __init__(self,
                 model: GruMultiStep,
                 training_normalized_dataframe: pd.DataFrame,
                 scaler: Normalizer,
                 percentile_value: int = 20):
        """
        
        :param model: GRU model (min gas price target)
        :param training_normalized_dataframe: Normalized dataframe used to train the model
        :param scaler: Scaler used to train the model
        """

        self.model = model
        self.scaler = scaler
        # Init the min and max slopes from training dataset
        predictions = self.model.predict(training_normalized_dataframe, self.scaler,
                                         normalize=False, denormalize=True, use_ground_truth=False)
        slopes = self.compute_slopes(predictions)
        self.min_slope = min(slopes)
        self.max_slope = max(slopes)
        print("Slopes fitting done from training set...")
        print(f"min slope: {self.min_slope} / max slop: {self.max_slope}")
        self.percentile_value = percentile_value

    @staticmethod
    def compute_slopes(predictions: pd.DataFrame) -> np.ndarray:
        predictions = predictions.copy()
        t = predictions.shape[1]
        slopes = np.empty(len(predictions))
        for i in range(len(predictions)):
            # We take the first element that is the slope (the second element is the intercept, we drop it)
            slope = np.polyfit(np.linspace(0, t, num=t), predictions.iloc[i].values, 1)[0]
            slopes[i] = slope
        return slopes

    def _scale_slope(self, slope):
        return (slope - self.min_slope) / (self.max_slope - self.min_slope)

    def _get_coeficient(self, slope):
        scaled_slop = self._scale_slope(slope)
        return np.exp(2 * scaled_slop - 2)

    def recommend(self, dataset: pd.DataFrame, urgency: int = 1.0):
        predictions = self.model.predict(dataset, self.scaler,
                                         normalize=False, denormalize=True, use_ground_truth=False)
        slopes = self.compute_slopes(predictions)
        c = self._get_coeficient(slopes)
        g = np.percentile(predictions.values, self.percentile_value, axis=1)
        res = g * c * urgency
        return pd.DataFrame(res, columns=["gas_price_recommendation"], index=predictions.index)
