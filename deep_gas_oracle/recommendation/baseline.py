import numpy as np
import pandas as pd


def geth_recommend(df_blocks: pd.DataFrame, lookback: int = 100, percentile: int = 60) -> pd.DataFrame:
    """

    Geth. The Ethereum client implementation in go, namely Geth, accounts
    for over 79% of all Ethereum clients. To recommend a gas price, Geth uses
    the minimum gas price of the previous blocks. It looks back at the 100 blocks
    preceding the current one and then uses the value of the 60th percentile of the
    minimum gas prices as the price recommendation.


    :param df_blocks:
    :param lookback:
    :param percentile:
    :return:
    """
    df_res = pd.DataFrame(data=np.zeros((len(df_blocks) - lookback, 1)),
                          columns=["gas_price_recommendation"],
                          index=df_blocks.index[lookback:])
    col_idx_min_gas_price = list(df_blocks.columns).index("min_gas_price")
    for i in range(lookback, len(df_blocks)):
        window = df_blocks.iloc[i - lookback: i, col_idx_min_gas_price].values.reshape(-1)
        df_res[df_res.index == df_res.index[i-lookback]] = np.percentile(window, percentile)
    return df_res
