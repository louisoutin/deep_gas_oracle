import pandas as pd


def resample_and_merge(df_blocks: pd.DataFrame,
                       df_prices: pd.DataFrame,
                       dict_params: dict,
                       freq: str = "5T"):

    df = resample(df_blocks, freq, dict_params)
    df_prices = resample(df_prices, freq, dict_params)

    # we add the 24h lagged variable of the mean gas price
    df["mean_gas_price_24h_lagged"] = df["mean_gas_price"].shift(288, axis=0)

    # we incorporate the eth price into our main DataFrame
    df["eth_usd_price"] = df_prices["eth_usd_price"]
    df["eth_usd_price"] = df["eth_usd_price"].ffill()
    return df.drop([df.index[i] for i in range(288)])  # We drop the first day because of the shift that includ NaNs.


def resample(df: pd.DataFrame,
             freq: str,
             dict_params: dict) -> pd.DataFrame:
    """

    :param df:
    :param freq: e: '5T' for 5 minutes
    :param dict_params:
    :return:
    """

    columns = []
    for c in df.columns:
        if c in list(dict_params.keys()):
            op = dict_params[c]
            if op == "mean":
                columns.append(df[c].resample(freq, label="right").mean())
            elif op == "last":
                columns.append(df[c].resample(freq, label="right").last())
            else:
                raise RuntimeError(f"{op} is not a valid resampling operation:"
                                   f" currently supported are 'mean' or 'last'")

    return pd.concat(columns, axis=1)


def clip_bounds(df: pd.DataFrame,
                dict_params: pd.DataFrame) -> pd.DataFrame:
    """

    :param df: the df to clip
    :param dict_params: ex : {col1: {'min': 0, 'max': 30}, col2: {'min': -10, 'max': 80}}
    :return:
    """
    for c in df.columns:
        if c in list(dict_params.keys()):
            low_bound = dict_params[c]["min"]
            up_bound = dict_params[c]["max"]
            df_c_clipped = df[c].clip(lower=low_bound, upper=up_bound,
                                      axis=0, inplace=False)
            df[c] = df_c_clipped
    return df


def clip_std(df: pd.DataFrame,
             dict_params: pd.DataFrame) -> pd.DataFrame:
    """

    :param df: the df to clip
    :param dict_params: ex : {col1: 1.5, col2: 2}
    :return:
    """
    for c in df.columns:
        if c in list(dict_params.keys()):
            std_mult = dict_params[c]
            while True:
                mean, std = df[c].mean(), df[c].std()
                low_bound = mean - std_mult * std
                up_bound = mean + std_mult * std
                df_c_clipped = df[c].clip(lower=low_bound, upper=up_bound,
                                          axis=0, inplace=False)
                if ((df_c_clipped - df[c]) ** 2).max() < 0.01:
                    break
                df[c] = df_c_clipped
    return df
