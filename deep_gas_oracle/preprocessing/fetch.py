import pandas as pd
from google.cloud.bigquery import Client


def fetch_eth_blocks(client: Client, start_date: str, end_date: str):

    sql = f"""
    SELECT blocks.timestamp, blocks.number, blocks.transaction_count, blocks.gas_limit, blocks.gas_used, AVG(txs.gas_price) AS mean_gas_price, MIN(txs.gas_price) AS min_gas_price, MAX(txs.gas_price) AS max_gas_price
    FROM `bigquery-public-data.ethereum_blockchain.blocks` AS blocks INNER JOIN `bigquery-public-data.ethereum_blockchain.transactions` AS txs ON blocks.timestamp=txs.block_timestamp
    WHERE DATE(blocks.timestamp) >= DATE('{start_date}') and DATE(blocks.timestamp) <= DATE('{end_date}')
    GROUP BY blocks.timestamp, blocks.number, blocks.transaction_count, blocks.gas_limit, blocks.gas_used
    ORDER BY blocks.timestamp
    """
    df = client.query(sql).to_dataframe()
    df[["mean_gas_price", "min_gas_price", "max_gas_price"]] = df[["mean_gas_price", "min_gas_price",
                                                                   "max_gas_price"]] / 1000000000
    df["gas_used"] = df["gas_used"] / 100000
    df["gas_limit"] = df["gas_limit"] / 100000
    df = df.rename(columns={'gas_used': 'gas_used_percentage'})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.set_index("timestamp")
    return df


def fetch_eth_price(path_to_csv: str, start_date: str, end_date: str):
    # https://www.kaggle.com/prasoonkottarathil/ethereum-historical-dataset?select=ETH_1min.csv
    df_eth_usd = pd.read_csv(path_to_csv)
    df_eth_usd = df_eth_usd.iloc[::-1]  # We need to "reverse" the index from DESC to ASC
    df_eth_usd = df_eth_usd[["Date", "Close"]]
    df_eth_usd = df_eth_usd.rename(columns={'Date': 'timestamp', 'Close': "eth_usd_price"})
    df_eth_usd["timestamp"] = pd.to_datetime(df_eth_usd["timestamp"])
    df_eth_usd = df_eth_usd.set_index("timestamp")
    df_eth_usd = df_eth_usd[(df_eth_usd.index >= start_date) & (df_eth_usd.index <= end_date)]
    return df_eth_usd
