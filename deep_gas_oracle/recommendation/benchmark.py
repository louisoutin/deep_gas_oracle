import numpy as np
import pandas as pd


def evaluate_recommender(df_blocks: pd.DataFrame, df_recommendations: pd.DataFrame) -> (pd.DataFrame, dict):

    # check if recommendation doesn't have the full range of the blocks to evaluate
    if df_recommendations.index[0] > df_blocks.index[0]:
        df_blocks = df_blocks.loc[df_recommendations.index[0]:].copy()

    start_block = df_blocks.loc[df_blocks.index[0], "number"]
    end_block = df_blocks.loc[df_blocks.index[-1], "number"]

    pendings = {}
    results = pd.DataFrame(np.zeros((len(df_blocks), 5)),
                           columns=["block_start",
                                    "inclusion_block",
                                    "blocks_waited",
                                    "price_recommended",
                                    "inclusion_price"])

    results["block_start"] = df_blocks["number"].values
    current_block = start_block
    last_bloc = start_block
    while (current_block <= end_block) or (len(list(pendings.keys())) != 0 and current_block <= last_bloc):
        while len(df_blocks[df_blocks["number"] == current_block]) == 0:
            # If the block is not present, we pass to the next one
            # (I got 10 absent blocks on the test period from the paper)
            current_block += 1
        price = df_blocks.loc[df_blocks["number"] == current_block, "min_gas_price"].values[0]
        while len(list(pendings.keys())) != 0 and min(list(pendings.values())) >= price:
            # process transaction
            block_txs = min(pendings, key=pendings.get)
            results[results["block_start"] == block_txs] = [block_txs,
                                                            current_block,
                                                            current_block - block_txs,
                                                            pendings[block_txs],
                                                            price]
            del pendings[block_txs]
        if current_block <= end_block:
            # get recommendation and add to the pendings
            curr_time = df_blocks[df_blocks["number"] == current_block].index[0]
            pendings[current_block] = df_recommendations[:curr_time].values[-1, 0]  # take last available recommendation
            last_bloc = current_block
        current_block += 1
    return results
