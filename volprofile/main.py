import plotly.graph_objects as go
import numpy as np
import pandas as pd
import math

from volprofile.cfg import TEST_PATH


def getVP(df: pd.DataFrame, nBins: int = 20):
    """
    Returns:
        df (pd.DataFrame): consists of `minPrice` and
        `maxPrice` and `aggregateVolume`
    """
    volume_seri = df['volume'].to_numpy()
    price_seri = df['price'].to_numpy()

    minPrice, maxPrice = np.min(price_seri), np.max(price_seri)

    step = (maxPrice - minPrice) / nBins
    idxs = (price_seri - minPrice) // step

    idxs[idxs >= nBins] = nBins - 1

    volumes = [0] * nBins
    res: pd.DataFrame = pd.DataFrame(
        columns=['minPrice', 'maxPrice', 'aggregateVolume'])

    for i, key in enumerate(idxs):
        volumes[int(key)] += volume_seri[i]

    for i in range(nBins):
        res.loc[i] = [getPrice(i, step, minPrice) - step / 2, # min
                      getPrice(i, step, minPrice) + step / 2, # max
                      volumes[i]]                             # aggVolume

    return res


def getVPWithOHLC(df: pd.DataFrame, nBins: int = 20):
    """
    parameters:
        - df: columns (open high low close volume etc.)
    Returns:
        - (pd.DataFrame): consists of `minPrice` and
        `maxPrice` and `aggregateVolume`
    """
    precision_multiplier = 5
    volume_seri = df['volume'].to_numpy()
    high_seri = df['high'].to_numpy()
    low_seri = df['low'].to_numpy()

    minPrice, maxPrice = np.min(low_seri), np.max(high_seri)

    # 5 in the next line would increase the precision of final results
    # in the last step before returning must convert it again to nBins
    step = (maxPrice - minPrice) / (precision_multiplier * nBins)
    max_idxs = (high_seri - minPrice) // step
    min_idxs = (low_seri - minPrice) // step

    min_idxs[min_idxs >= nBins * precision_multiplier] = nBins * precision_multiplier - 1
    max_idxs[max_idxs >= nBins * precision_multiplier] = nBins * precision_multiplier - 1

    volumes = [0] * nBins * precision_multiplier
    res: pd.DataFrame = pd.DataFrame(
        columns=['minPrice', 'maxPrice', 'aggregateVolume'])

    for i, min_idx in enumerate(min_idxs):
        for j in range(int(min_idx), int(max_idxs[i]) + 1):
            volumes[j] += volume_seri[i] / (-int(min_idx) + int(max_idxs[i]) + 1)

    reducedSumVolumes = np.add.reduceat(volumes, np.arange(0, len(volumes), precision_multiplier))
    step = step * precision_multiplier
    if len(reducedSumVolumes) != nBins:
        raise Exception("length of reduced volumes not consistent with number of bins")

    for i in range(nBins):
        res.loc[i] = [getPrice(i, step, minPrice) - step / 2,
                      getPrice(i, step, minPrice) + step / 2,
                      reducedSumVolumes[i]]

    return res


def getKMaxBars(volprofile_result: pd.DataFrame, k: int):
    return volprofile_result.nlargest(k, ['aggregateVolume'])


def getUnusualIncreasingBars(df: pd.DataFrame, isUpward: bool):
    if not isUpward:
        df = df.iloc[::-1]

    df['MA3'] = df['aggregateVolume'].rolling(3).mean()
    df['diff'] = (df['aggregateVolume'] / df['MA3']) * 100 - 100
    std = df['diff'].std()
    df['significance'] = df['diff'] > (math.pi / 2) * std

    return df[df['significance'] == True]


def plot(df: pd.DataFrame, price):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Bar(
        name='volumeProfile',
        x=df.aggregateVolume,
        y=(df.minPrice + df.maxPrice) / 2,
        orientation='h'), row=1, col=2)

    fig.add_trace(go.Scatter(name='close', y=price,
                  mode='lines', marker_color='#D2691E'), row=1, col=1)

    fig.show()


def getPrice(idx, step, minPrice):
    return minPrice + (2 * idx + 1) * step / 2


def _test():
    df = pd.read_csv(TEST_PATH)[-500:-50]
    df['price'] = (df['high'] + df['low']) / 2

    res = getVP(df)
    plot(res, df.price)

    res = getVPWithOHLC(df)
    plot(res, df.price)




if __name__ == "__main__":
    _test()
