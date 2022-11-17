
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from volprofile.cfg import path


def getVP(df: pd.DataFrame, nBins: int = 20):
    """
    Args:
        df (pd.DataFrame): must consist of two keys at least 1-`volume` 2-`price`
        nBins (int, optional): number of buckets. Defaults to 20.

    Returns:
        df (pd.DataFrame): consists of `minPrice` and `maxPrice` and `aggregateVolume`
    """
    volume_seri = df['volume'].to_numpy()
    price_seri = df['price'].to_numpy()

    if len(volume_seri) != len(price_seri):
        raise ValueError("len(volume) does not equal len(price)")

    minPrice, maxPrice = np.min(price_seri), np.max(price_seri)

    step = (maxPrice - minPrice) / nBins
    idxs = (price_seri - minPrice) // step

    idxs[idxs >= nBins] = nBins - 1
    idxs[idxs < 0] = 0

    volumes = [0] * nBins
    res: pd.DataFrame = pd.DataFrame(
        columns=['minPrice', 'maxPrice', 'aggregateVolume'])

    for i, key in enumerate(idxs):
        volumes[int(key)] += volume_seri[i]

    for i in range(nBins):
        res.loc[i] = [getPrice(i, step, minPrice) - step / 2,
                      getPrice(i, step, minPrice) + step / 2, volumes[i]]

    return res


def plot(df: pd.DataFrame, price):
    """plot output of getVP
    """
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
    df = pd.read_csv(path)[-500:-50]
    df['price'] = (df['high'] + df['low']) / 2

    res = getVP(df)

    plot(res, df.price)


if __name__ == "__main__":
    _test()
