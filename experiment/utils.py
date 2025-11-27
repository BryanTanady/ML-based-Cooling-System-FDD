# Training utils
import pandas as pd
import random
from pathlib import Path
from collections.abc import Mapping

from fdd_system.ML.common.config.system import *
from fdd_system.ML.common.config.data import *

def cvt_dict_feats_to_np(dict_features: list[dict[str, float]], feature_names: list[str]) -> np.ndarray:
    return np.array([
        [float(sample[f]) for f in feature_names]
        for sample in dict_features
    ])


def _parse_training_data(data_paths: list[str], label: int, window_size: int, stride: int, col_names: list[str]) -> list[RawAccWindow]:
    """Parse list of paths to data with same label.
    
    Args:
        data_paths: a list of paths to training data csv, all of same label eg:
            all represent streaming data from normal operating condition or airflow blocked.
        label: an integer that will be assigned to the given data.
        window_size: The size of one windowed buffer sent from microcontroller
        stride: sliding window's stride
    
    Returns:
        A list of dictionary, each dict entry represents a singleton window.
    """
    data = []

    for path in data_paths:
        x = pd.read_csv(path)        
        n = len(x)
        windowed_data = [
            RawAccWindow.from_dataframe_public_dset(x.iloc[i:i + window_size], label, col_names)
            for i in range(0, n - window_size + 1, stride)
        ]

        data.extend(windowed_data)
    return data

def prepare_training_data(training_data: Mapping[int, list[str] ], shuffle: bool, col_names: list[str]) -> list[RawAccWindow]:
    """Parse training data csv into list of dictionaries, each represents a single window of data.
    
    Args:
        training_data: a dict mapping fault type to list of paths.
        shuffle: set to true to shuffle the order of the windows
    
    Returns:
        A list of RawAccWindow.
    """
    result = []
    for label, paths in training_data.items():
        result.extend(_parse_training_data(paths, label, SensorConfig.WINDOW_SIZE, SensorConfig.STRIDE, col_names))

    if shuffle:
        random.shuffle(result)
    
    return result

def auto_stft_params(
    window_size: int | None = None,
    sampling_rate: int | None = None,
):
    if window_size is None:
        window_size = SensorConfig.WINDOW_SIZE
    if sampling_rate is None:
        sampling_rate = SensorConfig.SAMPLING_RATE

    Ws = int(window_size)

    if Ws < 32:
        nperseg = Ws
    else:
        target = max(16, Ws // 3)
        nperseg = 1 << (target.bit_length() - 1)
        if nperseg > Ws:
            nperseg //= 2

    noverlap = nperseg // 2

    nfft = 1 << (nperseg - 1).bit_length()
    nfft = max(64, nfft)
    nfft = min(4096, nfft)

    return int(nperseg), int(noverlap), int(nfft)
