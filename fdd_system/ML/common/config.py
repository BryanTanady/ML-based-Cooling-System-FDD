"""This file defines all involved data classess"""
from dataclasses import dataclass
import numpy as np
from typing import Optional
import pandas as pd


"""This file stores all configs for system components, such as fan, sensor (how the buffer is structured), etc."""

class FanConfig():
    NUM_BLADES = 5

class SensorConfig():
    SAMPLING_RATE = 800
    WINDOW_SIZE = 1000
    STRIDE = 500

    
@dataclass(kw_only=True)
class RawInput:
    """Base class for any raw input. Currently this seems useless, but we need to
    think what happens if we consider adding new sensor other than accelerometer.
    
    Args:
        device_id: which fan/system is sending this window coming from. Optional, because when there
            is one fan it's trivial.
        label: what operating condition does the input represent. It's optional, meaning it can be None
            because it's unlabelled. Furthermore, the integer should represent the value from classification.fault_types
    """
    device_id: Optional[int] = None
    label: Optional[int] = None

@dataclass
class RawAccWindow(RawInput):
    """Represents a single window of data from accelerometer.
    
    Args:
        acc_x,y,z: acceleration on each axis.
        timestamps: optional per-sample timestamps (seconds). When provided,
            preprocessing can resample onto a clean, uniform grid.
        sampling_rate_hz: optional sampling rate hint for this window. If not
            set, SensorConfig.SAMPLING_RATE is used.
        acc_mag: optional pre-computed magnitude channel for orientation-robust
            processing.
    """
    acc_x: np.ndarray
    acc_y: np.ndarray
    acc_z: np.ndarray
    timestamps: Optional[np.ndarray] = None
    sampling_rate_hz: Optional[float] = None
    acc_mag: Optional[np.ndarray] = None

    @classmethod
    def from_dataframe_public_dset(cls, df: pd.DataFrame, label_: int, col_names: list[str]):
        """Parse from dataframe to RawAccWindow.

        Args:
            col_names: list col names for acc_x, acc_y and z in the dataset

        NOTE: THIS PARSER IS CURRENTLY BASED ON PUBLIC DATASET CSV
        """
        assert len(col_names) == 3 
        
        x, y, z = col_names

        acc_x = df[x].to_numpy()
        acc_y = df[y].to_numpy()
        acc_z = df[z].to_numpy()

        return cls(
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            label=label_
        )

    
"""Define fault types recognized by the system"""
from enum import Enum

class OperatingCondition(Enum):
    """Operating condtiions recognized by the system."""
    NORMAL = 0
    BLOCKED_AIRFLOW = 1
    INTERFERENCE = 2
    IMBALANCE = 3
    UNKNOWN = 4
