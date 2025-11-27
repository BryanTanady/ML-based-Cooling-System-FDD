"""This file defines all involved data classess"""
from dataclasses import dataclass
import numpy as np
from typing import Optional
import pandas as pd

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
    """
    acc_x: np.ndarray
    acc_y: np.ndarray
    acc_z: np.ndarray

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

    

