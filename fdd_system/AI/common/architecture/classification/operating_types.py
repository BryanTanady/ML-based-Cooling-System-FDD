"""Define fault types recognized by the system"""
from enum import Enum

class OperatingCondition(Enum):
    """Operating condtiions recognized by the system."""
    NORMAL = 0
    BLOCKED_AIRFLOW = 1
    BLADE_ISSUE = 2
    POWER_ISSUE = 3
