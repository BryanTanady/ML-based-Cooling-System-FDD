"""This file stores all configs for system components, such as fan, sensor (how the buffer is structured), etc."""

class FanConfig():
    NUM_BLADES = 5

class SensorConfig():
    SAMPLING_RATE = 200
    WINDOW_SIZE = 150
    STRIDE = 50