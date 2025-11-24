"""This file stores all configs for system components, such as fan, sensor (how the buffer is structured), etc."""

class FanConfig():
    NUM_BLADES = 7

class SensorConfig():
    SAMPLING_RATE = 400
    WINDOW_SIZE = 100
    STRIDE = 50