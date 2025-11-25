# sim_writer.py
import serial
import time
import pandas as pd

PORT = "/dev/pts/10"  # the other end from SerialReader
TEST_FILE = "fdd_system/broker/sample_data/normal.csv"
BAUD = 9600

ser = serial.Serial(PORT, BAUD)

i = 0
try:
    while True:
        # line = f"{i}, {i+1}, {i+2}\n"
        # i += 1
        # ser.write(line.encode("utf-8"))
        # ser.flush()
    # time.sleep(0.02)

        df = pd.read_csv(TEST_FILE)
        for _, row in df.iterrows():
            acc_x, acc_y, acc_z = row["AccelerometerX"], row["AccelerometerY"], row["AccelerometerZ"]
            line = f"{acc_x}, {acc_y}, {acc_z}\n"  # SerialReader splits on newline
            print("Send: ", line.strip())

            ser.write(line.encode("utf-8"))
            ser.flush()
            time.sleep(0.01)
      
        
except KeyboardInterrupt:
    pass
finally:
    ser.close()
