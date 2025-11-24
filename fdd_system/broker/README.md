## Local test loop (broker + simulator)

1) Create paired PTYs (two ends of a virtual serial cable):
```bash
socat -d -d PTY,link=/tmp/ttySIM1,raw,echo=0 PTY,link=/tmp/ttySIM2,raw,echo=0
```
`/tmp/ttySIM1` and `/tmp/ttySIM2` are the two ends; pick one for the broker and one for the simulator. But look at the output in the terminal for the two used port ends. Then replace the following [CHANGEME] with those ports. 

2) Run the broker, pointing `--port` to one end:
```bash
python -m fdd_system.broker.main \
  --port [CHANGEME] \
  --baudrate 9600 \
  --model-path fdd_system/AI/training/ML/weights/rf_model.joblib \
  --loop-delay 0.05
```

3) Change the port on broker/simulator.py to the other end of the port. Then in another shell, run the simulator on the other end:
```bash
python fdd_system/broker/simulator.py
```
Simulator streams samples from `fdd_system/broker/sample_data/blocked.csv`; broker reads them, builds windows, and logs predictions.
