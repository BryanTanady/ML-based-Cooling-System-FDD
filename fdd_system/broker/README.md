## Broker module layout

Broker code is now consolidated into three source files:

- `main.py`: CLI and runtime orchestration
- `prediction_utils.py`: model/pipeline construction and prediction/calibration helpers
- `io_helpers.py`: serial reader, line parsing, window building, and alert transport

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
  --baudrate 115200 \
  --model-path experiment/weights/model_best.joblib \
  --model-format sklearn \
  --embedder ml2 \
  --preprocessor basic \
  --alert-api-url http://127.0.0.1:8001/api/alert \
  --asset-id FAN-01 \
  --loop-delay 0.05
```

ONNX + 1D CNN example:
```bash
python -m fdd_system.broker.main \
  --port [CHANGEME] \
  --baudrate 115200 \
  --input-format bin \
  --fs-hz 800 \
  --model-path experiment/weights/model_best_cnn1d.onnx \
  --model-format onnx \
  --embedder raw1dcnn \
  --preprocessor robust \
  --alert-api-url http://127.0.0.1:8001/api/alert \
  --asset-id FAN-01 \
  --alert-timeout 1.0
```

3) Change the port on broker/simulator.py to the other end of the port. Then in another shell, run the simulator on the other end:
```bash
python fdd_system/broker/simulator.py
```
Simulator streams samples from `fdd_system/broker/sample_data/blocked.csv`; broker reads them, builds windows, logs predictions, and forwards non-normal alerts to the backend API.
