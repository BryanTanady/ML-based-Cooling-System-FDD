## Broker module layout

Broker code is consolidated into:

- `main.py`: CLI and runtime orchestration
- `prediction_utils.py`: model/pipeline construction and prediction helpers
- `io_helpers.py`: serial reader, line parsing, window building, and alert transport
- `simulator.py`: Arduino protocol-9 binary stream simulator over PTY or TCP

## Local test loop (no Arduino)

1) Start the simulator in one shell (PTY serial mode):
```bash
python fdd_system/broker/simulator.py \
  --transport pty \
  --pty-link /tmp/ttyARDUINO \
  --fs-hz 800 \
  --source-csv experiment/data_did_1/normal/normal_1.csv \
  --loop
```
This emits the same protocol-9 frame layout as `microcontroller/arduino.ino`
(`[AA 55][x_lo x_hi y_lo y_hi z_lo z_hi][crc8(payload)]`) with firmware-like
interval scheduling/resync behavior.

2) Point a consumer to `/tmp/ttyARDUINO`.

Broker example:
```bash
python -m fdd_system.broker.main \
  --port /tmp/ttyARDUINO \
  --baudrate 115200 \
  --input-format bin \
  --fs-hz 800 \
  --model-path experiment/weights/model_best_cnn1d.onnx \
  --model-format onnx \
  --embedder raw1dcnn \
  --preprocessor robust \
  --alert-api-url http://127.0.0.1:8001/api/alert \
  --asset-id FAN-01
```

Data collection example:
```bash
sh data_collection/run_getData.sh \
  --time 10 \
  --label test_sim.csv \
  --port /tmp/ttyARDUINO \
  --baud 115200 \
  --fs 800
```

Optional TCP mode is still available:
```bash
python fdd_system/broker/simulator.py --transport tcp --host 127.0.0.1 --port 9999
```
