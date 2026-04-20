# Data Collection

Use `run_getData.sh` to record accelerometer samples from the Arduino serial stream into a CSV.

## What it does

- Starts `data_collection/getData.py`
- Reads binary protocol-9 frames (`AA 55 + xyz + crc`)
- Writes CSV with columns:
  - `idx`
  - `t_us`
  - `X`
  - `Y`
  - `Z`
  - `t_s`

If `--label` is just a filename (for example `normal_1.csv`), the output is written in `data_collection/`.

## Basic usage

From repo root:

```bash
sh data_collection/run_getData.sh \
  --time 30 \
  --label normal_1.csv \
  --port /dev/ttyACM0 \
  --baud 115200 \
  --fs 800
```

## Arguments

- `--time` (required): recording duration in seconds.
- `--label` (required): output CSV filename or path. `.csv` is added if missing.
- `--port`: serial port path (default: `/dev/ttyUSB0`).
- `--baud`: serial baud rate (default: `115200`).
- `--fs`: sampling rate in Hz used to generate `t_us`/`t_s` (default: `800`).
- `--timeout`: serial read timeout in seconds (default: `1.0`).

## Find your serial port (Linux)

```bash
find /dev -maxdepth 1 -type c \( -name 'ttyACM*' -o -name 'ttyUSB*' \) -print
```

If this prints nothing, no Arduino serial device is currently exposed.

## Save to a specific folder

```bash
sh data_collection/run_getData.sh \
  --time 20 \
  --label experiment/data_did_1/normal/normal_99.csv \
  --port /dev/ttyACM0 \
  --baud 115200 \
  --fs 800
```

## Expected result

At the end, you should see a message like:

```text
Wrote 8000 rows to /.../data_collection/normal_1.csv
```
