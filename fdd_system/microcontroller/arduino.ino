#include <Wire.h>
#include <avr/wdt.h>

// ----------------- Config -----------------
static const uint8_t ADXL_ADDR = 0x53;
static const uint8_t REG_DEVID = 0x00;
static const uint8_t REG_BW_RATE = 0x2C;
static const uint8_t REG_POWER_CTL = 0x2D;
static const uint8_t REG_DATA_FORMAT = 0x31;
static const uint8_t REG_DATAX0 = 0x32;

static const uint32_t BAUD_RATE = 115200;

// Pick 800 or 1600 here
static const uint16_t SAMPLING_RATE_HZ = 800;
static const uint32_t SAMPLE_INTERVAL_US = 1000000UL / SAMPLING_RATE_HZ;

static uint32_t lastRetryMs = 0;
static const uint32_t RETRY_INTERVAL_MS = 100;
static bool accel_ok = false;

static uint32_t nextSampleUs = 0;
static bool schedulerInit = false;
static uint32_t globalSampleIndex = 0;

// ----------------- Low-level I2C helpers -----------------
static bool i2c_write_reg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission() == 0);
}

static bool i2c_read_reg(uint8_t reg, uint8_t &val) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(ADXL_ADDR, (uint8_t)1) != 1) return false;
  val = Wire.read();
  return true;
}

static bool read_adxl345_raw(int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(REG_DATAX0);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom(ADXL_ADDR, (uint8_t)6) != 6) return false;

  uint8_t x0 = Wire.read();
  uint8_t x1 = Wire.read();
  uint8_t y0 = Wire.read();
  uint8_t y1 = Wire.read();
  uint8_t z0 = Wire.read();
  uint8_t z1 = Wire.read();

  x = (int16_t)((x1 << 8) | x0);
  y = (int16_t)((y1 << 8) | y0);
  z = (int16_t)((z1 << 8) | z0);
  return true;
}

// ----------------- CRC8 (Dallas/Maxim, poly 0x31) -----------------
static uint8_t crc8_maxim(const uint8_t *data, uint8_t len) {
  uint8_t crc = 0x00;
  for (uint8_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      if (crc & 0x01) crc = (crc >> 1) ^ 0x8C;  // reversed poly of 0x31
      else crc >>= 1;
    }
  }
  return crc;
}

// ----------------- ADXL345 init -----------------
static uint8_t bw_rate_value_for_odr(uint16_t hz) {
  // BW_RATE[3:0]
  // 400 -> 0x0C, 800 -> 0x0D, 1600 -> 0x0E
  if (hz >= 1600) return 0x0E;
  if (hz >= 800)  return 0x0D;
  return 0x0C; // fallback 400
}

static void try_init_accel() {
  uint8_t devid = 0;
  if (!i2c_read_reg(REG_DEVID, devid) || devid != 0xE5) {
    accel_ok = false;
    return;
  }

  if (!i2c_write_reg(REG_POWER_CTL, 0x00)) { accel_ok = false; return; }

  uint8_t bw = bw_rate_value_for_odr(SAMPLING_RATE_HZ);
  if (!i2c_write_reg(REG_BW_RATE, bw)) { accel_ok = false; return; }

  // FULL_RES=1, +/-2g
  if (!i2c_write_reg(REG_DATA_FORMAT, 0x08)) { accel_ok = false; return; }

  // MEASURE=1
  if (!i2c_write_reg(REG_POWER_CTL, 0x08)) { accel_ok = false; return; }

  accel_ok = true;
}

// ----------------- Setup / Loop -----------------
void setup() {
  wdt_enable(WDTO_2S);
  Serial.begin(BAUD_RATE);
  delay(200);

  Wire.begin();
  Wire.setClock(400000);
  Wire.setTimeout(10);

  try_init_accel();
}

static inline void put_u32_le(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

static inline void put_i16_le(uint8_t *p, int16_t v) {
  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)((uint16_t)v >> 8);
}

void loop() {
  wdt_reset();

  uint32_t nowMs = millis();
  uint32_t nowUs = micros();

  if (!accel_ok) {
    if (nowMs - lastRetryMs >= RETRY_INTERVAL_MS) {
      lastRetryMs = nowMs;
      try_init_accel();
    }
    return;
  }

  if (!schedulerInit) {
    nextSampleUs = nowUs + SAMPLE_INTERVAL_US;
    schedulerInit = true;
  }

  // If we fall behind due to host not reading, do not try to "catch up" forever.
  // Resync if late by more than 1 interval.
  if ((int32_t)(nowUs - nextSampleUs) >= (int32_t)SAMPLE_INTERVAL_US) {
    nextSampleUs = nowUs + SAMPLE_INTERVAL_US;
  }

  if ((int32_t)(nowUs - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_INTERVAL_US;

    int16_t x, y, z;
    if (!read_adxl345_raw(x, y, z)) {
      accel_ok = false;
      return;
    }

    // 9-byte frame: [AA 55] [x_lo x_hi y_lo y_hi z_lo z_hi] [crc]
    const uint8_t SYNC0 = 0xAA;
    const uint8_t SYNC1 = 0x55;

    uint8_t frame[9];
    frame[0] = SYNC0;
    frame[1] = SYNC1;

    put_i16_le(&frame[2], x);
    put_i16_le(&frame[4], y);
    put_i16_le(&frame[6], z);

    // CRC over payload only (bytes 2..7)
    frame[8] = crc8_maxim(&frame[2], 6);

    Serial.write(frame, sizeof(frame));

  }
}
