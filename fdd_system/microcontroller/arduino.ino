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

  // 1. Initialize Sensor
  if(!accel.begin()) {
    // Serial.println("No ADXL345 detected! Check wiring.");
    while(1);
  }
  accel.setRange(ADXL345_RANGE_2_G);

  
  // 2. Spin up the Fan (Remember: 0 is ON for your P-Channel)
  analogWrite(fanPin, 0);
  delay(3000); 

  // 3. Calibration Phase
  // Serial.println("Calibrating 'Normal' Vibration...");
  float totalVibration = 0;
  int samples = 100;

  for(int i=0; i<samples; i++) {
    totalVibration += measureVibration();
    delay(10);
  }
  
  baselineVibration = totalVibration / samples;

  accel.setDataRate(ADXL345_DATARATE_3200_HZ);
  

  // Serial.print("Baseline Vibration Level: ");
  // Serial.println(baselineVibration);
  // Serial.println("Sentinel Mode Active.");
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
  
  // --- CHANGE #2: Made "Stopped" alert less sensitive ---
  // We now only trigger if vibration is extremely low (was 1.0)x
  //if (currentVib < 0.5) { 
  //   Serial.println("⚠️ ALERT: FAN STOPPED! ⚠️");
  //}

  delay(1000/1600);
}

// Helper function to calculate "G-Force Intensity"
float measureVibration() {
  sensors_event_t event; 
  accel.getEvent(&event);
  
  
  char buffer[64];

  char fx[16];
  char fy[16];
  char fz[16];

  float ax = event.acceleration.x;
  float ay = event.acceleration.y;
  float az = event.acceleration.z;

  dtostrf(ax, 0, 2, fx);
  dtostrf(ay, 0, 2, fy);
  dtostrf(az, 0, 2, fz);

  sprintf(buffer, "x %s y %s z %s", fx, fy, fz);
  Serial.println(buffer);


  // Serial.println(event.acceleration.x);
  // Subtract gravity (approx 9.8 m/s^2) to see just the vibration
  return 0;
}
