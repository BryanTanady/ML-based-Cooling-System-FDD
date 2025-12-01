#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <avr/wdt.h>   // Comment out if not AVR

// ----------------- Globals & Config -----------------

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

const int fanPin = 9;

const unsigned long SAMPLE_INTERVAL_US = 625;  // target ~1600 Hz
unsigned long lastSampleUs             = 0;

unsigned long sampleCount    = 0;    // counts ALL sensor samples per RATE window
unsigned long lastRateReport = 0;
const uint16_t PRINT_EVERY_N = 1;

bool accel_ok = false;

// retry init timing
unsigned long lastRetryMs             = 0;
const unsigned long RETRY_INTERVAL_MS = 100;   // retry every 0.5 s

// timing to recover sampling rate
unsigned long firstSampleUs = 0;
bool haveFirstSample        = false;
uint32_t globalSampleIndex  = 0;    // monotonically increasing sample index

// ----------------- Helpers -----------------

void try_init_accel() {
  if (accel.begin()) {
    accel_ok = true;
    accel.setDataRate(ADXL345_DATARATE_1600_HZ);
    accel.setRange(ADXL345_RANGE_2_G);
  } else {
    accel_ok = false;
  }
}

// Emit one S line given ax, ay, az and current time.
void printSampleLine(float ax, float ay, float az, unsigned long nowUs) {
  if (!haveFirstSample) {
    firstSampleUs   = nowUs;
    haveFirstSample = true;
  }
  unsigned long tRelUs = nowUs - firstSampleUs;
  
  Serial.print("S ");
  Serial.print(tRelUs);
  Serial.print(' ');
  Serial.print(ax, 3);
  Serial.print(' ');
  Serial.print(ay, 3);
  Serial.print(' ');
  Serial.println(az, 3);
}

// ----------------- Arduino Setup / Loop -----------------

void setup() {
  // Enable watchdog with ~2s timeout (AVR only)
  wdt_enable(WDTO_2S);

  Serial.begin(115200);
  delay(500);

  pinMode(fanPin, OUTPUT);

  // Start fan (for your P-channel arrangement 0 = full on)
  analogWrite(fanPin, 0);

  Wire.begin();
  Wire.setClock(400000);  // fast I2C (needs proper pull-ups)
  Wire.setTimeout(50);    // 50 ms timeout for I2C operations

  // Initial attempt to init sensor
  try_init_accel();
}

void loop() {
  // Feed watchdog every iteration
  wdt_reset();

  unsigned long nowMs = millis();
  unsigned long nowUs = micros();

  // ----------------- If sensor not OK, keep retrying -----------------
  if (!accel_ok) {
    if (nowMs - lastRetryMs >= RETRY_INTERVAL_MS) {
      lastRetryMs = nowMs;
      try_init_accel();
    }
    return;
  }

  // ----------------- Normal sampling loop -----------------
  if (nowUs - lastSampleUs >= SAMPLE_INTERVAL_US) {
    lastSampleUs = nowUs;

    sensors_event_t event;
    accel.getEvent(&event);

    float ax = event.acceleration.x;
    float ay = event.acceleration.y;
    float az = event.acceleration.z;

    globalSampleIndex++;
    sampleCount++;

    static uint16_t printCounter = 0;
    if (++printCounter >= PRINT_EVERY_N) {
      printCounter = 0;
      printSampleLine(ax, ay, az, nowUs);
    }
  }

  // ----------------- Print effective sample rate -----------------
  // if (nowMs - lastRateReport >= 1000) {
  //   Serial.print("RATE ");
  //   Serial.println(sampleCount);   // samples / ~1s
  //   sampleCount    = 0;
  //   lastRateReport = nowMs;
  // }
}
