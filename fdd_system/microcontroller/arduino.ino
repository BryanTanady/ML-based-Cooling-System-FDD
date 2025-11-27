#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

// Create the sensor object
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

const int fanPin = 9;
float baselineVibration = 0;

// --- CHANGE #1: Made "Wobble" alert less sensitive ---
// We will only trigger if vibration is 3x higher than normal (was 2.0)
float sensitivity = 3.0; 

void setup() {
  Serial.begin(9600);
  pinMode(fanPin, OUTPUT);

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
  // Measure current vibration
  float currentVib = measureVibration();

  // Debug print (Use Serial Plotter!)
  //Serial.print("Vibration:");
  // Serial.println(currentVib);
  //Serial.print(",");
  //Serial.println(baselineVibration * sensitivity);

  // 4. Anomaly Check
  // If vibration is 3x higher than normal -> WOBBLE DETECTED
  if (currentVib > (baselineVibration * sensitivity)) {
    // Serial.println("⚠️ ALERT: HIGH VIBRATION DETECTED! ⚠️");
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
