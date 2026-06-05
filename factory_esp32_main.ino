#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "iPhone (9)";
const char* WIFI_PASSWORD = "okokokok";
const char* MQTT_BROKER = "172.20.10.7";
const int MQTT_PORT = 1883;

const char* TOPIC_INSPECTION_REQUEST = "factory/main/inspection/request";
const char* TOPIC_INSPECTION_RESULT = "factory/pi/inspection/result";
const char* TOPIC_AUX_READY = "factory/aux/release_ready";

// ===== BELT =====
#define RPWM 15
#define LPWM 2
#define REN 4
#define LEN 16

// ===== SENSORS =====
#define SENSOR_BOARD_IN 19     // LOW = co mach tai vi tri camera
#define SENSOR_ARM_PICK 21     // HIGH = den vi tri gap

// ===== BANNER SERVO =====
#define BANNER_SERVO_PIN 27
#define ARM_LED_PIN 23

// ===== ARM SERVOS =====
#define ARM_BASE_PIN 33
#define ARM_FORWARD_PIN 32
#define ARM_UPDOWN_PIN 25
#define ARM_GRIP_PIN 26

const unsigned long INSPECTION_TIMEOUT_MS = 10000;
const unsigned long AUX_TIMEOUT_MS = 8000;
const unsigned long BANNER_OPEN_MS = 500;
const unsigned long BANNER_SETTLE_MS = 250;
const unsigned long SENSOR_DEBOUNCE_MS = 80;
const unsigned long RESULT_SETTLE_MS = 1200;

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

Servo bannerServo;
Servo baseServo, forwardServo, upDownServo, gripServo;

enum MainState {
  WAIT_BOARD_AT_CAMERA,
  WAIT_PI_RESULT,
  RUN_TO_ARM_ZONE,
  ARM_HANDOFF,
  WAIT_AUX_RELEASE
};

MainState state = WAIT_BOARD_AT_CAMERA;

int cycleId = 0;
int pendingCycleId = 0;
String lastBoardType = "UNKNOWN";
String lastVerdict = "WAIT";
bool inspectionReady = false;
bool auxReleaseReady = false;

unsigned long stateStartedAt = 0;
unsigned long lastSensorTriggerAt = 0;
unsigned long lastDebugPrintAt = 0;

int bannerClosedAngle = 90;
int bannerOpenAngle = 0;

int baseHome = 88;
int forwardHome = 46;
int upDownHome = 138;
int gripHome = 60;

int baseNow = 90;
int forwardNow = 90;
int upDownNow = 90;
int gripNow = 90;
int speedDelay = 10;

void beltForward() {
  Serial.println("[MAIN] Belt forward");
  digitalWrite(LPWM, LOW);
  delayMicroseconds(10);
  digitalWrite(RPWM, HIGH);
}

void beltStop() {
  Serial.println("[MAIN] Belt stop");
  digitalWrite(RPWM, LOW);
  digitalWrite(LPWM, LOW);
}

void moveSmooth(Servo &servo, int &current, int target) {
  while (current != target) {
    current += (current < target) ? 1 : -1;
    servo.write(current);
    delay(speedDelay);
  }
  delay(150);
}

void armHome() {
  moveSmooth(baseServo, baseNow, baseHome);
  moveSmooth(forwardServo, forwardNow, forwardHome);
  moveSmooth(upDownServo, upDownNow, upDownHome);
  moveSmooth(gripServo, gripNow, gripHome);
}

void armSequence() {
  moveSmooth(forwardServo, forwardNow, 90);
  moveSmooth(upDownServo, upDownNow, 100);
  moveSmooth(gripServo, gripNow, 122);

  if (lastVerdict == "OK") {
    moveSmooth(baseServo, baseNow, 178);
    moveSmooth(forwardServo, forwardNow, 70);
    moveSmooth(gripServo, gripNow, 92);
  } else {
    moveSmooth(baseServo, baseNow, 144);
    moveSmooth(gripServo, gripNow, 92);
  }

  moveSmooth(upDownServo, upDownNow, 108);
  armHome();
}

void bannerRelease() {
  bannerServo.write(bannerOpenAngle);
  delay(BANNER_OPEN_MS);
  beltForward();
  delay(BANNER_SETTLE_MS);
  bannerServo.write(bannerClosedAngle);
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  String topicStr = String(topic);

  if (topicStr == TOPIC_INSPECTION_RESULT) {
    int p1 = message.indexOf(',');
    int p2 = message.indexOf(',', p1 + 1);
    int p3 = message.indexOf(',', p2 + 1);
    if (p1 < 0 || p2 < 0 || p3 < 0) {
      return;
    }

    int resultCycle = message.substring(0, p1).toInt();
    if (resultCycle != pendingCycleId) {
      return;
    }

    lastBoardType = message.substring(p1 + 1, p2);
    lastVerdict = message.substring(p2 + 1, p3);
    inspectionReady = true;
    Serial.print("[MAIN] PI result received. board=");
    Serial.print(lastBoardType);
    Serial.print(" verdict=");
    Serial.println(lastVerdict);
  }

  if (topicStr == TOPIC_AUX_READY) {
    auxReleaseReady = true;
    Serial.println("[MAIN] AUX release ready");
  }
}

void reconnectMQTT() {
  while (!mqttClient.connected()) {
    if (mqttClient.connect("esp32_main")) {
      mqttClient.subscribe(TOPIC_INSPECTION_RESULT);
      mqttClient.subscribe(TOPIC_AUX_READY);
    } else {
      delay(1500);
    }
  }
}

bool sensorTriggered(int pin, int activeState) {
  if (digitalRead(pin) != activeState) {
    return false;
  }
  if (millis() - lastSensorTriggerAt < SENSOR_DEBOUNCE_MS) {
    return false;
  }
  lastSensorTriggerAt = millis();
  return true;
}

void requestInspection() {
  cycleId++;
  pendingCycleId = cycleId;
  inspectionReady = false;
  lastBoardType = "UNKNOWN";
  lastVerdict = "WAIT";
  String payload = String(pendingCycleId);
  Serial.print("[MAIN] Request inspection cycle=");
  Serial.println(pendingCycleId);
  mqttClient.publish(TOPIC_INSPECTION_REQUEST, payload.c_str(), true);
}

void setup() {
  Serial.begin(115200);

  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(REN, OUTPUT);
  pinMode(LEN, OUTPUT);
  pinMode(ARM_LED_PIN, OUTPUT);
  digitalWrite(REN, HIGH);
  digitalWrite(LEN, HIGH);
  digitalWrite(ARM_LED_PIN, LOW);
  beltStop();

  pinMode(SENSOR_BOARD_IN, INPUT_PULLUP);
  pinMode(SENSOR_ARM_PICK, INPUT_PULLUP);

  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);

  bannerServo.attach(BANNER_SERVO_PIN, 500, 2400);
  bannerServo.write(bannerClosedAngle);

  delay(1500);
  baseServo.attach(ARM_BASE_PIN, 500, 2400);
  forwardServo.attach(ARM_FORWARD_PIN, 500, 2400);
  upDownServo.attach(ARM_UPDOWN_PIN, 500, 2400);
  gripServo.attach(ARM_GRIP_PIN, 500, 2400);
  delay(300);
  armHome();

  beltForward();
  state = WAIT_BOARD_AT_CAMERA;
  stateStartedAt = millis();
}

void loop() {
  if (!mqttClient.connected()) {
    reconnectMQTT();
  }
  mqttClient.loop();

  switch (state) {
    case WAIT_BOARD_AT_CAMERA:
      if (millis() - lastDebugPrintAt > 250) {
        lastDebugPrintAt = millis();
        Serial.print("[MAIN] WAIT_BOARD_AT_CAMERA sensor19=");
        Serial.print(digitalRead(SENSOR_BOARD_IN));
        Serial.print(" sensor21=");
        Serial.println(digitalRead(SENSOR_ARM_PICK));
      }
      if (sensorTriggered(SENSOR_BOARD_IN, LOW)) {
        Serial.println("[MAIN] Board detected at camera sensor -> stopping belt");
        beltStop();
        requestInspection();
        state = WAIT_PI_RESULT;
        stateStartedAt = millis();
      }
      break;

    case WAIT_PI_RESULT:
      if (millis() - lastDebugPrintAt > 500) {
        lastDebugPrintAt = millis();
        Serial.print("[MAIN] WAIT_PI_RESULT inspectionReady=");
        Serial.print(inspectionReady);
        Serial.print(" elapsed=");
        Serial.println(millis() - stateStartedAt);
      }
      if (inspectionReady && millis() - stateStartedAt >= RESULT_SETTLE_MS) {
        Serial.println("[MAIN] Inspection ready -> banner release");
        bannerRelease();
        state = RUN_TO_ARM_ZONE;
        stateStartedAt = millis();
      } else if (millis() - stateStartedAt > INSPECTION_TIMEOUT_MS) {
        Serial.println("[MAIN] Inspection timeout -> banner release");
        bannerRelease();
        state = RUN_TO_ARM_ZONE;
        stateStartedAt = millis();
      }
      break;

    case RUN_TO_ARM_ZONE:
      if (millis() - lastDebugPrintAt > 250) {
        lastDebugPrintAt = millis();
        Serial.print("[MAIN] RUN_TO_ARM_ZONE sensor21=");
        Serial.println(digitalRead(SENSOR_ARM_PICK));
      }
      if (sensorTriggered(SENSOR_ARM_PICK, HIGH)) {
        Serial.println("[MAIN] Board reached arm zone -> stopping belt");
        beltStop();
        state = ARM_HANDOFF;
        stateStartedAt = millis();
      }
      break;

    case ARM_HANDOFF:
      Serial.print("[MAIN] ARM_HANDOFF verdict=");
      Serial.println(lastVerdict);
      auxReleaseReady = false;
      digitalWrite(ARM_LED_PIN, HIGH);
      armSequence();
      digitalWrite(ARM_LED_PIN, LOW);
      state = WAIT_AUX_RELEASE;
      stateStartedAt = millis();
      break;

    case WAIT_AUX_RELEASE:
      if (millis() - lastDebugPrintAt > 500) {
        lastDebugPrintAt = millis();
        Serial.print("[MAIN] WAIT_AUX_RELEASE auxReleaseReady=");
        Serial.println(auxReleaseReady);
      }
      if (auxReleaseReady || millis() - stateStartedAt > AUX_TIMEOUT_MS) {
        if (auxReleaseReady) {
          Serial.println("[MAIN] AUX confirmed -> belt forward");
        } else {
          Serial.println("[MAIN] AUX timeout -> belt forward");
        }
        beltForward();
        state = WAIT_BOARD_AT_CAMERA;
        stateStartedAt = millis();
      }
      break;
  }
}
