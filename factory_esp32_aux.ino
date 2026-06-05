#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "iPhone (9)";
const char* WIFI_PASSWORD = "okokokok";
const char* MQTT_BROKER = "172.20.10.7";
const int MQTT_PORT = 1883;

const char* TOPIC_INSPECTION_RESULT = "factory/pi/inspection/result";
const char* TOPIC_AUX_READY = "factory/aux/release_ready";

#define TOUCH_SENSOR_PIN 25   // LOW = canh tay cham cam bien
#define SORT_SERVO_PIN 26

const unsigned long HOLD_AT_TARGET_MS = 1200;
const unsigned long SENSOR_DEBOUNCE_MS = 120;

Servo sortServo;
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

String boardQueue[12];
int queueHead = 0;
int queueTail = 0;
int queueCount = 0;

int currentPos = 90;
const int homeAngle = 90;
bool sequenceActive = false;
unsigned long sequenceStartedAt = 0;
unsigned long lastTouchAt = 0;

void enqueueBoard(const String& boardType) {
  if (queueCount >= 12) {
    return;
  }
  boardQueue[queueTail] = boardType;
  queueTail = (queueTail + 1) % 12;
  queueCount++;
}

String dequeueBoard() {
  if (queueCount == 0) {
    return "UNKNOWN";
  }
  String value = boardQueue[queueHead];
  queueHead = (queueHead + 1) % 12;
  queueCount--;
  return value;
}

int boardAngle(const String& boardType) {
  if (boardType == "M1") return 40;
  if (boardType == "M2") return 130;
  if (boardType == "TIMEOUT") return 170;
  return 150;
}

void moveSmooth(int target) {
  while (currentPos != target) {
    currentPos += (currentPos < target) ? 1 : -1;
    sortServo.write(currentPos);
    delay(10);
  }
  delay(100);
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

  if (String(topic) != TOPIC_INSPECTION_RESULT) {
    return;
  }

  int p1 = message.indexOf(',');
  int p2 = message.indexOf(',', p1 + 1);
  if (p1 < 0 || p2 < 0) {
    return;
  }

  String boardType = message.substring(p1 + 1, p2);
  enqueueBoard(boardType);
}

void reconnectMQTT() {
  while (!mqttClient.connected()) {
    if (mqttClient.connect("esp32_aux")) {
      mqttClient.subscribe(TOPIC_INSPECTION_RESULT);
    } else {
      delay(1500);
    }
  }
}

void setup() {
  pinMode(TOUCH_SENSOR_PIN, INPUT_PULLUP);
  sortServo.attach(SORT_SERVO_PIN, 500, 2400);
  sortServo.write(homeAngle);
  currentPos = homeAngle;

  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
}

void loop() {
  if (!mqttClient.connected()) {
    reconnectMQTT();
  }
  mqttClient.loop();

  bool touchActive = digitalRead(TOUCH_SENSOR_PIN) == LOW;

  if (!sequenceActive && touchActive && queueCount > 0 && millis() - lastTouchAt > SENSOR_DEBOUNCE_MS) {
    lastTouchAt = millis();
    String boardType = dequeueBoard();
    moveSmooth(boardAngle(boardType));
    mqttClient.publish(TOPIC_AUX_READY, boardType.c_str(), false);
    sequenceActive = true;
    sequenceStartedAt = millis();
  }

  if (sequenceActive && millis() - sequenceStartedAt > HOLD_AT_TARGET_MS) {
    moveSmooth(homeAngle);
    sequenceActive = false;
  }
}
