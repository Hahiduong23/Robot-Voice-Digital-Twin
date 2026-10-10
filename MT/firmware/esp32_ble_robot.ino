/* =========================================================================
 * ESP32 ROBOT VOICE CONTROLLER - BLUETOOTH LOW ENERGY (BLE) FIRMWARE
 * Protocol: Nordic UART Service (NUS) via Web Bluetooth API
 * Project: Keyword Spotting Robot Control (M5 SE_VoiceResNet)
 * Author: Nguyen Hai Duong - ICSLab (Advisor: Dr. Le Minh Huy)
 * =========================================================================
 * 
 * SO DO KET NOI PHAN CUNG (WIRING DIAGRAM):
 * ---------------------------------------------------------
 * ESP32 GPIO         Mach Cau H (L298N / TB6612)      Chuc Nang
 * ---------------------------------------------------------
 * GPIO 18 (PWM)  --> ENA (hoac PWMA)                  Toc do Banh Trai
 * GPIO 19        --> IN1                             Chieu quay Banh Trai
 * GPIO 21        --> IN2                             Chieu quay Banh Trai
 * GPIO 22        --> IN3                             Chieu quay Banh Phai
 * GPIO 23        --> IN4                             Chieu quay Banh Phai
 * GPIO 5  (PWM)  --> ENB (hoac PWMB)                  Toc do Banh Phai
 * GPIO 2         --> LED Onboard                      Bao trang thai BLE
 * GPIO 4         --> Coi Buzzer                       Tieng bip KhoiDong
 * GND            --> GND chung Mach Cau H & Nguon Pin
 * =========================================================================
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// Dinh nghia cac chan GPIO dieu khien dong co
#define PIN_ENA 18
#define PIN_IN1 19
#define PIN_IN2 21
#define PIN_IN3 22
#define PIN_IN4 23
#define PIN_ENB 5

#define PIN_LED 2
#define PIN_BUZZER 4

// Cau hinh PWM (ESP32 LEDC PWM)
#define PWM_FREQ 1000
#define PWM_RES 8 // 8-bit (0 - 255)
#define PWM_CH_A 0
#define PWM_CH_B 1

#define SPEED_FAST 220
#define SPEED_TURN 190
#define SPEED_SLOW 140

// Thoi gian Watchdog tu ngat dong co (mili-giay)
#define WATCHDOG_TIMEOUT_MS 2000

// Nordic UART Service (NUS) UUIDs
#define SERVICE_UUID           "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID_RX "6e400002-b5a3-f393-e0a9-e50e24dcca9e" // Nhan tu Web
#define CHARACTERISTIC_UUID_TX "6e400003-b5a3-f393-e0a9-e50e24dcca9e" // Gui len Web

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
unsigned long lastCommandTime = 0;
bool isMoving = false;

// Ham dieu khien dong co DC
void stopMotors() {
  ledcWrite(PIN_ENA, 0);
  ledcWrite(PIN_ENB, 0);
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);
  isMoving = false;
  Serial.println("[ROBOT] -> DUNG XE (STOP)");
}

void moveForward(int speed = SPEED_FAST) {
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  ledcWrite(PIN_ENA, speed);
  ledcWrite(PIN_ENB, speed);
  isMoving = true;
  lastCommandTime = millis();
  Serial.println("[ROBOT] -> TIEN THANG (FORWARD)");
}

void moveBackward(int speed = SPEED_SLOW) {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
  ledcWrite(PIN_ENA, speed);
  ledcWrite(PIN_ENB, speed);
  isMoving = true;
  lastCommandTime = millis();
  Serial.println("[ROBOT] -> LUI LAI (BACKWARD)");
}

void turnLeft(int speed = SPEED_TURN) {
  // Banh phai quay tien, banh trai quay lui (Xoay vi sai tai cho)
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  ledcWrite(PIN_ENA, speed);
  ledcWrite(PIN_ENB, speed);
  isMoving = true;
  lastCommandTime = millis();
  Serial.println("[ROBOT] -> RE TRAI (TURN LEFT)");
}

void turnRight(int speed = SPEED_TURN) {
  // Banh trai quay tien, banh phai quay lui
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
  ledcWrite(PIN_ENA, speed);
  ledcWrite(PIN_ENB, speed);
  isMoving = true;
  lastCommandTime = millis();
  Serial.println("[ROBOT] -> RE PHAI (TURN RIGHT)");
}

void beepBuzzer() {
  digitalWrite(PIN_BUZZER, HIGH);
  digitalWrite(PIN_LED, HIGH);
  delay(120);
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED, LOW);
  delay(80);
  digitalWrite(PIN_BUZZER, HIGH);
  digitalWrite(PIN_LED, HIGH);
  delay(150);
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED, HIGH); // Giu sang sau khi khoi dong
  Serial.println("[ROBOT] -> KHOI DONG (SYSTEM READY BEEP)");
}

// Xu ly byte lenh nhan tu trinh duyet Web
void executeCommand(char cmd) {
  switch (cmd) {
    case F: // Forward
      moveForward();
      break;
    case B: // Backward
      moveBackward();
      break;
    case L: // Turn Left
      turnLeft();
      break;
    case R: // Turn Right
      turnRight();
      break;
    case S: // Stop
      stopMotors();
      break;
    case K: // Khoi Dong
      beepBuzzer();
      break;
    default:
      Serial.printf("[CANH BAO] Ky tu khong hop le: %c\n", cmd);
      break;
  }
}

// Callback khi Web gui du lieu xuong qua Bluetooth
class MyCallbacks: public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String rxValue = pCharacteristic->getValue();
    if (rxValue.length() > 0) {
      char cmd = rxValue[0];
      Serial.printf("[BLE NHAN]: %c\n", cmd);
      executeCommand(cmd);

      // Phan hoi ACK nguoc lai Web
      if (pTxCharacteristic != NULL) {
        String ack = "ACK_" + String(cmd);
        pTxCharacteristic->setValue((uint8_t*)ack.c_str(), ack.length());
        pTxCharacteristic->notify();
      }
    }
  }
};

// Callback trang thai ket noi
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    digitalWrite(PIN_LED, HIGH);
    Serial.println("[BLE] Web Dashboard da ket noi thanh cong!");
  };

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    digitalWrite(PIN_LED, LOW);
    stopMotors();
    Serial.println("[BLE] Mat ket noi voi Web. Da dung xe an toan!");
  }
};

void setup() {
  Serial.begin(115200);
  Serial.println("\n========================================================");
  Serial.println("   KHOI DONG HE THONG ROBOT VOICE CONTROLLER (ESP32)");
  Serial.println("========================================================");

  // Khoi tao cac chan I/O
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  // Khoi tao PWM
  ledcAttach(PIN_ENA, PWM_FREQ, PWM_RES);
  ledcAttach(PIN_ENB, PWM_FREQ, PWM_RES);
  stopMotors();

  // Khoi tao BLE
  BLEDevice::init("ESP32_ROBOT_VOICE");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pTxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY
                      );
  pTxCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_RX,
                        BLECharacteristic::PROPERTY_WRITE |
                        BLECharacteristic::PROPERTY_WRITE_NR
                      );
  pRxCharacteristic->setCallbacks(new MyCallbacks());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Dang phat song voi ten: ESP32_ROBOT_VOICE");
  Serial.println("[BLE] San sang ket noi tu Web GitHub Pages!");
}

void loop() {
  // Safe Watchdog: Tu dong dung xe neu khong nhan lenh moi sau 2 giay
  if (isMoving && (millis() - lastCommandTime > WATCHDOG_TIMEOUT_MS)) {
    Serial.println("[WATCHDOG TIMEOUT] Khong co lenh moi. Tu dong ngat dong co!");
    stopMotors();
  }

  // Xu ly quang ba lai neu bi ngat ket noi
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    Serial.println("[BLE] Bat dau phat song tim thiet bi lai...");
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
  delay(10);
}
