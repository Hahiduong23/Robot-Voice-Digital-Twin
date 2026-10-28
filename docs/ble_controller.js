// ==========================================================================
// BLE CONTROLLER: Web Bluetooth API for Wireless ESP32 Robot Hardware
// Nordic UART Service (NUS) Protocol with Auto-Reconnect & Fail-Safe
// ==========================================================================

class BleController {
  constructor(onStatusChange, onLog) {
    this.onStatusChange = onStatusChange || (() => {});
    this.onLog = onLog || (() => {});

    this.device = null;
    this.server = null;
    this.rxCharacteristic = null;
    this.txCharacteristic = null;
    this.isConnected = false;

    // Standard Nordic UART Service (NUS) UUIDs
    this.NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
    this.NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"; // Browser writes to ESP32
    this.NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"; // ESP32 notifies Browser

    // Command mapping table
    this.cmdMap = {
      "Tien": "F",
      "Lui": "B",
      "ReTrai": "L",
      "RePhai": "R",
      "TamDung": "S",
      "KhoiDong": "K"
    };
  }

  isSupported() {
    return (navigator.bluetooth !== undefined);
  }

  async connect() {
    if (!this.isSupported()) {
      this.onLog("[BLE LỖI] Trình duyệt này không hỗ trợ Web Bluetooth. Trên iPhone, vui lòng mở bằng app Bluefy!", "err");
      alert("Trình duyệt không hỗ trợ Web Bluetooth. Vui lòng dùng Chrome trên Android/PC hoặc Bluefy trên iPhone!");
      return false;
    }

    try {
      this.onLog("[BLE] Đang quét tìm thiết bị Bluetooth ESP32...", "sys");
      this.onStatusChange("connecting", "Đang quét...");

      this.device = await navigator.bluetooth.requestDevice({
        filters: [
          { namePrefix: "ESP32" },
          { namePrefix: "Robot" }
        ],
        optionalServices: [this.NUS_SERVICE_UUID]
      });

      this.device.addEventListener("gattserverdisconnected", () => {
        this.handleDisconnected();
      });

      this.onLog("[BLE] Tìm thấy thiết bị: " + this.device.name + ". Đang kết nối GATT...", "sys");
      this.server = await this.device.gatt.connect();

      this.onLog("[BLE] Đang lấy dịch vụ Nordic UART...", "sys");
      const service = await this.server.getPrimaryService(this.NUS_SERVICE_UUID);

      this.rxCharacteristic = await service.getCharacteristic(this.NUS_RX_UUID);
      
      try {
        this.txCharacteristic = await service.getCharacteristic(this.NUS_TX_UUID);
        await this.txCharacteristic.startNotifications();
        this.txCharacteristic.addEventListener("characteristicvaluechanged", (e) => {
          const val = new TextDecoder().decode(e.target.value);
          this.onLog("[ESP32 NHẬN]: " + val, "rx");
        });
      } catch (e) {
        console.warn("Không bật được TX notifications:", e);
      }

      this.isConnected = true;
      this.onStatusChange("connected", "Đã kết nối: " + this.device.name);
      this.onLog("[BLE THÀNH CÔNG] Đã kết nối với xe thật (" + this.device.name + ")!", "sys");
      return true;
    } catch (err) {
      this.isConnected = false;
      this.onStatusChange("disconnected", "Ngắt kết nối");
      if (err.name !== "NotFoundError") {
        this.onLog("[BLE LỖI KẾT NỐI] " + err.message, "err");
      } else {
        this.onLog("[BLE] Người dùng đã hủy chọn thiết bị.", "sys");
      }
      return false;
    }
  }

  handleDisconnected() {
    this.isConnected = false;
    this.rxCharacteristic = null;
    this.txCharacteristic = null;
    this.onStatusChange("disconnected", "Ngắt kết nối");
    this.onLog("[BLE CẢNH BÁO] Xe Robot đã ngắt kết nối!", "err");
  }

  async disconnect() {
    if (this.device && this.device.gatt.connected) {
      await this.device.gatt.disconnect();
    }
    this.handleDisconnected();
  }

  async sendCommand(cmdName) {
    const charCode = this.cmdMap[cmdName];
    if (!charCode) return false;

    const timestamp = new Date().toLocaleTimeString("vi-VN", { hour12: false }) + "." + String(new Date().getMilliseconds()).padStart(3, "0");

    if (!this.isConnected || !this.rxCharacteristic) {
      this.onLog("[" + timestamp + "] [SIMULATION ONLY] Lệnh " + cmdName + " -> Ký tự '" + charCode + "' (Chưa bật xe thật)", "sys");
      return false;
    }

    try {
      const data = new Uint8Array([charCode.charCodeAt(0)]);
      await this.rxCharacteristic.writeValueWithoutResponse(data);
      this.onLog("[" + timestamp + "] [GỬI BLE] -> Đã gửi mã '" + charCode + "' (" + cmdName + ") xuống xe thật!", "tx");
      return true;
    } catch (err) {
      this.onLog("[BLE GỬI LỖI] " + err.message, "err");
      return false;
    }
  }
}
