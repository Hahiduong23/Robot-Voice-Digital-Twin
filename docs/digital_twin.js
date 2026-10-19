// ==========================================================================
// DIGITAL TWIN ARENA: 2D Robot Simulation Canvas with Physics & Telemetry
// Renders 60 FPS Interactive Robot Twin with Heading, Trails & Sensors
// ==========================================================================

class DigitalTwinArena {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.width = this.canvas.width;
    this.height = this.canvas.height;

    // Robot Kinematics (2-Wheel Differential Drive)
    this.x = this.width / 2;
    this.y = this.height / 2;
    this.angle = 0; // Radians, 0 points East (Right)
    this.speed = 0;
    this.targetSpeed = 0;
    this.angularVelocity = 0;
    this.targetAngularVelocity = 0;

    this.robotLength = 36;
    this.robotWidth = 24;

    // Operational State Machine
    // Must trigger "KhoiDong" first before moving. "TamDung" freezes and locks until "KhoiDong" again.
    this.isStarted = false;
    this.isPaused = false;
    this.state = "CHỜ KHỞI ĐỘNG";

    // Discrete step movement (di chuyển 1 đoạn ngắn nhất định)
    this.remainingDistance = 0;
    this.stepDistance = 55; // 55 pixels per voice step (~0.6m in physical scale)

    // Trail history
    this.trail = [];
    this.maxTrail = 60;

    // Headlight blink timer
    this.blinkTimer = 0;
    this.isBlinking = false;

    // Turn target angle (for 90 deg voice turns)
    this.targetTurnAngle = null;

    // Animation frame handle
    this.lastTime = performance.now();
    this.isRunning = true;

    // Bind loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  reset() {
    this.x = this.width / 2;
    this.y = this.height / 2;
    this.angle = 0;
    this.speed = 0;
    this.targetSpeed = 0;
    this.angularVelocity = 0;
    this.targetAngularVelocity = 0;
    this.targetTurnAngle = null;
    this.remainingDistance = 0;
    this.isStarted = false;
    this.isPaused = false;
    this.state = "CHỜ KHỞI ĐỘNG";
    this.trail = [];
  }

  // Handle Voice or Manual Command
  applyCommand(cmd) {
    // 1. LỆNH KHỞI ĐỘNG: Kích hoạt hệ thống hoặc mở khóa sau khi tạm dừng
    if (cmd === "KhoiDong") {
      this.isStarted = true;
      this.isPaused = false;
      this.isBlinking = true;
      this.blinkTimer = 35;
      this.targetSpeed = 0;
      this.targetAngularVelocity = 0;
      this.remainingDistance = 0;
      this.state = "SẴN SÀNG";
      return { ok: true, msg: "Đã khởi động robot! Sẵn sàng nhận lệnh điều khiển." };
    }

    // 2. LỆNH TẠM DỪNG: Phanh khẩn cấp & khóa toàn bộ chuyển động
    if (cmd === "TamDung") {
      this.isPaused = true;
      this.targetSpeed = 0;
      this.speed = 0;
      this.targetAngularVelocity = 0;
      this.angularVelocity = 0;
      this.targetTurnAngle = null;
      this.remainingDistance = 0;
      this.state = "TẠM DỪNG";
      return { ok: true, msg: "Xe đã tạm dừng! Toàn bộ chuyển động bị khóa cho đến khi nhận lệnh 'Khởi Động'." };
    }

    // 3. KIỂM TRA ĐIỀU KIỆN TIÊN QUYẾT: Phải khởi động trước và không bị tạm dừng
    if (!this.isStarted) {
      return { ok: false, msg: "Xe chưa khởi động! Vui lòng đọc lệnh 'Khởi Động' (hoặc phím I) trước." };
    }

    if (this.isPaused) {
      return { ok: false, msg: "Xe đang bị tạm dừng khóa! Vui lòng đọc lệnh 'Khởi Động' (hoặc phím I) để mở khóa tiếp tục chạy." };
    }

    // 4. CÁC LỆNH DI CHUYỂN (Chỉ chạy khi xe đã khởi động và không bị tạm dừng)
    switch (cmd) {
      case "Tien":
        // Di chuyển 1 đoạn ngắn cố định (discrete step)
        this.remainingDistance = this.stepDistance;
        this.targetSpeed = 85;
        this.targetAngularVelocity = 0;
        this.targetTurnAngle = null;
        this.state = "TIẾN";
        return { ok: true, msg: "Tiến 1 bước ngắn (" + this.stepDistance + "px)" };

      case "Lui":
        // Lùi 1 đoạn ngắn cố định (discrete step)
        this.remainingDistance = this.stepDistance;
        this.targetSpeed = -65;
        this.targetAngularVelocity = 0;
        this.targetTurnAngle = null;
        this.state = "LÙI";
        return { ok: true, msg: "Lùi 1 bước ngắn (" + this.stepDistance + "px)" };

      case "ReTrai":
        // Quay góc 90 độ sang trái
        this.remainingDistance = 0;
        this.targetTurnAngle = this.angle - (Math.PI / 2);
        this.targetAngularVelocity = -2.5;
        this.targetSpeed = 20;
        this.state = "RẼ TRÁI";
        return { ok: true, msg: "Rẽ trái 90°" };

      case "RePhai":
        // Quay góc 90 độ sang phải
        this.remainingDistance = 0;
        this.targetTurnAngle = this.angle + (Math.PI / 2);
        this.targetAngularVelocity = 2.5;
        this.targetSpeed = 20;
        this.state = "RẼ PHẢI";
        return { ok: true, msg: "Rẽ phải 90°" };

      default:
        return { ok: false, msg: "Lệnh không hỗ trợ: " + cmd };
    }
  }

  updatePhysics(dt) {
    // A. Xử lý quay theo góc đích 90 độ
    if (this.targetTurnAngle !== null) {
      const diff = this.targetTurnAngle - this.angle;
      if (Math.abs(diff) < 0.08) {
        this.angle = this.targetTurnAngle;
        this.targetTurnAngle = null;
        this.targetAngularVelocity = 0;
        this.angularVelocity = 0;
        this.targetSpeed = 0;
        if (this.isStarted && !this.isPaused) {
          this.state = "SẴN SÀNG";
        }
      } else {
        this.angularVelocity = Math.sign(diff) * 2.5;
      }
    } else {
      this.angularVelocity += (this.targetAngularVelocity - this.angularVelocity) * 0.15;
    }

    // B. Xử lý di chuyển tiến/lùi 1 đoạn ngắn nhất định (Step Distance)
    if (this.remainingDistance > 0) {
      const deltaDistance = Math.abs(this.speed * dt);
      this.remainingDistance -= deltaDistance;

      if (this.remainingDistance <= 0) {
        this.remainingDistance = 0;
        this.targetSpeed = 0;
        this.speed = 0;
        if (this.isStarted && !this.isPaused) {
          this.state = "SẴN SÀNG";
        }
      }
    }

    this.speed += (this.targetSpeed - this.speed) * 0.15;
    this.angle += this.angularVelocity * dt;

    // Cập nhật vị trí tọa độ
    this.x += Math.cos(this.angle) * this.speed * dt;
    this.y += Math.sin(this.angle) * this.speed * dt;

    // Giới hạn biên sa bàn
    const pad = 25;
    if (this.x < pad) { this.x = pad; this.targetSpeed = 0; this.remainingDistance = 0; }
    if (this.x > this.width - pad) { this.x = this.width - pad; this.targetSpeed = 0; this.remainingDistance = 0; }
    if (this.y < pad) { this.y = pad; this.targetSpeed = 0; this.remainingDistance = 0; }
    if (this.y > this.height - pad) { this.y = this.height - pad; this.targetSpeed = 0; this.remainingDistance = 0; }

    // Dấu vết quỹ đạo bánh xe
    if (Math.abs(this.speed) > 5) {
      this.trail.push({ x: this.x, y: this.y, alpha: 1.0 });
      if (this.trail.length > this.maxTrail) this.trail.shift();
    }

    if (this.blinkTimer > 0) this.blinkTimer--;
    else this.isBlinking = false;
  }

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // 1. Grid Background
    ctx.strokeStyle = "rgba(0, 242, 254, 0.06)";
    ctx.lineWidth = 1;
    const gridSize = 30;
    for (let x = 0; x < this.width; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.height); ctx.stroke();
    }
    for (let y = 0; y < this.height; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.width, y); ctx.stroke();
    }

    // 2. Arena Boundary Wall Glow
    ctx.strokeStyle = "rgba(0, 242, 254, 0.35)";
    ctx.lineWidth = 2;
    ctx.strokeRect(10, 10, this.width - 20, this.height - 20);

    // 3. Render Trail
    for (let i = 0; i < this.trail.length; i++) {
      const p = this.trail[i];
      const norm = (i + 1) / this.trail.length;
      ctx.fillStyle = "rgba(0, 242, 254, " + (norm * 0.35) + ")";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.5 * norm, 0, Math.PI * 2);
      ctx.fill();
    }

    // 4. Render Robot Body
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);

    // Headlight Cone (Sensor Range)
    const coneGrad = ctx.createRadialGradient(0, 0, 10, 80, 0, 80);
    coneGrad.addColorStop(0, "rgba(0, 242, 254, 0.3)");
    coneGrad.addColorStop(1, "rgba(0, 242, 254, 0.0)");
    ctx.fillStyle = coneGrad;
    ctx.beginPath();
    ctx.moveTo(10, 0);
    ctx.arc(0, 0, 85, -Math.PI / 6, Math.PI / 6);
    ctx.closePath();
    ctx.fill();

    // Wheels (Left & Right DC motors)
    ctx.fillStyle = "#1e293b";
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1;
    // Left wheel
    ctx.fillRect(-10, -this.robotWidth / 2 - 4, 20, 5);
    ctx.strokeRect(-10, -this.robotWidth / 2 - 4, 20, 5);
    // Right wheel
    ctx.fillRect(-10, this.robotWidth / 2 - 1, 20, 5);
    ctx.strokeRect(-10, this.robotWidth / 2 - 1, 20, 5);

    // Chassis Box
    ctx.fillStyle = "#0f172a";
    ctx.strokeStyle = (this.state === "TẠM DỪNG") ? "#ef4444" : "#00f2fe";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-this.robotLength / 2, -this.robotWidth / 2, this.robotLength, this.robotWidth, 6);
    ctx.fill();
    ctx.stroke();

    // Direction arrow
    ctx.fillStyle = "#00f2fe";
    ctx.beginPath();
    ctx.moveTo(12, 0);
    ctx.lineTo(2, -6);
    ctx.lineTo(2, 6);
    ctx.closePath();
    ctx.fill();

    // Headlights (LEDs)
    if (this.isBlinking && (this.blinkTimer % 6 < 3)) {
      ctx.fillStyle = "#10b981"; // Green blink
    } else if (this.state === "TẠM DỪNG") {
      ctx.fillStyle = "#ef4444"; // Brake red
    } else if (!this.isStarted) {
      ctx.fillStyle = "#f59e0b"; // Standby Amber
    } else {
      ctx.fillStyle = "#00f2fe"; // Neon cyan
    }
    ctx.beginPath();
    ctx.arc(this.robotLength / 2 - 2, -7, 2.5, 0, Math.PI * 2);
    ctx.arc(this.robotLength / 2 - 2, 7, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // Update HUD Metrics
    const hudPos = document.getElementById("hud-pos");
    const hudAngle = document.getElementById("hud-angle");
    const hudSpeed = document.getElementById("hud-speed");
    const stateEl = document.getElementById("hud-state");

    if (hudPos) hudPos.textContent = "(" + Math.round(this.x) + ", " + Math.round(this.y) + ")";
    let deg = Math.round((this.angle * 180 / Math.PI) % 360);
    if (deg < 0) deg += 360;
    if (hudAngle) hudAngle.textContent = deg + "°";
    if (hudSpeed) hudSpeed.textContent = Math.round(Math.abs(this.speed)) + " px/s";
    
    if (stateEl) {
      stateEl.textContent = this.state;
      if (this.state === "TẠM DỪNG") stateEl.className = "state-stop";
      else if (!this.isStarted) stateEl.className = "state-standby";
      else stateEl.className = "state-active";
    }
  }

  animate(now) {
    const dt = Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;

    this.updatePhysics(dt);
    this.render();

    if (this.isRunning) {
      requestAnimationFrame(this.animate);
    }
  }
}
