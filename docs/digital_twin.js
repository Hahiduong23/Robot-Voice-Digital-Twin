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
    this.state = "DỪNG";

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
    this.state = "DỪNG";
    this.trail = [];
  }

  // Handle Voice Command
  applyCommand(cmd) {
    switch (cmd) {
      case "Tien":
        this.targetSpeed = 100;
        this.targetAngularVelocity = 0;
        this.targetTurnAngle = null;
        this.state = "TIẾN";
        break;
      case "Lui":
        this.targetSpeed = -70;
        this.targetAngularVelocity = 0;
        this.targetTurnAngle = null;
        this.state = "LÙI";
        break;
      case "ReTrai":
        this.targetTurnAngle = this.angle - (Math.PI / 2); // 90 deg Left
        this.targetAngularVelocity = -2.4;
        this.targetSpeed = 30; // Small creep during turn
        this.state = "RẼ TRÁI";
        break;
      case "RePhai":
        this.targetTurnAngle = this.angle + (Math.PI / 2); // 90 deg Right
        this.targetAngularVelocity = 2.4;
        this.targetSpeed = 30;
        this.state = "RẼ PHẢI";
        break;
      case "TamDung":
        this.targetSpeed = 0;
        this.targetAngularVelocity = 0;
        this.targetTurnAngle = null;
        this.state = "TẠM DỪNG";
        break;
      case "KhoiDong":
        this.isBlinking = true;
        this.blinkTimer = 30;
        this.state = "SẴN SÀNG";
        break;
      default:
        break;
    }
  }

  updatePhysics(dt) {
    // Handle turning to target angle
    if (this.targetTurnAngle !== null) {
      const diff = this.targetTurnAngle - this.angle;
      if (Math.abs(diff) < 0.08) {
        this.angle = this.targetTurnAngle;
        this.targetTurnAngle = null;
        this.targetAngularVelocity = 0;
        this.angularVelocity = 0;
        this.targetSpeed = 0;
        this.state = "DỪNG";
      } else {
        this.angularVelocity = Math.sign(diff) * 2.4;
      }
    } else {
      // Smooth interpolation
      this.angularVelocity += (this.targetAngularVelocity - this.angularVelocity) * 0.15;
    }

    this.speed += (this.targetSpeed - this.speed) * 0.15;
    this.angle += this.angularVelocity * dt;

    // Displace position
    this.x += Math.cos(this.angle) * this.speed * dt;
    this.y += Math.sin(this.angle) * this.speed * dt;

    // Boundaries bounce
    const pad = 25;
    if (this.x < pad) { this.x = pad; this.targetSpeed = -this.targetSpeed * 0.5; }
    if (this.x > this.width - pad) { this.x = this.width - pad; this.targetSpeed = -this.targetSpeed * 0.5; }
    if (this.y < pad) { this.y = pad; this.targetSpeed = -this.targetSpeed * 0.5; }
    if (this.y > this.height - pad) { this.y = this.height - pad; this.targetSpeed = -this.targetSpeed * 0.5; }

    // Breadcrumb trail
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
    } else {
      ctx.fillStyle = "#00f2fe"; // Neon cyan
    }
    ctx.beginPath();
    ctx.arc(this.robotLength / 2 - 2, -7, 2.5, 0, Math.PI * 2);
    ctx.arc(this.robotLength / 2 - 2, 7, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // Update HUD Metrics
    document.getElementById("hud-pos").textContent = "(" + Math.round(this.x) + ", " + Math.round(this.y) + ")";
    let deg = Math.round((this.angle * 180 / Math.PI) % 360);
    if (deg < 0) deg += 360;
    document.getElementById("hud-angle").textContent = deg + "°";
    document.getElementById("hud-speed").textContent = Math.round(Math.abs(this.speed)) + " px/s";
    
    const stateEl = document.getElementById("hud-state");
    stateEl.textContent = this.state;
    stateEl.className = (this.state === "TẠM DỪNG") ? "state-stop" : "state-active";
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
