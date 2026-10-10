// ==========================================================================
// MAIN APPLICATION COORDINATOR: Master State & Real-Time Edge AI Loop
// Ties together: ONNX Web + Audio Engine + Digital Twin + Web Bluetooth
// ==========================================================================

let constants = null;
let ortSession = null;
let audioEngine = null;
let arena = null;
let ble = null;

let inferenceInterval = null;
let isProcessing = false;
let lastCommandTime = 0;
const COMMAND_COOLDOWN_MS = 1000; // Debounce 1.0s
const CONFIDENCE_THRESHOLD = 0.82; // 82% threshold for activation

// DOM Elements
const statusModel = document.getElementById("status-model");
const statusBle = document.getElementById("status-ble");
const statusMic = document.getElementById("status-mic");
const vadIndicator = document.getElementById("vad-indicator");

const btnToggleMic = document.getElementById("btn-toggle-mic");
const btnConnectBle = document.getElementById("btn-connect-ble");
const toggleDenoise = document.getElementById("toggle-denoise");
const btnResetSim = document.getElementById("btn-reset-sim");
const btnClearLog = document.getElementById("btn-clear-log");

const waveformCanvas = document.getElementById("waveform-canvas");
const spectrogramCanvas = document.getElementById("spectrogram-canvas");
const valRms = document.getElementById("val-rms");

const bannerEl = document.getElementById("detected-banner");
const bannerIcon = document.getElementById("banner-icon");
const bannerCmd = document.getElementById("banner-command");
const bannerConf = document.getElementById("banner-conf");
const gaugesContainer = document.getElementById("gauges-container");
const logTerminal = document.getElementById("log-terminal");

// Icons dictionary
const commandIcons = {
  "KhoiDong": "⚡",
  "Tien": "⬆️",
  "Lui": "⬇️",
  "ReTrai": "⬅️",
  "RePhai": "➡️",
  "TamDung": "🛑",
  "KhongXacDinh": "❓"
};

const commandNamesVi = {
  "KhoiDong": "KHỞI ĐỘNG (BÍP)",
  "Tien": "TIẾN THẲNG",
  "Lui": "LÙI LẠI",
  "ReTrai": "RẼ TRÁI",
  "RePhai": "RẼ PHẢI",
  "TamDung": "TẠM DỪNG (PHANH)",
  "KhongXacDinh": "KHÔNG XÁC ĐỊNH"
};

// Log logger helper
function addLog(message, type = "sys") {
  const entry = document.createElement("div");
  entry.className = "log-entry " + type;
  entry.textContent = message;
  logTerminal.appendChild(entry);
  logTerminal.scrollTop = logTerminal.scrollHeight;
}

// 1. Initialize Application
async function initApp() {
  try {
    addLog("[HỆ THỐNG] Đang tải hằng số âm học audio_constants.json...", "sys");
    const res = await fetch("audio_constants.json");
    constants = await res.json();

    // Setup Digital Twin Arena
    arena = new DigitalTwinArena("arena-canvas");

    // Setup Audio Engine
    audioEngine = new AudioEngine(constants);

    // Setup BLE Controller
    ble = new BleController(
      (state, text) => {
        const dot = statusBle.querySelector(".status-dot");
        const label = statusBle.querySelector(".status-text");
        label.textContent = "BLE: " + text;
        if (state === "connected") {
          dot.className = "status-dot online";
          btnConnectBle.classList.add("recording");
          btnConnectBle.querySelector("span").textContent = "Ngắt BLE";
        } else if (state === "connecting") {
          dot.className = "status-dot loading";
        } else {
          dot.className = "status-dot offline";
          btnConnectBle.classList.remove("recording");
          btnConnectBle.querySelector("span").textContent = "📡 Kết nối Xe Robot";
        }
      },
      (msg, type) => addLog(msg, type)
    );

    // Build 7 Gauge UI rows
    buildGaugesUI();

    // Load ONNX Model
    addLog("[HỆ THỐNG] Đang nạp mô hình M5_SE_VoiceResNet (ONNX WebAssembly)...", "sys");
    ort.env.wasm.numThreads = 1;
    ortSession = await ort.InferenceSession.create("models/m5_voice_resnet.onnx", {
      executionProviders: ["wasm"]
    });

    // Update status
    const modelDot = statusModel.querySelector(".status-dot");
    const modelText = statusModel.querySelector(".status-text");
    modelDot.className = "status-dot online";
    modelText.textContent = "M5 AI: Sẵn sàng";

    btnToggleMic.disabled = false;
    addLog("[HỆ THỐNG] Nạp mô hình M5 thành công! Dung lượng: ~1.2 MB. Đã sẵn sàng chạy Realtime.", "sys");

    // Start Visualizer RAF Loop
    startVisualizerLoop();
  } catch (err) {
    console.error("Init Error:", err);
    addLog("[LỖI KHỞI TẠO] " + err.message, "err");
    const modelDot = statusModel.querySelector(".status-dot");
    modelDot.className = "status-dot offline";
    statusModel.querySelector(".status-text").textContent = "Lỗi nạp AI";
  }
}

// Build Gauge DOM elements
function buildGaugesUI() {
  gaugesContainer.innerHTML = "";
  constants.class_names.forEach(cls => {
    const row = document.createElement("div");
    row.className = "gauge-row";
    row.id = "gauge-" + cls;

    const meta = document.createElement("div");
    meta.className = "gauge-meta";

    const name = document.createElement("span");
    name.className = "gauge-name";
    name.textContent = (commandIcons[cls] || "") + " " + (commandNamesVi[cls] || cls);

    const pct = document.createElement("span");
    pct.className = "gauge-pct";
    pct.textContent = "0.0%";

    meta.appendChild(name);
    meta.appendChild(pct);

    const track = document.createElement("div");
    track.className = "gauge-track";

    const bar = document.createElement("div");
    bar.className = "gauge-bar";
    track.appendChild(bar);

    row.appendChild(meta);
    row.appendChild(track);
    gaugesContainer.appendChild(row);
  });
}

// Visualizer Canvas Loop
function startVisualizerLoop() {
  function render() {
    if (audioEngine) {
      audioEngine.renderWaveform(waveformCanvas);
      audioEngine.renderSpectrogram(spectrogramCanvas);

      valRms.textContent = audioEngine.currentRmsDb.toFixed(1) + " dB";

      if (audioEngine.isVoiceDetected) {
        vadIndicator.textContent = "VAD: PHÁT HIỆN GIỌNG NÓI";
        vadIndicator.className = "live-indicator active";
      } else {
        vadIndicator.textContent = "VAD: YÊN TĨNH";
        vadIndicator.className = "live-indicator";
      }
    }
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

// Real-time AI Inference Loop
async function runInferenceStep() {
  if (!audioEngine || !audioEngine.isRecording || !ortSession || isProcessing) return;

  isProcessing = true;
  try {
    // 1. Extract exact Log-Mel Spectrogram tensor [1, 1, 128, 63]
    const melData = audioEngine.extractMelSpectrogram();
    const inputTensor = new ort.Tensor("float32", melData, [1, 1, 128, 63]);

    // 2. Run ONNX Inference
    const t0 = performance.now();
    const feeds = { input_mel: inputTensor };
    const results = await ortSession.run(feeds);
    const latency = (performance.now() - t0).toFixed(1);

    // 3. Output logits [1, 7] -> Softmax
    const logits = results.logits.data;
    const probs = softmax(logits);

    // 4. Update 7 Confidence Gauges
    let maxIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    constants.class_names.forEach((cls, i) => {
      const p = probs[i];
      const row = document.getElementById("gauge-" + cls);
      if (row) {
        const pctEl = row.querySelector(".gauge-pct");
        const barEl = row.querySelector(".gauge-bar");
        pctEl.textContent = (p * 100).toFixed(1) + "%";
        barEl.style.width = (p * 100) + "%";

        if (i === maxIdx && p >= CONFIDENCE_THRESHOLD && cls !== "KhongXacDinh") {
          row.classList.add("active");
        } else {
          row.classList.remove("active");
        }
      }
    });

    // 5. Trigger Command if valid and confident
    const predictedClass = constants.class_names[maxIdx];
    const now = performance.now();

    if (
      audioEngine.isVoiceDetected &&
      predictedClass !== "KhongXacDinh" &&
      maxProb >= CONFIDENCE_THRESHOLD &&
      (now - lastCommandTime) > COMMAND_COOLDOWN_MS
    ) {
      lastCommandTime = now;
      handleVoiceTrigger(predictedClass, maxProb, latency);
    }
  } catch (err) {
    console.error("Inference Error:", err);
  } finally {
    isProcessing = false;
  }
}

// Handle Confirmed Voice Trigger
function handleVoiceTrigger(cmd, conf, latency) {
  // Update Banner
  bannerIcon.textContent = commandIcons[cmd] || "⚡";
  bannerCmd.textContent = commandNamesVi[cmd] || cmd;
  bannerConf.textContent = (conf * 100).toFixed(1) + "%";
  bannerEl.classList.add("highlight");
  setTimeout(() => bannerEl.classList.remove("highlight"), 800);

  // Trigger Digital Twin 2D Robot
  if (arena) {
    arena.applyCommand(cmd);
  }

  // Trigger Physical BLE Robot
  if (ble) {
    ble.sendCommand(cmd);
  }

  // Play synthetic feedback audio beep
  playFeedbackBeep(cmd);
}

// Synthetic Audio Beep Feedback
function playFeedbackBeep(cmd) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    const freq = (cmd === "TamDung") ? 330 : 660;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);

    osc.start();
    osc.stop(ctx.currentTime + 0.15);
  } catch (e) {}
}

// Numerical Softmax helper
function softmax(arr) {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i];
  }
  let sum = 0;
  const exp = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    exp[i] = Math.exp(arr[i] - max);
    sum += exp[i];
  }
  for (let i = 0; i < arr.length; i++) {
    exp[i] /= sum;
  }
  return exp;
}

// EVENT LISTENERS
btnToggleMic.addEventListener("click", async () => {
  if (!audioEngine.isRecording) {
    const ok = await audioEngine.initMic();
    if (ok) {
      btnToggleMic.classList.add("recording");
      btnToggleMic.querySelector("span").textContent = "Tắt Microphone";
      statusMic.querySelector(".status-dot").className = "status-dot online";
      statusMic.querySelector(".status-text").textContent = "Micro: Đang thu";
      addLog("[MICROPHONE] Đã kích hoạt thu âm 16kHz & VAD liên tục.", "sys");

      // Start inference loop (every 250ms)
      inferenceInterval = setInterval(runInferenceStep, 250);
    }
  } else {
    audioEngine.stopMic();
    if (inferenceInterval) clearInterval(inferenceInterval);
    btnToggleMic.classList.remove("recording");
    btnToggleMic.querySelector("span").textContent = "Bật Microphone";
    statusMic.querySelector(".status-dot").className = "status-dot offline";
    statusMic.querySelector(".status-text").textContent = "Micro: Tắt";
    addLog("[MICROPHONE] Đã tắt thu âm.", "sys");
  }
});

btnConnectBle.addEventListener("click", async () => {
  if (!ble.isConnected) {
    await ble.connect();
  } else {
    await ble.disconnect();
  }
});

toggleDenoise.addEventListener("change", (e) => {
  if (audioEngine) {
    audioEngine.denoiseEnabled = e.target.checked;
    addLog("[RNNoise] Bộ lọc nhiễu môi trường: " + (e.target.checked ? "BẬT" : "TẮT"), "sys");
  }
});

btnResetSim.addEventListener("click", () => {
  if (arena) arena.reset();
  addLog("[ARENA] Đã đặt lại vị trí xe mô phỏng về giữa sa bàn.", "sys");
});

btnClearLog.addEventListener("click", () => {
  logTerminal.innerHTML = "";
  addLog("[LOG] Đã làm sạch nhật ký.", "sys");
});

// Manual Test Buttons for Hardware
document.querySelectorAll(".btn-hw").forEach(btn => {
  btn.addEventListener("click", () => {
    const cmd = btn.getAttribute("data-cmd");
    addLog("[THỬ NGHIỆM THỦ CÔNG] Người dùng kích hoạt lệnh: " + cmd, "sys");
    if (arena) arena.applyCommand(cmd);
    if (ble) ble.sendCommand(cmd);
    playFeedbackBeep(cmd);
  });
});

// Keyboard Controls (WASD / Space)
window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  const key = e.key.toUpperCase();
  if (key === "W") {
    if (arena) arena.applyCommand("Tien");
    if (ble) ble.sendCommand("Tien");
  } else if (key === "S") {
    if (arena) arena.applyCommand("Lui");
    if (ble) ble.sendCommand("Lui");
  } else if (key === "A") {
    if (arena) arena.applyCommand("ReTrai");
    if (ble) ble.sendCommand("ReTrai");
  } else if (key === "D") {
    if (arena) arena.applyCommand("RePhai");
    if (ble) ble.sendCommand("RePhai");
  } else if (key === " " || key === "SPACEBAR") {
    if (arena) arena.applyCommand("TamDung");
    if (ble) ble.sendCommand("TamDung");
  }
});

// Launch on Window Load
window.addEventListener("DOMContentLoaded", initApp);
