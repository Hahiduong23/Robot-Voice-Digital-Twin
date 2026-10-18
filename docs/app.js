// ==========================================================================
// MAIN APPLICATION COORDINATOR: SOTA BC-ResNet + Push-to-Talk 2.0s Normalization
// Completely eliminates RNNoise & Domain Shift. 100% human-controlled speech capture!
// ==========================================================================

let constants = null;
let ortSession = null;
let audioEngine = null;
let arena = null;
let ble = null;

let isProcessing = false;
let lastCommandTime = 0;
const COMMAND_COOLDOWN_MS = 600; // Debounce
const CONFIDENCE_THRESHOLD = 0.50; // Threshold for activation

// DOM Elements
const statusModel = document.getElementById("status-model");
const statusBle = document.getElementById("status-ble");
const statusMic = document.getElementById("status-mic");
const vadIndicator = document.getElementById("vad-indicator");

const btnPttMic = document.getElementById("btn-ptt-mic");
const pttText = document.getElementById("ptt-text");
const btnPlayback = document.getElementById("btn-playback");
const btnConnectBle = document.getElementById("btn-connect-ble");
const btnResetSim = document.getElementById("btn-reset-sim");
const btnClearLog = document.getElementById("btn-clear-log");

const waveformCanvas = document.getElementById("waveform-canvas");
const spectrogramCanvas = document.getElementById("spectrogram-canvas");
const valRms = document.getElementById("val-rms");
const valRecDuration = document.getElementById("val-rec-duration");
const valRecPeak = document.getElementById("val-rec-peak");

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

    // Load ONNX Model: BC-ResNet (Qualcomm AI Research)
    addLog("[HỆ THỐNG] Đang nạp mô hình SOTA BC-ResNet (Qualcomm) (ONNX WebAssembly)...", "sys");
    ort.env.wasm.numThreads = 1;
    ortSession = await ort.InferenceSession.create("models/bc_resnet.onnx?v=2800", {
      executionProviders: ["wasm"]
    });

    // Update status
    const modelDot = statusModel.querySelector(".status-dot");
    const modelText = statusModel.querySelector(".status-text");
    modelDot.className = "status-dot online";
    modelText.textContent = "BC-ResNet: Sẵn sàng";

    btnPttMic.disabled = false;
    addLog("[HỆ THỐNG] Nạp mô hình SOTA BC-ResNet (Data_FN 2,800 mẫu) thành công! Cơ chế Push-to-Talk 2.0s chuẩn hóa đã sẵn sàng!", "sys");

    // Start Visualizer Loop
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
    }
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

// ==========================================================================
// PUSH-TO-TALK STATE MACHINE & EVENT HANDLERS
// ==========================================================================
let isHoldActive = false;
let holdStartTime = 0;
let isClickModeActive = false; // For toggle click mode

async function handleStartSpeech() {
  if (isProcessing) return;

  if (!audioEngine.isMicOpen) {
    const ok = await audioEngine.initMic();
    if (!ok) {
      addLog("[LỖI] Không thể kích hoạt microphone từ trình duyệt.", "err");
      return;
    }
  }

  isHoldActive = true;
  holdStartTime = performance.now();
  audioEngine.startRecording();

  btnPttMic.classList.add("recording");
  pttText.textContent = "🔴 ĐANG THU ÂM... (THẢ ĐỂ GỬI LỆNH)";
  vadIndicator.textContent = "🔴 ĐANG THU ÂM KHẨU LỆNH...";
  vadIndicator.className = "live-indicator recording";

  const micDot = statusMic.querySelector(".status-dot");
  const micText = statusMic.querySelector(".status-text");
  micDot.className = "status-dot online";
  micText.textContent = "Micro: Đang thu...";
}

async function handleStopSpeech() {
  if (!isHoldActive || !audioEngine.isRecording) return;
  isHoldActive = false;

  btnPttMic.classList.remove("recording");
  pttText.textContent = "⚡ ĐANG CÂN 2S & NHẬN DIỆN...";
  vadIndicator.textContent = "⚡ ĐANG CÂN 2S & NHẬN DIỆN...";
  vadIndicator.className = "live-indicator processing";

  const micDot = statusMic.querySelector(".status-dot");
  const micText = statusMic.querySelector(".status-text");
  micDot.className = "status-dot online";
  micText.textContent = "Micro: Đang xử lý...";

  // Await W3C OfflineAudioContext Sinc anti-aliasing resampling & peak normalization
  const result = await audioEngine.stopRecordingAndProcess();

  micText.textContent = "Micro: Chờ lệnh";

  if (!result || result.isTooShort) {
    addLog("[CẢNH BÁO] Thời lượng ấn giữ quá ngắn (< 0.25s). Vui lòng ấn giữ trong khi nói khẩu lệnh.", "warn");
    vadIndicator.textContent = "⚠️ QUÁ NGẮN (< 0.25s)";
    vadIndicator.className = "live-indicator";
    pttText.textContent = "🎙️ ẤN GIỮ ĐỂ NÓI (THẢ ĐỂ GỬI LỆNH)";
    return;
  }

  // Update telemetry metrics
  if (valRecDuration) valRecDuration.textContent = result.durationSec.toFixed(2) + " s";
  if (valRecPeak) {
    const pb = (result.peakBefore !== undefined) ? result.peakBefore.toFixed(2) : "0.00";
    const pf = (result.finalPeak !== undefined) ? result.finalPeak.toFixed(2) : "0.00";
    valRecPeak.textContent = pb + " ➜ " + pf;
  }

  // SILENCE GUARD: User stayed silent while holding the button (peak < 0.025 or rms < 0.003)
  // Completely eliminates false 'Rẽ Phải' predictions on pure silence!
  if (result.isSilence) {
    addLog("[IM LẶNG] Không phát hiện tiếng nói (Đỉnh: " + result.peakBefore.toFixed(3) + ", RMS: " + result.rmsBefore.toFixed(4) + "). Đã bỏ qua suy luận để ngăn chặn dự đoán sai!", "warn");
    vadIndicator.textContent = "🤫 KHÔNG CÓ TIẾNG NÓI (IM LẶNG)";
    vadIndicator.className = "live-indicator";
    pttText.textContent = "🎙️ ẤN GIỮ ĐỂ NÓI (THẢ ĐỂ GỬI LỆNH)";

    // Reset all gauge bars
    constants.class_names.forEach(cls => {
      const row = document.getElementById("gauge-" + cls);
      if (row) {
        const pctEl = row.querySelector(".gauge-pct");
        const barEl = row.querySelector(".gauge-bar");
        pctEl.textContent = "0.0%";
        barEl.style.width = "0%";
        row.classList.remove("active");
      }
    });
    return;
  }

  // Enable audio playback button
  if (btnPlayback) btnPlayback.disabled = false;

  // Run AI Inference
  await runInferenceOnResult(result);

  vadIndicator.textContent = "🎙️ SẴN SÀNG THU ÂM";
  vadIndicator.className = "live-indicator";
  pttText.textContent = "🎙️ ẤN GIỮ ĐỂ NÓI (THẢ ĐỂ GỬI LỆNH)";
}

// AI Inference on Normalized, Center-Padded 2.0s Mel Spectrogram
async function runInferenceOnResult(result) {
  if (!ortSession || isProcessing) return;
  isProcessing = true;

  try {
    const inputTensor = new ort.Tensor("float32", result.melTensor, [1, 1, 128, 63]);
    const t0 = performance.now();
    const feeds = { input_mel: inputTensor };
    const ortResults = await ortSession.run(feeds);
    const latency = (performance.now() - t0).toFixed(1);

    const logits = ortResults.logits.data;
    const probs = softmax(logits);

    let maxIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    // Update 7 Gauges
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

    const predictedClass = constants.class_names[maxIdx];
    const now = performance.now();

    addLog(`[AI PHÂN LOẠI] 🎯 Kết quả: ${commandNamesVi[predictedClass] || predictedClass} (${(maxProb * 100).toFixed(1)}%) - Thời lượng: ${(result.durationSec * 1000).toFixed(0)}ms - Độ trễ AI: ${latency}ms`, "sys");

    if (
      predictedClass !== "KhongXacDinh" &&
      maxProb >= CONFIDENCE_THRESHOLD &&
      (now - lastCommandTime) > COMMAND_COOLDOWN_MS
    ) {
      lastCommandTime = now;
      handleVoiceTrigger(predictedClass, maxProb, latency);
    } else if (predictedClass === "KhongXacDinh" || maxProb < CONFIDENCE_THRESHOLD) {
      addLog(`[BỎ QUA] Lệnh là 'Không Xác Định' hoặc độ tin cậy < ${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}%. Bỏ qua kích hoạt robot.`, "warn");
    }
  } catch (err) {
    console.error("Inference Error:", err);
    addLog("[LỖI AI] " + err.message, "err");
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

// ==========================================================================
// EVENT LISTENERS: PUSH-TO-TALK (MOUSE + TOUCH + SPACEBAR)
// ==========================================================================

// Audio Playback Button: Nghe lại âm thanh AI vừa nhận
if (btnPlayback) {
  btnPlayback.addEventListener("click", () => {
    if (audioEngine) {
      addLog("[HỆ THỐNG] 🔊 Đang phát lại đúng đoạn 2.0s nạp vào mô hình AI...", "sys");
      audioEngine.playLastProcessedAudio();
    }
  });
}

// Desktop Mouse Events
btnPttMic.addEventListener("mousedown", (e) => {
  e.preventDefault();
  handleStartSpeech();
});

window.addEventListener("mouseup", (e) => {
  if (isHoldActive) {
    handleStopSpeech();
  }
});

// Mobile Touch Events (iOS Safari & Android Chrome)
btnPttMic.addEventListener("touchstart", (e) => {
  e.preventDefault();
  handleStartSpeech();
}, { passive: false });

btnPttMic.addEventListener("touchend", (e) => {
  e.preventDefault();
  handleStopSpeech();
}, { passive: false });

btnPttMic.addEventListener("touchcancel", (e) => {
  e.preventDefault();
  handleStopSpeech();
}, { passive: false });

// Keyboard: Spacebar Hold-to-Talk
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    e.preventDefault();
    if (!isHoldActive) {
      handleStartSpeech();
    }
  }
  // Keyboard manual drive controls
  if (["ArrowUp", "KeyW"].includes(e.code)) {
    if (arena) arena.applyCommand("Tien");
    if (ble && ble.isConnected) ble.sendCommand("Tien");
  } else if (["ArrowDown", "KeyS"].includes(e.code)) {
    if (arena) arena.applyCommand("Lui");
    if (ble && ble.isConnected) ble.sendCommand("Lui");
  } else if (["ArrowLeft", "KeyA"].includes(e.code)) {
    if (arena) arena.applyCommand("ReTrai");
    if (ble && ble.isConnected) ble.sendCommand("ReTrai");
  } else if (["ArrowRight", "KeyD"].includes(e.code)) {
    if (arena) arena.applyCommand("RePhai");
    if (ble && ble.isConnected) ble.sendCommand("RePhai");
  }
});

window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    e.preventDefault();
    if (isHoldActive) {
      handleStopSpeech();
    }
  }
});

// BLE Connect Button
btnConnectBle.addEventListener("click", async () => {
  if (!ble.isConnected) {
    await ble.connect();
  } else {
    await ble.disconnect();
  }
});

// Reset Sim Button
btnResetSim.addEventListener("click", () => {
  if (arena) {
    arena.reset();
    addLog("[MÔ PHỎNG] Đã đặt lại vị trí Robot về tâm sa bàn.", "sys");
  }
});

// Clear Log Button
btnClearLog.addEventListener("click", () => {
  logTerminal.innerHTML = "";
  addLog("[HỆ THỐNG] Đã xóa toàn bộ nhật ký.", "sys");
});

// Boot application when DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initApp();
});
