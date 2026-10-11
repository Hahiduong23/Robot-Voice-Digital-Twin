// ==========================================================================
// AUDIO ENGINE: Web Audio API + VAD-Gated Speech Center-Padding (16kHz)
// Generates exact [1, 1, 128, 63] tensors matching PyTorch torchaudio 100%
// ==========================================================================

class AudioEngine {
  constructor(constants) {
    this.constants = constants;
    this.sampleRate = constants.sample_rate || 16000;
    this.targetSamples = constants.target_samples || 32000; // 2.0s
    this.nFft = constants.n_fft || 1024;
    this.hopLength = constants.hop_length || 512;
    this.nMels = constants.n_mels || 128;
    this.topDb = constants.top_db || 80.0;
    this.hannWindow = new Float32Array(constants.hann_window);
    this.melFilterbank = constants.mel_filterbank; // 128 x 513

    this.audioCtx = null;
    this.mediaStream = null;
    this.processorNode = null;
    this.isRecording = false;
    this.denoiseEnabled = true;

    // 32,000 samples Ring Buffer (2.0s) for continuous live monitoring & visualizers
    this.ringBuffer = new Float32Array(this.targetSamples);
    this.ringWritePos = 0;

    // Rolling Spectrogram for Visualization (128 x 150)
    this.rollingSpec = [];
    this.maxRollingFrames = 100;

    // VAD & Real-Time Endpointing Engine
    this.currentRmsDb = -60.0;
    this.isVoiceDetected = false;
    this.vadThresholdDb = -42.0; // Default voice threshold in dB (configurable)
    this.vadState = "IDLE"; // 'IDLE' | 'RECORDING' | 'PROCESSING'

    // Pre-roll buffer (300ms = 4800 samples) to preserve initial speech consonants ('T', 'Kh', 'R'...)
    this.preRollSize = Math.round(this.sampleRate * 0.30);
    this.preRollBuffer = new Float32Array(this.preRollSize);
    this.preRollPos = 0;

    // Utterance accumulation
    this.recordedSpeechChunks = [];
    this.recordedSpeechLength = 0;
    this.silenceSamplesCount = 0;
    this.minSpeechSamples = Math.round(this.sampleRate * 0.25); // At least 250ms of speech (4000 samples)
    this.maxSpeechSamples = Math.round(this.sampleRate * 1.80); // Safety limit 1.8s
    this.silenceHangoverSamples = Math.round(this.sampleRate * 0.35); // 350ms silence to finalize speech

    // Callbacks
    this.onUtteranceComplete = null; // (melFlatTensor, centeredAudio, durationMs) => {}
    this.onVadStateChange = null; // (state: string, rmsDb: number) => {}
  }

  async initMic() {
    if (this.isRecording) return true;

    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate
      });

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });

      const sourceNode = this.audioCtx.createMediaStreamSource(this.mediaStream);

      // ScriptProcessor for universal browser compatibility (iOS Safari, Android Chrome, Mac, Windows)
      const bufferSize = 2048;
      this.processorNode = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);

      this.processorNode.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);
        this.processAudioBlock(inputData, e.inputBuffer.sampleRate);
      };

      sourceNode.connect(this.processorNode);
      this.processorNode.connect(this.audioCtx.destination);

      this.isRecording = true;
      this.vadState = "IDLE";
      return true;
    } catch (err) {
      console.error("[AUDIO ERR] Không thể mở microphone:", err);
      return false;
    }
  }

  stopMic() {
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close();
      this.audioCtx = null;
    }
    this.isRecording = false;
    this.vadState = "IDLE";
  }

  // Push samples into circular buffer with linear resampling if needed
  processAudioBlock(rawChunk, nativeSr) {
    let chunk = rawChunk;
    if (nativeSr !== this.sampleRate) {
      chunk = this.resampleLinear(rawChunk, nativeSr, this.sampleRate);
    }

    // Measure RMS (dB) on incoming audio
    let sumSq = 0;
    for (let i = 0; i < chunk.length; i++) {
      sumSq += chunk[i] * chunk[i];
    }
    const rms = Math.sqrt(sumSq / Math.max(chunk.length, 1));
    this.currentRmsDb = 20 * Math.log10(Math.max(rms, 1e-5));
    const isAboveThreshold = (this.currentRmsDb > this.vadThresholdDb);
    this.isVoiceDetected = isAboveThreshold;

    // Push into ring buffer for live waveform & rolling spectrogram display
    for (let i = 0; i < chunk.length; i++) {
      this.ringBuffer[this.ringWritePos] = chunk[i];
      this.ringWritePos = (this.ringWritePos + 1) % this.targetSamples;
    }

    // Push latest column for live rolling spectrogram rendering
    if (this.isRecording && this.ringWritePos % 512 === 0) {
      this.updateLiveSpectrogramColumn();
    }

    // VAD Speech Collector & Endpointing Engine
    this.handleVadChunk(chunk, isAboveThreshold);
  }

  handleVadChunk(chunk, isAboveThreshold) {
    if (this.vadState === "IDLE") {
      // Maintain circular pre-roll buffer in silence
      for (let i = 0; i < chunk.length; i++) {
        this.preRollBuffer[this.preRollPos] = chunk[i];
        this.preRollPos = (this.preRollPos + 1) % this.preRollSize;
      }

      if (isAboveThreshold) {
        // Voice onset detected!
        this.vadState = "RECORDING";
        this.silenceSamplesCount = 0;

        // Extract ordered pre-roll audio (300ms before onset)
        const preRoll = new Float32Array(this.preRollSize);
        const part1Len = this.preRollSize - this.preRollPos;
        preRoll.set(this.preRollBuffer.subarray(this.preRollPos), 0);
        preRoll.set(this.preRollBuffer.subarray(0, this.preRollPos), part1Len);

        this.recordedSpeechChunks = [preRoll, new Float32Array(chunk)];
        this.recordedSpeechLength = this.preRollSize + chunk.length;

        if (this.onVadStateChange) {
          this.onVadStateChange("RECORDING", this.currentRmsDb);
        }
      } else {
        if (this.onVadStateChange) {
          this.onVadStateChange("IDLE", this.currentRmsDb);
        }
      }
    } else if (this.vadState === "RECORDING") {
      // Accumulate audio during speech
      this.recordedSpeechChunks.push(new Float32Array(chunk));
      this.recordedSpeechLength += chunk.length;

      if (!isAboveThreshold) {
        this.silenceSamplesCount += chunk.length;
      } else {
        this.silenceSamplesCount = 0;
      }

      const silenceReached = (this.silenceSamplesCount >= this.silenceHangoverSamples);
      const maxReached = (this.recordedSpeechLength >= this.maxSpeechSamples);

      if (silenceReached || maxReached) {
        this.vadState = "PROCESSING";
        if (this.onVadStateChange) {
          this.onVadStateChange("PROCESSING", this.currentRmsDb);
        }

        // Merge accumulated chunks
        const speech = new Float32Array(this.recordedSpeechLength);
        let offset = 0;
        for (const c of this.recordedSpeechChunks) {
          speech.set(c, offset);
          offset += c.length;
        }

        // Reset buffer
        this.recordedSpeechChunks = [];
        this.recordedSpeechLength = 0;
        this.silenceSamplesCount = 0;

        // Check if duration is meaningful (>= minSpeechSamples + preRollSize)
        if (speech.length >= this.minSpeechSamples + this.preRollSize) {
          let effectiveSpeech = speech;
          // Trim the trailing hangover silence
          if (silenceReached && speech.length > this.silenceHangoverSamples) {
            const trimTail = Math.floor(this.silenceHangoverSamples * 0.7);
            effectiveSpeech = speech.subarray(0, speech.length - trimTail);
          }

          // CENTER-PAD INTO 32,000 SAMPLES (2.0s) EXACTLY MATCHING PREPROCESS_OFFLINE.PY
          const centeredAudio = new Float32Array(this.targetSamples);
          if (effectiveSpeech.length < this.targetSamples) {
            const padTotal = this.targetSamples - effectiveSpeech.length;
            const padLeft = Math.floor(padTotal / 2);
            centeredAudio.set(effectiveSpeech, padLeft);
          } else {
            const start = Math.floor((effectiveSpeech.length - this.targetSamples) / 2);
            centeredAudio.set(effectiveSpeech.subarray(start, start + this.targetSamples), 0);
          }

          // Extract exact Log-Mel Spectrogram [1, 1, 128, 63]
          const melTensor = this.extractMelSpectrogram(centeredAudio);
          const durationMs = (effectiveSpeech.length / this.sampleRate) * 1000;

          if (this.onUtteranceComplete) {
            this.onUtteranceComplete(melTensor, centeredAudio, durationMs);
          }
        }

        this.vadState = "IDLE";
        if (this.onVadStateChange) {
          this.onVadStateChange("IDLE", this.currentRmsDb);
        }
      } else {
        if (this.onVadStateChange) {
          this.onVadStateChange("RECORDING", this.currentRmsDb);
        }
      }
    }
  }

  // Fast Linear Resampler
  resampleLinear(input, srcSr, dstSr) {
    const ratio = dstSr / srcSr;
    const newLen = Math.round(input.length * ratio);
    const output = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const srcIdx = i / ratio;
      const idxFloor = Math.floor(srcIdx);
      const frac = srcIdx - idxFloor;
      const nextIdx = Math.min(idxFloor + 1, input.length - 1);
      output[i] = input[idxFloor] * (1 - frac) + input[nextIdx] * frac;
    }
    return output;
  }

  // Extract ordered 32,000 samples from circular buffer
  getLatestAudioBuffer() {
    const output = new Float32Array(this.targetSamples);
    const part1Len = this.targetSamples - this.ringWritePos;
    output.set(this.ringBuffer.subarray(this.ringWritePos), 0);
    output.set(this.ringBuffer.subarray(0, this.ringWritePos), part1Len);
    return output;
  }

  // Compute exact [1, 1, 128, 63] Log-Mel Spectrogram matching PyTorch torchaudio 100%
  extractMelSpectrogram(customAudio = null) {
    const audio = customAudio || this.getLatestAudioBuffer();
    // Center padding: 512 at each end (1024 // 2) matching torchaudio
    const padLen = this.nFft / 2;
    const paddedAudio = new Float32Array(audio.length + this.nFft);
    paddedAudio.set(audio, padLen);

    const numFrames = 63; // Math.floor((32000 - 1024) / 512) + 1 with padding
    const numMels = this.nMels; // 128
    const numFreqs = (this.nFft / 2) + 1; // 513

    // Matrix [128, 63]
    const melEnergies = [];
    for (let m = 0; m < numMels; m++) {
      melEnergies.push(new Float32Array(numFrames));
    }

    const frameSignal = new Float32Array(this.nFft);
    const fftReal = new Float32Array(this.nFft);
    const fftImag = new Float32Array(this.nFft);
    const powerSpectrum = new Float32Array(numFreqs);

    for (let t = 0; t < numFrames; t++) {
      const start = t * this.hopLength;

      // Windowing
      for (let i = 0; i < this.nFft; i++) {
        frameSignal[i] = paddedAudio[start + i] * this.hannWindow[i];
        fftReal[i] = frameSignal[i];
        fftImag[i] = 0;
      }

      // Fast Real FFT (Cooley-Tukey Radix-2)
      this.fft(fftReal, fftImag);

      // Power Spectrum: |FFT|^2
      for (let f = 0; f < numFreqs; f++) {
        const r = fftReal[f];
        const im = fftImag[f];
        powerSpectrum[f] = r * r + im * im;
      }

      // Mel Filterbank dot-product
      for (let m = 0; m < numMels; m++) {
        let sum = 0;
        const row = this.melFilterbank[m];
        for (let f = 0; f < numFreqs; f++) {
          sum += powerSpectrum[f] * row[f];
        }
        melEnergies[m][t] = sum;
      }
    }

    // Convert to dB: 10 * log10(max(x, 1e-10)), top_db = 80.0
    let maxDb = -Infinity;
    for (let m = 0; m < numMels; m++) {
      for (let t = 0; t < numFrames; t++) {
        const val = 10.0 * Math.log10(Math.max(melEnergies[m][t], 1e-10));
        melEnergies[m][t] = val;
        if (val > maxDb) maxDb = val;
      }
    }

    const minDb = maxDb - this.topDb;
    // Flat 1D Float32Array for ONNX Tensor [1, 1, 128, 63]
    const flatTensor = new Float32Array(1 * 1 * numMels * numFrames);
    let idx = 0;
    for (let m = 0; m < numMels; m++) {
      for (let t = 0; t < numFrames; t++) {
        flatTensor[idx++] = Math.max(melEnergies[m][t], minDb);
      }
    }

    return flatTensor;
  }

  // Live Spectrogram Column updater for rolling visualizer
  updateLiveSpectrogramColumn() {
    const audio = this.getLatestAudioBuffer();
    const frameSignal = new Float32Array(this.nFft);
    const fftReal = new Float32Array(this.nFft);
    const fftImag = new Float32Array(this.nFft);
    const powerSpectrum = new Float32Array(this.nFft / 2 + 1);

    const start = Math.max(0, audio.length - this.nFft);
    for (let i = 0; i < this.nFft; i++) {
      frameSignal[i] = audio[start + i] * this.hannWindow[i];
      fftReal[i] = frameSignal[i];
      fftImag[i] = 0;
    }
    this.fft(fftReal, fftImag);
    for (let f = 0; f < powerSpectrum.length; f++) {
      powerSpectrum[f] = fftReal[f] * fftReal[f] + fftImag[f] * fftImag[f];
    }

    const col = new Float32Array(this.nMels);
    let maxEnergy = 1e-10;
    for (let m = 0; m < this.nMels; m++) {
      let sum = 0;
      const row = this.melFilterbank[m];
      for (let f = 0; f < powerSpectrum.length; f++) {
        sum += powerSpectrum[f] * row[f];
      }
      col[m] = sum;
      if (sum > maxEnergy) maxEnergy = sum;
    }

    const normCol = new Float32Array(this.nMels);
    for (let m = 0; m < this.nMels; m++) {
      const db = 10 * Math.log10(Math.max(col[m], 1e-10));
      normCol[m] = Math.max(0, Math.min(1, (db + 60) / 60));
    }

    this.rollingSpec.push(normCol);
    if (this.rollingSpec.length > this.maxRollingFrames) {
      this.rollingSpec.shift();
    }
  }

  // Fast In-Place Cooley-Tukey Radix-2 FFT (N = 1024)
  fft(real, imag) {
    const n = real.length;
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
      if (i < j) {
        let tr = real[i]; real[i] = real[j]; real[j] = tr;
        let ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
      }
      let k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const angle = -2 * Math.PI / len;
      const wStepR = Math.cos(angle);
      const wStepI = Math.sin(angle);

      for (let i = 0; i < n; i += len) {
        let wr = 1.0;
        let wi = 0.0;
        for (let m = 0; m < halfLen; m++) {
          const uR = real[i + m];
          const uI = imag[i + m];
          const vR = real[i + m + halfLen] * wr - imag[i + m + halfLen] * wi;
          const vI = real[i + m + halfLen] * wi + imag[i + m + halfLen] * wr;

          real[i + m] = uR + vR;
          imag[i + m] = uI + vI;
          real[i + m + halfLen] = uR - vR;
          imag[i + m + halfLen] = uI - vI;

          const nextWr = wr * wStepR - wi * wStepI;
          wi = wr * wStepI + wi * wStepR;
          wr = nextWr;
        }
      }
    }
  }

  // Draw Waveform onto HTML5 Canvas
  renderWaveform(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    if (!this.isRecording) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('Microphone đang tắt. Bấm [Bật Microphone] để bắt đầu', width / 2, height / 2 + 4);
      return;
    }

    const audio = this.getLatestAudioBuffer();
    const step = Math.ceil(audio.length / width);
    const ampScale = (height / 2) * 0.9;

    if (this.vadState === 'RECORDING') {
      ctx.strokeStyle = '#f59e0b'; // Amber when speaking
      ctx.lineWidth = 2.0;
    } else {
      ctx.strokeStyle = '#38bdf8'; // Cyan when listening
      ctx.lineWidth = 1.2;
    }

    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const idx = x * step;
      const y = (height / 2) - (audio[idx] * ampScale);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Draw Rolling Mel Spectrogram onto HTML5 Canvas
  renderSpectrogram(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (this.rollingSpec.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('Đang chờ dữ liệu phổ Log-Mel Spectrogram...', width / 2, height / 2 + 4);
      return;
    }

    const numCols = this.rollingSpec.length;
    const colWidth = width / this.maxRollingFrames;
    const rowHeight = height / this.nMels;

    for (let c = 0; c < numCols; c++) {
      const col = this.rollingSpec[c];
      const x = (width - (numCols - c) * colWidth);
      for (let m = 0; m < this.nMels; m++) {
        const y = height - (m + 1) * rowHeight;
        const normVal = Math.min(Math.max(col[m], 0), 1);

        let r = 0, g = 0, b = 0;
        if (normVal < 0.33) {
          const t = normVal / 0.33;
          r = Math.floor(40 * t);
          b = Math.floor(120 * t);
        } else if (normVal < 0.66) {
          const t = (normVal - 0.33) / 0.33;
          r = Math.floor(40 + 200 * t);
          g = Math.floor(80 * t);
          b = Math.floor(120 * (1 - t));
        } else {
          const t = (normVal - 0.66) / 0.34;
          r = 240 + Math.floor(15 * t);
          g = Math.floor(80 + 175 * t);
          b = Math.floor(200 * t);
        }

        ctx.fillStyle = "rgb(" + r + "," + g + "," + b + ")";
        ctx.fillRect(x, y, colWidth + 0.5, rowHeight + 0.5);
      }
    }
  }
}
