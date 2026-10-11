// ==========================================================================
// AUDIO ENGINE: Web Audio API (16kHz) + Push-to-Talk + Peak Normalization + Center-Pad 2.0s
// Generates exact [1, 1, 128, 63] tensors matching PyTorch training 100%
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
    this.isRecording = false; // Is active recording session (holding button)
    this.isMicOpen = false;    // Is mic hardware opened

    // Accumulator for current push-to-talk speech
    this.recordedChunks = [];
    this.recordedSamplesCount = 0;

    // Display buffer (32,000 samples) for waveform & spectrogram
    this.displayBuffer = new Float32Array(this.targetSamples);
    this.displayWritePos = 0;

    // Rolling Spectrogram for Visualization (128 x 100)
    this.rollingSpec = [];
    this.maxRollingFrames = 100;

    this.currentPeak = 0.0;
    this.currentRmsDb = -60.0;
  }

  async initMic() {
    if (this.isMicOpen) return true;
    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.sampleRate
      });

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true // Hardware AGC helps mobile mics reach healthy input levels
        }
      });

      const sourceNode = this.audioCtx.createMediaStreamSource(this.mediaStream);
      const bufferSize = 2048;
      this.processorNode = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);

      this.processorNode.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        this.processIncomingBlock(inputData, e.inputBuffer.sampleRate);
      };

      sourceNode.connect(this.processorNode);
      this.processorNode.connect(this.audioCtx.destination);

      this.isMicOpen = true;
      return true;
    } catch (err) {
      console.error("[AUDIO ERR] Không thể mở microphone:", err);
      return false;
    }
  }

  // Push-to-Talk: Start collecting speech
  startRecording() {
    this.recordedChunks = [];
    this.recordedSamplesCount = 0;
    this.isRecording = true;
  }

  // Push-to-Talk: Stop collecting, normalize, center-pad & extract Mel Spectrogram
  stopRecordingAndProcess() {
    this.isRecording = false;

    if (this.recordedChunks.length === 0 || this.recordedSamplesCount < 2400) {
      // Too short (< 0.15s)
      return null;
    }

    // 1. Merge all chunks into single Float32Array
    const rawAudio = new Float32Array(this.recordedSamplesCount);
    let offset = 0;
    for (const chunk of this.recordedChunks) {
      rawAudio.set(chunk, offset);
      offset += chunk.length;
    }
    this.recordedChunks = [];
    this.recordedSamplesCount = 0;

    const durationSec = rawAudio.length / this.sampleRate;

    // 2. Soft trim silence at start and end
    let startIdx = 0;
    let endIdx = rawAudio.length - 1;
    const trimThreshold = 0.015;

    while (startIdx < endIdx && Math.abs(rawAudio[startIdx]) < trimThreshold) {
      startIdx++;
    }
    while (endIdx > startIdx && Math.abs(rawAudio[endIdx]) < trimThreshold) {
      endIdx--;
    }

    // Retain 80ms safety margin around trimmed speech
    const margin = Math.round(this.sampleRate * 0.08);
    startIdx = Math.max(0, startIdx - margin);
    endIdx = Math.min(rawAudio.length - 1, endIdx + margin);

    const trimmed = rawAudio.subarray(startIdx, endIdx + 1);

    // 3. Peak Normalization to 0.75 (Exact match to Data_MT training distribution!)
    let maxPeak = 0;
    for (let i = 0; i < trimmed.length; i++) {
      const v = Math.abs(trimmed[i]);
      if (v > maxPeak) maxPeak = v;
    }

    const normalized = new Float32Array(trimmed.length);
    let finalPeak = maxPeak;
    if (maxPeak > 0.005) {
      const gain = Math.min(0.75 / maxPeak, 15.0);
      for (let i = 0; i < trimmed.length; i++) {
        normalized[i] = trimmed[i] * gain;
      }
      finalPeak = maxPeak * gain;
    } else {
      normalized.set(trimmed);
    }

    // 4. Exact Center-Padding into 32,000 samples (2.0s) matching preprocess_offline.py
    const centeredAudio = new Float32Array(this.targetSamples);
    if (normalized.length < this.targetSamples) {
      const padTotal = this.targetSamples - normalized.length;
      const padLeft = Math.floor(padTotal / 2);
      centeredAudio.set(normalized, padLeft);
    } else {
      const start = Math.floor((normalized.length - this.targetSamples) / 2);
      centeredAudio.set(normalized.subarray(start, start + this.targetSamples), 0);
    }

    // 5. Update displayBuffer with the centered audio so waveform visualizer displays exact 2.0s input!
    this.displayBuffer.set(centeredAudio);

    // 6. Extract exact [1, 1, 128, 63] Log-Mel Spectrogram
    const melTensor = this.extractMelSpectrogram(centeredAudio);

    return {
      melTensor,
      centeredAudio,
      durationSec,
      maxPeak,
      finalPeak
    };
  }

  processIncomingBlock(rawChunk, nativeSr) {
    let chunk = rawChunk;
    if (nativeSr !== this.sampleRate) {
      chunk = this.resampleLinear(rawChunk, nativeSr, this.sampleRate);
    }

    // Measure live metrics
    let maxP = 0;
    let sumSq = 0;
    for (let i = 0; i < chunk.length; i++) {
      const absV = Math.abs(chunk[i]);
      if (absV > maxP) maxP = absV;
      sumSq += chunk[i] * chunk[i];
    }
    this.currentPeak = maxP;
    const rms = Math.sqrt(sumSq / Math.max(chunk.length, 1));
    this.currentRmsDb = 20 * Math.log10(Math.max(rms, 1e-5));

    // If recording: accumulate audio and update live display
    if (this.isRecording) {
      this.recordedChunks.push(new Float32Array(chunk));
      this.recordedSamplesCount += chunk.length;

      for (let i = 0; i < chunk.length; i++) {
        this.displayBuffer[this.displayWritePos] = chunk[i];
        this.displayWritePos = (this.displayWritePos + 1) % this.targetSamples;
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

  // Compute exact [1, 1, 128, 63] Log-Mel Spectrogram matching PyTorch torchaudio 100%
  extractMelSpectrogram(audio) {
    const padLen = this.nFft / 2;
    const paddedAudio = new Float32Array(audio.length + this.nFft);
    paddedAudio.set(audio, padLen);

    const numFrames = 63;
    const numMels = this.nMels; // 128
    const numFreqs = (this.nFft / 2) + 1; // 513

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

      for (let i = 0; i < this.nFft; i++) {
        frameSignal[i] = paddedAudio[start + i] * this.hannWindow[i];
        fftReal[i] = frameSignal[i];
        fftImag[i] = 0;
      }

      this.fft(fftReal, fftImag);

      for (let f = 0; f < numFreqs; f++) {
        powerSpectrum[f] = fftReal[f] * fftReal[f] + fftImag[f] * fftImag[f];
      }

      for (let m = 0; m < numMels; m++) {
        let sum = 0;
        const row = this.melFilterbank[m];
        for (let f = 0; f < numFreqs; f++) {
          sum += powerSpectrum[f] * row[f];
        }
        melEnergies[m][t] = sum;
      }
    }

    let maxDb = -Infinity;
    for (let m = 0; m < numMels; m++) {
      for (let t = 0; t < numFrames; t++) {
        const val = 10.0 * Math.log10(Math.max(melEnergies[m][t], 1e-10));
        melEnergies[m][t] = val;
        if (val > maxDb) maxDb = val;
      }
    }

    const minDb = maxDb - this.topDb;
    const flatTensor = new Float32Array(1 * 1 * numMels * numFrames);
    let idx = 0;
    for (let m = 0; m < numMels; m++) {
      for (let t = 0; t < numFrames; t++) {
        flatTensor[idx++] = Math.max(melEnergies[m][t], minDb);
      }
    }

    // Push latest spectrogram for rolling view
    this.rollingSpec = [];
    for (let t = 0; t < numFrames; t++) {
      const col = new Float32Array(numMels);
      for (let m = 0; m < numMels; m++) {
        col[m] = (melEnergies[m][t] - minDb) / this.topDb;
      }
      this.rollingSpec.push(col);
    }

    return flatTensor;
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

    const audio = this.displayBuffer;
    const step = Math.ceil(audio.length / width);
    const ampScale = (height / 2) * 0.9;

    if (this.isRecording) {
      ctx.strokeStyle = '#ef4444'; // Red glowing when holding down / recording
      ctx.lineWidth = 2.2;
    } else {
      ctx.strokeStyle = '#00f2fe'; // Cyan when finished / ready
      ctx.lineWidth = 1.5;
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

  // Draw Log-Mel Spectrogram onto HTML5 Canvas
  renderSpectrogram(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    if (this.rollingSpec.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('Đang chờ khẩu lệnh để tạo phổ Log-Mel Spectrogram [128x63]...', width / 2, height / 2 + 4);
      return;
    }

    const numCols = this.rollingSpec.length;
    const colWidth = width / numCols;
    const rowHeight = height / this.nMels;

    for (let c = 0; c < numCols; c++) {
      const col = this.rollingSpec[c];
      const x = c * colWidth;
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
