const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const resultStatus = document.getElementById("resultStatus");
const playback = document.getElementById("playback");
const fileMeta = document.getElementById("fileMeta");
const sampleRateEl = document.getElementById("sampleRate");
const durationEl = document.getElementById("duration");
const resultsEl = document.getElementById("results");

const API_ENDPOINT = "/api/analyze";
const USE_MOCK = false;

let audioContext;
let mediaStream;
let processor;
let sourceNode;
let gainNode;
let audioBuffers = [];
let recordingStart = 0;
let timerId;
let wavBlob;
let wavUrl;

startBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
analyzeBtn.addEventListener("click", analyzeAudio);

function setStatus(text) {
  statusEl.textContent = text;
}

function setResultStatus(text) {
  resultStatus.textContent = text;
}

async function startRecording() {
  if (mediaStream) {
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    setStatus("Microphone blocked");
    return;
  }

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  sampleRateEl.textContent = `${audioContext.sampleRate} Hz`;

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  gainNode = audioContext.createGain();
  gainNode.gain.value = 0;

  audioBuffers = [];
  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    audioBuffers.push(new Float32Array(input));
  };

  sourceNode.connect(processor);
  processor.connect(gainNode);
  gainNode.connect(audioContext.destination);

  recordingStart = Date.now();
  timerId = setInterval(updateDuration, 200);

  setStatus("Recording");
  startBtn.disabled = true;
  stopBtn.disabled = false;
  analyzeBtn.disabled = true;
  playback.removeAttribute("src");
  fileMeta.textContent = "Recording...";
}

function stopRecording() {
  if (!mediaStream) {
    return;
  }

  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;

  if (processor) {
    processor.disconnect();
  }
  if (sourceNode) {
    sourceNode.disconnect();
  }
  if (gainNode) {
    gainNode.disconnect();
  }

  clearInterval(timerId);
  updateDuration();

  const samples = flattenBuffers(audioBuffers);
  wavBlob = encodeWav(samples, audioContext.sampleRate);
  if (wavUrl) {
    URL.revokeObjectURL(wavUrl);
  }
  wavUrl = URL.createObjectURL(wavBlob);

  playback.src = wavUrl;
  fileMeta.textContent = describeFile(wavBlob);

  audioContext.close();
  audioContext = null;

  setStatus("Stopped");
  startBtn.disabled = false;
  stopBtn.disabled = true;
  analyzeBtn.disabled = false;
}

function updateDuration() {
  if (!recordingStart) {
    durationEl.textContent = "00:00";
    return;
  }
  const elapsed = Date.now() - recordingStart;
  durationEl.textContent = formatDuration(elapsed);
}

function formatDuration(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function flattenBuffers(buffers) {
  const length = buffers.reduce((sum, buffer) => sum + buffer.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  buffers.forEach((buffer) => {
    result.set(buffer, offset);
    offset += buffer.length;
  });
  return result;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function describeFile(blob) {
  const sizeKb = Math.round(blob.size / 1024);
  return `recording.wav (${sizeKb} KB)`;
}

async function analyzeAudio() {
  if (!wavBlob) {
    return;
  }

  setResultStatus("Uploading");
  analyzeBtn.disabled = true;
  resultsEl.innerHTML = "";

  if (USE_MOCK) {
    renderResults({ happy: 0.62, neutral: 0.2, sad: 0.1, angry: 0.08 });
    setResultStatus("Mocked");
    analyzeBtn.disabled = false;
    return;
  }

  const formData = new FormData();
  formData.append("file", wavBlob, "recording.wav");

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const data = await response.json();
    const normalized = normalizeResults(data);
    if (!normalized) {
      throw new Error("Response missing emotion scores");
    }

    renderResults(normalized);
    setResultStatus("Complete");
  } catch (error) {
    resultsEl.innerHTML = "<div class=\"empty\">Upload failed. Check API.</div>";
    setResultStatus("Failed");
  } finally {
    analyzeBtn.disabled = false;
  }
}

function normalizeResults(data) {
  if (!data) {
    return null;
  }
  if (data.emotions && typeof data.emotions === "object") {
    return data.emotions;
  }
  if (data.scores && typeof data.scores === "object") {
    return data.scores;
  }
  if (Array.isArray(data)) {
    return arrayToMap(data);
  }
  if (Array.isArray(data.results)) {
    return arrayToMap(data.results);
  }
  if (typeof data === "object") {
    const values = Object.values(data);
    if (values.every((value) => typeof value === "number")) {
      return data;
    }
  }
  return null;
}

function arrayToMap(items) {
  const map = {};
  items.forEach((item) => {
    if (typeof item === "string") {
      map[item] = 0;
      return;
    }
    if (item && typeof item === "object") {
      const label = item.label || item.emotion || item.name;
      const score = item.score ?? item.probability ?? item.value;
      if (label) {
        map[label] = Number(score) || 0;
      }
    }
  });
  return map;
}

function renderResults(scores) {
  const entries = Object.entries(scores)
    .map(([label, value]) => [label, clamp01(value)])
    .sort((a, b) => b[1] - a[1]);

  resultsEl.innerHTML = "";

  if (!entries.length) {
    resultsEl.innerHTML = "<div class=\"empty\">No results.</div>";
    return;
  }

  entries.forEach(([label, value], index) => {
    const row = document.createElement("div");
    row.className = "result-row";

    const title = document.createElement("div");
    title.className = "result-title";
    title.innerHTML = `<span>${label}</span><span>${Math.round(
      value * 100
    )}%</span>`;

    const bar = document.createElement("div");
    bar.className = "bar";

    const fill = document.createElement("span");
    fill.style.width = `${Math.round(value * 100)}%`;
    fill.style.transitionDelay = `${index * 80}ms`;

    bar.appendChild(fill);
    row.appendChild(title);
    row.appendChild(bar);
    resultsEl.appendChild(row);
  });
}

function clamp01(value) {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

window.addEventListener("beforeunload", () => {
  if (wavUrl) {
    URL.revokeObjectURL(wavUrl);
  }
});
