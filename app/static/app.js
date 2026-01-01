const startCta = document.getElementById("startCta");
const profileSection = document.getElementById("profileSection");
const profileForm = document.getElementById("profileForm");
const profileError = document.getElementById("profileError");
const recordSection = document.getElementById("recordSection");
const recordBtn = document.getElementById("recordBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("emotionLabel");
const confidenceEl = document.getElementById("emotionConfidence");
const scoreList = document.getElementById("scoreList");
const liveToggle = document.getElementById("liveToggle");
const confidenceCanvas = document.getElementById("confidenceChart");
const scoresCanvas = document.getElementById("scoresChart");
const modal = document.getElementById("analysisModal");
const cancelAnalysis = document.getElementById("cancelAnalysis");

let audioCtx;
let mediaStream;
let sourceNode;
let processorNode;
let buffers = [];
let isRecording = false;
let bufferLength = 0;
let lastSentAt = 0;
let isSending = false;
let confidenceChart;
let scoresChart;
let profileData = {};
let autoStopTimer;
const DEFAULT_LABELS = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "ps",
  "sad",
];

const WINDOW_SEC = 3;
const MIN_SEND_INTERVAL_MS = 1500;
const MAX_RECORD_MS = 4500;
const MIN_SAMPLES = 8000;

const setStatus = (text) => {
  statusEl.textContent = text;
};

const showModal = () => modal.classList.add("modal--visible");
const hideModal = () => modal.classList.remove("modal--visible");

const resetUI = () => {
  labelEl.textContent = "-";
  confidenceEl.textContent = "Waiting for audio";
  scoreList.innerHTML = "";
  DEFAULT_LABELS.forEach((label) => {
    const row = document.createElement("div");
    row.className = "score";
    row.innerHTML = `<strong>${label}</strong><span>0.0%</span>`;
    scoreList.appendChild(row);
  });
  updateChart(0);
  setStatus("Ready for a new recording.");
};

const initChart = () => {
  if (!window.Chart || !confidenceCanvas) {
    return;
  }

  confidenceChart = new Chart(confidenceCanvas, {
    type: "doughnut",
    data: {
      labels: ["Confidence", "Remaining"],
      datasets: [
        {
          data: [0, 100],
          backgroundColor: ["#111827", "#e5e7eb"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: "70%",
      plugins: { legend: { display: false } },
    },
  });

  if (scoresCanvas) {
    scoresChart = new Chart(scoresCanvas, {
      type: "bar",
      data: {
        labels: [],
        datasets: [
          {
            label: "Score",
            data: [],
            backgroundColor: "#111827",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#6b7280" } },
          y: { beginAtZero: true, max: 1, ticks: { color: "#6b7280" } },
        },
      },
    });
  }
};

const updateChart = (confidence) => {
  if (!confidenceChart) return;
  confidenceChart.data.datasets[0].data = [
    Math.round(confidence * 100),
    Math.round((1 - confidence) * 100),
  ];
  confidenceChart.data.datasets[0].backgroundColor = [
    "#111827",
    confidence === 0 ? "#e5e7eb" : "#f3f4f6",
  ];
  confidenceChart.update();
};

const startRecording = async () => {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  buffers = [];
  bufferLength = 0;
  lastSentAt = 0;

  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const chunk = new Float32Array(input);
    buffers.push(chunk);
    bufferLength += chunk.length;

    if (liveToggle.checked) {
      maybeSendLive(audioCtx.sampleRate);
    }
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioCtx.destination);

  isRecording = true;
  recordBtn.textContent = "Stop Recording";
  resetBtn.disabled = true;
  setStatus("Recording... speak clearly for a few seconds.");

  clearTimeout(autoStopTimer);
  autoStopTimer = setTimeout(() => {
    if (isRecording) {
      stopRecording();
    }
  }, MAX_RECORD_MS);
};

const stopRecording = async () => {
  clearTimeout(autoStopTimer);
  processorNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());
  await audioCtx.close();

  isRecording = false;
  recordBtn.textContent = "Start Recording";
  setStatus("Analysing speech...");

  const samples = flattenBuffers(buffers);
  if (samples.length < MIN_SAMPLES) {
    setStatus("Recording too short. Please try again.");
    resetBtn.disabled = false;
    return;
  }
  await sendPrediction(samples, audioCtx.sampleRate);
  resetBtn.disabled = false;
};

const flattenBuffers = (chunks) => {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
};

const getLatestSamples = (chunks, sampleCount) => {
  const result = new Float32Array(sampleCount);
  let offset = sampleCount;
  for (let i = chunks.length - 1; i >= 0 && offset > 0; i--) {
    const chunk = chunks[i];
    const copyCount = Math.min(chunk.length, offset);
    offset -= copyCount;
    result.set(chunk.subarray(chunk.length - copyCount), offset);
  }
  return result;
};

const trimBuffers = (maxSamples) => {
  while (bufferLength > maxSamples && buffers.length > 1) {
    const removed = buffers.shift();
    bufferLength -= removed.length;
  }
};

const maybeSendLive = async (sampleRate) => {
  const windowSamples = Math.floor(sampleRate * WINDOW_SEC);
  const now = Date.now();
  if (bufferLength < windowSamples) {
    return;
  }
  if (isSending || now - lastSentAt < MIN_SEND_INTERVAL_MS) {
    return;
  }
  isSending = true;
  lastSentAt = now;
  const samples = getLatestSamples(buffers, windowSamples);
  await sendPrediction(samples, sampleRate, true);
  isSending = false;
  trimBuffers(windowSamples * 4);
};

const sendPrediction = async (samples, sampleRate, isLive = false) => {
  const wav = encodeWav(samples, sampleRate);
  const blob = new Blob([wav], { type: "audio/wav" });
  const formData = new FormData();
  formData.append("file", blob, "recording.wav");

  if (!isLive) showModal();
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);
    const data = await response.json();
    renderResult(data);
    setStatus(isLive ? "Live prediction updated." : "Prediction complete.");
  } catch (error) {
    setStatus("Prediction timed out or failed. Please try again.");
  } finally {
    if (!isLive) hideModal();
  }
};

const encodeWav = (samples, sampleRate) => {
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

  floatTo16BitPCM(view, 44, samples);
  return buffer;
};

const floatTo16BitPCM = (view, offset, input) => {
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset + i * 2, s, true);
  }
};

const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

const renderResult = (data) => {
  if (!data || !data.label) {
    setStatus("No prediction returned.");
    return;
  }
  labelEl.textContent = data.label;
  confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  const entries = Object.entries(data.scores || {}).sort((a, b) => b[1] - a[1]);
  scoreList.innerHTML = "";
  entries.forEach(([label, score]) => {
    const row = document.createElement("div");
    row.className = "score";
    row.innerHTML = `<strong>${label}</strong><span>${(score * 100).toFixed(1)}%</span>`;
    scoreList.appendChild(row);
  });
  updateChart(data.confidence);
  if (scoresChart) {
    scoresChart.data.labels = entries.map(([label]) => label);
    scoresChart.data.datasets[0].data = entries.map(([, score]) => score);
    scoresChart.update();
  }
};

const selectChip = (event) => {
  const button = event.target.closest(".chip");
  if (!button) return;
  const group = button.dataset.group;
  document
    .querySelectorAll(`.chip[data-group="${group}"]`)
    .forEach((chip) => chip.classList.remove("chip--active"));
  button.classList.add("chip--active");
};

const validateProfile = () => {
  const lang = document.querySelector('.chip[data-group="lang"].chip--active')
    ?.textContent;
  const age = document.querySelector('.chip[data-group="age"].chip--active')
    ?.textContent;
  const gender = document.querySelector('.chip[data-group="gender"].chip--active')
    ?.textContent;
  if (!lang || !age || !gender) {
    profileError.textContent =
      "Please select language, age group, and gender to continue.";
    return false;
  }
  profileError.textContent = "";
  profileData = { lang, age, gender };
  return true;
};

startCta.addEventListener("click", () => {
  profileSection.classList.remove("demo--hidden");
  profileSection.scrollIntoView({ behavior: "smooth" });
});

profileForm.addEventListener("click", selectChip);

profileForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!validateProfile()) {
    return;
  }
  profileSection.style.display = "none";
  recordSection.classList.remove("demo--hidden");
  recordSection.scrollIntoView({ behavior: "smooth" });
});

recordBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

resetBtn.addEventListener("click", () => {
  resetUI();
});

cancelAnalysis.addEventListener("click", () => {
  hideModal();
});

initChart();
resetUI();
