// Pure-JS MNIST CNN inference matching mnist_digit_recognizer.h5

const statusEl = document.getElementById("model-status");
const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
const previewCanvas = document.getElementById("preview-canvas");
const previewCtx = previewCanvas.getContext("2d");
const canvasHint = document.getElementById("canvas-hint");

const brushSlider = document.getElementById("brush-size");
const brushValue = document.getElementById("brush-value");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");

const predictedDigitEl = document.getElementById("predicted-digit");
const predictedConfEl = document.getElementById("predicted-confidence");
const inferenceTimeEl = document.getElementById("inference-time");
const probBarsEl = document.getElementById("prob-bars");

const tabButtons = document.querySelectorAll(".tab-btn");
const drawPanel = document.getElementById("draw-panel");
const uploadPanel = document.getElementById("upload-panel");
const imageUpload = document.getElementById("image-upload");
const dropzone = document.getElementById("dropzone");
const autoInvertCheckbox = document.getElementById("auto-invert");
const uploadPreviewWrap = document.getElementById("upload-preview-wrap");
const uploadPreviewImg = document.getElementById("upload-preview");

let modelWeights = null;
let isDrawing = false;
let hasInk = false;
let lastPoint = null;
let lastUploadedImage = null;
let livePredictTimer = null;

// Build probability bars 0-9
const barRows = [];
for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "prob-row";
  row.innerHTML = `
    <span class="prob-digit">${d}</span>
    <div class="prob-track"><div class="prob-fill"></div></div>
    <span class="prob-pct">0.0%</span>
  `;
  probBarsEl.appendChild(row);
  barRows.push({
    row,
    fill: row.querySelector(".prob-fill"),
    pct: row.querySelector(".prob-pct"),
  });
}

function initCanvas() {
  drawCtx.fillStyle = "#000000";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  previewCtx.fillStyle = "#000000";
  previewCtx.fillRect(0, 0, 28, 28);
  hasInk = false;
  canvasHint.classList.remove("is-hidden");
}

function resetPredictionUI() {
  predictedDigitEl.textContent = "—";
  predictedConfEl.textContent = "Draw or select a digit";
  inferenceTimeEl.textContent = "—";
  for (const { row, fill, pct } of barRows) {
    row.classList.remove("is-top");
    fill.style.width = "0%";
    pct.textContent = "0.0%";
  }
}

async function loadModel() {
  try {
    const [metaRes, binRes] = await Promise.all([
      fetch("assets/model_meta.json"),
      fetch("assets/model_weights.bin"),
    ]);
    if (!metaRes.ok || !binRes.ok) {
      throw new Error("Failed to fetch model artifacts");
    }
    const meta = await metaRes.json();
    const buffer = await binRes.arrayBuffer();
    const allFloats = new Float32Array(buffer);

    const tensors = {};
    for (const t of meta.tensors) {
      tensors[t.name] = allFloats.subarray(t.offset, t.offset + t.length);
    }
    modelWeights = tensors;

    statusEl.textContent = "Model ready";
    statusEl.classList.remove("status-loading");
    statusEl.classList.add("status-ready");
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Model load failed";
    statusEl.classList.remove("status-loading");
    statusEl.classList.add("status-error");
  }
}

// CNN Forward Pass matching Keras Sequential:
// Conv2D(32, 3x3, relu) -> MaxPool(2x2) -> Conv2D(64, 3x3, relu) -> MaxPool(2x2) -> Flatten -> Dense(128, relu) -> Dense(10, softmax)
function conv2dRelu(input, inH, inW, inC, kernel, bias, kSize, outC) {
  const outH = inH - kSize + 1;
  const outW = inW - kSize + 1;
  const output = new Float32Array(outH * outW * outC);

  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      const outBase = (oh * outW + ow) * outC;
      for (let f = 0; f < outC; f++) {
        let sum = bias[f];
        for (let kh = 0; kh < kSize; kh++) {
          for (let kw = 0; kw < kSize; kw++) {
            const inBase = ((oh + kh) * inW + (ow + kw)) * inC;
            const kBase = ((kh * kSize + kw) * inC) * outC;
            for (let c = 0; c < inC; c++) {
              sum += input[inBase + c] * kernel[kBase + c * outC + f];
            }
          }
        }
        output[outBase + f] = sum > 0 ? sum : 0;
      }
    }
  }
  return { data: output, h: outH, w: outW, c: outC };
}

function maxPool2x2(input, inH, inW, channels) {
  const outH = Math.floor(inH / 2);
  const outW = Math.floor(inW / 2);
  const output = new Float32Array(outH * outW * channels);

  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      const outBase = (oh * outW + ow) * channels;
      const ih = oh * 2;
      const iw = ow * 2;
      const idx00 = (ih * inW + iw) * channels;
      const idx01 = (ih * inW + (iw + 1)) * channels;
      const idx10 = ((ih + 1) * inW + iw) * channels;
      const idx11 = ((ih + 1) * inW + (iw + 1)) * channels;
      for (let c = 0; c < channels; c++) {
        let m = input[idx00 + c];
        if (input[idx01 + c] > m) m = input[idx01 + c];
        if (input[idx10 + c] > m) m = input[idx10 + c];
        if (input[idx11 + c] > m) m = input[idx11 + c];
        output[outBase + c] = m;
      }
    }
  }
  return { data: output, h: outH, w: outW, c: channels };
}

function dense(input, kernel, bias, inSize, outSize, applyRelu) {
  const output = new Float32Array(outSize);
  for (let j = 0; j < outSize; j++) {
    let sum = bias[j];
    for (let i = 0; i < inSize; i++) {
      sum += input[i] * kernel[i * outSize + j];
    }
    output[j] = applyRelu && sum < 0 ? 0 : sum;
  }
  return output;
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < logits.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

function runInference(input28) {
  const t0 = performance.now();
  const c1 = conv2dRelu(input28, 28, 28, 1, modelWeights.conv1_kernel, modelWeights.conv1_bias, 3, 32);
  const p1 = maxPool2x2(c1.data, c1.h, c1.w, c1.c);
  const c2 = conv2dRelu(p1.data, p1.h, p1.w, p1.c, modelWeights.conv2_kernel, modelWeights.conv2_bias, 3, 64);
  const p2 = maxPool2x2(c2.data, c2.h, c2.w, c2.c);
  const d1 = dense(p2.data, modelWeights.dense1_kernel, modelWeights.dense1_bias, 1600, 128, true);
  const logits = dense(d1, modelWeights.dense2_kernel, modelWeights.dense2_bias, 128, 10, false);
  const probs = softmax(logits);
  const elapsed = performance.now() - t0;
  return { probs, elapsed };
}

// MNIST-style preprocessing for drawn canvas: crop bounding box, scale to 20x20, center in 28x28
function preprocessDrawnCanvas() {
  const w = drawCanvas.width;
  const h = drawCanvas.height;
  const imgData = drawCtx.getImageData(0, 0, w, h).data;

  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = imgData[(y * w + x) * 4];
      if (v > 12) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX || maxY < minY) return null;

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const targetW = Math.max(1, Math.round(bw * scale));
  const targetH = Math.max(1, Math.round(bh * scale));

  // Draw cropped & scaled digit onto 28x28 preview canvas
  previewCtx.fillStyle = "#000000";
  previewCtx.fillRect(0, 0, 28, 28);

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = targetW;
  tmpCanvas.height = targetH;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.drawImage(drawCanvas, minX, minY, bw, bh, 0, 0, targetW, targetH);

  // Compute center of mass of the scaled 20x20 patch
  const scaled = tmpCtx.getImageData(0, 0, targetW, targetH).data;
  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const v = scaled[(y * targetW + x) * 4];
      mass += v;
      cx += x * v;
      cy += y * v;
    }
  }
  const comX = mass > 0 ? cx / mass : targetW / 2;
  const comY = mass > 0 ? cy / mass : targetH / 2;
  const offsetX = Math.round(14 - comX);
  const offsetY = Math.round(14 - comY);

  previewCtx.drawImage(tmpCanvas, offsetX, offsetY);
  const finalData = previewCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    input[i] = finalData[i * 4] / 255.0;
  }
  return input;
}

function updatePredictionDisplay(probs, elapsed) {
  let topDigit = 0;
  let topProb = -1;
  for (let d = 0; d < 10; d++) {
    if (probs[d] > topProb) {
      topProb = probs[d];
      topDigit = d;
    }
  }

  predictedDigitEl.textContent = String(topDigit);
  predictedConfEl.textContent = `${(topProb * 100).toFixed(2)}% confidence`;
  inferenceTimeEl.textContent = `${elapsed.toFixed(1)} ms`;

  for (let d = 0; d < 10; d++) {
    const pctVal = probs[d] * 100;
    const { row, fill, pct } = barRows[d];
    row.classList.toggle("is-top", d === topDigit);
    fill.style.width = `${pctVal.toFixed(1)}%`;
    pct.textContent = `${pctVal.toFixed(1)}%`;
  }
}

function predictFromCanvas() {
  if (!modelWeights) return;
  const input = preprocessDrawnCanvas();
  if (!input) {
    resetPredictionUI();
    return;
  }
  const { probs, elapsed } = runInference(input);
  updatePredictionDisplay(probs, elapsed);
}

function scheduleLivePredict() {
  clearTimeout(livePredictTimer);
  livePredictTimer = setTimeout(predictFromCanvas, 60);
}

// Drawing handlers (Pointer Events)
function getCanvasPoint(e) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * drawCanvas.width,
    y: ((e.clientY - rect.top) / rect.height) * drawCanvas.height,
  };
}

function drawStroke(from, to) {
  drawCtx.strokeStyle = "#ffffff";
  drawCtx.lineWidth = Number(brushSlider.value);
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.beginPath();
  drawCtx.moveTo(from.x, from.y);
  drawCtx.lineTo(to.x, to.y);
  drawCtx.stroke();
}

drawCanvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  drawCanvas.setPointerCapture(e.pointerId);
  isDrawing = true;
  hasInk = true;
  canvasHint.classList.add("is-hidden");
  const pt = getCanvasPoint(e);
  lastPoint = pt;
  drawStroke(pt, pt);
  scheduleLivePredict();
});

drawCanvas.addEventListener("pointermove", (e) => {
  if (!isDrawing) return;
  const pt = getCanvasPoint(e);
  drawStroke(lastPoint, pt);
  lastPoint = pt;
  scheduleLivePredict();
});

function stopDrawing() {
  if (!isDrawing) return;
  isDrawing = false;
  lastPoint = null;
  predictFromCanvas();
}

drawCanvas.addEventListener("pointerup", stopDrawing);
drawCanvas.addEventListener("pointercancel", stopDrawing);

brushSlider.addEventListener("input", () => {
  brushValue.textContent = `${brushSlider.value}px`;
});

clearBtn.addEventListener("click", () => {
  initCanvas();
  resetPredictionUI();
});

predictBtn.addEventListener("click", predictFromCanvas);

document.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "c" && !e.metaKey && !e.ctrlKey && !e.altKey) {
    initCanvas();
    resetPredictionUI();
  }
});

// Sample digit generator
document.querySelectorAll(".sample-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const digit = btn.dataset.digit;
    initCanvas();
    hasInk = true;
    canvasHint.classList.add("is-hidden");
    drawCtx.fillStyle = "#ffffff";
    drawCtx.font = "bold 200px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    drawCtx.textAlign = "center";
    drawCtx.textBaseline = "middle";
    drawCtx.fillText(digit, drawCanvas.width / 2, drawCanvas.height / 2 + 8);
    predictFromCanvas();
  });
});

// Mode tabs
tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode;
    tabButtons.forEach((b) => {
      const active = b === btn;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", String(active));
    });
    drawPanel.classList.toggle("hidden", mode !== "draw");
    uploadPanel.classList.toggle("hidden", mode !== "upload");
  });
});

// Image upload preprocessing (matches Cell 11 of mnist.ipynb)
function predictUploadedImage(img) {
  if (!modelWeights || !img) return;
  previewCtx.drawImage(img, 0, 0, 28, 28);
  const data = previewCtx.getImageData(0, 0, 28, 28).data;
  const grays = new Float32Array(28 * 28);

  let borderSum = 0;
  let borderCount = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const idx = (y * 28 + x) * 4;
      const g = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      grays[y * 28 + x] = g;
      if (x === 0 || x === 27 || y === 0 || y === 27) {
        borderSum += g;
        borderCount++;
      }
    }
  }

  const shouldInvert = autoInvertCheckbox.checked && borderSum / borderCount > 127;
  const input = new Float32Array(28 * 28);
  const outImg = previewCtx.createImageData(28, 28);
  for (let i = 0; i < 28 * 28; i++) {
    const val = shouldInvert ? 255 - grays[i] : grays[i];
    input[i] = val / 255.0;
    const byte = Math.max(0, Math.min(255, Math.round(val)));
    outImg.data[i * 4] = byte;
    outImg.data[i * 4 + 1] = byte;
    outImg.data[i * 4 + 2] = byte;
    outImg.data[i * 4 + 3] = 255;
  }
  previewCtx.putImageData(outImg, 0, 0);

  const { probs, elapsed } = runInference(input);
  updatePredictionDisplay(probs, elapsed);
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    lastUploadedImage = img;
    uploadPreviewImg.src = url;
    uploadPreviewWrap.classList.remove("hidden");
    predictUploadedImage(img);
  };
  img.src = url;
}

imageUpload.addEventListener("change", (e) => handleFile(e.target.files[0]));
autoInvertCheckbox.addEventListener("change", () => {
  if (lastUploadedImage) predictUploadedImage(lastUploadedImage);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

initCanvas();
resetPredictionUI();
loadModel();
