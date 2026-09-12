const fileInput = document.querySelector('#file');
const dropzone = document.querySelector('#dropzone');
const preview = document.querySelector('#preview');
const button = document.querySelector('#analyze');
const status = document.querySelector('#status');

// Image Comparison Slider elements
let currentImgSliderMode = 'blueprint'; // 'blueprint' or 'heatmap'
let cachedAnalysisResult = null;

const imgSliderRange = document.querySelector('#img-slider-range');
const imgSliderHandle = document.querySelector('#img-slider-handle');
const imgSliderBtn = document.querySelector('#img-slider-btn');
const imgSliderOrig = document.querySelector('#img-slider-orig');
const imgSliderTarget = document.querySelector('#img-slider-target');
const btnModeBlueprint = document.querySelector('#img-slider-mode-blueprint');
const btnModeHeatmap = document.querySelector('#img-slider-mode-heatmap');

function setImgSliderPosition(val) {
  if (imgSliderOrig) {
    imgSliderOrig.style.clipPath = `polygon(0 0, ${val}% 0, ${val}% 100%, 0 100%)`;
  }
  if (imgSliderHandle) imgSliderHandle.style.left = val + '%';
  if (imgSliderBtn) imgSliderBtn.style.left = val + '%';
}

function updateSliderImages() {
  if (!cachedAnalysisResult) return;
  if (imgSliderOrig) {
    // 100% UNTOUCHED RAW ORIGINAL IMAGE
    imgSliderOrig.src = cachedAnalysisResult.original_image;
  }
  if (imgSliderTarget) {
    imgSliderTarget.src = currentImgSliderMode === 'blueprint'
      ? cachedAnalysisResult.blueprint_image
      : cachedAnalysisResult.heatmap_image;
  }
  setImgSliderPosition(imgSliderRange ? imgSliderRange.value : 50);
}

if (imgSliderRange) {
  imgSliderRange.addEventListener('input', () => {
    setImgSliderPosition(imgSliderRange.value);
  });
}

if (btnModeBlueprint && btnModeHeatmap) {
  btnModeBlueprint.addEventListener('click', () => {
    currentImgSliderMode = 'blueprint';
    btnModeBlueprint.classList.add('active');
    btnModeHeatmap.classList.remove('active');
    updateSliderImages();
  });

  btnModeHeatmap.addEventListener('click', () => {
    currentImgSliderMode = 'heatmap';
    btnModeHeatmap.classList.add('active');
    btnModeBlueprint.classList.remove('active');
    updateSliderImages();
  });
}

function choose(file) {
  if (!file) return;
  fileInput._file = file;
  preview.src = URL.createObjectURL(file);
  const wrapper = document.querySelector('#preview-wrapper');
  if (wrapper) wrapper.hidden = false;
  button.disabled = false;
  status.textContent = 'Target acquired: ' + file.name + ' — Ready for deep forensic scan.';
}

fileInput.addEventListener('change', () => choose(fileInput.files[0]));

['dragover', 'drop'].forEach(event => {
  dropzone.addEventListener(event, e => {
    e.preventDefault();
    if (event === 'drop') choose(e.dataTransfer.files[0]);
  });
});

button.addEventListener('click', async () => {
  const file = fileInput._file || fileInput.files[0];
  if (!file) return;
  button.disabled = true;
  status.textContent = 'Executing ONNX feature extraction & Noise Blueprint decomposition…';
  const data = new FormData();
  data.append('file', file);
  try {
    const response = await fetch('/analyze', { method: 'POST', body: data });
    const r = await response.json();
    if (!response.ok) throw Error(r.error || 'Analysis failed');
    cachedAnalysisResult = r;
    render(r);
    updateSliderImages();
    status.textContent = 'Forensic scan complete. Evidence generated in order: 01 Original, 02 Noise Blueprint, 03 Heatmap.';
  } catch (error) {
    status.textContent = 'Scan error: ' + error.message;
  } finally {
    button.disabled = false;
  }
});

function render(r) {
  const resultsSec = document.querySelector('#results');
  resultsSec.hidden = false;
  
  const engineBadge = document.querySelector('#engine-badge');
  if (engineBadge) {
    engineBadge.textContent = 'Engine: ' + (r.engine || 'ONNX Runtime v1.29');
  }

  const verdictBanner = document.querySelector('#verdict-banner');
  const verdict = document.querySelector('#verdict');
  const verdictExpl = document.querySelector('#verdict-expl');
  const scanStatus = document.querySelector('#scan-status');

  verdictBanner.classList.remove('verdict-banner-fake', 'verdict-banner-real', 'verdict-banner-demo', 'verdict-banner-uncertain');

  // Exact requested wording:
  // - VERIFIED AUTHENTIC MEDIA — REAL
  // - VERIFIED AUTHENTIC MEDIA — FAKE
  // - AUTHENTICITY COULD NOT BE VERIFIED
  const verdictText = r.overall_verdict_text || (
    r.prediction === 'REAL' ? 'VERIFIED AUTHENTIC MEDIA — REAL' : 'DEEPFAKE DETECTED — FAKE'
  );

  verdict.textContent = verdictText;

  if (r.prediction === 'REAL') {
    verdictBanner.classList.add('verdict-banner-real');
    verdictExpl.textContent = 'Spatial facial landmarks, high-frequency PRNU noise distribution, and optical chromatic balance match authentic hardware sensor capture.';
    if (scanStatus) scanStatus.textContent = '✓ VERIFIED AUTHENTIC MEDIA';
  } else {
    // FAKE (or any non-REAL — no uncertain/yellow state)
    verdictBanner.classList.add('verdict-banner-fake');
    verdictExpl.textContent = 'Generative synthesis boundaries, residual noise anomalies, and neural feature irregularities confirm deepfake manipulation.';
    if (scanStatus) scanStatus.textContent = '▲ CRITICAL: DEEPFAKE CONFIRMED';
  }

  // Update Real vs Fake percentages
  const realPctElem = document.querySelector('#real-pct');
  const fakePctElem = document.querySelector('#fake-pct');
  if (realPctElem) realPctElem.textContent = (r.real_percentage !== undefined ? r.real_percentage : (100.0 - r.fake_percentage)).toFixed(1) + '%';
  if (fakePctElem) fakePctElem.textContent = (r.fake_percentage !== undefined ? r.fake_percentage : r.suspicion_score).toFixed(1) + '%';

  // Update confidence text
  const confVal = r.confidence;
  document.querySelector('#confidence').textContent = confVal.toFixed(1) + '%';
  document.querySelector('#confidence-label').textContent = 'AI CONFIDENCE';

  // Animate SVG circular gauge
  const gaugeFill = document.querySelector('#gauge-fill');
  if (gaugeFill) {
    const radius = 50;
    const circumference = 2 * Math.PI * radius; // ~314.16
    const offset = circumference - (confVal / 100) * circumference;
    gaugeFill.style.strokeDasharray = `${circumference}`;
    gaugeFill.style.strokeDashoffset = `${offset}`;
    
    if (r.prediction === 'FAKE') {
      gaugeFill.style.stroke = '#ff2a5f';
    } else if (r.prediction === 'REAL') {
      gaugeFill.style.stroke = '#00ff87';
    } else {
      gaugeFill.style.stroke = '#f4d35e';
    }
  }

  // Signals telemetry
  const signals = [
    ['ResNet18 CNN Spatial Features', r.cnn_score, r.cnn_score > 50 ? 'SYNTHESIS ARTIFACT' : 'NATURAL OPTICS'],
    ['LSTM Sequence Feature Signal', r.lstm_score, r.lstm_score > 50 ? 'SEQUENCE ANOMALY' : 'CONSISTENT OPTICS'],
    ['Noise Blueprint Variance', r.noise_score, r.noise_score > 45 ? 'SYNTHETIC STIPPLING' : 'NATURAL SENSOR GRAIN'],
    ['Colour Balance & HSV Residuals', r.color_score, r.color_score > 35 ? 'BOUNDARY SHIFT' : 'BALANCED OPTICS'],
    ['Grayscale Luminance Discrepancy', r.grayscale_score, r.grayscale_score > 40 ? 'EDGE ANOMALY' : 'UNIFORM LIGHTING'],
    ['Composite Suspicion Index', r.suspicion_score, r.suspicion_score >= 60 ? 'HIGH MANIPULATION RISK' : (r.suspicion_score >= 38 ? 'INTERMEDIATE ZONE' : 'AUTHENTIC MEDIA')]
  ];

  document.querySelector('#signals').innerHTML = signals.map(([name, value, statusTag]) => {
    const realImpact = (100 - value).toFixed(1);
    const fakeImpact = value.toFixed(1);
    const isFakeTag = statusTag.includes('MANIPULATION') || statusTag.includes('SYNTHESIS') || statusTag.includes('SYNTHETIC') || statusTag.includes('SHIFT') || statusTag.includes('ANOMALY');
    const isUncertainTag = statusTag.includes('INTERMEDIATE');
    const pillClass = isFakeTag ? 'tag-fake-pill' : (isUncertainTag ? 'tag-uncertain-pill' : 'tag-real-pill');
    return `
      <div class="signal-item">
        <div class="signal-label-row">
          <span class="signal-name">${name}</span>
          <span class="signal-tag ${pillClass}">${statusTag}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${value > 50 ? 'bar-fake' : (value > 38 ? 'bar-uncertain' : 'bar-real')}" style="width:${Math.min(value, 100)}%"></div>
        </div>
        <div class="signal-impact-row">
          <span class="impact-real">Real Impact: ${realImpact}%</span>
          <span class="impact-fake">Fake Impact: ${fakeImpact}%</span>
          <b class="signal-val">${value.toFixed(1)}%</b>
        </div>
      </div>
    `;
  }).join('');

  // 3-Way Comparative Evidence Suite in strict order: 01 Original -> 02 Noise Blueprint -> 03 Heatmap
  document.querySelector('#original').src = r.original_image;
  document.querySelector('#blueprint').src = r.blueprint_image;
  document.querySelector('#heatmap').src = r.heatmap_image;

  resultsSec.scrollIntoView({ behavior: 'smooth' });
}
