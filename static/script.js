const fileInput = document.querySelector('#file');
const dropzone = document.querySelector('#dropzone');
const preview = document.querySelector('#preview');
const button = document.querySelector('#analyze');
const status = document.querySelector('#status');

function choose(file) {
  if (!file) return;
  fileInput._file = file;
  preview.src = URL.createObjectURL(file);
  const wrapper = document.querySelector('#preview-wrapper');
  if (wrapper) wrapper.hidden = false;
  button.disabled = false;
  status.textContent = 'Target acquired: ' + file.name;
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
  status.textContent = 'Executing ONNX ResNet-LSTM feature extraction & Noise Blueprint decomposition…';
  const data = new FormData();
  data.append('file', file);
  try {
    const response = await fetch('/analyze', { method: 'POST', body: data });
    const r = await response.json();
    if (!response.ok) throw Error(r.error || 'Analysis failed');
    render(r);
    status.textContent = 'Forensic scan complete. Sequence verified.';
  } catch (error) {
    status.textContent = 'Scan error: ' + error.message;
  } finally {
    button.disabled = false;
  }
});

function render(r) {
  const resultsSec = document.querySelector('#results');
  resultsSec.hidden = false;
  const demo = r.demonstration_mode;
  
  const engineBadge = document.querySelector('#engine-badge');
  if (engineBadge) {
    engineBadge.textContent = 'Engine: ' + (r.engine || 'ONNX Runtime v1.29');
  }

  const verdictBanner = document.querySelector('#verdict-banner');
  const verdict = document.querySelector('#verdict');
  const verdictExpl = document.querySelector('#verdict-expl');
  const scanStatus = document.querySelector('#scan-status');

  verdictBanner.classList.remove('verdict-banner-fake', 'verdict-banner-real', 'verdict-banner-demo');

  if (demo) {
    verdict.textContent = 'DEMONSTRATION MODE';
    verdictBanner.classList.add('verdict-banner-demo');
    verdictExpl.textContent = 'Forensic heuristic signals calculated. Model weights pending.';
    if (scanStatus) scanStatus.textContent = '● FORENSIC HEURISTIC MODE';
  } else if (r.prediction === 'DEEPFAKE') {
    verdict.textContent = 'DEEPFAKE DETECTED';
    verdictBanner.classList.add('verdict-banner-fake');
    verdictExpl.textContent = 'High probability of facial synthesis or generative manipulation detected across neural feature and noise signatures.';
    if (scanStatus) scanStatus.textContent = '▲ CRITICAL: MANIPULATION FOUND';
  } else {
    verdict.textContent = 'REAL / AUTHENTIC MEDIA';
    verdictBanner.classList.add('verdict-banner-real');
    verdictExpl.textContent = 'Visual features, frequency distributions, and sensor noise consistency align with authentic camera capture.';
    if (scanStatus) scanStatus.textContent = '✓ VERIFIED AUTHENTIC';
  }

  // Update Real vs Fake percentages
  const realPctElem = document.querySelector('#real-pct');
  const fakePctElem = document.querySelector('#fake-pct');
  if (realPctElem) realPctElem.textContent = (r.real_percentage !== undefined ? r.real_percentage : (100.0 - r.fake_percentage)).toFixed(1) + '%';
  if (fakePctElem) fakePctElem.textContent = (r.fake_percentage !== undefined ? r.fake_percentage : r.suspicion_score).toFixed(1) + '%';

  // Update confidence text
  const confVal = r.confidence;
  document.querySelector('#confidence').textContent = confVal.toFixed(1) + '%';
  document.querySelector('#confidence-label').textContent = demo ? 'EVIDENCE INTENSITY' : 'AI CONFIDENCE';

  // Animate SVG circular gauge
  const gaugeFill = document.querySelector('#gauge-fill');
  if (gaugeFill) {
    const radius = 50;
    const circumference = 2 * Math.PI * radius; // ~314.16
    const offset = circumference - (confVal / 100) * circumference;
    gaugeFill.style.strokeDasharray = `${circumference}`;
    gaugeFill.style.strokeDashoffset = `${offset}`;
    
    if (r.prediction === 'DEEPFAKE') {
      gaugeFill.style.stroke = '#ff2a5f';
    } else if (r.prediction === 'REAL') {
      gaugeFill.style.stroke = '#00ff87';
    } else {
      gaugeFill.style.stroke = '#f4d35e';
    }
  }

  // Signals telemetry & Parameter breakdown for Real % vs Fake %
  const signals = [
    ['ResNet18 CNN Spatial Features', r.cnn_score, r.cnn_score > 50 ? 'DEEPFAKE INDICATOR' : 'REAL COMPATIBLE'],
    ['LSTM Sequence Memory Signal', r.lstm_score, r.lstm_score > 50 ? 'DEEPFAKE INDICATOR' : 'REAL COMPATIBLE'],
    ['Noise Blueprint Residual Variance', r.noise_score, r.noise_score > 45 ? 'SYNTHETIC STIPPLING' : 'NATURAL GRAIN'],
    ['Colour Balance & HSV Residuals', r.color_score, r.color_score > 35 ? 'BOUNDARY SHIFT' : 'BALANCED OPTICS'],
    ['Grayscale Luminance Discrepancy', r.grayscale_score, r.grayscale_score > 40 ? 'EDGE ANOMALY' : 'UNIFORM LIGHTING'],
    ['Composite Suspicion Index', r.suspicion_score, r.suspicion_score >= 50 ? 'HIGH DEEPFAKE RISK' : 'AUTHENTIC MEDIA']
  ];

  document.querySelector('#signals').innerHTML = signals.map(([name, value, statusTag]) => {
    const realImpact = (100 - value).toFixed(1);
    const fakeImpact = value.toFixed(1);
    const isFakeTag = statusTag.includes('DEEPFAKE') || statusTag.includes('SYNTHETIC') || statusTag.includes('SHIFT') || statusTag.includes('ANOMALY');
    return `
      <div class="signal-item">
        <div class="signal-label-row">
          <span class="signal-name">${name}</span>
          <span class="signal-tag ${isFakeTag ? 'tag-fake-pill' : 'tag-real-pill'}">${statusTag}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${value > 50 ? 'bar-fake' : 'bar-real'}" style="width:${Math.min(value, 100)}%"></div>
        </div>
        <div class="signal-impact-row">
          <span class="impact-real">Real Impact: ${realImpact}%</span>
          <span class="impact-fake">Fake Impact: ${fakeImpact}%</span>
          <b class="signal-val">${value.toFixed(1)}%</b>
        </div>
      </div>
    `;
  }).join('');

  // 3-Way Comparative Evidence Suite (No overlays)
  document.querySelector('#original').src = r.original_image;
  document.querySelector('#blueprint').src = r.blueprint_image;
  document.querySelector('#heatmap').src = r.heatmap_image;

  resultsSec.scrollIntoView({ behavior: 'smooth' });
}

