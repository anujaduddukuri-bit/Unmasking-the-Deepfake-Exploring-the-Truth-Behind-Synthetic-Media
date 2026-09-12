const fileInput = document.querySelector('#file');
const dropzone = document.querySelector('#dropzone');
const preview = document.querySelector('#preview');
const previewWrapper = document.querySelector('#preview-wrapper');
const button = document.querySelector('#analyze');
const status = document.querySelector('#status');
const progressContainer = document.querySelector('#progress-container');

// Step elements
const step1 = document.querySelector('#step-1');
const step2 = document.querySelector('#step-2');
const step3 = document.querySelector('#step-3');
const step4 = document.querySelector('#step-4');
const step5 = document.querySelector('#step-5');

function setStep(stepEl, state) {
  if (!stepEl) return;
  stepEl.classList.remove('active', 'done');
  if (state === 'active') {
    stepEl.classList.add('active');
    const icon = stepEl.querySelector('.step-icon');
    if (icon) icon.textContent = '⏳';
  } else if (state === 'done') {
    stepEl.classList.add('done');
    const icon = stepEl.querySelector('.step-icon');
    if (icon) icon.textContent = '✅';
  }
}

function choose(file) {
  if (!file) return;
  fileInput._file = file;

  // Visual feedback on dropzone
  const dropTitle = document.querySelector('.drop-title');
  const dropSub = document.querySelector('.drop-sub');
  if (dropTitle) dropTitle.textContent = `📁 ${file.name}`;
  if (dropSub) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    dropSub.textContent = `${sizeMb} MB · Video ready — click button below to analyze`;
  }
  if (dropzone) {
    dropzone.style.borderColor = 'var(--cyan, #00f0ff)';
    dropzone.style.background = 'rgba(0, 240, 255, 0.08)';
  }

  // Show uploaded video preview if element exists
  const previewEl = document.querySelector('#preview');
  const previewWrap = document.querySelector('#preview-wrapper');
  if (previewEl) {
    try {
      const videoUrl = URL.createObjectURL(file);
      previewEl.src = videoUrl;
      if (previewWrap) {
        previewWrap.hidden = false;
        previewWrap.style.display = 'block';
      }
      previewEl.style.display = 'block';
      previewEl.load();
    } catch (e) {
      console.warn('Video preview load warning:', e);
    }
  }

  // Guaranteed button enable
  if (button) {
    button.disabled = false;
    button.removeAttribute('disabled');
  }

  if (status) {
    status.textContent = `Selected: ${file.name} — Ready for forensic sequence scan.`;
  }

  if (step1) {
    setStep(step1, 'done');
  }
}

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    choose(fileInput.files[0]);
  }
});

['dragenter', 'dragover'].forEach(event => {
  dropzone.addEventListener(event, e => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--cyan, #00f0ff)';
    dropzone.style.background = 'rgba(0, 240, 255, 0.12)';
  });
});

['dragleave', 'dragend'].forEach(event => {
  dropzone.addEventListener(event, e => {
    e.preventDefault();
    if (!fileInput._file) {
      dropzone.style.borderColor = '';
      dropzone.style.background = '';
    }
  });
});

dropzone.addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
    choose(e.dataTransfer.files[0]);
  }
});

button.addEventListener('click', async () => {
  const file = fileInput._file || fileInput.files[0];
  if (!file) return;

  button.disabled = true;
  if (progressContainer) {
    progressContainer.hidden = false;
    progressContainer.style.display = 'block';
  }
  
  // Step 1: Video uploaded
  setStep(step1, 'done');

  // Step 2: Extracting frames (active)
  setStep(step2, 'active');
  if (status) status.textContent = 'Step 1/4: Seeking & extracting keyframes uniformly from video stream…';

  const t1 = setTimeout(() => {
    setStep(step2, 'done');
    setStep(step3, 'active');
    if (status) status.textContent = 'Step 2/4: Running ONNX neural feature extraction & temporal consistency check…';
  }, 2200);

  const t2 = setTimeout(() => {
    setStep(step3, 'done');
    setStep(step4, 'active');
    if (status) status.textContent = 'Step 3/4: Generating Noise Blueprints & Thermal Heatmaps across all frames…';
  }, 5500);

  const data = new FormData();
  data.append('file', file);

  try {
    const response = await fetch('/analyze-video', { method: 'POST', body: data });
    clearTimeout(t1);
    clearTimeout(t2);

    const r = await response.json();
    if (!response.ok) throw Error(r.error || 'Video analysis failed');

    try {
      sessionStorage.setItem('video_report_' + r.job_id, JSON.stringify(r));
      sessionStorage.setItem('last_video_report', JSON.stringify(r));
    } catch (e) {
      console.warn('Could not cache report in sessionStorage:', e);
    }

    // Mark steps complete
    setStep(step2, 'done');
    setStep(step3, 'done');
    setStep(step4, 'done');
    setStep(step5, 'done');

    if (status) status.textContent = 'Analysis complete! Redirecting to Forensic Video Dossier…';

    const targetUrl = r.redirect_url || ('/video-results/' + r.job_id);
    setTimeout(() => {
      window.location.href = targetUrl;
    }, 700);
  } catch (error) {
    clearTimeout(t1);
    clearTimeout(t2);
    if (status) status.textContent = 'Analysis Error: ' + error.message;
    button.disabled = false;
  }
});
