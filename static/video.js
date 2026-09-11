const fileInput = document.querySelector('#file');
const dropzone = document.querySelector('#dropzone');
const preview = document.querySelector('#preview');
const button = document.querySelector('#analyze');
const status = document.querySelector('#status');
const progressContainer = document.querySelector('#progress-container');

function choose(file) {
  if (!file) return;
  fileInput._file = file;
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  button.disabled = false;
  if (status) status.textContent = 'Selected: ' + file.name + ' (Ready to extract up to 30 frames at max 30 FPS)';
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
  if (progressContainer) progressContainer.hidden = false;
  status.textContent = 'Extracting separate frames (max 30 FPS) & running ONNX sequence forensics…';

  const data = new FormData();
  data.append('file', file);

  try {
    const response = await fetch('/analyze-video', { method: 'POST', body: data });
    const r = await response.json();
    if (!response.ok) throw Error(r.error || 'Video analysis failed');
    
    status.textContent = 'Analysis complete! Redirecting to dedicated Video Forensics Dashboard…';
    
    // Redirect to the dedicated video forensics page
    if (r.redirect_url) {
      setTimeout(() => {
        window.location.href = r.redirect_url;
      }, 600);
    } else {
      window.location.href = '/video-results/' + r.job_id;
    }
  } catch (error) {
    status.textContent = 'Error: ' + error.message;
    button.disabled = false;
  }
});

