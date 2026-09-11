/**
 * Interactive controller for the Dedicated Video Forensics Report Page
 */
document.addEventListener('DOMContentLoaded', () => {
  if (typeof VIDEO_DATA === 'undefined') return;

  const frames = VIDEO_DATA.frames || [];
  let currentModalFrameNum = 1;
  let currentModalTab = 'blueprint';

  // Render the SVG timeline chart
  renderTimelineChart(VIDEO_DATA);

  // Sorting & Filtering logic
  const sortSelect = document.querySelector('#frame-sort');
  const filterBtns = document.querySelectorAll('.btn-filter');
  const framesGrid = document.querySelector('#frames-grid');

  if (sortSelect) {
    sortSelect.addEventListener('change', () => {
      applyFilterAndSort();
    });
  }

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyFilterAndSort();
    });
  });

  function applyFilterAndSort() {
    const activeFilterBtn = document.querySelector('.btn-filter.active');
    const filterType = activeFilterBtn ? activeFilterBtn.dataset.filter : 'all';
    const sortVal = sortSelect ? sortSelect.value : 'chrono';

    const cards = Array.from(document.querySelectorAll('.frame-card'));

    // Filter
    cards.forEach(card => {
      const suspicion = parseFloat(card.dataset.suspicion) || 0;
      if (filterType === 'high' && suspicion < 50) {
        card.style.display = 'none';
      } else {
        card.style.display = '';
      }
    });

    // Sort
    const visibleCards = cards.slice();
    visibleCards.sort((a, b) => {
      const numA = parseInt(a.dataset.number, 10);
      const numB = parseInt(b.dataset.number, 10);
      const suspA = parseFloat(a.dataset.suspicion) || 0;
      const suspB = parseFloat(b.dataset.suspicion) || 0;
      const noiseA = parseFloat(a.dataset.noise) || 0;
      const noiseB = parseFloat(b.dataset.noise) || 0;

      if (sortVal === 'suspicion-desc') return suspB - suspA;
      if (sortVal === 'noise-desc') return noiseB - noiseA;
      return numA - numB;
    });

    visibleCards.forEach(card => framesGrid.appendChild(card));
  }

  // Modal controller functions
  window.openFrameModal = function(frameNum) {
    const frame = frames.find(f => f.number === frameNum);
    if (!frame) return;

    currentModalFrameNum = frameNum;
    updateModalView(frame, currentModalTab);

    const modal = document.querySelector('#inspector-modal');
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
  };

  window.closeFrameModal = function() {
    const modal = document.querySelector('#inspector-modal');
    modal.hidden = true;
    document.body.style.overflow = '';
  };

  function updateModalView(frame, tab) {
    document.querySelector('#modal-frame-title').textContent = `INSPECTING FRAME #${String(frame.number).padStart(2, '0')}`;
    document.querySelector('#modal-frame-time').textContent = `⏱️ ${frame.timestamp.toFixed(2)}s`;

    // Forensic Telemetry Signals grid for inspected frame
    const modalFooter = document.querySelector('.modal-metrics-footer');
    if (modalFooter) {
      const signalsData = [
        ['Noise Blueprint Residual Variance', frame.noise_score, frame.noise_score > 45 ? 'SYNTHETIC STIPPLING' : 'NATURAL GRAIN'],
        ['Colour Balance & HSV Residuals', frame.color_score, frame.color_score > 35 ? 'BOUNDARY SHIFT' : 'BALANCED OPTICS'],
        ['Grayscale Luminance Discrepancy', frame.grayscale_score, frame.grayscale_score > 40 ? 'EDGE ANOMALY' : 'UNIFORM LIGHTING'],
        ['Frame Composite Suspicion Index', frame.suspicion_score, frame.suspicion_score >= 50 ? 'SUSPICIOUS FRAME' : 'AUTHENTIC FRAME']
      ];
      
      modalFooter.innerHTML = `
        <div class="signals-hud-grid" style="width: 100%; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
          ${signalsData.map(([name, val, tag]) => {
            const realImp = (100 - val).toFixed(1);
            const fakeImp = val.toFixed(1);
            const isFake = tag.includes('SYNTHETIC') || tag.includes('SHIFT') || tag.includes('ANOMALY') || tag.includes('SUSPICIOUS');
            return `
              <div class="signal-item" style="padding: 10px 12px; background: rgba(2, 6, 23, 0.8);">
                <div class="signal-label-row">
                  <span class="signal-name" style="font-size:0.8rem;">${name}</span>
                  <span class="signal-tag ${isFake ? 'tag-fake-pill' : 'tag-real-pill'}" style="font-size:0.62rem;">${tag}</span>
                </div>
                <div class="bar-track" style="height:6px;">
                  <div class="bar-fill ${val > 50 ? 'bar-fake' : 'bar-real'}" style="width:${Math.min(val, 100)}%"></div>
                </div>
                <div class="signal-impact-row" style="font-size:0.72rem;">
                  <span class="impact-real">Real Impact: ${realImp}%</span>
                  <span class="impact-fake">Fake Impact: ${fakeImp}%</span>
                  <b class="signal-val">${val.toFixed(1)}%</b>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      `;
    }

    const singleView = document.querySelector('#modal-single-view');
    const compView = document.querySelector('#modal-compare-view');
    const activeImg = document.querySelector('#modal-active-img');
    const caption = document.querySelector('#modal-view-caption');

    if (tab === 'side-by-side') {
      singleView.hidden = true;
      compView.hidden = false;
      document.querySelector('#modal-comp-original').src = frame.original_image;
      document.querySelector('#modal-comp-blueprint').src = frame.blueprint_image;
      document.querySelector('#modal-comp-heatmap').src = frame.heatmap_image;
    } else {
      compView.hidden = true;
      singleView.hidden = false;

      if (tab === 'blueprint') {
        activeImg.src = frame.blueprint_image;
        caption.textContent = 'NOISE BLUEPRINT: High-frequency image noise rendered as granular sand stippling on an architectural navy blueprint canvas, exposing generative seams and synthetic grain anomalies.';
      } else if (tab === 'original') {
        activeImg.src = frame.original_image;
        caption.textContent = 'ORIGINAL: Unmodified frame extracted uniformly from the video stream.';
      } else if (tab === 'heatmap') {
        activeImg.src = frame.heatmap_image;
        caption.textContent = 'HEATMAP: Thermal Jet colormap highlighting suspicious regions (Blue: low, Yellow: elevated, Red: highest relative anomaly).';
      }
    }

    // Update prev/next buttons
    const prevBtn = document.querySelector('#modal-prev-btn');
    const nextBtn = document.querySelector('#modal-next-btn');
    prevBtn.disabled = frame.number <= 1;
    nextBtn.disabled = frame.number >= frames.length;
  }

  // Modal tab clicks
  const modalTabs = document.querySelectorAll('.modal-tab');
  modalTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      modalTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentModalTab = tab.dataset.tab;
      const frame = frames.find(f => f.number === currentModalFrameNum);
      if (frame) updateModalView(frame, currentModalTab);
    });
  });

  // Modal prev/next navigation
  document.querySelector('#modal-prev-btn').addEventListener('click', () => {
    if (currentModalFrameNum > 1) {
      currentModalFrameNum--;
      const frame = frames.find(f => f.number === currentModalFrameNum);
      if (frame) updateModalView(frame, currentModalTab);
    }
  });

  document.querySelector('#modal-next-btn').addEventListener('click', () => {
    if (currentModalFrameNum < frames.length) {
      currentModalFrameNum++;
      const frame = frames.find(f => f.number === currentModalFrameNum);
      if (frame) updateModalView(frame, currentModalTab);
    }
  });

  // Keyboard navigation
  window.addEventListener('keydown', e => {
    const modal = document.querySelector('#inspector-modal');
    if (!modal || modal.hidden) return;
    if (e.key === 'Escape') closeFrameModal();
    if (e.key === 'ArrowLeft') document.querySelector('#modal-prev-btn').click();
    if (e.key === 'ArrowRight') document.querySelector('#modal-next-btn').click();
  });
});

/**
 * Render an interactive SVG chart of the video timeline
 */
function renderTimelineChart(data) {
  const container = document.querySelector('#timeline-chart');
  if (!container) return;

  const timeline = data.timeline || [];
  if (!timeline.length) {
    container.innerHTML = '<p class="caution">No timeline points available.</p>';
    return;
  }

  const w = container.clientWidth || 980;
  const h = 200;
  const pad = { top: 20, right: 30, bottom: 35, left: 45 };
  const innerW = w - pad.left - pad.right;
  const innerH = h - pad.top - pad.bottom;

  const n = timeline.length;
  const x = i => pad.left + (n <= 1 ? 0 : (i / (n - 1)) * innerW);
  const y = val => pad.top + innerH - (Math.min(100, Math.max(0, val)) / 100) * innerH;

  // Build SVG path strings
  let suspicionPath = '';
  let noisePath = '';
  let colorPath = '';

  timeline.forEach((pt, i) => {
    const px = x(i);
    const pySusp = y(pt.suspicion);
    const pyNoise = y(pt.noise);
    const pyColor = y(pt.color);

    if (i === 0) {
      suspicionPath += `M ${px} ${pySusp}`;
      noisePath += `M ${px} ${pyNoise}`;
      colorPath += `M ${px} ${pyColor}`;
    } else {
      suspicionPath += ` L ${px} ${pySusp}`;
      noisePath += ` L ${px} ${pyNoise}`;
      colorPath += ` L ${px} ${pyColor}`;
    }
  });

  // Y-axis gridlines
  let gridLines = '';
  [0, 25, 50, 75, 100].forEach(val => {
    const gy = y(val);
    gridLines += `
      <line x1="${pad.left}" y1="${gy}" x2="${w - pad.right}" y2="${gy}" stroke="#1c3445" stroke-dasharray="3,3" />
      <text x="${pad.left - 8}" y="${gy + 4}" fill="#7694a9" font-size="10" text-anchor="end">${val}%</text>
    `;
  });

  // Interactive points
  let pointsHtml = '';
  timeline.forEach((pt, i) => {
    const px = x(i);
    const pySusp = y(pt.suspicion);
    const isPeak = pt.frame === data.peak_frame;
    pointsHtml += `
      <circle cx="${px}" cy="${pySusp}" r="${isPeak ? 6 : 3.5}" fill="${isPeak ? '#ff4757' : '#00f0ff'}" stroke="#071018" stroke-width="1.5" class="chart-pt" data-frame="${pt.frame}" data-susp="${pt.suspicion}" data-ts="${pt.timestamp}">
        <title>Frame #${pt.frame} (${pt.timestamp}s): Suspicion ${pt.suspicion.toFixed(1)}%</title>
      </circle>
    `;
  });

  const svg = `
    <svg viewBox="0 0 ${w} ${h}" class="timeline-svg" style="width:100%;height:${h}px">
      ${gridLines}
      <path d="${colorPath}" fill="none" stroke="#ff5268" stroke-width="1.5" opacity="0.65" />
      <path d="${noisePath}" fill="none" stroke="#f4d35e" stroke-width="1.8" opacity="0.75" />
      <path d="${suspicionPath}" fill="none" stroke="#00f0ff" stroke-width="2.5" />
      ${pointsHtml}
      <text x="${pad.left}" y="${h - 10}" fill="#7694a9" font-size="10">Frame #01 (0.0s)</text>
      <text x="${w - pad.right}" y="${h - 10}" fill="#7694a9" font-size="10" text-anchor="end">Frame #${data.frame_count} (${data.duration_seconds}s)</text>
    </svg>
  `;

  container.innerHTML = svg;

  // Add click to jump to frame card
  container.querySelectorAll('.chart-pt').forEach(pt => {
    pt.addEventListener('click', () => {
      const frameNum = pt.dataset.frame;
      const targetCard = document.querySelector(`#frame-card-${frameNum}`);
      if (targetCard) {
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        targetCard.classList.add('highlight-pulse');
        setTimeout(() => targetCard.classList.remove('highlight-pulse'), 1800);
      }
    });
  });
}
