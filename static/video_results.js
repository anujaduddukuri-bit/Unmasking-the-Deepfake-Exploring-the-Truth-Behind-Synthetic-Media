/**
 * Interactive controller for the Dedicated Video Forensics Report Page
 * Supports interactive Before/After comparison slider, frame scrubbing, timeline SVG, and filtering.
 */
document.addEventListener('DOMContentLoaded', () => {
  if (typeof VIDEO_DATA === 'undefined') return;

  const frames = VIDEO_DATA.frames || [];
  let currentModalFrameNum = 1;
  let currentModalTab = 'original';
  let modalSliderMode = 'blueprint'; // 'blueprint' or 'heatmap'

  let activeFeaturedFrameNum = 1;
  let featuredSliderMode = 'blueprint'; // 'blueprint' or 'heatmap'
  let featuredViewMode = 'slider'; // 'slider' or 'side'

  // Render SVG timeline chart
  renderTimelineChart(VIDEO_DATA);

  // --------------------------------------------------------------------------
  // Helper: Clip-path forensic comparison wiper slider
  // --------------------------------------------------------------------------
  function setClipSlider(val, origImg, handleBar, handleBtn) {
    const clamped = Math.max(0, Math.min(100, val));
    if (origImg) {
      origImg.style.clipPath = `polygon(0 0, ${clamped}% 0, ${clamped}% 100%, 0 100%)`;
    }
    if (handleBar) handleBar.style.left = `${clamped}%`;
    if (handleBtn) handleBtn.style.left = `${clamped}%`;
  }

  // --------------------------------------------------------------------------
  // FEATURED VIDEO FRAME FORENSICS CONSOLE
  // --------------------------------------------------------------------------
  const featuredFrameScrubber = document.querySelector('#featured-frame-scrubber');
  const scrubberPrevBtn = document.querySelector('#scrubber-prev-btn');
  const scrubberNextBtn = document.querySelector('#scrubber-next-btn');
  const scrubberFrameNum = document.querySelector('#scrubber-frame-num');
  const scrubberFrameTime = document.querySelector('#scrubber-frame-time');

  const featuredSliderRange = document.querySelector('#featured-slider-range');
  const featuredSliderOrig = document.querySelector('#featured-slider-orig');
  const featuredSliderTarget = document.querySelector('#featured-slider-target');
  const featuredSliderHandle = document.querySelector('#featured-slider-handle');
  const featuredSliderBtn = document.querySelector('#featured-slider-btn');

  const featSliderView = document.querySelector('#feat-slider-view');
  const featSideView = document.querySelector('#feat-side-view');

  const featModeBlueprint = document.querySelector('#feat-mode-blueprint');
  const featModeHeatmap = document.querySelector('#feat-mode-heatmap');
  const featModeSide = document.querySelector('#feat-mode-side');

  function selectFeaturedFrame(frameNum) {
    const num = parseInt(frameNum, 10);
    const frame = frames.find(f => f.number === num);
    if (!frame) return;

    activeFeaturedFrameNum = num;

    // Update Headings and Badges
    const headEl = document.querySelector('#active-frame-heading');
    const subEl = document.querySelector('#active-frame-sub');
    const badgeEl = document.querySelector('#active-frame-verdict-badge');

    if (headEl) headEl.textContent = `FRAME #${String(frame.number).padStart(2, '0')} FORENSIC INSPECTION`;
    if (subEl) subEl.textContent = `Timestamp: ${frame.timestamp.toFixed(2)}s · Slide horizontally to inspect untouched raw original vs forensic anomaly overlays`;
    if (badgeEl) {
      badgeEl.textContent = `${frame.frame_verdict} (${frame.suspicion_score.toFixed(1)}%)`;
      badgeEl.className = `frame-badge ${frame.frame_verdict === 'FAKE' ? 'badge-high' : (frame.frame_verdict === 'REAL' ? 'badge-low' : 'badge-med')}`;
    }

    // Update Scrubber state
    if (scrubberFrameNum) scrubberFrameNum.textContent = `Frame ${String(frame.number).padStart(2, '0')} / ${String(frames.length).padStart(2, '0')}`;
    if (scrubberFrameTime) scrubberFrameTime.textContent = `(${frame.timestamp.toFixed(2)}s)`;
    if (featuredFrameScrubber && parseInt(featuredFrameScrubber.value, 10) !== num) {
      featuredFrameScrubber.value = num;
    }
    if (scrubberPrevBtn) scrubberPrevBtn.disabled = num <= 1;
    if (scrubberNextBtn) scrubberNextBtn.disabled = num >= frames.length;

    // Update Wiper Slider Images (100% untouched raw original frame on left)
    if (featuredSliderOrig) {
      featuredSliderOrig.src = frame.original_image;
    }
    if (featuredSliderTarget) {
      featuredSliderTarget.src = featuredSliderMode === 'blueprint' ? frame.blueprint_image : frame.heatmap_image;
    }
    setClipSlider(featuredSliderRange ? featuredSliderRange.value : 50, featuredSliderOrig, featuredSliderHandle, featuredSliderBtn);

    // Update 3-Way Side-by-Side Images
    const sideOrig = document.querySelector('#feat-side-orig');
    const sideBlue = document.querySelector('#feat-side-blue');
    const sideHeat = document.querySelector('#feat-side-heat');
    if (sideOrig) sideOrig.src = frame.original_image;
    if (sideBlue) sideBlue.src = frame.blueprint_image;
    if (sideHeat) sideHeat.src = frame.heatmap_image;

    // Update Active Frame Forensic Telemetry HUD
    updateTelemetryItem('noise', frame.noise_score, frame.noise_score > 45 ? 'SYNTHETIC STIPPLING' : 'NATURAL SENSOR GRAIN', frame.noise_score > 45);
    updateTelemetryItem('color', frame.color_score, frame.color_score > 35 ? 'BOUNDARY SHIFT' : 'BALANCED OPTICS', frame.color_score > 35);
    updateTelemetryItem('gray', frame.grayscale_score, frame.grayscale_score > 40 ? 'EDGE ANOMALY' : 'UNIFORM LIGHTING', frame.grayscale_score > 40);
    updateTelemetryItem('susp', frame.suspicion_score, frame.frame_verdict, frame.suspicion_score >= 50);

    // Highlight card in grid if visible
    document.querySelectorAll('.frame-card').forEach(c => c.classList.remove('active-inspected-card'));
    const activeCard = document.querySelector(`#frame-card-${num}`);
    if (activeCard) activeCard.classList.add('active-inspected-card');
  }

  function updateTelemetryItem(prefix, val, tagText, isFake) {
    const tag = document.querySelector(`#feat-tele-${prefix}-tag`);
    const bar = document.querySelector(`#feat-tele-${prefix}-bar`);
    const realText = document.querySelector(`#feat-tele-${prefix}-real`);
    const fakeText = document.querySelector(`#feat-tele-${prefix}-fake`);
    const valText = document.querySelector(`#feat-tele-${prefix}-val`);

    if (tag) {
      tag.textContent = tagText;
      tag.className = `signal-tag ${isFake ? 'tag-fake-pill' : 'tag-real-pill'}`;
    }
    if (bar) {
      bar.style.width = `${Math.min(val, 100)}%`;
      bar.className = `bar-fill ${isFake ? 'bar-fake' : 'bar-real'}`;
    }
    if (realText) realText.textContent = `Real: ${(100 - val).toFixed(1)}%`;
    if (fakeText) fakeText.textContent = `Fake: ${val.toFixed(1)}%`;
    if (valText) valText.textContent = `${val.toFixed(1)}%`;
  }

  // Scrubber Events
  if (featuredFrameScrubber) {
    featuredFrameScrubber.addEventListener('input', () => {
      selectFeaturedFrame(featuredFrameScrubber.value);
    });
  }
  if (scrubberPrevBtn) {
    scrubberPrevBtn.addEventListener('click', () => {
      if (activeFeaturedFrameNum > 1) selectFeaturedFrame(activeFeaturedFrameNum - 1);
    });
  }
  if (scrubberNextBtn) {
    scrubberNextBtn.addEventListener('click', () => {
      if (activeFeaturedFrameNum < frames.length) selectFeaturedFrame(activeFeaturedFrameNum + 1);
    });
  }

  // Featured Wiper Slider Drag Event
  if (featuredSliderRange) {
    featuredSliderRange.addEventListener('input', () => {
      setClipSlider(featuredSliderRange.value, featuredSliderOrig, featuredSliderHandle, featuredSliderBtn);
    });
  }

  // Featured Mode Toggle Buttons
  if (featModeBlueprint && featModeHeatmap && featModeSide) {
    featModeBlueprint.addEventListener('click', () => {
      featuredSliderMode = 'blueprint';
      featuredViewMode = 'slider';
      featModeBlueprint.classList.add('active');
      featModeHeatmap.classList.remove('active');
      featModeSide.classList.remove('active');
      if (featSliderView) { featSliderView.hidden = false; featSliderView.style.display = 'block'; }
      if (featSideView) { featSideView.hidden = true; featSideView.style.display = 'none'; }
      selectFeaturedFrame(activeFeaturedFrameNum);
    });

    featModeHeatmap.addEventListener('click', () => {
      featuredSliderMode = 'heatmap';
      featuredViewMode = 'slider';
      featModeHeatmap.classList.add('active');
      featModeBlueprint.classList.remove('active');
      featModeSide.classList.remove('active');
      if (featSliderView) { featSliderView.hidden = false; featSliderView.style.display = 'block'; }
      if (featSideView) { featSideView.hidden = true; featSideView.style.display = 'none'; }
      selectFeaturedFrame(activeFeaturedFrameNum);
    });

    featModeSide.addEventListener('click', () => {
      featuredViewMode = 'side';
      featModeSide.classList.add('active');
      featModeBlueprint.classList.remove('active');
      featModeHeatmap.classList.remove('active');
      if (featSliderView) { featSliderView.hidden = true; featSliderView.style.display = 'none'; }
      if (featSideView) { featSideView.hidden = false; featSideView.style.display = 'grid'; }
      selectFeaturedFrame(activeFeaturedFrameNum);
    });
  }

  // Expose selectFeaturedFrame globally
  window.selectFeaturedFrame = selectFeaturedFrame;

  // Initialize Featured Console on Frame 1 immediately
  if (frames.length > 0) {
    selectFeaturedFrame(1);
  }

  // --------------------------------------------------------------------------
  // Sorting & Filtering logic for frame cards
  // --------------------------------------------------------------------------
  const sortSelect = document.querySelector('#frame-sort');
  const filterBtns = document.querySelectorAll('.filter-group .btn-filter');
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
    const activeFilterBtn = document.querySelector('.filter-group .btn-filter.active');
    const filterType = activeFilterBtn ? activeFilterBtn.dataset.filter : 'all';
    const sortVal = sortSelect ? sortSelect.value : 'chrono';

    const cards = Array.from(document.querySelectorAll('.frame-card'));

    cards.forEach(card => {
      const suspicion = parseFloat(card.dataset.suspicion) || 0;
      if (filterType === 'high' && suspicion < 40) {
        card.style.display = 'none';
      } else {
        card.style.display = '';
      }
    });

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

  // --------------------------------------------------------------------------
  // Modal Slider controls setup
  // --------------------------------------------------------------------------
  const modalSliderRange = document.querySelector('#slider-range');
  const modalSliderOrig = document.querySelector('#slider-orig-img');
  const modalSliderTarget = document.querySelector('#slider-target-img');
  const modalSliderHandle = document.querySelector('#slider-handle');
  const modalSliderHandleBtn = document.querySelector('#slider-handle-btn');
  const modalSliderBtnBlueprint = document.querySelector('#slider-mode-blueprint');
  const modalSliderBtnHeatmap = document.querySelector('#slider-mode-heatmap');

  if (modalSliderRange) {
    modalSliderRange.addEventListener('input', () => {
      setClipSlider(modalSliderRange.value, modalSliderOrig, modalSliderHandle, modalSliderHandleBtn);
    });
  }

  if (modalSliderBtnBlueprint && modalSliderBtnHeatmap) {
    modalSliderBtnBlueprint.addEventListener('click', () => {
      modalSliderMode = 'blueprint';
      modalSliderBtnBlueprint.classList.add('active');
      modalSliderBtnHeatmap.classList.remove('active');
      const frame = frames.find(f => f.number === currentModalFrameNum);
      if (frame) updateModalView(frame, currentModalTab);
    });

    modalSliderBtnHeatmap.addEventListener('click', () => {
      modalSliderMode = 'heatmap';
      modalSliderBtnHeatmap.classList.add('active');
      modalSliderBtnBlueprint.classList.remove('active');
      const frame = frames.find(f => f.number === currentModalFrameNum);
      if (frame) updateModalView(frame, currentModalTab);
    });
  }

  // --------------------------------------------------------------------------
  // Modal controller functions
  // --------------------------------------------------------------------------
  window.openFrameModal = function(frameNum) {
    const num = parseInt(frameNum, 10);
    const frame = frames.find(f => f.number === num);
    if (!frame) return;

    currentModalFrameNum = num;
    currentModalTab = 'original';

    const modalTabs = document.querySelectorAll('.modal-tab');
    modalTabs.forEach(t => {
      if (t.dataset.tab === 'original') {
        t.classList.add('active');
      } else {
        t.classList.remove('active');
      }
    });

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
        ['Noise Blueprint Variance', frame.noise_score, frame.noise_score > 45 ? 'SYNTHETIC STIPPLING' : 'NATURAL SENSOR GRAIN'],
        ['Colour Balance & HSV', frame.color_score, frame.color_score > 35 ? 'BOUNDARY SHIFT' : 'BALANCED OPTICS'],
        ['Grayscale Luminance Discrepancy', frame.grayscale_score, frame.grayscale_score > 40 ? 'EDGE ANOMALY' : 'UNIFORM LIGHTING'],
        ['Frame Verdict', frame.suspicion_score, frame.frame_verdict]
      ];

      modalFooter.innerHTML = `
        <div class="signals-hud-grid" style="width: 100%; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
          ${signalsData.map(([name, val, tag]) => {
            const realImp = (100 - val).toFixed(1);
            const fakeImp = val.toFixed(1);
            const isFake = tag === 'FAKE' || tag.includes('SYNTHETIC') || tag.includes('SHIFT') || tag.includes('ANOMALY');
            const isUncertain = tag === 'UNCERTAIN';
            const pillClass = isFake ? 'tag-fake-pill' : (isUncertain ? 'tag-uncertain-pill' : 'tag-real-pill');
            return `
              <div class="signal-item" style="padding: 10px 12px; background: rgba(2, 6, 23, 0.8);">
                <div class="signal-label-row">
                  <span class="signal-name" style="font-size:0.8rem;">${name}</span>
                  <span class="signal-tag ${pillClass}" style="font-size:0.62rem;">${tag}</span>
                </div>
                <div class="bar-track" style="height:6px;">
                  <div class="bar-fill ${val > 50 ? 'bar-fake' : (val > 38 ? 'bar-uncertain' : 'bar-real')}" style="width:${Math.min(val, 100)}%"></div>
                </div>
                <div class="signal-impact-row" style="font-size:0.72rem;">
                  <span class="impact-real">Real: ${realImp}%</span>
                  <span class="impact-fake">Fake: ${fakeImp}%</span>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      `;
    }

    const singleView = document.querySelector('#modal-single-view');
    const compView = document.querySelector('#modal-compare-view');
    const sliderView = document.querySelector('#modal-slider-view');
    const activeImg = document.querySelector('#modal-active-img');
    const caption = document.querySelector('#modal-view-caption');

    // Pre-populate comparison and slider images so they never show broken image state
    const compOrig = document.querySelector('#modal-comp-original');
    const compBlue = document.querySelector('#modal-comp-blueprint');
    const compHeat = document.querySelector('#modal-comp-heatmap');
    if (compOrig && frame.original_image) compOrig.src = frame.original_image;
    if (compBlue && frame.blueprint_image) compBlue.src = frame.blueprint_image;
    if (compHeat && frame.heatmap_image) compHeat.src = frame.heatmap_image;

    if (tab === 'slider') {
      singleView.hidden = true;
      singleView.style.display = 'none';
      compView.hidden = true;
      compView.style.display = 'none';
      sliderView.hidden = false;
      sliderView.style.display = 'block';

      if (modalSliderOrig) modalSliderOrig.src = frame.original_image;
      if (modalSliderTarget) modalSliderTarget.src = modalSliderMode === 'blueprint' ? frame.blueprint_image : frame.heatmap_image;
      setClipSlider(modalSliderRange ? modalSliderRange.value : 50, modalSliderOrig, modalSliderHandle, modalSliderHandleBtn);
    } else if (tab === 'side-by-side') {
      sliderView.hidden = true;
      sliderView.style.display = 'none';
      singleView.hidden = true;
      singleView.style.display = 'none';
      compView.hidden = false;
      compView.style.display = 'grid';
    } else {
      sliderView.hidden = true;
      sliderView.style.display = 'none';
      compView.hidden = true;
      compView.style.display = 'none';
      singleView.hidden = false;
      singleView.style.display = 'block';

      if (tab === 'original') {
        activeImg.src = frame.original_image;
        caption.textContent = '01 ORIGINAL: Untouched raw frame extracted directly from the video stream without any noise, heatmap, or blueprint processing.';
      } else if (tab === 'blueprint') {
        activeImg.src = frame.blueprint_image;
        caption.textContent = '02 NOISE BLUEPRINT: Multi-scale residual noise stippling rendered on an architectural navy blueprint canvas, exposing synthetic seams.';
      } else if (tab === 'heatmap') {
        activeImg.src = frame.heatmap_image;
        caption.textContent = '03 HEATMAP: Thermal Jet colormap highlighting suspicious localized anomaly gradients.';
      }
    }

    // Update prev/next buttons
    const prevBtn = document.querySelector('#modal-prev-btn');
    const nextBtn = document.querySelector('#modal-next-btn');
    if (prevBtn) prevBtn.disabled = frame.number <= 1;
    if (nextBtn) nextBtn.disabled = frame.number >= frames.length;
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
  const mPrevBtn = document.querySelector('#modal-prev-btn');
  const mNextBtn = document.querySelector('#modal-next-btn');
  if (mPrevBtn) {
    mPrevBtn.addEventListener('click', () => {
      if (currentModalFrameNum > 1) {
        currentModalFrameNum--;
        const frame = frames.find(f => f.number === currentModalFrameNum);
        if (frame) updateModalView(frame, currentModalTab);
      }
    });
  }
  if (mNextBtn) {
    mNextBtn.addEventListener('click', () => {
      if (currentModalFrameNum < frames.length) {
        currentModalFrameNum++;
        const frame = frames.find(f => f.number === currentModalFrameNum);
        if (frame) updateModalView(frame, currentModalTab);
      }
    });
  }

  // Keyboard navigation
  window.addEventListener('keydown', e => {
    const modal = document.querySelector('#inspector-modal');
    if (!modal || modal.hidden) return;
    if (e.key === 'Escape') closeFrameModal();
    if (e.key === 'ArrowLeft' && mPrevBtn) mPrevBtn.click();
    if (e.key === 'ArrowRight' && mNextBtn) mNextBtn.click();
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
    let ptColor = '#00ff87';
    if (pt.suspicion >= 65) ptColor = '#ff2a5f';
    else if (pt.suspicion >= 38) ptColor = '#f4d35e';

    pointsHtml += `
      <circle cx="${px}" cy="${pySusp}" r="${isPeak ? 6 : 3.5}" fill="${isPeak ? '#ff2a5f' : ptColor}" stroke="#071018" stroke-width="1.5" class="chart-pt" data-frame="${pt.frame}" data-susp="${pt.suspicion}" data-ts="${pt.timestamp}">
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

  // Add click to jump to frame in Featured Forensics Console & scroll to card
  container.querySelectorAll('.chart-pt').forEach(pt => {
    pt.addEventListener('click', () => {
      const frameNum = parseInt(pt.dataset.frame, 10);
      if (window.selectFeaturedFrame) {
        window.selectFeaturedFrame(frameNum);
      }
      const targetCard = document.querySelector(`#frame-card-${frameNum}`);
      if (targetCard) {
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        targetCard.classList.add('highlight-pulse');
        setTimeout(() => targetCard.classList.remove('highlight-pulse'), 1800);
      }
    });
  });
}
