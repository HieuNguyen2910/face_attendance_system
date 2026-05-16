// static/attendance/js/manage.js
// Quản lý giao diện Manage - add / edit / delete + camera capture (live mode)

(() => {
  const tbody = document.getElementById('userTBody');
  const btnAdd = document.getElementById('btnAdd');
  const searchInput = document.getElementById('searchInput');
  const deptFilter = document.getElementById('deptFilter');
  const modalRoot = document.getElementById('modalRoot');

  let users = {};
  let selectedRowId = null;

  // --- Helpers: modal creation ---
  function showModal(html, opts = {}) {
    modalRoot.style.display = 'block';
    const wrap = document.createElement('div');
    wrap.className = 'app-modal';
    const boxClass = opts.boxClass ? `modal-box ${opts.boxClass}` : 'modal-box';
    wrap.innerHTML = `<div class="${boxClass}">${html}</div>`;
    modalRoot.appendChild(wrap);
    function close() {
      try { modalRoot.removeChild(wrap); } catch(e) {}
      modalRoot.style.display = modalRoot.children.length ? 'block' : 'none';
      if (opts.onClose) opts.onClose();
    }
    return { container: wrap, close };
  }

  function showConfirm(message, onConfirm, onCancel) {
    const html = `
      <style>
        .modal-btn { padding:8px 14px; border-radius:6px; border:none; cursor:pointer; font-size:0.9rem; font-weight:500; transition:0.15s; }
        .modal-btn.primary { background:#007bff; color:white; }
        .modal-btn.primary:hover { background:#0063d6; }
        .modal-btn.warning { background:#ff9800; color:white; }
        .modal-btn.warning:hover { background:#e68900; }
        .modal-btn.danger { background:#dc3545; color:white; }
        .modal-btn.danger:hover { background:#b02a37; }
      </style>
      <div><strong>Xác nhận</strong></div>
      <div style="margin-top:8px">${message}</div>
      <div class="modal-actions">
        <button id="confirmNo" class="modal-btn warning">Hủy</button>
        <button id="confirmYes" class="modal-btn danger">Xóa</button>
      </div>
    `;
    const m = showModal(html);
    m.container.querySelector('#confirmYes').onclick = () => { m.close(); onConfirm && onConfirm(); };
    m.container.querySelector('#confirmNo').onclick = () => { m.close(); onCancel && onCancel(); };
  }

  function showPrompt(title, placeholder, onOk, onCancel) {
    const html = `
      <style>
        .modal-btn { padding:8px 14px; border-radius:6px; border:none; cursor:pointer; font-size:0.9rem; font-weight:500; transition:0.15s; }
        .modal-btn.primary { background:#007bff; color:white; }
        .modal-btn.primary:hover { background:#0063d6; }
        .modal-btn.warning { background:#ff9800; color:white; }
        .modal-btn.warning:hover { background:#e68900; }
        #promptInput { width:100%; padding:8px 10px; border-radius:6px; border:1px solid #ccc; font-size:0.9rem; margin-top:6px; box-sizing:border-box; }
        #promptInput:focus { border-color:#007bff; box-shadow:0 0 0 2px rgba(0,123,255,0.2); outline:none; }
      </style>
      <div><strong>${title}</strong></div>
      <div class="modal-row" style="margin-top:8px">
        <input id="promptInput" placeholder="${placeholder}">
      </div>
      <div class="modal-actions">
        <button id="promptCancel" class="modal-btn warning">Hủy</button>
        <button id="promptOk" class="modal-btn primary">OK</button>
      </div>
    `;
    const m = showModal(html);
    const input = m.container.querySelector('#promptInput');
    input.focus();
    m.container.querySelector('#promptOk').onclick = () => { const v = input.value.trim(); m.close(); onOk && onOk(v); };
    m.container.querySelector('#promptCancel').onclick = () => { m.close(); onCancel && onCancel(); };
  }

  function readCaptureModes(container) {
    return {
      bare: Boolean(container.querySelector('#inpModeBare')?.checked),
      glasses: Boolean(container.querySelector('#inpModeGlasses')?.checked),
    };
  }

  // --- Camera modal UI (live continuous mode) ---
  function createCameraUI(title = 'Đăng ký khuôn mặt', options = {}) {
    const maxEmbeddings = options.maxEmbeddings || 50;
    const html = `
      <style>
        .camera-modal-shell {
          width: min(88vw, 960px) !important;
          max-width: 960px !important;
          max-height: calc(100vh - 28px) !important;
          padding: 10px !important;
          overflow: hidden !important;
        }

        #camWrapper {
          display: flex;
          flex-direction: column;
          gap: 12px;
          height: min(82vh, 760px);
          min-height: 0;
          text-align: left;
          padding: 4px;
        }

        #camHeader {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 16px;
          margin-bottom: 14px;
        }

        #camEyebrow {
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #0f766e;
          margin-bottom: 6px;
        }

        #camHeader h3 { margin:0; font-size:1.4rem; color:#102a43; }

        #camSubtext { margin:8px 0 0; color:#486581; font-size:0.95rem; line-height:1.5; }

        #captureCount {
          min-width: 120px;
          padding: 10px 14px;
          border-radius: 999px;
          background: linear-gradient(135deg, #0f766e, #14b8a6);
          color: #fff;
          font-weight: 700;
          text-align: center;
          box-shadow: 0 10px 22px rgba(20,184,166,0.25);
          white-space: nowrap;
        }

        #captureProgress {
          width: 100%;
          height: 10px;
          border-radius: 999px;
          background: #d9e2ec;
          overflow: hidden;
        }

        #captureProgressBar {
          width: 0%;
          height: 100%;
          background: linear-gradient(90deg, #14b8a6, #0ea5e9);
          border-radius: inherit;
          transition: width 0.25s ease;
        }

        #camShell {
          display: grid;
          grid-template-columns: minmax(0, 0.98fr) minmax(300px, 0.9fr);
          gap: 14px;
          flex: 1;
          min-height: 0;
          overflow: hidden;
        }

        #videoStage {
          position: relative;
          border-radius: 24px;
          overflow: hidden;
          background:
            radial-gradient(circle at top, rgba(56,189,248,0.22), transparent 42%),
            linear-gradient(145deg, #08121d, #17212d 65%, #101828);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 18px 40px rgba(15,23,42,0.25);
          min-height: 320px;
        }

        #regVideo {
          width: 100%;
          height: 100%;
          min-height: 320px;
          background: #000;
          object-fit: cover;
          transform: scaleX(-1) !important;
          display: block;
        }

        #faceGuide {
          position: absolute;
          top: 10%; right: 14%; bottom: 17%; left: 14%;
          border-radius: 36% 36% 42% 42% / 26% 26% 54% 54%;
          border: 3px solid rgba(56,189,248,0.92);
          box-shadow: 0 0 0 999px rgba(15,23,42,0.22), 0 0 30px rgba(56,189,248,0.4);
          pointer-events: none;
          animation: faceGuidePulse 2.2s ease-in-out infinite;
          transition: border-color 0.3s, box-shadow 0.3s;
        }
        #faceGuide.guide-ok {
          border-color: #22c55e;
          box-shadow: 0 0 0 999px rgba(15,23,42,0.22), 0 0 30px rgba(34,197,94,0.55);
          animation: none;
        }
        #faceGuide.guide-too-small {
          border-color: #f59e0b;
          box-shadow: 0 0 0 999px rgba(15,23,42,0.22), 0 0 30px rgba(245,158,11,0.55);
        }
        #faceGuide.guide-too-large {
          border-color: #ef4444;
          box-shadow: 0 0 0 999px rgba(15,23,42,0.22), 0 0 30px rgba(239,68,68,0.55);
        }
        #faceSizeHint {
          position: absolute;
          bottom: 12%; left: 50%;
          transform: translateX(-50%);
          background: rgba(15,23,42,0.72);
          color: #fff;
          font-size: 12px;
          padding: 3px 12px;
          border-radius: 20px;
          pointer-events: none;
          white-space: nowrap;
          opacity: 0;
          transition: opacity 0.3s;
        }
        #faceSizeHint.visible { opacity: 1; }

        #faceGuide::before, #faceGuide::after {
          content: "";
          position: absolute;
          border: 4px solid rgba(255,255,255,0.9);
          width: 22px; height: 22px;
        }
        #faceGuide::before { top:-3px; left:-3px; border-right:0; border-bottom:0; border-radius:16px 0 0 0; }
        #faceGuide::after  { right:-3px; bottom:-3px; border-left:0; border-top:0; border-radius:0 0 16px 0; }

        #captureFlash {
          position: absolute; inset: 0;
          background: rgba(255,255,255,0.96);
          opacity: 0; pointer-events: none;
          transition: opacity 0.16s ease;
        }

        #videoHint {
          position: absolute;
          left: 18px; right: 18px; bottom: 18px;
          padding: 14px 16px;
          border-radius: 18px;
          background: rgba(15,23,42,0.62);
          color: #e2e8f0;
          font-size: 0.92rem;
          line-height: 1.45;
          backdrop-filter: blur(12px);
          box-shadow: 0 10px 24px rgba(15,23,42,0.28);
        }

        #camInfoPanel {
          display: flex;
          flex-direction: column;
          gap: 14px;
          min-height: 0;
          overflow-y: auto;
          padding-right: 4px;
        }

        #camModeBadge {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          width: fit-content;
          padding: 8px 12px;
          border-radius: 999px;
          font-size: 0.82rem;
          font-weight: 700;
          background: #ecfeff;
          color: #0f766e;
          border: 1px solid rgba(20,184,166,0.18);
        }
        #camModeBadge::before {
          content: "";
          width: 10px; height: 10px;
          border-radius: 50%;
          background: #14b8a6;
          box-shadow: 0 0 0 6px rgba(20,184,166,0.15);
        }

        #camStatusCard {
          padding: 18px;
          border-radius: 22px;
          background: linear-gradient(160deg, #eff6ff, #f8fafc 72%);
          border: 1px solid rgba(148,163,184,0.18);
          box-shadow: 0 12px 28px rgba(15,23,42,0.08);
        }
        #camStatusLabel {
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #64748b;
          margin-bottom: 8px;
        }
        #camLastStatus {
          font-size: 1rem;
          font-weight: 600;
          color: #0f172a;
          line-height: 1.4;
          min-height: 1.4em;
        }
        #camStatusSub {
          margin-top: 8px;
          color: #475569;
          font-size: 0.9rem;
          line-height: 1.5;
        }

        #camTips {
          padding: 16px 18px;
          border-radius: 22px;
          background: #f8fafc;
          border: 1px dashed #cbd5e1;
          color: #334155;
          font-size: 0.92rem;
          line-height: 1.55;
        }
        #camTips strong { display:block; margin-bottom:6px; color:#0f172a; }

        #captureThumbs {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
          max-height: 172px;
          overflow-y: auto;
          align-content: start;
        }

        .capture-thumb {
          position: relative;
          border-radius: 16px;
          overflow: hidden;
          background: #0f172a;
          min-height: 96px;
          box-shadow: 0 12px 26px rgba(15,23,42,0.16);
        }
        .capture-thumb img { width:100%; height:100%; object-fit:cover; display:block; }
        .capture-thumb span {
          position: absolute;
          left: 8px; right: 8px; bottom: 8px;
          padding: 5px 7px;
          border-radius: 10px;
          background: rgba(15,23,42,0.72);
          color: #fff;
          font-size: 0.72rem;
          font-weight: 700;
          text-align: center;
          line-height: 1.2;
        }

        #camActions {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          flex-shrink: 0;
          padding-top: 10px;
          border-top: 1px solid rgba(226,232,240,0.95);
          background: #fff;
        }

        #camStepHelp { color:#475569; font-size:0.9rem; line-height:1.45; }

        #camDoneBtn {
          padding: 12px 18px;
          border: none;
          border-radius: 12px;
          background: linear-gradient(135deg, #0284c7, #0ea5e9);
          color: #fff;
          cursor: pointer;
          font-size: 0.94rem;
          font-weight: 700;
          transition: transform 0.18s ease, box-shadow 0.18s ease;
          box-shadow: 0 12px 26px rgba(14,165,233,0.24);
        }
        #camDoneBtn:hover { transform: translateY(-1px); }

        #camCancel {
          padding: 10px 16px;
          border: none;
          border-radius: 12px;
          background: linear-gradient(135deg, #ef4444, #dc2626);
          color: #fff;
          cursor: pointer;
          font-size: 0.92rem;
          font-weight: 700;
          transition: transform 0.18s ease, box-shadow 0.18s ease;
          box-shadow: 0 12px 26px rgba(220,38,38,0.25);
        }
        #camCancel:hover { transform: translateY(-1px); }

        #camActionButtons { display:flex; gap:10px; align-items:center; }

        @keyframes faceGuidePulse {
          0%, 100% { transform: scale(0.985); opacity: 0.82; }
          50% { transform: scale(1.01); opacity: 1; }
        }

        @media (max-width: 920px) {
          #camWrapper { height: min(86vh, 820px); }
          #camShell { grid-template-columns: 1fr; overflow-y: auto; }
          #regVideo { min-height: 300px; }
          #captureThumbs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          #faceGuide { top:8%; right:12%; bottom:15%; left:12%; }
        }

        @media (max-width: 640px) {
          .camera-modal-shell { width: min(96vw, 720px) !important; padding: 8px !important; }
          #camWrapper { height: min(88vh, 860px); }
          #camHeader { flex-direction: column; }
          #captureCount { min-width: 0; }
          #camActions { flex-direction: column; align-items: stretch; }
          #camActionButtons { width:100%; justify-content: stretch; }
          #camActionButtons button { flex: 1; }
        }
      </style>

      <div id="camWrapper">
        <div id="camHeader">
          <div>
            <div id="camEyebrow">Thu thập embedding đa góc</div>
            <h3 id="camTitle">${title}</h3>
            <p id="camSubtext">Xoay mặt tự nhiên sang trái, phải, ngẩng, cúi và thay đổi khoảng cách. Hệ thống tự động lưu khi phát hiện góc mới.</p>
          </div>
          <div id="captureCount">0 / ${maxEmbeddings} góc</div>
        </div>

        <div id="captureProgress">
          <div id="captureProgressBar"></div>
        </div>

        <div id="camShell">
          <div id="videoStage">
            <video id="regVideo" autoplay playsinline></video>
            <div id="faceGuide"></div>
            <div id="faceSizeHint"></div>
            <div id="captureFlash"></div>
            <div id="videoHint">Xoay mặt sang các góc khác nhau — hệ thống tự nhận ra góc mới và lưu ngay.</div>
          </div>

          <div id="camInfoPanel">
            <div id="camModeBadge">Đang chuẩn bị...</div>

            <div id="camStatusCard">
              <div id="camStatusLabel">Trạng thái</div>
              <div id="camLastStatus">Đang khởi động camera...</div>
              <div id="camStatusSub">Đưa mặt vào khung và bắt đầu xoay nhẹ</div>
            </div>

            <div id="camTips">
              <strong>Hướng dẫn lấy đủ góc</strong>
              Xoay trái/phải ~20°, ngẩng/cúi nhẹ cằm, thử đứng xa rồi gần. Mỗi góc lạ hệ thống tự lưu ngay. Nhấn <em>Dừng &amp; Hoàn tất</em> khi đã hài lòng.
            </div>

            <div id="captureThumbs"></div>
          </div>
        </div>

        <div id="camActions">
          <div id="camStepHelp">Xoay mặt tự nhiên — hệ thống phát hiện và lưu tự động khi đủ khác biệt.</div>
          <div id="camActionButtons">
            <button id="camDoneBtn" type="button">Dừng &amp; Hoàn tất</button>
            <button id="camCancel" type="button">Huỷ</button>
          </div>
        </div>
      </div>
    `;

    return showModal(html, { boxClass: 'camera-modal-shell' });
  }

  async function startCamera(videoEl) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    videoEl.srcObject = stream;
    await new Promise(res => videoEl.onloadedmetadata = res);
    return stream;
  }

  function captureBlobFromVideo(videoEl) {
    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  }

  function updateCaptureProgress(barEl, countEl, done, total) {
    const percent = total ? (done / total) * 100 : 0;
    barEl.style.width = `${percent}%`;
    countEl.textContent = `${done} / ${total} góc`;
  }

  function triggerCameraFlash(flashEl) {
    flashEl.style.opacity = '1';
    setTimeout(() => { flashEl.style.opacity = '0'; }, 140);
  }

  function estimateCanvasQuality(canvas, ctx) {
    const { width, height } = canvas;
    const imageData = ctx.getImageData(0, 0, width, height).data;
    const stride = 4;
    let detailScore = 0;
    let brightnessScore = 0;
    let samples = 0;

    for (let y = 0; y < height - 4; y += 4) {
      for (let x = 0; x < width - 4; x += 4) {
        const index = (y * width + x) * stride;
        const rightIndex = (y * width + (x + 4)) * stride;
        const downIndex = ((y + 4) * width + x) * stride;
        const gray = 0.299 * imageData[index] + 0.587 * imageData[index + 1] + 0.114 * imageData[index + 2];
        const grayRight = 0.299 * imageData[rightIndex] + 0.587 * imageData[rightIndex + 1] + 0.114 * imageData[rightIndex + 2];
        const grayDown = 0.299 * imageData[downIndex] + 0.587 * imageData[downIndex + 1] + 0.114 * imageData[downIndex + 2];
        detailScore += Math.abs(gray - grayRight) + Math.abs(gray - grayDown);
        brightnessScore += gray;
        samples += 1;
      }
    }

    if (!samples) return -Infinity;
    const avgDetail = detailScore / samples;
    const avgBrightness = brightnessScore / samples;
    const brightnessPenalty = Math.abs(avgBrightness - 138) * 0.35;
    return avgDetail - brightnessPenalty;
  }

  async function captureBestBlobFromVideo(videoEl, attempts = 3, delayMs = 120) {
    let best = null;
    for (let i = 0; i < attempts; i++) {
      const canvas = document.createElement('canvas');
      canvas.width = videoEl.videoWidth || 640;
      canvas.height = videoEl.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      const quality = estimateCanvasQuality(canvas, ctx);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
      if (blob && (!best || quality > best.quality)) {
        best = { blob, quality };
      }
      if (i < attempts - 1) {
        await new Promise(r => setTimeout(r, delayMs));
      }
    }
    return best ? best.blob : null;
  }

  function appendThumbnail(thumbsEl, label, blob) {
    const item = document.createElement('div');
    item.className = 'capture-thumb';
    const url = URL.createObjectURL(blob);
    item.innerHTML = `<img src="${url}" alt="${label}"><span>${label}</span>`;
    thumbsEl.prepend(item);
    while (thumbsEl.children.length > 12) {
      thumbsEl.removeChild(thumbsEl.lastChild);
    }
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function startFaceGuidePolling(videoEl, guideEl, hintEl, detectUrl) {
    let active = true;
    const offCanvas = document.createElement('canvas');

    async function poll() {
      while (active) {
        try {
          offCanvas.width = videoEl.videoWidth || 640;
          offCanvas.height = videoEl.videoHeight || 480;
          offCanvas.getContext('2d').drawImage(videoEl, 0, 0, offCanvas.width, offCanvas.height);
          const blob = await new Promise(r => offCanvas.toBlob(r, 'image/jpeg', 0.7));
          const fd = new FormData();
          fd.append('image', blob, 'preview.jpg');
          const res = await fetch(detectUrl, { method: 'POST', body: fd });
          const data = await res.json();

          guideEl.classList.remove('guide-ok', 'guide-too-small', 'guide-too-large');
          if (data.status === 'no_face') {
            hintEl.textContent = '';
            hintEl.classList.remove('visible');
          } else if (data.state === 'ok') {
            guideEl.classList.add('guide-ok');
            hintEl.textContent = '✓ Khoảng cách tốt';
            hintEl.classList.add('visible');
          } else if (data.state === 'too_small') {
            guideEl.classList.add('guide-too-small');
            hintEl.textContent = '↔ Lại gần hơn';
            hintEl.classList.add('visible');
          } else if (data.state === 'too_large') {
            guideEl.classList.add('guide-too-large');
            hintEl.textContent = '↔ Lùi ra xa hơn';
            hintEl.classList.add('visible');
          }
        } catch (_) {}
        await sleep(600);
      }
      guideEl.classList.remove('guide-ok', 'guide-too-small', 'guide-too-large');
      hintEl.classList.remove('visible');
    }

    poll();
    return () => { active = false; };
  }

  // --- Live continuous enrollment ---
  async function captureEnrollmentLive(camModal, options = {}) {
    const { userId, maxEmbeddings = 50, modeBadgeText = 'Đang thu' } = options;

    const videoEl      = camModal.container.querySelector('#regVideo');
    const countEl      = camModal.container.querySelector('#captureCount');
    const progressBarEl= camModal.container.querySelector('#captureProgressBar');
    const doneBtnEl    = camModal.container.querySelector('#camDoneBtn');
    const cancelBtnEl  = camModal.container.querySelector('#camCancel');
    const flashEl      = camModal.container.querySelector('#captureFlash');
    const thumbsEl     = camModal.container.querySelector('#captureThumbs');
    const statusEl     = camModal.container.querySelector('#camLastStatus');
    const modeBadgeEl  = camModal.container.querySelector('#camModeBadge');

    if (modeBadgeEl) modeBadgeEl.textContent = modeBadgeText;

    let stream;
    try {
      stream = await startCamera(videoEl);
    } catch (err) {
      camModal.close();
      throw new Error('Không thể truy cập camera: ' + err.message);
    }

    const guideEl = camModal.container.querySelector('#faceGuide');
    const hintEl  = camModal.container.querySelector('#faceSizeHint');
    const stopPolling = window.apiUrls?.detectFace
      ? startFaceGuidePolling(videoEl, guideEl, hintEl, window.apiUrls.detectFace)
      : () => {};

    let sessionCount = 0;
    let stopped = false;
    let cancelled = false;

    function updateProgress(n) {
      sessionCount = n;
      countEl.textContent = `${n} / ${maxEmbeddings} góc`;
      progressBarEl.style.width = `${Math.min(100, (n / maxEmbeddings) * 100)}%`;
    }

    doneBtnEl.onclick  = () => { stopped = true; };
    cancelBtnEl.onclick = () => { cancelled = true; stopped = true; };

    updateProgress(0);
    if (statusEl) statusEl.innerHTML = 'Hướng mặt vào camera và bắt đầu xoay...';

    // Sequential capture loop — mỗi iteration chờ server trả về rồi mới chụp tiếp
    while (!stopped && sessionCount < maxEmbeddings) {
      try {
        const blob = await captureBestBlobFromVideo(videoEl, 2, 80);
        if (!blob || stopped) break;

        const fd = new FormData();
        fd.append('user_id', userId);
        fd.append('image', blob, 'frame.jpg');

        const res  = await fetch(window.apiUrls.registerFrame, { method: 'POST', body: fd });
        const data = await res.json();

        if (data.status === 'added') {
          triggerCameraFlash(flashEl);
          const newCount = sessionCount + 1;
          updateProgress(newCount);
          appendThumbnail(thumbsEl, `#${newCount}`, blob);
          if (statusEl) statusEl.innerHTML =
            `<span style="color:#166534;font-weight:600">✓ Đã lưu góc mới</span>` +
            `<small style="color:#6b7280;margin-left:6px">(${(data.max_similarity * 100).toFixed(0)}% tương đồng)</small>`;
          if (newCount >= maxEmbeddings) stopped = true;

        } else if (data.status === 'skipped') {
          if (statusEl) statusEl.innerHTML =
            `<span style="color:#92400e">→ Góc này đã có</span>` +
            `<small style="color:#6b7280;margin-left:6px">(${(data.max_similarity * 100).toFixed(0)}%) — xoay mặt sang hướng khác</small>`;
          // Cập nhật count từ DB nếu server trả về (trường hợp supplement sau lần trước)
          if (typeof data.total_embeddings === 'number' && data.total_embeddings > sessionCount) {
            updateProgress(data.total_embeddings);
          }

        } else if (data.status === 'no_face') {
          if (statusEl) statusEl.innerHTML =
            `<span style="color:#9ca3af">⌀ Không phát hiện khuôn mặt — đưa mặt vào giữa khung</span>`;
        }
      } catch (e) {
        console.warn('Frame capture error:', e);
      }

      if (!stopped) await sleep(300);
    }

    stopPolling();
    try { stream.getTracks().forEach(t => t.stop()); } catch(e) {}
    camModal.close();

    if (cancelled) throw new Error('Người dùng hủy camera');
    return sessionCount;
  }

  // Orchestrate bare → glasses sessions
  async function runRegistrationSessions(userId, modes, maxEmbeddings) {
    if (modes.bare) {
      const cam = createCameraUI('Đăng ký — Không kính', { maxEmbeddings });
      await captureEnrollmentLive(cam, { userId, maxEmbeddings, modeBadgeText: 'Không kính' });
    }

    if (modes.glasses) {
      const proceed = await new Promise(resolve => {
        const html = `
          <style>
            .modal-btn { padding:8px 14px; border-radius:6px; border:none; cursor:pointer; font-size:0.9rem; font-weight:500; transition:0.15s; }
            .modal-btn.primary { background:#007bff; color:white; }
            .modal-btn.primary:hover { background:#0063d6; }
            .modal-btn.warning { background:#ff9800; color:white; }
            .modal-btn.warning:hover { background:#e68900; }
          </style>
          <div><strong>Chuyển sang chế độ có kính</strong></div>
          <div style="margin:10px 0 4px;color:#475569;line-height:1.55">
            Đeo kính vào và điều chỉnh thoải mái. Bấm <strong>Tiếp tục</strong> để bắt đầu thu embedding khi đeo kính.
          </div>
          <div class="modal-actions">
            <button id="btnSkip" class="modal-btn warning">Bỏ qua</button>
            <button id="btnNext" class="modal-btn primary">Tiếp tục</button>
          </div>
        `;
        const m = showModal(html);
        m.container.querySelector('#btnNext').onclick = () => { m.close(); resolve(true); };
        m.container.querySelector('#btnSkip').onclick = () => { m.close(); resolve(false); };
      });

      if (proceed) {
        const cam = createCameraUI('Đăng ký — Có kính', { maxEmbeddings });
        await captureEnrollmentLive(cam, { userId, maxEmbeddings, modeBadgeText: 'Có kính' });
      }
    }
  }

  // --- API calls ---
  async function apiFetchUsers() {
    try {
      const res = await fetch(window.apiUrls.listUsers);
      const json = await res.json();
      if (json && json.status === 'ok' && Array.isArray(json.users)) return json.users;
      if (Array.isArray(json)) return json;
      if (json && Array.isArray(json.users)) return json.users;
      return json.users || json;
    } catch (e) {
      console.error('apiFetchUsers lỗi:', e);
      return [];
    }
  }

  async function apiRegisterEmployee({ user_id, name, position, department, password, images }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    fd.append('name', name);
    fd.append('position', position);
    fd.append('department', department || '');
    if (password) fd.append('password', password);
    if (images && images.length) {
      images.forEach((b, i) => fd.append('image', b, `img${i}.jpg`));
    }
    const res = await fetch(window.apiUrls.registerEmployee, { method: 'POST', body: fd });
    return await res.json();
  }

  async function apiReplaceFace({ user_id, images }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    images.forEach((b, i) => fd.append('image', b, `img${i}.jpg`));
    const res = await fetch(window.apiUrls.replaceFace, { method: 'POST', body: fd });
    return await res.json();
  }

  async function apiClearEmbeddings({ user_id }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    const res = await fetch(window.apiUrls.clearEmbeddings, { method: 'POST', body: fd });
    return await res.json();
  }

  async function apiUpdateUser({ user_id, name, position, department }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    fd.append('name', name);
    fd.append('position', position);
    fd.append('department', department || '');
    const res = await fetch(window.apiUrls.updateUser, { method: 'POST', body: fd });
    return await res.json();
  }

  async function apiDeleteUser({ user_id }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    const res = await fetch(window.apiUrls.deleteUser, { method: 'POST', body: fd });
    return await res.json();
  }

  // --- Dept filter & table ---
  function renderDeptFilter() {
    if (!deptFilter) return;
    const current = deptFilter.value;
    const depts = new Set();
    Object.values(users).forEach(m => { if (m.department) depts.add(m.department); });
    deptFilter.innerHTML = '<option value="">Tất cả phòng ban</option>';
    Array.from(depts).sort().forEach(d => {
      const opt = document.createElement('option');
      opt.value = d;
      opt.textContent = d;
      if (d === current) opt.selected = true;
      deptFilter.appendChild(opt);
    });
  }

  function renderTable(filter = '') {
    tbody.innerHTML = '';
    const f = filter.trim().toLowerCase();
    const deptVal = deptFilter ? deptFilter.value : '';
    const ids = Object.keys(users).sort((a, b) => a.localeCompare(b));
    let shown = 0;

    ids.forEach(id => {
      const meta = users[id] || {};
      const name = meta.name || meta.ten_nv || '';
      const pos  = meta.position || meta.chuc_vu || '';
      const dept = meta.department || '';

      if (deptVal && dept !== deptVal) return;
      const rowText = `${id} ${name} ${pos} ${dept}`.toLowerCase();
      if (f && !rowText.includes(f)) return;

      shown++;
      const tr = document.createElement('tr');
      tr.dataset.id = id;
      tr.innerHTML = `
        <td class="col-id"><span class="employee-id">${escapeHtml(id)}</span></td>
        <td class="col-name" style="font-weight:500;">${escapeHtml(name)}</td>
        <td class="col-pos">${escapeHtml(pos)}</td>
        <td class="col-dept">${escapeHtml(dept) || '<span style="color:var(--text-muted)">—</span>'}</td>
        <td class="col-actions">
          <button class="btn-row view" title="Xem lịch sử chấm công">📋 Lịch sử</button>
          <button class="btn-row edit" title="Sửa thông tin">✏️ Sửa</button>
          <button class="btn-row del"  title="Xóa nhân viên">🗑️ Xóa</button>
        </td>
      `;

      tr.querySelector('.btn-row.view').onclick = (e) => {
        e.stopPropagation();
        const url = (window.employeeHistoryUrlTemplate || '').replace('__ID__', encodeURIComponent(id));
        window.location.href = url;
      };
      tr.querySelector('.btn-row.edit').onclick = (e) => { e.stopPropagation(); openEditForm(id); };
      tr.querySelector('.btn-row.del').onclick = (e) => {
        e.stopPropagation();
        showConfirm(`Xóa nhân viên <strong>${escapeHtml(id)}</strong> — ${escapeHtml(name)}?`, async () => {
          try {
            const res = await apiDeleteUser({ user_id: id });
            if (res.status === 'ok') { selectedRowId = null; await refreshUsers(); }
            else alert('Xóa lỗi: ' + (res.message || JSON.stringify(res)));
          } catch (err) { alert('Lỗi xóa: ' + err.message); }
        });
      };

      tr.onclick = () => {
        document.querySelectorAll('#userTBody tr').forEach(r => r.classList.remove('selected'));
        tr.classList.add('selected');
        selectedRowId = id;
      };

      tbody.appendChild(tr);
    });

    if (shown === 0) {
      const empty = document.createElement('tr');
      empty.innerHTML = `<td colspan="5" class="empty">
        <div class="empty-state">
          <div class="empty-state-icon">🔍</div>
          <div class="empty-state-text">Không tìm thấy nhân viên</div>
          <div class="empty-state-sub">Thử thay đổi bộ lọc hoặc từ khoá tìm kiếm</div>
        </div>
      </td>`;
      tbody.appendChild(empty);
    }
  }

  // --- Add form ---
  function openAddForm() {
    const html = `
      <style>
        .modal-btn { padding:8px 14px; border-radius:6px; border:none; cursor:pointer; font-size:0.9rem; font-weight:500; transition:0.15s; }
        .modal-btn.primary { background:#007bff; color:white; }
        .modal-btn.primary:hover { background:#0063d6; }
        .modal-btn.warning { background:#ff9800; color:white; }
        .modal-btn.warning:hover { background:#e68900; }

        .capture-mode-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; flex:1; }
        .capture-mode-option { display:flex; gap:12px; align-items:flex-start; padding:14px 16px; border-radius:18px; border:1px solid #cbd5e1; background:linear-gradient(180deg,#f8fafc,#eef6ff); box-shadow:inset 0 1px 0 rgba(255,255,255,0.9); }
        .capture-mode-option input { margin-top:4px; flex:none; }
        .capture-mode-title { display:block; font-weight:700; color:#0f172a; margin-bottom:4px; }
        .capture-mode-desc { display:block; color:#475569; line-height:1.45; font-size:0.88rem; }
        .capture-plan-note { margin-top:12px; padding:14px 16px; border-radius:18px; background:#f8fafc; color:#334155; line-height:1.55; font-size:0.9rem; }
        @media (max-width:720px) { .capture-mode-grid { grid-template-columns:1fr; } }
      </style>

      <div><strong>Thêm nhân viên mới</strong></div>

      <div class="modal-row" style="margin-top:8px">
        <label style="width:80px">Mã ID</label>
        <input id="inpId" placeholder="VD: nv001">
      </div>
      <div class="modal-row">
        <label style="width:80px">Họ tên</label>
        <input id="inpName" placeholder="Họ tên đầy đủ">
      </div>
      <div class="modal-row">
        <label style="width:80px">Vị trí</label>
        <input id="inpPos" placeholder="VD: Kỹ sư, Nhân viên...">
      </div>
      <div class="modal-row">
        <label style="width:80px">Phòng ban</label>
        <input id="inpDept" placeholder="VD: Kỹ thuật, Nhân sự, Kế toán...">
      </div>
      <div class="modal-row">
        <label style="width:80px">Mật khẩu</label>
        <input id="inpPassword" type="password" placeholder="Mật khẩu đăng nhập (tuỳ chọn)">
      </div>
      <div class="modal-row">
        <label style="width:80px">Tối đa</label>
        <input id="inpMaxEmb" type="number" min="5" max="100" value="50" style="width:64px">
        <span style="color:#64748b;font-size:0.88rem;margin-left:6px">góc / chế độ</span>
      </div>

      <div class="modal-row" style="align-items:flex-start">
        <label style="width:80px">Chế độ</label>
        <div class="capture-mode-grid">
          <label class="capture-mode-option">
            <input id="inpModeBare" type="checkbox" checked>
            <span>
              <span class="capture-mode-title">Không kính</span>
              <span class="capture-mode-desc">Xoay mặt tự nhiên, hệ thống tự lưu các góc mới.</span>
            </span>
          </label>
          <label class="capture-mode-option">
            <input id="inpModeGlasses" type="checkbox">
            <span>
              <span class="capture-mode-title">Có kính</span>
              <span class="capture-mode-desc">Tiếp tục thu sau khi đeo kính vào.</span>
            </span>
          </label>
        </div>
      </div>

      <div class="capture-plan-note">
        Hệ thống tự động nhận ra và lưu khi phát hiện góc mặt chưa có trong DB. Xoay mặt trái/phải, ngẩng/cúi nhẹ và thay đổi khoảng cách để phủ đủ các góc.
      </div>

      <div class="modal-actions">
        <button id="btnCancel" class="modal-btn warning">Hủy</button>
        <button id="btnContinue" class="modal-btn primary">Bắt đầu đăng ký</button>
      </div>
    `;

    const m = showModal(html);
    m.container.querySelector('#btnCancel').onclick = () => m.close();
    m.container.querySelector('#btnContinue').onclick = async () => {
      const id      = m.container.querySelector('#inpId').value.trim();
      const name    = m.container.querySelector('#inpName').value.trim();
      const pos     = m.container.querySelector('#inpPos').value.trim();
      const dept    = m.container.querySelector('#inpDept').value.trim();
      const password= m.container.querySelector('#inpPassword').value.trim();
      const maxEmb  = Math.max(5, Math.min(100, parseInt(m.container.querySelector('#inpMaxEmb').value) || 50));
      const modes   = readCaptureModes(m.container);

      if (!id || !name || !pos) return alert('Vui lòng nhập đầy đủ ID, họ tên và vị trí');
      if (!modes.bare && !modes.glasses) return alert('Chọn ít nhất 1 chế độ đăng ký: không kính hoặc có kính');
      m.close();

      // Bước 1: tạo nhân viên (không có ảnh)
      try {
        const regRes = await apiRegisterEmployee({ user_id: id, name, position: pos, department: dept, password, images: [] });
        if (regRes.status !== 'ok') {
          alert('Lỗi tạo nhân viên: ' + (regRes.message || JSON.stringify(regRes)));
          return;
        }
      } catch (err) {
        alert('Lỗi kết nối: ' + err.message);
        return;
      }

      // Bước 2: thu embedding qua camera
      try {
        await runRegistrationSessions(id, modes, maxEmb);
        alert(`Đăng ký thành công! Đã thêm nhân viên ${name}.`);
        await refreshUsers();
      } catch (err) {
        if (err.message && err.message.includes('hủy')) {
          alert('Đã hủy thu ảnh. Thông tin nhân viên đã được tạo nhưng chưa có embedding khuôn mặt.');
        } else {
          alert(err.message || String(err));
        }
        await refreshUsers();
      }
    };
  }

  function openEditSelectedOrPrompt() {
    if (selectedRowId) {
      openEditForm(selectedRowId);
    } else {
      showPrompt('Nhập ID cần sửa', 'Mã ID', (id) => {
        if (!id) return;
        if (!users[id]) return alert('Không tìm thấy ID: ' + id);
        openEditForm(id);
      });
    }
  }

  // --- Edit form ---
  function openEditForm(id) {
    const meta = users[id] || {};
    const html = `
      <style>
        .modal-btn { padding:8px 14px; border-radius:6px; border:none; cursor:pointer; font-size:0.9rem; font-weight:500; transition:0.15s; }
        .modal-btn.primary { background:#007bff; color:white; }
        .modal-btn.primary:hover { background:#0063d6; }
        .modal-btn.warning { background:#ff9800; color:white; }
        .modal-btn.warning:hover { background:#e68900; }
        .modal-btn.success { background:#16a34a; color:white; }
        .modal-btn.success:hover { background:#15803d; }
        .modal-btn.danger2 { background:#7c3aed; color:white; }
        .modal-btn.danger2:hover { background:#6d28d9; }

        .capture-mode-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; flex:1; }
        .capture-mode-option { display:flex; gap:12px; align-items:flex-start; padding:14px 16px; border-radius:18px; border:1px solid #cbd5e1; background:linear-gradient(180deg,#f8fafc,#eef6ff); box-shadow:inset 0 1px 0 rgba(255,255,255,0.9); }
        .capture-mode-option input { margin-top:4px; flex:none; }
        .capture-mode-title { display:block; font-weight:700; color:#0f172a; margin-bottom:4px; }
        .capture-mode-desc { display:block; color:#475569; line-height:1.45; font-size:0.88rem; }
        .capture-plan-note { margin-top:12px; padding:14px 16px; border-radius:18px; background:#f8fafc; color:#334155; line-height:1.55; font-size:0.9rem; }
        @media (max-width:720px) { .capture-mode-grid { grid-template-columns:1fr; } }
      </style>

      <div><strong>Sửa nhân viên</strong></div>

      <div class="modal-row" style="margin-top:8px">
        <label style="width:80px">Mã ID</label>
        <input id="inpId" value="${escapeHtml(id)}" disabled>
      </div>
      <div class="modal-row">
        <label style="width:80px">Họ tên</label>
        <input id="inpName" value="${escapeHtml(meta.name || '')}">
      </div>
      <div class="modal-row">
        <label style="width:80px">Vị trí</label>
        <input id="inpPos" value="${escapeHtml(meta.position || '')}">
      </div>
      <div class="modal-row">
        <label style="width:80px">Phòng ban</label>
        <input id="inpDept" value="${escapeHtml(meta.department || '')}">
      </div>

      <div style="margin:8px 0 4px;color:#64748b;font-size:0.88rem;border-top:1px solid #e2e8f0;padding-top:10px">
        Chọn chế độ và số lượng góc tối đa khi thu thêm hoặc đăng ký lại khuôn mặt.
      </div>
      <div class="modal-row">
        <label style="width:80px">Tối đa</label>
        <input id="inpMaxEmb" type="number" min="5" max="100" value="50" style="width:64px">
        <span style="color:#64748b;font-size:0.88rem;margin-left:6px">góc / chế độ</span>
      </div>

      <div class="modal-row" style="align-items:flex-start">
        <label style="width:80px">Chế độ</label>
        <div class="capture-mode-grid">
          <label class="capture-mode-option">
            <input id="inpModeBare" type="checkbox" checked>
            <span>
              <span class="capture-mode-title">Không kính</span>
              <span class="capture-mode-desc">Bổ sung hoặc đăng ký lại góc khi không đeo kính.</span>
            </span>
          </label>
          <label class="capture-mode-option">
            <input id="inpModeGlasses" type="checkbox">
            <span>
              <span class="capture-mode-title">Có kính</span>
              <span class="capture-mode-desc">Bổ sung hoặc đăng ký lại góc khi đeo kính.</span>
            </span>
          </label>
        </div>
      </div>

      <div class="capture-plan-note">
        <strong>Bổ sung góc mặt</strong> — thêm vào embedding đã có (giữ nguyên DB cũ, thu thêm góc còn thiếu).<br>
        <strong>Đăng ký lại từ đầu</strong> — xóa toàn bộ embedding cũ rồi thu lại hoàn toàn mới.
      </div>

      <div class="modal-actions" style="flex-wrap:wrap;gap:8px">
        <button id="btnCancel"    class="modal-btn warning">Hủy</button>
        <button id="btnSaveMeta"  class="modal-btn primary">Lưu thông tin</button>
        <button id="btnAddFace"   class="modal-btn success">Bổ sung góc mặt</button>
        <button id="btnResetFace" class="modal-btn danger2">Đăng ký lại từ đầu</button>
      </div>
    `;

    const m = showModal(html);

    m.container.querySelector('#btnCancel').onclick = () => m.close();

    // Chỉ lưu metadata
    m.container.querySelector('#btnSaveMeta').onclick = async () => {
      const name = m.container.querySelector('#inpName').value.trim();
      const pos  = m.container.querySelector('#inpPos').value.trim();
      const dept = m.container.querySelector('#inpDept').value.trim();
      if (!name || !pos) return alert('Vui lòng nhập đầy đủ họ tên & vị trí');
      try {
        const res = await apiUpdateUser({ user_id: id, name, position: pos, department: dept });
        if (res.status === 'ok') { alert('Cập nhật thành công'); m.close(); await refreshUsers(); }
        else alert('Cập nhật lỗi: ' + (res.message || JSON.stringify(res)));
      } catch (err) { alert('Lỗi cập nhật: ' + err.message); }
    };

    // Bổ sung góc mặt (giữ embedding cũ)
    m.container.querySelector('#btnAddFace').onclick = async () => {
      const modes  = readCaptureModes(m.container);
      const maxEmb = Math.max(5, Math.min(100, parseInt(m.container.querySelector('#inpMaxEmb').value) || 50));
      if (!modes.bare && !modes.glasses) return alert('Chọn ít nhất 1 chế độ');
      m.close();
      try {
        await runRegistrationSessions(id, modes, maxEmb);
        alert('Bổ sung embedding thành công!');
        await refreshUsers();
      } catch (err) {
        alert(err.message || String(err));
        await refreshUsers();
      }
    };

    // Đăng ký lại từ đầu (xóa cũ, thu mới)
    m.container.querySelector('#btnResetFace').onclick = async () => {
      const modes  = readCaptureModes(m.container);
      const maxEmb = Math.max(5, Math.min(100, parseInt(m.container.querySelector('#inpMaxEmb').value) || 50));
      if (!modes.bare && !modes.glasses) return alert('Chọn ít nhất 1 chế độ');
      m.close();
      // Xóa embedding cũ trước
      try {
        await apiClearEmbeddings({ user_id: id });
      } catch (err) {
        alert('Lỗi xóa embedding cũ: ' + err.message);
        return;
      }
      // Thu mới
      try {
        await runRegistrationSessions(id, modes, maxEmb);
        alert('Đăng ký lại khuôn mặt thành công!');
        await refreshUsers();
      } catch (err) {
        alert(err.message || String(err));
        await refreshUsers();
      }
    };
  }

  function openDeleteSelectedOrPrompt() {
    if (selectedRowId) {
      const meta = users[selectedRowId] || {};
      showConfirm(`Bạn có chắc muốn xóa ID: <strong>${selectedRowId}</strong> — ${meta.name || ''}?`, async () => {
        try {
          const res = await apiDeleteUser({ user_id: selectedRowId });
          if (res.status === 'ok') { alert('Đã xóa ' + selectedRowId); selectedRowId = null; await refreshUsers(); }
          else alert('Xóa lỗi: ' + (res.message || JSON.stringify(res)));
        } catch (err) { alert('Lỗi xóa: ' + err.message); }
      });
    } else {
      showPrompt('Nhập ID cần xóa', 'Mã ID', (id) => {
        if (!id) return;
        if (!users[id]) return alert('Không tìm thấy ID: ' + id);
        const meta = users[id];
        showConfirm(`Bạn có chắc muốn xóa ID:${id} — ${meta.name || ''}?`, async () => {
          try {
            const res = await apiDeleteUser({ user_id: id });
            if (res.status === 'ok') { alert('Đã xóa ' + id); selectedRowId = null; await refreshUsers(); }
            else alert('Xóa lỗi: ' + (res.message || JSON.stringify(res)));
          } catch (err) { alert('Lỗi xóa: ' + err.message); }
        });
      });
    }
  }

  function escapeHtml(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  async function refreshUsers() {
    const raw = await apiFetchUsers();
    const out = {};
    try {
      if (Array.isArray(raw)) {
        raw.forEach(u => {
          if (!u) return;
          const id = u.user_id || u.id || String(u[0] || '').trim();
          if (!id) return;
          out[id] = {
            name: u.name || u.ten_nv || '',
            position: u.position || u.chuc_vu || '',
            department: u.department || '',
            vectors: u.vectors || []
          };
        });
      } else if (typeof raw === 'object' && raw !== null) {
        Object.keys(raw).forEach(k => {
          const v = raw[k];
          if (Array.isArray(v)) {
            out[k] = { name:'', position:'', department:'', vectors:v };
          } else if (typeof v === 'object' && v !== null) {
            out[k] = { name: v.name || v.ten_nv || '', position: v.position || v.chuc_vu || '', department: v.department || '', vectors: v.vectors || [] };
          } else {
            out[k] = { name: String(v || ''), position:'', department:'' };
          }
        });
      }
    } catch (e) { console.error('refreshUsers parse error:', e); }

    users = out;
    selectedRowId = null;
    renderDeptFilter();
    renderTable(searchInput ? searchInput.value : '');
  }

  // --- Event bindings ---
  if (btnAdd)      btnAdd.onclick    = () => openAddForm();
  if (searchInput) searchInput.oninput = () => renderTable(searchInput.value);
  if (deptFilter)  deptFilter.onchange = () => renderTable(searchInput ? searchInput.value : '');

  refreshUsers();
})();
