document.addEventListener('DOMContentLoaded', () => {
  const video        = document.getElementById('video');
  const overlay      = document.getElementById('overlay');
  const ctx          = overlay.getContext('2d');
  const clockEl      = document.getElementById('realtimeClock');
  const faceHintBar  = document.getElementById('faceHintBar');
  const faceHintText = document.getElementById('faceHintText');
  const captureBtn   = document.getElementById('captureBtn');
  const resultBanner = document.getElementById('resultBanner');

  const DETECT_INTERVAL_MS  = 700;   // how often to poll face size
  const CAPTURE_COOLDOWN_MS = 3000;  // lock button after capture attempt

  const tmpCanvas = document.createElement('canvas');
  const tmpCtx    = tmpCanvas.getContext('2d');

  let detectTimer     = null;
  let isCameraReady   = false;
  let isCapturing     = false;
  let faceSizeOk      = false;

  // ── Clock ───────────────────────────────────────────────────────────────────

  function updateClock() {
    clockEl.textContent = new Date().toLocaleTimeString('vi-VN', {
      hour12: false,
      timeZone: 'Asia/Bangkok',
    });
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── Overlay: oval guide ─────────────────────────────────────────────────────

  function drawOvalGuide(color) {
    const w = overlay.width;
    const h = overlay.height;
    ctx.clearRect(0, 0, w, h);

    const cx   = w / 2;
    const cy   = h / 2;
    const rx   = w * 0.28;
    const ry   = h * 0.40;

    // Dim area outside oval
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();

    // Oval border
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.stroke();
  }

  function adjustOverlay() {
    overlay.width         = video.videoWidth;
    overlay.height        = video.videoHeight;
    overlay.style.width   = video.clientWidth  + 'px';
    overlay.style.height  = video.clientHeight + 'px';
    drawOvalGuide(faceSizeOk ? '#10b981' : '#94a3b8');
  }

  // ── Face hint update ────────────────────────────────────────────────────────

  function setHint(state, text) {
    faceHintBar.className = 'face-hint-bar hint-' + state;
    faceHintText.textContent = text;
    faceSizeOk = (state === 'ok');
    captureBtn.disabled = !faceSizeOk || isCapturing;
    drawOvalGuide(faceSizeOk ? '#10b981' : '#94a3b8');
  }

  // ── Periodic face-size detection ────────────────────────────────────────────

  async function detectFaceSize() {
    if (!isCameraReady || isCapturing) return;
    if (video.videoWidth === 0) return;

    const scale     = 320 / video.videoWidth;
    tmpCanvas.width  = 320;
    tmpCanvas.height = Math.round(video.videoHeight * scale);
    tmpCtx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);

    const blob = await new Promise(r => tmpCanvas.toBlob(r, 'image/jpeg', 0.8));
    if (!blob) return;

    try {
      const fd  = new FormData();
      fd.append('image', blob, 'frame.jpg');
      const res  = await fetch(window.detectFaceUrl, { method: 'POST', body: fd });
      const data = await res.json();

      if (data.status === 'no_face' || !data.state) {
        setHint('none', 'Không phát hiện khuôn mặt');
      } else if (data.state === 'too_small') {
        setHint('small', 'Lại gần camera hơn');
      } else if (data.state === 'too_large') {
        setHint('large', 'Lùi ra xa hơn');
      } else {
        setHint('ok', 'Khuôn mặt phù hợp');
      }
    } catch (_) {
      // network error — keep last state
    }
  }

  // ── Show result ─────────────────────────────────────────────────────────────

  function showResult(success, lines) {
    resultBanner.className = 'result-banner ' + (success ? 'success' : 'fail');
    resultBanner.innerHTML = lines.map(l => `<div>${l}</div>`).join('');
  }

  function clearResult() {
    resultBanner.className = 'result-banner';
    resultBanner.innerHTML = '';
  }

  // ── Capture & submit ────────────────────────────────────────────────────────

  captureBtn.addEventListener('click', async () => {
    if (isCapturing || !faceSizeOk) return;
    isCapturing = true;
    captureBtn.disabled = true;
    captureBtn.classList.add('loading');
    captureBtn.innerHTML = 'Đang xử lý...';
    clearResult();

    // Capture at full camera resolution
    const scale     = 640 / video.videoWidth;
    tmpCanvas.width  = 640;
    tmpCanvas.height = Math.round(video.videoHeight * scale);
    tmpCtx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);

    const blob = await new Promise(r => tmpCanvas.toBlob(r, 'image/jpeg', 0.92));

    try {
      const fd = new FormData();
      fd.append('image', blob, 'capture.jpg');
      const res  = await fetch(window.selfAttendanceUrl, { method: 'POST', body: fd });
      const data = await res.json();

      if (data.status === 'ok') {
        const typeLabel = data.type === 'checkin' ? 'Check-in' : 'Check-out';
        const typeIcon  = data.type === 'checkin' ? 'login' : 'logout';
        const statusVal = data.status_in || data.status_out || '';
        resultBanner.className = 'result-banner success';
        resultBanner.innerHTML =
          `<div class="rb-title"><span class="material-symbols-outlined">check_circle</span>Chấm công thành công</div>` +
          `<div class="rb-row"><span class="material-symbols-outlined">${typeIcon}</span>${typeLabel} lúc <strong>${data.time}</strong></div>` +
          (statusVal ? `<div class="rb-row"><span class="material-symbols-outlined">schedule</span>Trạng thái: <strong>${statusVal}</strong></div>` : '');
        drawOvalGuide('#10b981');
      } else {
        resultBanner.className = 'result-banner fail';
        resultBanner.innerHTML =
          `<div class="rb-title"><span class="material-symbols-outlined">cancel</span>Chấm công thất bại</div>` +
          `<div class="rb-row"><span class="material-symbols-outlined">info</span>${data.message || 'Không nhận diện được'}</div>`;
      }
    } catch (err) {
      resultBanner.className = 'result-banner fail';
      resultBanner.innerHTML =
        `<div class="rb-title"><span class="material-symbols-outlined">wifi_off</span>Lỗi kết nối</div>` +
        `<div class="rb-row"><span class="material-symbols-outlined">info</span>${String(err)}</div>`;
    }

    // Cooldown before next attempt
    setTimeout(() => {
      isCapturing = false;
      captureBtn.classList.remove('loading');
      captureBtn.innerHTML = '<span class="material-symbols-outlined">photo_camera</span>Chấm công';
      // Re-enable only if face is still in good position (next detect cycle will update)
    }, CAPTURE_COOLDOWN_MS);
  });

  // ── Start camera ─────────────────────────────────────────────────────────────

  async function startCamera(retried = false) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.addEventListener('loadeddata', () => {
        isCameraReady = true;
        adjustOverlay();
        setHint('none', 'Đặt khuôn mặt vào khung oval');
        detectTimer = setInterval(detectFaceSize, DETECT_INTERVAL_MS);
      });
      window.addEventListener('resize', () => {
        if (isCameraReady) adjustOverlay();
      });
    } catch (err) {
      if (!retried && (err.name === 'NotAllowedError' || err.name === 'NotReadableError')) {
        // Camera có thể đang bị server giữ — yêu cầu giải phóng rồi thử lại
        faceHintText.textContent = 'Đang giải phóng camera, vui lòng chờ...';
        try {
          await fetch(window.cameraReleaseUrl, { method: 'POST' });
          await new Promise(r => setTimeout(r, 800));
        } catch (_) {}
        return startCamera(true);
      }
      faceHintText.textContent = err.name === 'NotAllowedError'
        ? 'Camera đang được trang Admin sử dụng. Vui lòng đóng trang Chấm công Admin rồi thử lại.'
        : 'Không thể mở camera: ' + err.message;
      faceHintBar.className = 'face-hint-bar hint-none';
    }
  }

  startCamera();
});
