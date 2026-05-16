document.addEventListener('DOMContentLoaded', () => {
  const videoCanvas    = document.getElementById('videoCanvas');
  const overlay        = document.getElementById('overlay');
  const ctx            = overlay.getContext('2d');
  const recognizedIdEl = document.getElementById('recognizedId');
  const realtimeClock  = document.getElementById('realtimeClock');
  const autoToggleBtn  = document.getElementById('autoToggleBtn');
  const autoModeCard   = document.getElementById('autoModeCard');
  const autoModeBadge  = document.getElementById('autoModeBadge');
  const liveLogBody    = document.getElementById('liveLogBody');
  const clearLogBtn    = document.getElementById('clearLogBtn');
  const dots = [
    document.getElementById('dot1'),
    document.getElementById('dot2'),
    document.getElementById('dot3'),
  ];

  const mirror               = true;
  const CONSECUTIVE_REQUIRED = 3;
  const COOLDOWN_MS          = 30000;
  const TRACK_FADE_MS        = 1500;   // remove track from overlay after this long without update

  // WebSocket URL — auto-detect ws:// or wss://
  const wsScheme    = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const WS_DISPLAY  = `${wsScheme}://${window.location.host}/ws/camera-display/`;
  const WS_RECONNECT_DELAY = 3000;

  // Track state received from server
  // {id -> {name, box:{x,y,w,h}, lastSeen: timestamp}}
  let serverTracks   = {};
  let frameWidth     = 640;   // server's processing frame dimensions
  let frameHeight    = 480;
  let cameraOnline   = false;

  let autoMode     = false;
  let pendingId    = null;
  let pendingCount = 0;
  let cooldownMap  = {};
  let logCount     = 0;
  let ws           = null;

  // ── Clock ─────────────────────────────────────────────────────────────────

  function updateClock() {
    if (realtimeClock) {
      realtimeClock.textContent = new Date().toLocaleTimeString('vi-VN', {
        hour12: false, timeZone: 'Asia/Bangkok',
      });
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── Overlay ────────────────────────────────────────────────────────────────

  function adjustOverlaySize() {
    if (!videoCanvas.width) return;
    overlay.width          = videoCanvas.width;
    overlay.height         = videoCanvas.height;
    overlay.style.width    = videoCanvas.clientWidth  + 'px';
    overlay.style.height   = videoCanvas.clientHeight + 'px';
  }

  // ── Auto-mode ──────────────────────────────────────────────────────────────

  function setAutoMode(on) {
    autoMode = on;
    if (autoToggleBtn) autoToggleBtn.classList.toggle('on', on);
    if (autoModeCard)  autoModeCard.classList.toggle('is-on', on);
    if (autoModeBadge) {
      autoModeBadge.textContent = on ? 'Bật' : 'Tắt';
      autoModeBadge.className   = 'auto-badge ' + (on ? 'on' : 'off');
    }
    if (!on) {
      pendingId    = null;
      pendingCount = 0;
      updateProgressDots(0);
      serverTracks = {};
    }
  }

  if (autoToggleBtn) {
    autoToggleBtn.addEventListener('click', () => setAutoMode(!autoMode));
  }

  // ── Progress dots ──────────────────────────────────────────────────────────

  function updateProgressDots(count) {
    dots.forEach((dot, i) => {
      if (!dot) return;
      dot.classList.remove('active', 'done');
      if (count > i + 1)      dot.classList.add('done');
      else if (count === i + 1) dot.classList.add('active');
    });
  }

  // ── Log ───────────────────────────────────────────────────────────────────

  function addLogEntry(data) {
    if (!liveLogBody) return;
    const emptyRow = liveLogBody.querySelector('.empty-log-row');
    if (emptyRow) emptyRow.remove();

    logCount++;
    const typeLabel = data.type === 'checkin' ? 'Check-in' : 'Check-out';
    const typeColor = data.type === 'checkin' ? 'var(--primary)' : 'var(--warning, #f59e0b)';
    const statusVal = data.status_in || data.status_out || '';
    const statusColor = (statusVal === 'Đúng giờ' || statusVal === 'Bình thường')
      ? 'var(--success)' : 'var(--warning, #f59e0b)';

    const row = document.createElement('tr');
    row.className = 'log-flash';
    row.innerHTML = `
      <td>${logCount}</td>
      <td>${data.user_id || ''}</td>
      <td>${data.name    || ''}</td>
      <td style="color:${typeColor}; font-weight:600;">${typeLabel}</td>
      <td>${data.time    || ''}</td>
      <td style="color:${statusColor};">${statusVal}</td>
    `;
    liveLogBody.insertBefore(row, liveLogBody.firstChild);
    const rows = liveLogBody.querySelectorAll('tr:not(.empty-log-row)');
    if (rows.length > 20) rows[rows.length - 1].remove();
  }

  if (clearLogBtn) {
    clearLogBtn.addEventListener('click', () => {
      if (!liveLogBody) return;
      liveLogBody.innerHTML =
        '<tr class="empty-log-row"><td colspan="6" class="empty">Chưa có chấm công nào trong phiên này</td></tr>';
      logCount = 0;
    });
  }

  // ── Auto attendance ────────────────────────────────────────────────────────

  async function triggerAutoAttendance(userId) {
    const fd = new FormData();
    fd.append('user_id', userId);
    try {
      const res  = await fetch(window.autoAttendanceUrl, { method: 'POST', body: fd });
      const data = await res.json();
      if (data.status === 'ok') {
        cooldownMap[userId] = Date.now();
        addLogEntry(data);
      }
    } catch (err) {
      console.error('auto_attendance error:', err);
    }
  }

  function handleAutoAttendance(primaryName) {
    if (!autoMode) return;
    if (!primaryName || primaryName === 'Unknown' || primaryName === 'Spoof') {
      pendingId    = null;
      pendingCount = 0;
      updateProgressDots(0);
      return;
    }

    if (primaryName !== pendingId) {
      pendingId    = primaryName;
      pendingCount = 1;
    } else {
      pendingCount++;
    }
    updateProgressDots(Math.min(pendingCount, CONSECUTIVE_REQUIRED));
    if (pendingCount < CONSECUTIVE_REQUIRED) return;

    const lastTime = cooldownMap[pendingId];
    if (lastTime && (Date.now() - lastTime) < COOLDOWN_MS) {
      pendingCount = CONSECUTIVE_REQUIRED;
      return;
    }

    const idToSend = pendingId;
    pendingId    = null;
    pendingCount = 0;
    updateProgressDots(0);
    triggerAutoAttendance(idToSend);
  }

  // ── Recognised ID display ──────────────────────────────────────────────────

  function updateRecognizedId(name) {
    if (!recognizedIdEl) return;
    recognizedIdEl.classList.remove('recognized', 'unknown', 'spoof');
    if (name === 'Spoof') {
      recognizedIdEl.textContent = 'Cảnh báo: Giả mạo!';
      recognizedIdEl.classList.add('spoof');
    } else {
      recognizedIdEl.textContent = name;
      recognizedIdEl.classList.add(name === 'Unknown' ? 'unknown' : 'recognized');
    }
  }

  // ── Scale server boxes to video dimensions ─────────────────────────────────

  function scaleBox(box) {
    if (!videoCanvas.width) return box;
    const sx = videoCanvas.width  / frameWidth;
    const sy = videoCanvas.height / frameHeight;
    return { x: box.x * sx, y: box.y * sy, w: box.w * sx, h: box.h * sy };
  }

  // ── Apply server tracks ────────────────────────────────────────────────────

  function applyServerTracks(tracks) {
    const now    = Date.now();
    const liveIds = new Set(tracks.map(t => t.id));

    // Remove stale tracks
    for (const id in serverTracks) {
      if (!liveIds.has(Number(id))) delete serverTracks[id];
    }

    // Update / insert
    for (const t of tracks) {
      serverTracks[t.id] = {
        name:     t.name,
        box:      scaleBox(t.box),
        lastSeen: now,
      };
    }
  }

  // ── Pick primary (largest real face) ──────────────────────────────────────

  function choosePrimary() {
    let best = null;
    let bestArea = 0;
    for (const id in serverTracks) {
      const trk  = serverTracks[id];
      const area = trk.box.w * trk.box.h;
      const known = trk.name !== 'Unknown' && trk.name !== 'Spoof';
      if (!best || (known && !bestKnown) || (known === bestKnown && area > bestArea)) {
        best      = trk;
        bestArea  = area;
        bestKnown = known;
      }
    }
    return best;
  }
  let bestKnown = false;

  // ── Draw overlay ───────────────────────────────────────────────────────────

  function drawOverlay() {
    adjustOverlaySize();
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!autoMode) return;

    if (!cameraOnline) {
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, overlay.width, overlay.height);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 18px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Camera offline', overlay.width / 2, overlay.height / 2);
      ctx.restore();
      return;
    }

    for (const id in serverTracks) {
      const trk   = serverTracks[id];
      let { x, y, w, h } = trk.box;
      const label = trk.name;
      const color = label === 'Spoof' ? '#ff3b30'
                  : label === 'Unknown' ? '#ffcc00'
                  : '#00ff88';

      ctx.save();
      if (mirror) {
        ctx.translate(overlay.width, 0);
        ctx.scale(-1, 1);
        x = overlay.width - x - w;
      }

      const padX = w * 0.15;
      const padY = h * 0.15;
      x -= padX / 2; y -= padY / 2;
      w += padX;     h += padY;

      const cx = x + w / 2;
      const cy = y + h / 2 - h * 0.03;
      const rx = w * 0.50;
      const ry = w * 0.72;

      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 3;
      ctx.stroke();

      ctx.font         = 'bold 25px Arial';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'bottom';
      ctx.shadowColor  = 'rgba(0,0,0,0.85)';
      ctx.shadowBlur   = 5;
      ctx.fillStyle    = color;
      ctx.fillText(label, cx, cy - ry - 8);
      ctx.restore();
    }
  }

  // ── WebSocket setup ────────────────────────────────────────────────────────

  function connectWs() {
    ws = new WebSocket(WS_DISPLAY);

    ws.onopen = () => {
      console.log('[WS] connected to camera-display');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'status') {
        cameraOnline = data.online;
        if (!data.online) serverTracks = {};
        return;
      }

      // Display frame — arrives at camera FPS (CPU path, low latency)
      if (data.type === 'frame') {
        cameraOnline = true;
        drawFrame(data.frame_data);
        return;
      }

      // Inference result — arrives at GPU inference rate (slower)
      if (data.type === 'result' && autoMode) {
        cameraOnline = true;
        frameWidth  = data.frame_width  || frameWidth;
        frameHeight = data.frame_height || frameHeight;
        applyServerTracks(data.tracks || []);

        const primary = choosePrimary();
        const name    = primary ? primary.name : 'Unknown';
        updateRecognizedId(name);
        handleAutoAttendance(name);
      }
    };

    ws.onclose = () => {
      console.log('[WS] disconnected, reconnecting...');
      cameraOnline = false;
      serverTracks = {};
      setTimeout(connectWs, WS_RECONNECT_DELAY);
    };

    ws.onerror = (err) => {
      console.error('[WS] error:', err);
      ws.close();
    };
  }

  // ── Render loop ────────────────────────────────────────────────────────────

  function renderLoop() {
    drawOverlay();
    requestAnimationFrame(renderLoop);
  }

  // ── Draw server frame on videoCanvas ──────────────────────────────────────

  let videoCvCtx = null;

  function drawFrame(base64data) {
    if (!videoCvCtx) videoCvCtx = videoCanvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      // Only resize canvas when dimensions actually change — resizing every frame
      // clears the canvas and triggers browser layout reflow, causing stuttering.
      if (videoCanvas.width !== img.naturalWidth || videoCanvas.height !== img.naturalHeight) {
        videoCanvas.width  = img.naturalWidth;
        videoCanvas.height = img.naturalHeight;
        adjustOverlaySize();
      }
      videoCvCtx.drawImage(img, 0, 0);
    };
    img.src = 'data:image/jpeg;base64,' + base64data;
  }

  window.addEventListener('resize', adjustOverlaySize);

  // Đóng WebSocket ngay khi rời trang — đảm bảo server giải phóng camera
  window.addEventListener('beforeunload', () => {
    if (ws) { ws.onclose = null; ws.close(); }
  });

  setAutoMode(true);
  connectWs();
  requestAnimationFrame(renderLoop);
});
