document.addEventListener('DOMContentLoaded', () => {
  // ======= DOM ELEMENTS =======
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const ctx = overlay.getContext('2d');
  const recognizedIdEl = document.getElementById('recognizedId');
  const checkinBtn = document.getElementById('checkinBtn');
  const checkoutBtn = document.getElementById('checkoutBtn');
  const realtimeClock = document.getElementById('realtimeClock');
  const statusBox = document.getElementById('statusBox');

  // ======= RECOGNITION & TRACKING STATE =======
  let lastRecognition = { name: "Unknown", box: null };
  let lastRecognizedName = "Unknown";
  let hasCheckedIn = false;

  // Object tracking khi server không trả bbox
  let trackerBox = null;
  let trackerName = "Unknown";
  let velocity = { vx: 0, vy: 0 };
  let missingFrames = 0;
  const MAX_MISSING = 10; // số frame tối đa dùng tracking

  // ======= CONFIG =======
  const RESIZE_WIDTH = 640;
  const REQUEST_INTERVAL = 100; // ms
  let lastRequestTime = 0;
  let isRequesting = false;

  // ======= TMP CANVAS =======
  const tmpCanvas = document.createElement('canvas');
  const tmpCtx = tmpCanvas.getContext('2d');

  // ======= HELPER FUNCTIONS =======
  function videoReady() {
    return video && video.readyState >= 2;
  }

  function canvasToBlob(canvas) {
    return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
  }

  function updateClock() {
    const now = new Date();
    realtimeClock.textContent = now.toLocaleTimeString();
  }
  setInterval(updateClock, 1000);
  updateClock();

  function adjustOverlaySize() {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    overlay.style.width = video.clientWidth + "px";
    overlay.style.height = video.clientHeight + "px";
  }

  // ======= BUTTONS =======
  function updateButtons(name) {
    if (name === "Unknown") {
      checkinBtn.disabled = true;
      checkoutBtn.disabled = true;
    } else if (!hasCheckedIn) {
      checkinBtn.disabled = false;
      checkoutBtn.disabled = true;
    } else {
      checkinBtn.disabled = false;
      checkoutBtn.disabled = false;
    }
    checkinBtn.style.opacity = checkinBtn.disabled ? 0.6 : 1;
    checkoutBtn.style.opacity = checkoutBtn.disabled ? 0.6 : 1;
  }

  async function checkUserCheckinState(userId) {
    if (!userId || userId === "Unknown") {
      hasCheckedIn = false;
      updateButtons(userId);
      return;
    }
    try {
      const fd = new FormData();
      fd.append("user_id", userId);
      const res = await fetch(checkinStatusUrl, { method: "POST", body: fd });
      const data = await res.json();
      hasCheckedIn = (data.status === "already");
      updateButtons(userId);
    } catch(err) {
      hasCheckedIn = false;
      updateButtons(userId);
    }
  }

  async function updateRecognizedId(name) {
    recognizedIdEl.textContent = name;
    recognizedIdEl.classList.remove('recognized', 'unknown');
    recognizedIdEl.classList.add(name === "Unknown" ? 'unknown' : 'recognized');

    if (name !== lastRecognizedName) {
      lastRecognizedName = name;
      await checkUserCheckinState(name);
    } else {
      updateButtons(name);
    }
  }

  // ======= OBJECT TRACKING (MƯỢT NHẸ, KO LỆCH BBOX) =======
  function updateTracker(box, nameFromServer) {
    if (box) {
      // Server trả bbox → reset tracker
      trackerBox = { ...box };
      trackerName = nameFromServer;
      velocity = { vx: 0, vy: 0 };
      missingFrames = 0;
    } else if (trackerBox) {
      // Server không trả bbox → dùng tracking
      missingFrames++;
      if (missingFrames > MAX_MISSING) {
        trackerBox = null;
        trackerName = "Unknown";
        velocity = { vx: 0, vy: 0 };
      } else {
        // Dự đoán theo velocity (mượt, damping 0.8)
        trackerBox.x += velocity.vx;
        trackerBox.y += velocity.vy;
        velocity.vx *= 0.8;
        velocity.vy *= 0.8;
      }
    }
    // Cập nhật lastRecognition để vẽ
    lastRecognition.box = trackerBox;
    lastRecognition.name = trackerBox ? trackerName : "Unknown";
  }

  function calculateVelocity(newBox) {
    if (!trackerBox) return;
    const prevCx = trackerBox.x + trackerBox.w / 2;
    const prevCy = trackerBox.y + trackerBox.h / 2;
    const newCx = newBox.x + newBox.w / 2;
    const newCy = newBox.y + newBox.h / 2;
    velocity.vx = newCx - prevCx;
    velocity.vy = newCy - prevCy;
  }

  // ======= DRAW OVERLAY =======
  function drawOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const box = lastRecognition.box;
    if (!box) return;

    let { x, y, w, h } = box;

    // Nếu video mirror
    const mirror = true; // set true nếu video là selfie
    ctx.save();
    if (mirror) {
      ctx.translate(overlay.width, 0);
      ctx.scale(-1, 1);
      x = overlay.width - x - w; // mirror box
    }

    const padX = w * 0.1;
    const padY = h * 0.1;
    x -= padX / 2;
    y -= padY / 2;
    w += padX;
    h += padY;

    ctx.beginPath();
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;
    ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Vẽ text đúng chiều
    ctx.fillStyle = "lime";
    ctx.font = "28px Arial";
    ctx.textAlign = "center";
    ctx.fillText(lastRecognition.name, x + w / 2, y - 15);

    ctx.restore();
  }


  // ======= SERVER REQUEST =======
  async function maybeSendRequest() {
    const now = Date.now();
    if (isRequesting || now - lastRequestTime < REQUEST_INTERVAL) return;
    if (!videoReady()) return;

    lastRequestTime = now;
    isRequesting = true;

    const scale = RESIZE_WIDTH / video.videoWidth;
    tmpCanvas.width = RESIZE_WIDTH;
    tmpCanvas.height = Math.round(video.videoHeight * scale);
    tmpCtx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);

    try {
      const blob = await canvasToBlob(tmpCanvas);
      if (!blob) throw new Error("Failed to create blob");

      const fd = new FormData();
      fd.append('image', blob, 'frame.jpg');

      const res = await fetch(recognizeUrl, { method: 'POST', body: fd });
      const data = await res.json();

      if (data.status === 'ok' && data.box) {
        const scaleX = video.videoWidth / tmpCanvas.width;
        const scaleY = video.videoHeight / tmpCanvas.height;
        const serverBox = {
          x: data.box.x * scaleX,
          y: data.box.y * scaleY,
          w: data.box.w * scaleX,
          h: data.box.h * scaleY
        };
        calculateVelocity(serverBox);
        updateTracker(serverBox, data.name);
        await updateRecognizedId(data.name);
      } else {
        updateTracker(null, "Unknown");
        await updateRecognizedId("Unknown");
      }
    } catch (err) {
      console.error(err);
    } finally {
      isRequesting = false;
    }
  }

  // ======= MAIN LOOP =======
  async function realtimeLoop() {
    drawOverlay();
    maybeSendRequest();
    requestAnimationFrame(realtimeLoop);
  }

  // ======= CHECK-IN / CHECK-OUT =======
  async function handleCheck(url) {
    const userId = lastRecognition.name !== "Unknown" ? lastRecognition.name : null;
    if (!userId) {
      statusBox.textContent = "❌ Chưa nhận diện được ID!";
      return;
    }
    const fd = new FormData();
    fd.append('user_id', userId);
    try {
      const res = await fetch(url, { method: 'POST', body: fd });
      const data = await res.json();
      if (data.status === "ok" || data.status === "already") {
        hasCheckedIn = (url === checkinUrl);
        statusBox.textContent = data.message || `${userId} ${url === checkinUrl ? 'check-in' : 'check-out'} lúc ${data.time}`;
      } else {
        statusBox.textContent = `❌ Lỗi: ${data.message || "Không xác định"}`;
      }
      updateButtons(userId);
    } catch (err) {
      statusBox.textContent = "❌ Lỗi server: " + err;
    }
  }

  checkinBtn.addEventListener('click', () => handleCheck(checkinUrl));
  checkoutBtn.addEventListener('click', () => handleCheck(checkoutUrl));

  // ======= START CAMERA =======
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.addEventListener('loadeddata', () => {
        adjustOverlaySize();
        requestAnimationFrame(realtimeLoop);
      });
      window.addEventListener('resize', adjustOverlaySize);
    } catch (err) {
      recognizedIdEl.textContent = "Không thể mở camera: " + err;
    }
  }
  startCamera();

  // ======= LOAD HISTORY TABLE =======
  const table = document.querySelector('#historyTable');
  if (table) {
    const tableBody = table.querySelector('tbody');
    const apiHistoryUrl = table.dataset.apiUrl;
    async function loadHistory() {
      try {
        const res = await fetch(apiHistoryUrl);
        const data = await res.json();
        tableBody.innerHTML = '';
        if (data.status === 'ok' && data.data.length > 0) {
          data.data.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${item.date || item[0]}</td><td>${item.user_id || item[1]}</td>`;
            tableBody.appendChild(row);
          });
        } else {
          tableBody.innerHTML = '<tr><td colspan="2">Chưa có dữ liệu</td></tr>';
        }
      } catch (err) {
        tableBody.innerHTML = '<tr><td colspan="2">Lỗi tải dữ liệu</td></tr>';
      }
    }
    loadHistory();
  }
});
