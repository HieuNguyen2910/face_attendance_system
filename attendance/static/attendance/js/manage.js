// static/attendance/js/manage.js
// Quản lý giao diện Manage - add / edit / delete + camera capture
// Tác giả: (sửa theo yêu cầu của bạn)

(() => {
  // Root elements
  const tbody = document.getElementById('userTBody');
  const btnAdd = document.getElementById('btnAdd');
  const btnEdit = document.getElementById('btnEdit');
  const btnDelete = document.getElementById('btnDelete');
  const searchInput = document.getElementById('searchInput');
  const modalRoot = document.getElementById('modalRoot');

  // In-memory users map: { id: { name, position, vectors? } }
  let users = {};
  let selectedRowId = null;

  // --- Helpers: modal creation ---
  function showModal(html, opts = {}) {
    modalRoot.style.display = 'block';
    const wrap = document.createElement('div');
    wrap.className = 'app-modal';
    wrap.innerHTML = `<div class="modal-box">${html}</div>`;
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
        .modal-btn {
          padding: 8px 14px;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          font-size: 0.9rem;
          font-weight: 500;
          transition: 0.15s;
        }

        .modal-btn.primary {
          background: #007bff;
          color: white;
        }
        .modal-btn.primary:hover {
          background: #0063d6;
        }

        .modal-btn.warning {
          background: #ff9800;
          color: white;
        }
        .modal-btn.warning:hover {
          background: #e68900;
        }

        .modal-btn.danger {
          background: #dc3545;
          color: white;
        }
        .modal-btn.danger:hover {
          background: #b02a37;
        }
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
        .modal-btn {
          padding: 8px 14px;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          font-size: 0.9rem;
          font-weight: 500;
          transition: 0.15s;
        }

        .modal-btn.primary {
          background: #007bff;
          color: white;
        }
        .modal-btn.primary:hover {
          background: #0063d6;
        }

        .modal-btn.warning {
          background: #ff9800;
          color: white;
        }
        .modal-btn.warning:hover {
          background: #e68900;
        }

        /* Optional: style input trong prompt */
        #promptInput {
          width: 100%;
          padding: 8px 10px;
          border-radius: 6px;
          border: 1px solid #ccc;
          font-size: 0.9rem;
          margin-top: 6px;
          box-sizing: border-box;
        }
        #promptInput:focus {
          border-color: #007bff;
          box-shadow: 0 0 0 2px rgba(0,123,255,0.2);
          outline: none;
        }
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

  // --- Camera modal & capture routine ---
  function createCameraUI(title = 'Đăng ký khuôn mặt') {
    const html = `
      <style>
        #camWrapper {
          text-align: center;
          padding: 10px;
        }

        #camWrapper h3 {
          margin-bottom: 10px;
          font-size: 1.2rem;
          font-weight: 600;
        }

        #regVideo {
          width: 100%;
          max-height: 480px;
          background: #000;
          border-radius: 8px;
          object-fit: cover;
          transform: scaleX(-1) !important;
        }

        #camInstruction {
          font-weight: 600;
          margin: 10px 0;
          font-size: 1rem;
          color: #333;
        }

        #camActions {
          display: flex;
          gap: 10px;
          justify-content: center;
          margin-top: 10px;
        }

        #camCancel {
          padding: 8px 14px;
          border: none;
          border-radius: 6px;
          background: #ff4444;
          color: #fff;
          cursor: pointer;
          font-size: 0.9rem;
          transition: 0.2s;
        }

        #camCancel:hover {
          background: #d63030;
        }
      </style>

      <div id="camWrapper">
        <h3 id="camTitle">${title}</h3>

        <video id="regVideo" autoplay playsinline></video>

        <p id="camInstruction">Chuẩn bị...</p>

        <div id="camActions">
          <button id="camCancel">Huỷ</button>
        </div>
      </div>
    `;

    return showModal(html);
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

  async function captureThreeFacesWithCountdown(camModal) {
    const videoEl = camModal.container.querySelector('#regVideo');
    const instructionEl = camModal.container.querySelector('#camInstruction');
    const cancelBtn = camModal.container.querySelector('#camCancel');

    let stream;
    try {
      stream = await startCamera(videoEl);
    } catch (err) {
      camModal.close();
      throw new Error('Không thể truy cập camera: ' + err.message);
    }

    let cancelled = false;
    cancelBtn.onclick = () => { cancelled = true; try { stream.getTracks().forEach(t => t.stop()); } catch(e){}; camModal.close(); };

    const steps = [
      { label: 'Nhìn CHÍNH DIỆN (giữ cố định)', seconds: 5 },
      { label: 'Nhìn SANG TRÁI', seconds: 5 },
      { label: 'Nhìn SANG PHẢI', seconds: 5 }
    ];

    await new Promise(r => setTimeout(r, 500));

    const blobs = [];
    for (let i = 0; i < steps.length && !cancelled; i++) {
      const step = steps[i];
      const start = Date.now();
      while (Date.now() - start < step.seconds * 1000 && !cancelled) {
        const remain = Math.ceil((step.seconds * 1000 - (Date.now() - start)) / 1000);
        instructionEl.textContent = `${step.label} — còn ${remain} giây...`;
        await new Promise(r => setTimeout(r, 250));
      }
      if (cancelled) break;
      instructionEl.textContent = `Đang chụp ${step.label}...`;
      const blob = await captureBlobFromVideo(videoEl);
      if (!blob) {
        try { stream.getTracks().forEach(t => t.stop()); } catch(e){}
        camModal.close();
        throw new Error('Không chụp được ảnh, thử lại');
      }
      blobs.push(blob);
      await new Promise(r => setTimeout(r, 350));
    }

    try { stream.getTracks().forEach(t => t.stop()); } catch(e){}
    camModal.close();
    if (cancelled) throw new Error('Người dùng hủy camera');
    return blobs;
  }

  // --- API calls (dùng window.apiUrls từ HTML) ---
  async function apiFetchUsers() {
    try {
      const res = await fetch(window.apiUrls.listUsers);
      const json = await res.json();
      // api trả về { status: 'ok', users: [...] } hoặc trực tiếp users array
      if (json && json.status === 'ok' && Array.isArray(json.users)) {
        return json.users;
      }
      if (Array.isArray(json)) return json;
      if (json && Array.isArray(json.users)) return json.users;
      // fallback: nếu là object map (cũ)
      return json.users || json;
    } catch (e) {
      console.error('apiFetchUsers lỗi:', e);
      return [];
    }
  }

  async function apiRegisterEmployee({ user_id, name, position, images }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    fd.append('name', name);
    fd.append('position', position);
    images.forEach((b, i) => fd.append('image', b, `img${i}.jpg`));
    const res = await fetch(window.apiUrls.registerEmployee, { method: 'POST', body: fd });
    return await res.json();
  }

  async function apiUpdateUser({ user_id, name, position }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    fd.append('name', name);
    fd.append('position', position);
    const res = await fetch(window.apiUrls.updateUser, { method: 'POST', body: fd });
    return await res.json();
  }

  async function apiDeleteUser({ user_id }) {
    const fd = new FormData();
    fd.append('user_id', user_id);
    const res = await fetch(window.apiUrls.deleteUser, { method: 'POST', body: fd });
    return await res.json();
  }

  // --- Render table ---
  function renderTable(filter = '') {
    tbody.innerHTML = '';
    const f = filter.trim().toLowerCase();
    
    Object.keys(users).sort((a,b)=>a.localeCompare(b)).forEach(id => {
      const meta = users[id] || {};
      const name = meta.name || meta.ten_nv || '';
      const pos = meta.position || meta.chuc_vu || '';
      const rowText = `${id} ${name} ${pos}`.toLowerCase();
      if (f && !rowText.includes(f)) return;
      const tr = document.createElement('tr');
      tr.dataset.id = id;
      tr.innerHTML = `
        <td class="col-id">${id}</td>
        <td class="col-name">${name}</td>
        <td class="col-pos">${pos}</td>
      `;
      tr.onclick = () => {
        document.querySelectorAll('#userTBody tr').forEach(r => r.classList.remove('selected'));
        tr.classList.add('selected');
        selectedRowId = id;
      };
      // tr.querySelector('.selectRowBtn').onclick = (ev) => {
      //   ev.stopPropagation();
      //   document.querySelectorAll('#userTBody tr').forEach(r => r.classList.remove('selected'));
      //   tr.classList.add('selected');
      //   selectedRowId = id;
      // };
      tbody.appendChild(tr);
    });
  }

  // --- Operations: Add / Edit / Delete ---
  function openAddForm() {
    const html = `
      <style>
        .modal-btn {
          padding: 8px 14px;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          font-size: 0.9rem;
          font-weight: 500;
          transition: 0.15s;
        }

        .modal-btn.primary {
          background: #007bff;
          color: white;
        }
        .modal-btn.primary:hover {
          background: #0063d6;
        }

        .modal-btn.warning {
          background: #ff9800;
          color: white;
        }
        .modal-btn.warning:hover {
          background: #e68900;
        }
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
        <input id="inpPos" placeholder="Chức vụ / phòng ban">
      </div>

      <div class="modal-actions">
        <button id="btnCancel" class="modal-btn warning">Hủy</button>
        <button id="btnContinue" class="modal-btn primary">Tiếp tục (Mở camera)</button>
      </div>
    `;

    const m = showModal(html);
    m.container.querySelector('#btnCancel').onclick = () => m.close();
    m.container.querySelector('#btnContinue').onclick = async () => {
      const id = m.container.querySelector('#inpId').value.trim();
      const name = m.container.querySelector('#inpName').value.trim();
      const pos = m.container.querySelector('#inpPos').value.trim();
      if (!id || !name || !pos) return alert('Vui lòng nhập đầy đủ ID, họ tên và vị trí');
      m.close();
      const cam = createCameraUI('Chụp ảnh đăng ký cho ' + name);
      try {
        const blobs = await captureThreeFacesWithCountdown(cam);
        const res = await apiRegisterEmployee({ user_id: id, name, position: pos, images: blobs });
        if (res.status === 'ok') {
          alert('Đăng ký thành công: ' + name);
          await refreshUsers();
        } else {
          alert('Đăng ký lỗi: ' + (res.message || JSON.stringify(res)));
        }
      } catch (err) {
        alert(err.message || String(err));
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

  function openEditForm(id) {
    const meta = users[id] || {};
    const html = `
      <style>
        .modal-btn {
          padding: 8px 14px;
          border-radius: 6px;
          border: none;
          cursor: pointer;
          font-size: 0.9rem;
          font-weight: 500;
          transition: 0.15s;
        }

        .modal-btn.primary {
          background: #007bff;
          color: white;
        }
        .modal-btn.primary:hover {
          background: #0063d6;
        }

        .modal-btn.warning {
          background: #ff9800;
          color: white;
        }
        .modal-btn.warning:hover {
          background: #e68900;
        }
      </style>

      <div><strong>Sửa nhân viên</strong></div>

      <div class="modal-row" style="margin-top:8px">
        <label style="width:80px">Mã ID</label>
        <input id="inpId" value="${id}" disabled>
      </div>

      <div class="modal-row">
        <label style="width:80px">Họ tên</label>
        <input id="inpName" value="${escapeHtml(meta.name || meta.ten_nv || '')}">
      </div>

      <div class="modal-row">
        <label style="width:80px">Vị trí</label>
        <input id="inpPos" value="${escapeHtml(meta.position || meta.chuc_vu || '')}">
      </div>

      <div style="margin-top:6px;color:#666;font-size:0.9rem">
        Bạn có thể chỉ sửa thông tin (name/position) hoặc muốn cập nhật ảnh thì dùng 'Thay ảnh' bên dưới.
      </div>

      <div class="modal-actions">
        <button id="btnCancel" class="modal-btn warning">Hủy</button>
        <button id="btnReplace" class="modal-btn primary">Lưu thay đổi</button>
        <button id="btnReplacePhoto" class="modal-btn primary">Thay ảnh (mở camera)</button>
      </div>
    `;
    const m = showModal(html);
    m.container.querySelector('#btnCancel').onclick = () => m.close();
    m.container.querySelector('#btnReplace').onclick = async () => {
      const name = m.container.querySelector('#inpName').value.trim();
      const pos = m.container.querySelector('#inpPos').value.trim();
      if (!name || !pos) return alert('Vui lòng nhập đầy đủ họ tên & vị trí');
      try {
        const res = await apiUpdateUser({ user_id: id, name, position: pos });
        if (res.status === 'ok') {
          alert('Cập nhật thành công');
          m.close();
          await refreshUsers();
        } else {
          alert('Cập nhật lỗi: ' + (res.message || JSON.stringify(res)));
        }
      } catch (err) {
        alert('Lỗi khi cập nhật: ' + err.message);
      }
    };
    m.container.querySelector('#btnReplacePhoto').onclick = async () => {
      const cam = createCameraUI('Chụp ảnh mới cho ' + (meta.name || id));
      try {
        const blobs = await captureThreeFacesWithCountdown(cam);
        const res = await apiRegisterEmployee({ user_id: id, name: meta.name || '', position: meta.position || '', images: blobs });
        if (res.status === 'ok') {
          alert('Cập nhật ảnh thành công');
          m.close();
          await refreshUsers();
        } else {
          alert('Lỗi cập nhật ảnh: ' + (res.message || JSON.stringify(res)));
        }
      } catch (err) {
        alert(err.message || String(err));
      }
    };
  }

  function openDeleteSelectedOrPrompt() {
    if (selectedRowId) {
      const meta = users[selectedRowId] || {};
      showConfirm(`Bạn có chắc muốn xóa ID: <strong>${selectedRowId}</strong> — ${meta.name || ''}?`, async () => {
        try {
          const res = await apiDeleteUser({ user_id: selectedRowId });
          if (res.status === 'ok') {
            alert('Đã xóa ' + selectedRowId);
            selectedRowId = null;
            await refreshUsers();
          } else {
            alert('Xóa lỗi: ' + (res.message || JSON.stringify(res)));
          }
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
            if (res.status === 'ok') {
              alert('Đã xóa ' + id);
              selectedRowId = null;
              await refreshUsers();
            } else {
              alert('Xóa lỗi: ' + (res.message || JSON.stringify(res)));
            }
          } catch (err) { alert('Lỗi xóa: ' + err.message); }
        });
      });
    }
  }

  // --- Utilities ---
  function escapeHtml(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // --- Refresh users map & render ---
  async function refreshUsers() {
    const raw = await apiFetchUsers(); // raw nhiều khả năng là array
    const out = {};

    try {
      if (Array.isArray(raw)) {
        // raw = [{user_id, name, position}, ...]
        raw.forEach(u => {
          if (!u) return;
          const id = u.user_id || u.id || String(u[0] || '').trim();
          if (!id) return;
          out[id] = {
            name: u.name || u.ten_nv || '',
            position: u.position || u.chuc_vu || '',
            vectors: u.vectors || []
          };
        });
      } else if (typeof raw === 'object' && raw !== null) {
        // raw is already a map: { id: {name, position, ...}, ... }
        Object.keys(raw).forEach(k => {
          const v = raw[k];
          if (Array.isArray(v)) {
            out[k] = { name: '', position: '', vectors: v };
          } else if (typeof v === 'object' && v !== null) {
            out[k] = {
              name: v.name || v.ten_nv || '',
              position: v.position || v.chuc_vu || '',
              vectors: v.vectors || []
            };
          } else {
            out[k] = { name: String(v || ''), position: '' };
          }
        });
      }
    } catch (e) {
      console.error('refreshUsers parse error:', e);
    }

    users = out;
    selectedRowId = null; // reset selection when refresh
    renderTable(searchInput.value || '');
  }

  // --- Event bindings ---
  btnAdd.onclick = () => openAddForm();
  btnEdit.onclick = () => openEditSelectedOrPrompt();
  btnDelete.onclick = () => openDeleteSelectedOrPrompt();
  searchInput.oninput = () => renderTable(searchInput.value);

  // --- Init ---
  refreshUsers();

})();
