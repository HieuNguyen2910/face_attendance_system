document.addEventListener('DOMContentLoaded', () => {
  const urls  = window.historyApiUrls;
  const thead = document.getElementById('historyThead');
  const tbody = document.getElementById('historyTBody');

  // ── Table header templates ────────────────────────────────
  const DAY_HEADERS = `<tr>
    <th>Ngày</th><th>ID</th><th>Họ tên</th>
    <th>Check-in</th><th>Trạng thái vào</th>
    <th>Check-out</th><th>Trạng thái ra</th>
  </tr>`;

  const MONTH_HEADERS = `<tr>
    <th>Họ tên</th><th>ID</th>
    <th>Số ngày chấm công</th>
    <th>Số ngày đúng giờ</th>
    <th id="thLate" style="cursor:pointer; user-select:none; white-space:nowrap;">
      Số ngày muộn <span id="lateSortIcon">↕</span>
    </th>
    <th>Lịch sử</th>
  </tr>`;

  // ── State ─────────────────────────────────────────────────
  let currentMode       = 'day';
  let lateSortAsc       = true;
  let cachedDayData     = null;   // raw records for day view
  let cachedMonthData   = null;   // aggregated per-employee for month view
  let searchQuery       = '';

  // ── Mode tabs ─────────────────────────────────────────────
  document.querySelectorAll('.hist-mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const mode = tab.dataset.mode;
      if (mode === currentMode) return;
      currentMode = mode;

      document.querySelectorAll('.hist-mode-tab').forEach(t => {
        const active = t === tab;
        t.style.borderBottomColor = active ? 'var(--primary)' : 'transparent';
        t.style.color      = active ? 'var(--primary)' : 'var(--text-muted)';
        t.style.fontWeight = active ? '600' : '400';
      });
      document.querySelectorAll('.hist-ctrl').forEach(c => {
        c.style.display = c.dataset.ctrlfor === mode ? 'flex' : 'none';
      });

      // clear search on tab switch
      document.getElementById('searchInput').value = '';
      searchQuery = '';

      if (mode === 'day') {
        thead.innerHTML = DAY_HEADERS;
        loadByDay();
      } else {
        thead.innerHTML = MONTH_HEADERS;
        attachLateSort();
        loadByMonth();
      }
    });
  });

  // ── Late-sort ─────────────────────────────────────────────
  function attachLateSort() {
    const th = document.getElementById('thLate');
    if (th) {
      th.addEventListener('click', () => {
        lateSortAsc = !lateSortAsc;
        document.getElementById('lateSortIcon').textContent = lateSortAsc ? '↑' : '↓';
        if (cachedMonthData) renderMonthRows(applyMonthSearch(cachedMonthData));
      });
    }
  }

  // ── Helpers ───────────────────────────────────────────────
  function statusClass(s) {
    if (s === 'Đúng giờ') return 'status on-time';
    if (s === 'Muộn')     return 'status late';
    return 'status';
  }
  function loading(cols) {
    tbody.innerHTML = `<tr><td colspan="${cols}" class="empty">Đang tải dữ liệu...</td></tr>`;
  }
  function noData(cols, msg) {
    tbody.innerHTML = `<tr><td colspan="${cols}" class="empty">${msg || 'Không có dữ liệu'}</td></tr>`;
  }

  // ── Search filters ────────────────────────────────────────
  function applyDaySearch(data) {
    if (!searchQuery) return data;
    const q = searchQuery.toLowerCase();
    return data.filter(r =>
      (r.user_id || '').toLowerCase().includes(q) ||
      (r.name    || '').toLowerCase().includes(q)
    );
  }

  function applyMonthSearch(employees) {
    if (!searchQuery) return employees;
    const q = searchQuery.toLowerCase();
    return employees.filter(e =>
      (e.user_id || '').toLowerCase().includes(q) ||
      (e.name    || '').toLowerCase().includes(q)
    );
  }

  // ── Render: day view ──────────────────────────────────────
  function renderDayRows(data, dateStr) {
    if (!data || data.length === 0) { noData(7); return; }
    tbody.innerHTML = '';
    for (const r of data) {
      const d = r._dateDisplay || dateStr || '';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${d}</td>
        <td>${r.user_id || ''}</td>
        <td>${r.name    || ''}</td>
        <td class="time-cell">${r.checkin  || '—'}</td>
        <td><span class="${statusClass(r.status_in)}">${r.status_in  || '—'}</span></td>
        <td class="time-cell">${r.checkout || '—'}</td>
        <td><span class="status">${r.status_out || '—'}</span></td>
      `;
      tbody.appendChild(tr);
    }
  }

  // ── Render: month summary view ────────────────────────────
  function aggregateMonth(records) {
    const map = {};
    for (const r of records) {
      if (!map[r.user_id]) {
        map[r.user_id] = { user_id: r.user_id, name: r.name, total: 0, onTime: 0, late: 0 };
      }
      const e = map[r.user_id];
      e.total++;
      if (r.status_in === 'Đúng giờ') e.onTime++;
      if (r.status_in === 'Muộn')     e.late++;
    }
    return Object.values(map);
  }

  function renderMonthRows(employees) {
    if (!employees || employees.length === 0) { noData(6); return; }
    const sorted = [...employees].sort((a, b) =>
      lateSortAsc ? a.late - b.late : b.late - a.late
    );
    tbody.innerHTML = '';
    for (const e of sorted) {
      const histUrl = urls.employeeHistory.replace('__ID__', encodeURIComponent(e.user_id));
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight:500;">${e.name    || ''}</td>
        <td>${e.user_id}</td>
        <td style="text-align:center;">${e.total}</td>
        <td style="text-align:center;"><span class="status on-time">${e.onTime}</span></td>
        <td style="text-align:center;"><span class="status${e.late > 0 ? ' late' : ''}">${e.late}</span></td>
        <td style="text-align:center;">
          <a href="${histUrl}" class="btn btn-outline"
             style="padding:4px 12px; font-size:12px; white-space:nowrap;">Xem</a>
        </td>
      `;
      tbody.appendChild(tr);
    }
  }

  // ── Load by day ───────────────────────────────────────────
  async function loadByDay() {
    const dd   = String(document.getElementById('dayInput').value || '').padStart(2, '0');
    const mm   = String(document.getElementById('monthDaySelect').value).padStart(2, '0');
    const yyyy = document.getElementById('yearDayInput').value;
    if (!dd || dd === '00' || !mm || !yyyy || String(yyyy).length !== 4) {
      noData(7, 'Ngày không hợp lệ'); return;
    }
    const apiDate    = `${yyyy}-${mm}-${dd}`;
    const displayDate = `${dd}/${mm}/${yyyy}`;
    loading(7);
    try {
      const res  = await fetch(`${urls.byDay}?date=${encodeURIComponent(apiDate)}`);
      const data = await res.json();
      if (data.status === 'ok') {
        // Attach display date to each record for search re-render
        cachedDayData = (data.data || []).map(r => ({ ...r, _dateDisplay: displayDate }));
        renderDayRows(applyDaySearch(cachedDayData));
      } else {
        cachedDayData = [];
        noData(7);
      }
    } catch { noData(7, 'Lỗi kết nối'); }
  }

  // ── Load by month ─────────────────────────────────────────
  async function loadByMonth() {
    const month = document.getElementById('monthSelect').value;
    const year  = document.getElementById('yearMonthInput').value;
    if (!month || !year) { noData(6, 'Vui lòng chọn tháng và năm'); return; }
    loading(6);
    cachedMonthData = null;
    lateSortAsc = true;
    const icon = document.getElementById('lateSortIcon');
    if (icon) icon.textContent = '↕';
    try {
      const res  = await fetch(`${urls.byMonth}?month=${month}&year=${encodeURIComponent(year)}`);
      const data = await res.json();
      if (data.status === 'ok') {
        cachedMonthData = aggregateMonth(data.data || []);
        renderMonthRows(applyMonthSearch(cachedMonthData));
      } else {
        noData(6);
      }
    } catch { noData(6, 'Lỗi kết nối'); }
  }

  // ── Search input ──────────────────────────────────────────
  document.getElementById('searchInput').addEventListener('input', function () {
    searchQuery = this.value.trim();
    if (currentMode === 'day' && cachedDayData) {
      renderDayRows(applyDaySearch(cachedDayData));
    } else if (currentMode === 'month' && cachedMonthData) {
      renderMonthRows(applyMonthSearch(cachedMonthData));
    }
  });

  // ── Pre-fill current date/month ───────────────────────────
  const now = new Date();
  document.getElementById('dayInput').value       = now.getDate();
  document.getElementById('monthDaySelect').value = now.getMonth() + 1;
  document.getElementById('yearDayInput').value   = now.getFullYear();
  document.getElementById('monthSelect').value    = now.getMonth() + 1;
  document.getElementById('yearMonthInput').value = now.getFullYear();

  // Auto-load today on page open
  loadByDay();

  // ── Form handlers ─────────────────────────────────────────
  document.getElementById('filterDayForm').addEventListener('submit', e => {
    e.preventDefault();
    document.getElementById('searchInput').value = '';
    searchQuery = '';
    loadByDay();
  });
  document.getElementById('filterMonthForm').addEventListener('submit', e => {
    e.preventDefault();
    document.getElementById('searchInput').value = '';
    searchQuery = '';
    loadByMonth();
  });
});
