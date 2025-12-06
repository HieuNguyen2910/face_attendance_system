document.addEventListener('DOMContentLoaded', () => {
  const table = document.querySelector('#historyTable');
  const tableBody = table.querySelector('tbody');
  const apiUrl = table.dataset.apiUrl;

  const dayInput = document.getElementById('dayInput');
  const monthInput = document.getElementById('monthInput');
  const yearInput = document.getElementById('yearInput');

  // Hàm ghép dd/mm/yyyy từ 3 input
  function getDateString() {
    return `${dayInput.value.padStart(2,'0')}/${monthInput.value.padStart(2,'0')}/${yearInput.value}`;
  }

  // Hàm chuyển dd/mm/yyyy -> yyyy-mm-dd
  function formatDateForApi(dateStr) {
    const parts = dateStr.split('/');
    if (parts.length !== 3) return null;
    return `${parts[2]}-${parts[1].padStart(2,'0')}-${parts[0].padStart(2,'0')}`;
  }

  // Điền ngày hiện tại khi load trang
  const now = new Date();
  dayInput.value = now.getDate().toString().padStart(2,'0');
  monthInput.value = (now.getMonth()+1).toString().padStart(2,'0');
  yearInput.value = now.getFullYear();

  // ================= Load dữ liệu theo ngày =================
  async function loadHistoryByDay() {
    const dateStr = getDateString();
    const apiDate = formatDateForApi(dateStr);
    if (!apiDate) {
      tableBody.innerHTML = '<tr><td colspan="6" class="empty">Ngày không hợp lệ</td></tr>';
      return;
    }

    tableBody.innerHTML = '<tr><td colspan="6" class="empty">Đang tải dữ liệu...</td></tr>';

    try {
      const res = await fetch(`${apiUrl}?date=${encodeURIComponent(apiDate)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      tableBody.innerHTML = '';
      if (data.status === 'ok' && data.data && data.data.length > 0) {
        for (const info of data.data) {
          const row = document.createElement('tr');
          row.innerHTML = `
            <td>${dateStr}</td>
            <td>${info.user_id}</td>
            <td>${info.checkin || '-'}</td>
            <td>${info.status_in || '-'}</td>
            <td>${info.checkout || '-'}</td>
            <td>${info.status_out || '-'}</td>
          `;
          tableBody.appendChild(row);
        }
      } else {
        tableBody.innerHTML = '<tr><td colspan="6" class="empty">Chưa có dữ liệu</td></tr>';
      }
    } catch (err) {
      console.error(err);
      tableBody.innerHTML = '<tr><td colspan="6" class="empty">Lỗi tải dữ liệu</td></tr>';
    }
  }

  // Load mặc định ngày hôm nay
  loadHistoryByDay();

  // Xem theo ngày
  document.getElementById('filterDateForm').addEventListener('submit', e => {
    e.preventDefault();
    loadHistoryByDay();
  });

  // Xem theo ID
  document.getElementById('searchByIdForm').addEventListener('submit', async e => {
      e.preventDefault();
      const userId = document.getElementById('userIdInput').value.trim();
      if (!userId) return;

      try {
          // Gọi API đúng URL
          const res = await fetch(`/attendance/api/check_user/${encodeURIComponent(userId)}/`);
          const data = await res.json();

          if (data.exists) {
              // ID tồn tại → chuyển sang trang chi tiết
              window.location.href = `/history/id/${encodeURIComponent(userId)}/`;
          } else {
              // ID không tồn tại → báo lỗi
              alert('ID không tồn tại');
          }
      } catch (err) {
          console.error(err);
          alert('Lỗi kết nối tới server');
      }
  });


});
