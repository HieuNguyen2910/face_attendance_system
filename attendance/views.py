import json

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

from .models import Employee, Embedding, Attendance, WorkSchedule
from . import face_recognition as fr
from . import consumers as _consumers


# ========== VIEW HTML ==========

def home(request):
    users = Employee.objects.all()
    return render(request, "attendance/index.html", {"nhanviens": users})


def manage(request):
    users = Employee.objects.all()
    return render(request, "attendance/manage.html", {"nhanviens": users})


def history(request):
    """
    Hiển thị lịch sử tất cả các ngày từ DB
    """
    rows = Attendance.objects.select_related("user").order_by("-date")
    return render(request, "attendance/history.html", {"chamcongs": rows})


def history_by_id(request, user_id):
    """
    Lịch sử chấm công 1 người
    """
    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "User not found"})

    records = Attendance.objects.filter(user=user).order_by("-date")

    return render(request, "attendance/history_id.html", {
        "user_id": user.user_id,
        "name": user.name,
        "position": user.position,
        "records": records
    })


# ========== API ==========

@csrf_exempt
def api_recognize(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    if "image" not in request.FILES:
        return JsonResponse({"status": "fail", "message": "No image"})

    img = request.FILES['image'].read()
    res = fr.recognize_from_image_bytes_with_box(img)
    return JsonResponse(res)


@csrf_exempt
def api_detect_face(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    if "image" not in request.FILES:
        return JsonResponse({"status": "no_face"})
    img = request.FILES["image"].read()
    return JsonResponse(fr.detect_face_size_for_preview(img))


@csrf_exempt
def api_register(request):
    """
    Đăng ký embedding cho nhân viên (không thêm metadata).
    Nếu user chưa có trong Employee → lỗi.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    images = request.FILES.getlist("image")

    if not user_id or not images:
        return JsonResponse({"status": "fail", "message": "Missing fields"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "User not found"})

    vectors = []
    for img in images:
        emb = fr.embedding_from_image_bytes(img.read())
        if emb is not None:
            vectors.append(emb.tolist())

    if not vectors:
        return JsonResponse({"status": "fail", "message": "Không lấy được embedding"})

    # Lưu từng vector vào bảng Embeddings
    for vec in vectors:
        Embedding.objects.create(user=user, vector=json.dumps(vec))

    return JsonResponse({
        "status": "ok",
        "saved_vectors": len(vectors),
        "message": f"Đã lưu {len(vectors)} embedding cho {user_id}",
        "user_id": user_id
    })


@csrf_exempt
def api_register_employee(request):
    """
    Tạo/cập nhật nhân viên. Nếu có images → thay toàn bộ embedding.
    Nếu không có images → chỉ tạo/cập nhật thông tin (embedding thu sau qua api_register_frame).
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    name = request.POST.get("name")
    position = request.POST.get("position")
    department = request.POST.get("department", "").strip()
    images = request.FILES.getlist("image")

    if not (user_id and name and position):
        return JsonResponse({"status": "fail", "message": "Missing fields"})

    user, _ = Employee.objects.get_or_create(
        user_id=user_id,
        defaults={"name": name, "position": position, "department": department}
    )
    user.name = name
    user.position = position
    user.department = department
    user.save()

    # Tạo/cập nhật tài khoản CustomUser nếu có password
    password = request.POST.get("password", "").strip()
    if password:
        from django.contrib.auth.hashers import make_password
        from .models import CustomUser
        CustomUser.objects.update_or_create(
            employee=user,
            defaults={
                "username": user_id,
                "password": make_password(password),
                "role": "employee",
                "is_active": True,
            }
        )

    if not images:
        return JsonResponse({
            "status": "ok",
            "message": f"Đã tạo nhân viên {user_id}",
            "user_id": user_id,
        })

    # Có images → thay toàn bộ embedding (batch mode cũ)
    Embedding.objects.filter(user=user).delete()
    vectors = []
    for img in images:
        emb = fr.embedding_from_image_bytes(img.read())
        if emb is not None:
            vectors.append(emb.tolist())

    if not vectors:
        return JsonResponse({"status": "fail", "message": "Không lấy được embedding từ ảnh đã cung cấp"})

    for vec in vectors:
        Embedding.objects.create(user=user, vector=json.dumps(vec))

    return JsonResponse({
        "status": "ok",
        "saved_vectors": len(vectors),
        "message": f"Đã đăng ký {user_id} với {len(vectors)} embedding",
        "user_id": user_id,
    })


@csrf_exempt
def api_register_frame(request):
    """Thu một frame, trích embedding, so sánh đa dạng, lưu nếu đủ khác biệt."""
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    user_id = request.POST.get("user_id")
    if not user_id:
        return JsonResponse({"status": "fail", "message": "user_id required"})
    if "image" not in request.FILES:
        return JsonResponse({"status": "no_face"})
    img = request.FILES["image"].read()
    return JsonResponse(fr.check_and_register_frame(img, user_id))


@csrf_exempt
def api_clear_embeddings(request):
    """Xóa toàn bộ embedding của một nhân viên."""
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    user_id = request.POST.get("user_id")
    if not user_id:
        return JsonResponse({"status": "fail", "message": "user_id required"})
    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "User not found"})
    deleted, _ = Embedding.objects.filter(user=user).delete()
    return JsonResponse({"status": "ok", "deleted": deleted})


@csrf_exempt
def api_checkin(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    if not user_id:
        return JsonResponse({"status": "fail", "message": "user_id required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "User not found"})

    now = timezone.localtime()
    today = now.date()

    record, created = Attendance.objects.get_or_create(
        user=user,
        date=today
    )

    if record.checkin:
        return JsonResponse({
            "status": "already",
            "time": record.checkin.strftime("%H:%M:%S"),
            "message": f"{user.user_id} đã check-in lúc {record.checkin.strftime('%H:%M:%S')}"
        })

    schedule = WorkSchedule.get()
    cutoff = schedule.start_time
    current = now.time().replace(second=0, microsecond=0)
    status_in = "Đúng giờ" if current <= cutoff else "Muộn"

    record.checkin = now.time()
    record.status_in = status_in
    record.save()

    return JsonResponse({
        "status": "ok",
        "time": now.strftime("%H:%M:%S"),
        "status_in": status_in
    })


@csrf_exempt
def api_checkout(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    if not user_id:
        return JsonResponse({"status": "fail", "message": "user_id required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "User not found"})

    now = timezone.localtime()
    today = now.date()

    record, created = Attendance.objects.get_or_create(
        user=user,
        date=today
    )

    schedule = WorkSchedule.get()
    current = now.time().replace(second=0, microsecond=0)
    status_out = "Sớm" if current < schedule.end_time else "Bình thường"

    record.checkout = now.time()
    record.status_out = status_out
    record.save()

    return JsonResponse({
        "status": "ok",
        "time": now.strftime("%H:%M:%S"),
        "status_out": status_out
    })

@csrf_exempt
def api_employee_self_attendance(request):
    """
    Nhân viên tự chấm công: gửi ảnh → nhận diện → xác minh khớp session → ghi nhận.
    Trả lỗi nếu không nhận diện được hoặc khuôn mặt không khớp tài khoản.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    from .auth_views import get_current_user
    user = get_current_user(request)
    if not user or not user.employee:
        return JsonResponse({"status": "fail", "message": "Chưa đăng nhập hoặc không phải nhân viên"})

    employee = user.employee

    if "image" not in request.FILES:
        return JsonResponse({"status": "fail", "message": "Thiếu ảnh"})

    img_bytes = request.FILES["image"].read()
    res = fr.recognize_from_image_bytes_with_box(img_bytes)

    if res.get("status") == "no_face":
        return JsonResponse({"status": "fail", "message": "Không phát hiện khuôn mặt"})

    recognized_id = res.get("name", "Unknown")
    if recognized_id in ("Unknown", "Spoof", None, ""):
        return JsonResponse({"status": "fail", "message": "Không nhận diện được khuôn mặt"})

    if recognized_id != employee.user_id:
        return JsonResponse({"status": "fail", "message": "Khuôn mặt không khớp tài khoản này"})

    now = timezone.localtime()
    today = now.date()
    schedule = WorkSchedule.get()
    current = now.time().replace(second=0, microsecond=0)

    record, _ = Attendance.objects.get_or_create(user=employee, date=today)

    if not record.checkin:
        status_in = "Đúng giờ" if current <= schedule.start_time else "Muộn"
        record.checkin = now.time()
        record.status_in = status_in
        record.save()
        return JsonResponse({
            "status": "ok",
            "type": "checkin",
            "time": now.strftime("%H:%M:%S"),
            "user_id": employee.user_id,
            "name": employee.name,
            "status_in": status_in,
        })
    else:
        status_out = "Sớm" if current < schedule.end_time else "Bình thường"
        record.checkout = now.time()
        record.status_out = status_out
        record.save()
        return JsonResponse({
            "status": "ok",
            "type": "checkout",
            "time": now.strftime("%H:%M:%S"),
            "user_id": employee.user_id,
            "name": employee.name,
            "status_out": status_out,
        })


@csrf_exempt
def api_auto_attendance(request):
    """
    Tự động chấm công:
    - Chưa check-in hôm nay → ghi check-in
    - Đã check-in rồi → cập nhật check-out (liên tục, lần cuối = checkout cuối cùng)
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    if not user_id:
        return JsonResponse({"status": "fail", "message": "user_id required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "User not found"})

    now = timezone.localtime()
    today = now.date()
    schedule = WorkSchedule.get()
    current = now.time().replace(second=0, microsecond=0)

    record, _ = Attendance.objects.get_or_create(user=user, date=today)

    if not record.checkin:
        status_in = "Đúng giờ" if current <= schedule.start_time else "Muộn"
        record.checkin = now.time()
        record.status_in = status_in
        record.save()
        return JsonResponse({
            "status": "ok",
            "type": "checkin",
            "time": now.strftime("%H:%M:%S"),
            "user_id": user.user_id,
            "name": user.name,
            "status_in": status_in,
        })
    else:
        status_out = "Sớm" if current < schedule.end_time else "Bình thường"
        record.checkout = now.time()
        record.status_out = status_out
        record.save()
        return JsonResponse({
            "status": "ok",
            "type": "checkout",
            "time": now.strftime("%H:%M:%S"),
            "user_id": user.user_id,
            "name": user.name,
            "status_out": status_out,
        })


@csrf_exempt
def api_checkin_status(request):
    """
    POST {'user_id': ...}
    Trả về trạng thái hôm nay đã check-in chưa
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    if not user_id:
        return JsonResponse({"status": "fail", "message": "user_id required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "user not found"})

    today = timezone.localdate()
    try:
        record = Attendance.objects.get(user=user, date=today)
    except Attendance.DoesNotExist:
        return JsonResponse({"status": "not_yet"})

    if record.checkin:
        return JsonResponse({"status": "already", "time": record.checkin.strftime("%H:%M:%S")})
    else:
        return JsonResponse({"status": "not_yet"})


@csrf_exempt
def api_list_users(request):
    users = Employee.objects.all().values("user_id", "name", "position", "department")
    return JsonResponse({"status": "ok", "users": list(users)})

@csrf_exempt
def api_update_user(request):
    """
    POST:
      - user_id
      - name
      - position
    Cập nhật metadata (name, position) cho user_id đã tồn tại.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get('user_id')
    name = request.POST.get('name')
    position = request.POST.get('position')
    department = request.POST.get('department', '').strip()

    if not user_id or not name or not position:
        return JsonResponse({"status": "fail", "message": "user_id, name and position required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "user not found"})

    user.name = name
    user.position = position
    user.department = department
    user.save()

    return JsonResponse({"status": "ok", "message": "Cập nhật thành công", "user_id": user_id})



@csrf_exempt
def api_delete_user(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    if not user_id:
        return JsonResponse({"status": "fail", "message": "user_id required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "User not found"})

    user.delete()
    return JsonResponse({"status": "ok", "deleted": user_id})


@csrf_exempt
def api_replace_face(request):
    """
    POST multipart/form-data:
      - user_id
      - image (one or more)
    Thay thế vectors hiện tại của user_id bằng các vectors mới (xóa embeddings cũ).
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    images = request.FILES.getlist("image")

    if not user_id or not images:
        return JsonResponse({"status": "fail", "message": "user_id and images required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "user not found"})

    new_vectors = []
    for img in images:
        emb = fr.embedding_from_image_bytes(img.read())
        if emb is not None:
            new_vectors.append(emb.tolist())

    if not new_vectors:
        return JsonResponse({"status": "fail", "message": "Không nhận diện được khuôn mặt"})

    # Xóa embeddings cũ và lưu embeddings mới
    Embedding.objects.filter(user=user).delete()
    for vec in new_vectors:
        Embedding.objects.create(user=user, vector=json.dumps(vec))

    return JsonResponse({
        "status": "ok",
        "saved_vectors": len(new_vectors),
        "message": f"Cập nhật ảnh thành công, đã lưu {len(new_vectors)} embedding",
        "user_id": user_id
    })




# HISTORY API

@csrf_exempt
def api_history(request):
    rows = Attendance.objects.select_related("user").order_by("-date")
    data = [
        {
            "date": r.date.strftime("%Y-%m-%d"),
            "user_id": r.user.user_id,
            "checkin": r.checkin.strftime("%H:%M:%S") if r.checkin else None,
            "status_in": r.status_in,
            "checkout": r.checkout.strftime("%H:%M:%S") if r.checkout else None,
            "status_out": r.status_out,
        }
        for r in rows
    ]
    return JsonResponse({"status": "ok", "data": data})


@csrf_exempt
def api_history_by_day(request):
    date = request.GET.get("date")
    if not date:
        return JsonResponse({"status": "fail", "message": "date required"})

    records = Attendance.objects.filter(date=date).select_related("user")

    data = [
        {
            "user_id": r.user.user_id,
            "name": r.user.name,
            "checkin": r.checkin.strftime("%H:%M:%S") if r.checkin else None,
            "status_in": r.status_in,
            "checkout": r.checkout.strftime("%H:%M:%S") if r.checkout else None,
            "status_out": r.status_out,
        }
        for r in records
    ]

    return JsonResponse({"status": "ok", "data": data})


@csrf_exempt
def api_history_by_month(request):
    month = request.GET.get("month")
    year = request.GET.get("year")
    if not month or not year:
        return JsonResponse({"status": "fail", "message": "month and year required"})

    try:
        month = int(month)
        year = int(year)
    except ValueError:
        return JsonResponse({"status": "fail", "message": "invalid month or year"})

    records = Attendance.objects.filter(
        date__month=month, date__year=year
    ).select_related("user").order_by("date", "user__user_id")

    data = [
        {
            "date": r.date.strftime("%Y-%m-%d"),
            "user_id": r.user.user_id,
            "name": r.user.name,
            "checkin": r.checkin.strftime("%H:%M:%S") if r.checkin else None,
            "status_in": r.status_in,
            "checkout": r.checkout.strftime("%H:%M:%S") if r.checkout else None,
            "status_out": r.status_out,
        }
        for r in records
    ]
    return JsonResponse({"status": "ok", "data": data})


@csrf_exempt
def api_history_by_year(request):
    year = request.GET.get("year")
    if not year:
        return JsonResponse({"status": "fail", "message": "year required"})

    try:
        year = int(year)
    except ValueError:
        return JsonResponse({"status": "fail", "message": "invalid year"})

    records = Attendance.objects.filter(
        date__year=year
    ).select_related("user").order_by("date", "user__user_id")

    data = [
        {
            "date": r.date.strftime("%Y-%m-%d"),
            "user_id": r.user.user_id,
            "name": r.user.name,
            "checkin": r.checkin.strftime("%H:%M:%S") if r.checkin else None,
            "status_in": r.status_in,
            "checkout": r.checkout.strftime("%H:%M:%S") if r.checkout else None,
            "status_out": r.status_out,
        }
        for r in records
    ]
    return JsonResponse({"status": "ok", "data": data})


@csrf_exempt
def api_check_user(request, user_id):
    exists = Employee.objects.filter(user_id=user_id).exists()
    return JsonResponse({"exists": exists})



@csrf_exempt
def api_history_by_id(request, user_id):
    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "user not found"})

    records = Attendance.objects.filter(user=user)

    year_param = request.GET.get("year")
    month_param = request.GET.get("month")
    if year_param:
        try:
            records = records.filter(date__year=int(year_param))
        except ValueError:
            pass
    if month_param:
        try:
            records = records.filter(date__month=int(month_param))
        except ValueError:
            pass

    records = records.order_by("-date")

    result = [
        {
            "date": r.date.strftime("%Y-%m-%d"),
            "checkin": r.checkin.strftime("%H:%M:%S") if r.checkin else None,
            "status_in": r.status_in,
            "checkout": r.checkout.strftime("%H:%M:%S") if r.checkout else None,
            "status_out": r.status_out,
        }
        for r in records
    ]

    return JsonResponse({
        "status": "ok",
        "user_id": user_id,
        "name": user.name,
        "position": user.position,
        "records": result
    })


@csrf_exempt
def api_camera_release(request):
    """Giải phóng camera server-side nếu không có WebSocket client nào đang kết nối."""
    if _consumers._display_client_count > 0:
        return JsonResponse({"released": False, "message": "Camera đang được admin sử dụng"})

    reader = _consumers._local_reader
    task   = _consumers._local_reader_task
    if reader is not None:
        reader.stop()
    if task is not None and not task.done():
        task.cancel()
    _consumers._local_reader      = None
    _consumers._local_reader_task = None
    return JsonResponse({"released": True})
