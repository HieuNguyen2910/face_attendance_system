# attendance/views.py

import json
from datetime import datetime

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

from .models import Employee, Embedding, Attendance
from . import face_recognition as fr


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
        "user_id": user_id
    })


@csrf_exempt
def api_register_employee(request):
    """
    Đăng ký nhân viên mới + embeddings
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    user_id = request.POST.get("user_id")
    name = request.POST.get("name")
    position = request.POST.get("position")
    images = request.FILES.getlist("image")

    if not (user_id and name and position and images):
        return JsonResponse({"status": "fail", "message": "Missing fields"})

    user, _ = Employee.objects.get_or_create(
        user_id=user_id,
        defaults={"name": name, "position": position}
    )

    # update metadata nếu user đã tồn tại
    user.name = name
    user.position = position
    user.save()

    # Xóa embeddings cũ
    Embedding.objects.filter(user=user).delete()

    vectors = []
    for img in images:
        emb = fr.embedding_from_image_bytes(img.read())
        if emb is not None:
            vectors.append(emb.tolist())

    if not vectors:
        return JsonResponse({"status": "fail", "message": "Không lấy được embedding"})

    for vec in vectors:
        Embedding.objects.create(user=user, vector=json.dumps(vec))

    return JsonResponse({
        "status": "ok",
        "message": f"Đã thêm {len(vectors)} ảnh cho {user_id}"
    })


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

    now = datetime.now()
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

    status_in = "Đúng giờ" if now.hour < 8 or (now.hour == 8 and now.minute <= 30) else "Muộn"

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

    now = datetime.now()
    today = now.date()

    record, created = Attendance.objects.get_or_create(
        user=user,
        date=today
    )

    status_out = "Sớm" if now.hour < 18 else "Bình thường"

    record.checkout = now.time()
    record.status_out = status_out
    record.save()

    return JsonResponse({
        "status": "ok",
        "time": now.strftime("%H:%M:%S"),
        "status_out": status_out
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

    today = datetime.now().date()
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
    users = Employee.objects.all().values("user_id", "name", "position")
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

    if not user_id or not name or not position:
        return JsonResponse({"status": "fail", "message": "user_id, name and position required"})

    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return JsonResponse({"status": "fail", "message": "user not found"})

    user.name = name
    user.position = position
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

    return JsonResponse({"status": "ok", "message": "Cập nhật ảnh thành công", "user_id": user_id})




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

    records = Attendance.objects.filter(user=user).order_by("-date")

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



