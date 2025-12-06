# # attendance/views.py
# import os
# import json
# from datetime import datetime
# from django.shortcuts import render
# from django.http import JsonResponse, HttpResponseBadRequest
# from django.views.decorators.csrf import csrf_exempt
# from django.conf import settings

# # import functions from face_recognition module
# from . import face_recognition as fr

# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ATTEND_PATH = os.path.join(BASE_DIR, "attendance.json")
# EMBED_PATH = os.path.join(BASE_DIR, "embeddings.json")

# # ---------- Helper JSON ----------
# def load_json(path):
#     if not os.path.exists(path):
#         return {}
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             content = f.read().strip()
#             if not content:
#                 return {}
#             return json.loads(content)
#     except json.JSONDecodeError:
#         return {}

# import json
# import os

# def save_json(path, data):
#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
#     with open(path, "w", encoding="utf-8") as f:
#         f.write("{\n")
#         users = list(data.items())
#         for i, (user_id, val) in enumerate(users):
#             # Lấy dữ liệu
#             if isinstance(val, dict):
#                 name = val.get("name", "")
#                 position = val.get("position", "")
#                 vectors = val.get("vectors", [])
#             elif isinstance(val, list):
#                 # fallback: chỉ list vector cũ
#                 name = ""
#                 position = ""
#                 vectors = val
#             else:
#                 name = ""
#                 position = ""
#                 vectors = []

#             # serialize từng vector 1 dòng
#             vectors_lines = [json.dumps(vec, ensure_ascii=False, separators=(',', ':')) for vec in vectors]

#             # Viết user block
#             f.write(f'  "{user_id}": {{\n')
#             f.write(f'    "name": {json.dumps(name)},\n')
#             f.write(f'    "position": {json.dumps(position)},\n')
#             f.write(f'    "vectors": [\n')
#             # thêm 4 spaces indent cho vector
#             f.write(",\n".join("      " + line for line in vectors_lines))
#             f.write("\n    ]\n  }")
            
#             # dấu , nếu không phải user cuối cùng
#             if i != len(users) - 1:
#                 f.write(",\n")
#             else:
#                 f.write("\n")
#         f.write("}\n")


# # attendance/views.py


# def save_attendance(ATTEND_PATH,data):
#     os.makedirs(os.path.dirname(ATTEND_PATH) or ".", exist_ok=True)
#     with open(ATTEND_PATH, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)




# # ---------- Views ----------
# def home(request):
#     users = load_json(EMBED_PATH)
#     # pass users keys only (ids)
#     return render(request, "attendance/index.html", {"nhanviens": list(users.keys())})

# def manage(request):
#     users = load_json(EMBED_PATH)
#     return render(request, "attendance/manage.html", {"nhanviens": users})

# def history(request):
#     att = load_json(ATTEND_PATH)
#     # att is dict {timestamp: user}
#     # convert to list of tuples sorted desc
#     rows = sorted(att.items(), reverse=True)
#     return render(request, "attendance/history.html", {"chamcongs": rows})

# def history_by_id(request, user_id):
#     """
#     Trang HTML: xem lịch sử chấm công của 1 user theo ID
#     """
#     att = load_json(ATTEND_PATH)
#     users = load_json(EMBED_PATH)

#     info = users.get(user_id, {})
#     name = info.get("name", "")
#     position = info.get("position", "")

#     # Lấy danh sách record của user
#     records = []
#     for date, users_data in att.items():
#         if user_id in users_data:
#             record = users_data[user_id]
#             records.append({
#                 "date": date,
#                 "checkin": record.get("checkin"),
#                 "status_in": record.get("status_in"),
#                 "checkout": record.get("checkout"),
#                 "status_out": record.get("status_out"),
#             })

#     # sắp xếp theo date giảm dần
#     records.sort(key=lambda x: x["date"], reverse=True)

#     return render(request, "attendance/history_id.html", {
#         "user_id": user_id,
#         "name": name,
#         "position": position,
#         "records": records
#     })

# # ---------- API endpoints ----------
# @csrf_exempt
# def api_recognize(request):
#     """
#     POST multipart/form-data với key 'image' (file).
#     Trả JSON {status, name, similarity, box}
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")
#     if 'image' not in request.FILES:
#         return JsonResponse({"status": "fail", "message": "No image provided"})
    
#     img = request.FILES['image'].read()
#     res = fr.recognize_from_image_bytes_with_box(img)
#     return JsonResponse(res)


# @csrf_exempt
# def api_register(request):
#     """
#     POST multipart/form-data:
#       - user_id
#       - optional: name, position
#       - image: one or more image files
#     Lưu vào EMBED_PATH với cấu trúc mới:
#       users[user_id] = { "name": ..., "position": ..., "vectors": [...] }
#     Nếu name/position không được gửi (fallback), vẫn giữ hành vi cũ nhưng
#     lưu vectors dưới key user_id as list (để tương thích).
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")

#     user_id = request.POST.get('user_id')
#     name = request.POST.get('name')
#     position = request.POST.get('position')
#     images = request.FILES.getlist('image')

#     if not user_id or not images:
#         return JsonResponse({"status": "fail", "message": "user_id and image(s) required"})

#     vectors = []
#     for imgfile in images:
#         img_bytes = imgfile.read()
#         emb = fr.embedding_from_image_bytes(img_bytes)
#         if emb is not None:
#             vectors.append(emb.tolist())

#     if not vectors:
#         return JsonResponse({"status": "fail", "message": "Không nhận diện được khuôn mặt"})

#     # load existing embeddings
#     data = load_json(EMBED_PATH)

#     # nếu có name/position thì lưu metadata + vectors
#     if name is not None and position is not None:
#         data[user_id] = {
#             "name": name,
#             "position": position,
#             "vectors": vectors
#         }
#     else:
#         # fallback: giữ hành vi cũ (chỉ list vectors)
#         data[user_id] = vectors

#     save_json(EMBED_PATH, data)

#     return JsonResponse({
#         "status": "ok",
#         "message": f"Đã lưu {len(vectors)} vectors cho {user_id}",
#         "saved_vectors": len(vectors),
#         "user_id": user_id
#     })



# @csrf_exempt
# def api_checkin(request):
#     """
#     Ghi nhận chấm công check-in (đầu ngày)
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")

#     try:
#         body = request.POST or json.loads(request.body.decode('utf-8'))
#         user_id = body.get('user_id')
#     except:
#         user_id = None

#     if not user_id:
#         return JsonResponse({"status": "fail", "message": "user_id required"})

#     now = datetime.now()
#     today = now.strftime("%Y-%m-%d")
#     current_time = now.strftime("%H:%M:%S")

#     data = load_json(ATTEND_PATH)
#     if today not in data:
#         data[today] = {}

#     # Nếu đã check-in rồi
#     if user_id in data[today] and "checkin" in data[today][user_id]:
#         old_time = data[today][user_id]["checkin"]
#         return JsonResponse({
#             "status": "already",
#             "message": f"{user_id} đã check-in lúc {old_time}"
#         })

#     # Xác định đi đúng giờ hay muộn
#     status_in = "Đúng giờ" if now.hour < 8 or (now.hour == 8 and now.minute <= 30) else "Muộn"

#     data[today][user_id] = {
#         "checkin": current_time,
#         "status_in": status_in
#     }

#     save_attendance(ATTEND_PATH, data)

#     return JsonResponse({
#         "status": "ok",
#         "user": user_id,
#         "time": current_time,
#         "status_in": status_in
#     })


# @csrf_exempt
# def api_list_users(request):
#     users = load_json(EMBED_PATH)
#     return JsonResponse({"status":"ok","users": users})

# @csrf_exempt
# def api_delete_user(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")
#     user_id = request.POST.get('user_id')
#     if not user_id:
#         return JsonResponse({"status":"fail","message":"user_id required"})
#     users = load_json(EMBED_PATH)
#     if user_id in users:
#         deleted_meta = users[user_id]
#         del users[user_id]
#         save_json(EMBED_PATH, users)
#         return JsonResponse({"status":"ok","deleted": user_id, "deleted_meta": deleted_meta})
#     return JsonResponse({"status":"fail","message":"not found"})


# @csrf_exempt
# def api_checkout(request):
#     """
#     Ghi nhận chấm công check-out (về)
#     Có thể ấn nhiều lần, mỗi lần sẽ ghi đè thời gian cũ
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")

#     try:
#         body = request.POST or json.loads(request.body.decode('utf-8'))
#         user_id = body.get('user_id')
#     except:
#         user_id = None

#     if not user_id:
#         return JsonResponse({"status": "fail", "message": "user_id required"})

#     now = datetime.now()
#     today = now.strftime("%Y-%m-%d")
#     current_time = now.strftime("%H:%M:%S")

#     data = load_json(ATTEND_PATH)
#     if today not in data:
#         data[today] = {}

#     if user_id not in data[today]:
#         data[today][user_id] = {}

#     # Xác định trạng thái check-out
#     status_out = "Sớm" if now.hour < 18 else "Bình thường"

#     data[today][user_id]["checkout"] = current_time
#     data[today][user_id]["status_out"] = status_out

#     save_attendance(ATTEND_PATH, data)

#     return JsonResponse({
#         "status": "ok",
#         "user": user_id,
#         "time": current_time,
#         "status_out": status_out
#     })



# @csrf_exempt
# def api_history(request):
#     """
#     GET: Trả danh sách lịch sử chấm công từ attendance.json
#     """
#     if request.method != "GET":
#         return HttpResponseBadRequest("GET required")

#     att = load_json(ATTEND_PATH)
#     rows = sorted(att.items(), reverse=True)  # [(time, user), ...]
#     return JsonResponse({"status": "ok", "data": rows})


# @csrf_exempt
# def api_checkin_status(request):
#     """
#     POST {'user_id': ...}
#     Trả về trạng thái hôm nay đã check-in chưa
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")
#     user_id = request.POST.get('user_id')
#     if not user_id:
#         return JsonResponse({"status":"fail","message":"user_id required"})
#     data = load_json(ATTEND_PATH)
#     today = datetime.now().strftime("%Y-%m-%d")
#     if today in data and user_id in data[today] and "checkin" in data[today][user_id]:
#         return JsonResponse({"status":"already","time":data[today][user_id]["checkin"]})
#     return JsonResponse({"status":"not_yet"})


# @csrf_exempt
# def api_register_employee(request):
#     """
#     Đăng ký nhân viên mới: id + name + position + 3 ảnh
#     Lưu metadata + embeddings vào embeddings.json
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")

#     user_id = request.POST.get("user_id")
#     name = request.POST.get("name")
#     position = request.POST.get("position")
#     images = request.FILES.getlist("image")

#     if not user_id or not name or not position:
#         return JsonResponse({"status": "fail", "message": "Missing fields"})

#     if not images:
#         return JsonResponse({"status": "fail", "message": "Images required"})

#     # Load data
#     data = load_json(EMBED_PATH)
    
#     vectors = []
#     for img in images:
#         emb = fr.embedding_from_image_bytes(img.read())
#         if emb is not None:
#             vectors.append(emb.tolist())

#     if not vectors:
#         return JsonResponse({"status": "fail", "message": "Không lấy được embedding"})

#     # Lưu đầy đủ metadata + vectors
#     data[user_id] = {
#         "name": name,
#         "position": position,
#         "vectors": vectors
#     }

#     save_json(EMBED_PATH, data)

#     return JsonResponse({
#         "status": "ok",
#         "message": f"Đã lưu {len(vectors)} ảnh cho {user_id}",
#         "user": user_id
#     })


# @csrf_exempt
# def api_update_user(request):
#     """
#     POST:
#       - user_id
#       - name
#       - position
#     Cập nhật metadata (name, position) cho user_id đã tồn tại.
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")

#     user_id = request.POST.get('user_id')
#     name = request.POST.get('name')
#     position = request.POST.get('position')

#     if not user_id or not name or not position:
#         return JsonResponse({"status":"fail","message":"user_id, name and position required"})

#     data = load_json(EMBED_PATH)
#     if user_id not in data:
#         return JsonResponse({"status":"fail","message":"user not found"})

#     # nếu dữ liệu hiện tại là list (cũ) -> chuyển sang object
#     if isinstance(data[user_id], list):
#         # promote to object but keep vectors
#         data[user_id] = {
#             "name": name,
#             "position": position,
#             "vectors": data[user_id]
#         }
#     else:
#         # cập nhật trường name/position
#         data[user_id]["name"] = name
#         data[user_id]["position"] = position
#         # đảm bảo tồn tại vectors key
#         if "vectors" not in data[user_id]:
#             data[user_id]["vectors"] = data[user_id].get("vectors", [])

#     save_json(EMBED_PATH, data)
#     return JsonResponse({"status":"ok","message":"Cập nhật thành công","user_id":user_id})


# @csrf_exempt
# def api_replace_face(request):
#     """
#     POST multipart/form-data:
#       - user_id
#       - image (one or more)
#     Thay thế vectors hiện tại của user_id bằng các vectors mới.
#     """
#     if request.method != "POST":
#         return HttpResponseBadRequest("POST required")

#     user_id = request.POST.get('user_id')
#     images = request.FILES.getlist('image')

#     if not user_id or not images:
#         return JsonResponse({"status":"fail","message":"user_id and images required"})

#     data = load_json(EMBED_PATH)
#     if user_id not in data:
#         return JsonResponse({"status":"fail","message":"user not found"})

#     new_vectors = []
#     for img in images:
#         emb = fr.embedding_from_image_bytes(img.read())
#         if emb is not None:
#             new_vectors.append(emb.tolist())

#     if not new_vectors:
#         return JsonResponse({"status":"fail","message":"Không nhận diện được khuôn mặt"})

#     # nếu hiện tại lưu dưới dạng list (cũ) hoặc object, ta ghi lại thành object
#     data[user_id] = {
#         "name": data[user_id].get("name") if isinstance(data[user_id], dict) else "",
#         "position": data[user_id].get("position") if isinstance(data[user_id], dict) else "",
#         "vectors": new_vectors
#     }

#     save_json(EMBED_PATH, data)
#     return JsonResponse({"status":"ok","message":"Cập nhật ảnh thành công","user_id":user_id})



# @csrf_exempt
# def api_history_by_day(request):
#     date = request.GET.get("date")
#     if not date:
#         return JsonResponse({"status": "fail", "message": "date required"})

#     data = load_json(ATTEND_PATH)

#     if date not in data:
#         return JsonResponse({"status": "ok", "data": []})

#     result = []
#     for user_id, info in data[date].items():
#         result.append({
#             "user_id": user_id,
#             "checkin": info.get("checkin"),
#             "status_in": info.get("status_in"),
#             "checkout": info.get("checkout"),
#             "status_out": info.get("status_out"),
#         })

#     return JsonResponse({"status": "ok", "data": result})


# @csrf_exempt
# def api_history_by_id(request, user_id):
#     att = load_json(ATTEND_PATH)
#     users = load_json(EMBED_PATH)

#     # Lấy metadata
#     info = users.get(user_id, {})
#     name = info.get("name", "")
#     position = info.get("position", "")

#     result = []

#     for date, users_data in att.items():
#         if user_id in users_data:
#             result.append({
#                 "date": date,
#                 "checkin": users_data[user_id].get("checkin"),
#                 "status_in": users_data[user_id].get("status_in"),
#                 "checkout": users_data[user_id].get("checkout"),
#                 "status_out": users_data[user_id].get("status_out"),
#             })

#     result = sorted(result, key=lambda x: x["date"], reverse=True)

#     return JsonResponse({
#         "status": "ok",
#         "user_id": user_id,
#         "name": name,
#         "position": position,
#         "records": result
#     })















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



