
# # attendance/face_recognition.py
# import io
# import json
# import time

# import numpy as np
# import cv2
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity

# import torch

# # RetinaFace & ArcFace imports (như code gốc)
# from arcface_torch.backbones import get_model
# from retinaface_torch.models.retinaface import RetinaFace
# from retinaface_torch.data import cfg_mnet, cfg_re50
# from retinaface_torch.utils.box_utils import decode
# from retinaface_torch.layers.functions.prior_box import PriorBox

# # Django models (sử dụng ORM thay cho JSON)
# from .models import Employee, Embedding

# # CONFIG — chỉnh đường dẫn nếu cần
# import os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ARCFACE_MODEL = "r50"
# ARCFACE_WEIGHT = os.path.join(BASE_DIR, "arcface_torch", "backbones", "backbone1.pth")
# RETINA_MODEL_PATH = os.path.join(BASE_DIR, "retinaface_torch", "weights", "Resnet50_Final.pth")  # hoặc mobilenet0.25_Final.pth
# RETINA_NETWORK = "resnet50"  # mobile0.25  resnet50

# THRESHOLD = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ----------------- Model loading -----------------
# def load_model_retina(model_path, network_name):
#     cfg = cfg_mnet if network_name == "mobile0.25" else cfg_re50
#     net = RetinaFace(cfg=cfg, phase="test")
#     pretrained = torch.load(model_path, map_location=device)
#     # Map state dict keys
#     if "state_dict" in pretrained:
#         state = {k.replace("module.", ""): v for k, v in pretrained["state_dict"].items()}
#     else:
#         state = {k.replace("module.", ""): v for k, v in pretrained.items()}
#     net.load_state_dict(state, strict=False)
#     net.to(device).eval()
#     return net, cfg


# def load_model_arc(model_name, weight_path):
#     net = get_model(model_name, fp16=False)
#     net.load_state_dict(torch.load(weight_path, map_location=device))
#     net.to(device).eval()
#     return net


# # Load once
# net_arc = load_model_arc(ARCFACE_MODEL, ARCFACE_WEIGHT)
# net_retina, cfg = load_model_retina(RETINA_MODEL_PATH, RETINA_NETWORK)


# # ----------------- Image helpers -----------------
# def bytes_to_cv2_image(image_bytes):
#     """Convert bytes (uploaded image) to OpenCV BGR image."""
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     arr = np.array(img)  # RGB
#     bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#     return bgr


# def get_embedding_from_face(face_bgr):
#     """
#     Input: face image in BGR (OpenCV), arbitrary size.
#     Output: L2-normalized 1D numpy array embedding.
#     """
#     face = cv2.resize(face_bgr, (112, 112))
#     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#     face = np.transpose(face, (2, 0, 1)).astype(np.float32)
#     img = torch.from_numpy(face).unsqueeze(0).float().to(device)
#     # normalize as in training
#     img.div_(255).sub_(0.5).div_(0.5)
#     with torch.no_grad():
#         feat = net_arc(img).cpu().numpy()
#     feat = feat.squeeze()
#     # Normalize to unit vector
#     denom = np.linalg.norm(feat)
#     if denom > 0:
#         feat = feat / denom
#     return feat


# def detect_face_box(frame_bgr):
#     """
#     Return box [x1, y1, x2, y2] (ints) of the best detected face or None.
#     Uses RetinaFace model loaded above.
#     """
#     img = np.float32(frame_bgr)
#     img -= (104, 117, 123)
#     img = img.transpose(2, 0, 1)
#     img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

#     im_height, im_width = frame_bgr.shape[:2]
#     scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)

#     with torch.no_grad():
#         loc, conf, _ = net_retina(img_tensor)
#         priors = PriorBox(cfg, image_size=(im_height, im_width)).forward().to(device)
#         boxes = decode(loc.data.squeeze(0), priors.data, cfg["variance"])
#         boxes = boxes * scale
#         scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

#     inds = np.where(scores > 0.6)[0]
#     if len(inds) == 0:
#         return None
#     best_idx = inds[np.argmax(scores[inds])]
#     box = boxes[best_idx].int().cpu().numpy()
#     return box  # x1,y1,x2,y2


# def embedding_from_image_bytes(image_bytes):
#     """
#     Nhận image bytes (JPG/PNG), phát hiện face, trả embedding (1D numpy array) hoặc None nếu không thấy.
#     """
#     try:
#         frame = bytes_to_cv2_image(image_bytes)
#     except Exception as e:
#         # không parse được ảnh
#         print("bytes_to_cv2_image error:", e)
#         return None

#     box = detect_face_box(frame)
#     if box is None:
#         return None
#     x1, y1, x2, y2 = box
#     h, w = frame.shape[:2]
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)
#     face = frame[y1:y2, x1:x2]
#     if face.size == 0:
#         return None
#     emb = get_embedding_from_face(face)
#     return emb


# # ----------------- DB helpers (ORM) -----------------
# def load_all_embeddings_from_db():
#     """
#     Trả về tuple (names_list, embs_list)
#     - names_list: list string (user_id)
#     - embs_list: list of vectors (list or np.array)
#     Lấy từ bảng Embedding. Nếu nhiều vector cùng user thì user xuất hiện nhiều lần.
#     """
#     names = []
#     embs = []
#     qs = Embedding.objects.select_related("user").all()
#     for e in qs:
#         try:
#             vec = json.loads(e.vector)
#             # convert to numpy array
#             vec_arr = np.array(vec, dtype=np.float32)
#             # if not normalized, normalize
#             denom = np.linalg.norm(vec_arr)
#             if denom > 0:
#                 vec_arr = vec_arr / denom
#             names.append(e.user.user_id)
#             embs.append(vec_arr)
#         except Exception as ex:
#             # bỏ qua record hỏng
#             print("Invalid embedding record id=", getattr(e, 'id', None), "error:", ex)
#             continue
#     return names, embs


# def add_embedding_to_db(user_id, vector):
#     """
#     Lưu vector (numpy array hoặc list) vào bảng Embedding cho user_id.
#     Nếu user không tồn tại -> raise or return False.
#     """
#     try:
#         user = Employee.objects.get(user_id=user_id)
#     except Employee.DoesNotExist:
#         return False, "user_not_found"

#     # ensure list
#     if isinstance(vector, np.ndarray):
#         vec_list = vector.tolist()
#     else:
#         vec_list = vector

#     Embedding.objects.create(user=user, vector=json.dumps(vec_list, ensure_ascii=False))
#     return True, None


# # ----------------- Matching / Register (DB-based) -----------------
# def recognize_from_image_bytes(image_bytes):
#     """
#     Return dict similar to cũ:
#       {"status": "no_face"}  (nếu không detect face)
#       {"status": "unknown", "similarity": None} (nếu DB rỗng)
#       {"status": "ok", "name": matched_name, "similarity": best_sim}
#     matched_name là user_id tương ứng (giữ như cũ)
#     """
#     frame = bytes_to_cv2_image(image_bytes)
#     box = detect_face_box(frame)
#     if box is None:
#         return {"status": "no_face"}

#     x1, y1, x2, y2 = box
#     h, w = frame.shape[:2]
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)
#     face = frame[y1:y2, x1:x2]
#     if face.size == 0:
#         return {"status": "no_face"}

#     emb = get_embedding_from_face(face)

#     names, db_embs = load_all_embeddings_from_db()
#     if not db_embs:
#         return {"status": "unknown", "similarity": None}

#     db_embs_arr = np.array(db_embs)
#     sims = cosine_similarity([emb], db_embs_arr)[0]
#     best_idx = int(np.argmax(sims))
#     best_sim = float(sims[best_idx])
#     matched_name = names[best_idx] if best_sim > THRESHOLD else "Unknown"
#     return {"status": "ok", "name": matched_name, "similarity": best_sim}


# def register_image_for_user(image_bytes, user_id):
#     """
#     Nhận image bytes, user_id -> trích embedding, lưu vào DB (Embedding table).
#     Trả về dict: {"status":"ok", "user": user_id, "vector": [...] } or fail message.
#     """
#     if user_id is None:
#         return {"status": "fail", "message": "user_id required"}

#     emb = embedding_from_image_bytes(image_bytes)
#     if emb is None:
#         return {"status": "fail", "message": "Không nhận diện được khuôn mặt"}

#     ok, err = add_embedding_to_db(user_id, emb)
#     if not ok:
#         return {"status": "fail", "message": err}
#     return {"status": "ok", "user": user_id, "vector": emb.tolist()}


# def recognize_from_image_bytes_with_box(image_bytes):
#     """
#     Nhận image bytes, trả kết quả nhận diện kèm bounding box.
#     Trả về dict: {status, name, similarity, box: {x,y,w,h}}. Giữ format như cũ.
#     """
#     frame = bytes_to_cv2_image(image_bytes)
#     box = detect_face_box(frame)
#     if box is None:
#         return {"status": "no_face"}

#     x1, y1, x2, y2 = box
#     h, w = frame.shape[:2]
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)
#     face = frame[y1:y2, x1:x2]
#     if face.size == 0:
#         return {"status": "no_face"}

#     emb = get_embedding_from_face(face)

#     names, db_embs = load_all_embeddings_from_db()
#     if not db_embs:
#         matched_name = "Unknown"
#         best_sim = None
#     else:
#         db_embs_arr = np.array(db_embs)
#         sims = cosine_similarity([emb], db_embs_arr)[0]
#         best_idx = int(np.argmax(sims))
#         best_sim = float(sims[best_idx])
#         matched_name = names[best_idx] if best_sim > THRESHOLD else "Unknown"

#     return {
#         "status": "ok",
#         "name": matched_name,
#         "similarity": best_sim,
#         "box": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
#     }



































# attendance/face_recognition.py
import io
import json
import time

import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import torch

# RetinaFace & ArcFace imports (như code gốc)
from arcface_torch.backbones import get_model
from retinaface_torch.models.retinaface import RetinaFace
from retinaface_torch.data import cfg_mnet, cfg_re50
from retinaface_torch.utils.box_utils import decode
from retinaface_torch.layers.functions.prior_box import PriorBox

# Django models (sử dụng ORM thay cho JSON)
from .models import Employee, Embedding

# CONFIG — chỉnh đường dẫn nếu cần
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCFACE_MODEL = "r50"
ARCFACE_WEIGHT = os.path.join(BASE_DIR, "arcface_torch", "backbones", "backbone1.pth")
RETINA_MODEL_PATH = os.path.join(BASE_DIR, "retinaface_torch", "weights", "Resnet50_Final.pth")  # hoặc mobilenet0.25_Final.pth
RETINA_NETWORK = "resnet50"  # mobile0.25  resnet50

THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Model loading -----------------
def load_model_retina(model_path, network_name):
    cfg = cfg_mnet if network_name == "mobile0.25" else cfg_re50
    net = RetinaFace(cfg=cfg, phase="test")
    pretrained = torch.load(model_path, map_location=device)
    # Map state dict keys
    if "state_dict" in pretrained:
        state = {k.replace("module.", ""): v for k, v in pretrained["state_dict"].items()}
    else:
        state = {k.replace("module.", ""): v for k, v in pretrained.items()}
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return net, cfg


def load_model_arc(model_name, weight_path):
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.to(device).eval()
    return net


# Load once
net_arc = load_model_arc(ARCFACE_MODEL, ARCFACE_WEIGHT)
net_retina, cfg = load_model_retina(RETINA_MODEL_PATH, RETINA_NETWORK)


# ----------------- Image helpers -----------------
def bytes_to_cv2_image(image_bytes):
    """Convert bytes (uploaded image) to OpenCV BGR image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)  # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def get_embedding_from_face(face_bgr):
    """
    Input: face image in BGR (OpenCV), arbitrary size.
    Output: L2-normalized 1D numpy array embedding.
    """
    face = cv2.resize(face_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.transpose(face, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(face).unsqueeze(0).float().to(device)
    # normalize as in training
    img.div_(255).sub_(0.5).div_(0.5)
    with torch.no_grad():
        feat = net_arc(img).cpu().numpy()
    feat = feat.squeeze()
    # Normalize to unit vector
    denom = np.linalg.norm(feat)
    if denom > 0:
        feat = feat / denom
    return feat


def detect_face_box(frame_bgr):
    """
    Trả về:
      box: np.array([x1, y1, x2, y2], dtype=int) hoặc None
      landmarks: np.array([[x1,y1],...,[x5,y5]]) hoặc None
    """
    img = np.float32(frame_bgr)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    im_height, im_width = frame_bgr.shape[:2]
    scale = torch.tensor([im_width, im_height, im_width, im_height], dtype=torch.float32).to(device)

    with torch.no_grad():
        loc, conf, landms = net_retina(img_tensor)
        priors = PriorBox(cfg, image_size=(im_height, im_width)).forward().to(device)
        boxes = decode(loc.squeeze(0), priors, cfg["variance"])
        boxes = boxes * scale  # Tensor
        scores = conf.squeeze(0)[:, 1].cpu().numpy()  # lớp face
        landms = landms.squeeze(0).cpu().numpy()  # shape: (num_boxes, 10)

    inds = np.where(scores > 0.6)[0]
    if len(inds) == 0:
        return None, None

    best_idx = inds[np.argmax(scores[inds])]
    box = boxes[best_idx].cpu().numpy()  # có thể shape (4,)
    box = box.astype(int)

    # Chắc chắn convert thành [x1, y1, x2, y2]
    if box.shape == (4,):
        x1, y1, x2, y2 = box
    elif box.shape == (2,2):
        x1, y1, x2, y2 = box.flatten()
    else:
        x1, y1, x2, y2 = box[:4]

    landmark = landms[best_idx].reshape((5, 2))
    return np.array([x1, y1, x2, y2], dtype=int), landmark



def embedding_from_image_bytes(image_bytes):
    frame = bytes_to_cv2_image(image_bytes)
    box, landmarks = detect_face_box(frame)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Cắt mặt + align
    face = frame[y1:y2, x1:x2]
    aligned_face = align_face(face, landmarks)
    if aligned_face is None or aligned_face.size == 0:
        return None

    emb = get_embedding_from_face(aligned_face)
    return emb



# ----------------- DB helpers (ORM) -----------------
def load_all_embeddings_from_db():
    """
    Trả về tuple (names_list, embs_list)
    - names_list: list string (user_id)
    - embs_list: list of vectors (list or np.array)
    Lấy từ bảng Embedding. Nếu nhiều vector cùng user thì user xuất hiện nhiều lần.
    """
    names = []
    embs = []
    qs = Embedding.objects.select_related("user").all()
    for e in qs:
        try:
            vec = json.loads(e.vector)
            # convert to numpy array
            vec_arr = np.array(vec, dtype=np.float32)
            # if not normalized, normalize
            denom = np.linalg.norm(vec_arr)
            if denom > 0:
                vec_arr = vec_arr / denom
            names.append(e.user.user_id)
            embs.append(vec_arr)
        except Exception as ex:
            # bỏ qua record hỏng
            print("Invalid embedding record id=", getattr(e, 'id', None), "error:", ex)
            continue
    return names, embs


def add_embedding_to_db(user_id, vector):
    """
    Lưu vector (numpy array hoặc list) vào bảng Embedding cho user_id.
    Nếu user không tồn tại -> raise or return False.
    """
    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return False, "user_not_found"

    # ensure list
    if isinstance(vector, np.ndarray):
        vec_list = vector.tolist()
    else:
        vec_list = vector

    Embedding.objects.create(user=user, vector=json.dumps(vec_list, ensure_ascii=False))
    return True, None


# ----------------- Matching / Register (DB-based) -----------------
def recognize_from_image_bytes(image_bytes):
    """
    Return dict similar to cũ:
      {"status": "no_face"}  (nếu không detect face)
      {"status": "unknown", "similarity": None} (nếu DB rỗng)
      {"status": "ok", "name": matched_name, "similarity": best_sim}
    matched_name là user_id tương ứng (giữ như cũ)
    """
    frame = bytes_to_cv2_image(image_bytes)
    box, landmarks = detect_face_box(frame)
    if box is None:
        return {"status": "no_face"}

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return {"status": "no_face"}

    emb = get_embedding_from_face(face)

    names, db_embs = load_all_embeddings_from_db()
    if not db_embs:
        return {"status": "unknown", "similarity": None}

    db_embs_arr = np.array(db_embs)
    sims = cosine_similarity([emb], db_embs_arr)[0]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    matched_name = names[best_idx] if best_sim > THRESHOLD else "Unknown"
    return {"status": "ok", "name": matched_name, "similarity": best_sim}


def register_image_for_user(image_bytes, user_id):
    """
    Nhận image bytes, user_id -> trích embedding, lưu vào DB (Embedding table).
    Trả về dict: {"status":"ok", "user": user_id, "vector": [...] } or fail message.
    """
    if user_id is None:
        return {"status": "fail", "message": "user_id required"}

    emb = embedding_from_image_bytes(image_bytes)
    if emb is None:
        return {"status": "fail", "message": "Không nhận diện được khuôn mặt"}

    ok, err = add_embedding_to_db(user_id, emb)
    if not ok:
        return {"status": "fail", "message": err}
    return {"status": "ok", "user": user_id, "vector": emb.tolist()}


def recognize_from_image_bytes_with_box(image_bytes):
    """
    Nhận image bytes, trả kết quả nhận diện kèm bounding box.
    Trả về dict: {status, name, similarity, box: {x,y,w,h}}. Giữ format như cũ.
    """
    frame = bytes_to_cv2_image(image_bytes)
    box, landmarks = detect_face_box(frame)
    if box is None:
        return {"status": "no_face"}

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return {"status": "no_face"}

    emb = get_embedding_from_face(face)

    names, db_embs = load_all_embeddings_from_db()
    if not db_embs:
        matched_name = "Unknown"
        best_sim = None
    else:
        db_embs_arr = np.array(db_embs)
        sims = cosine_similarity([emb], db_embs_arr)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        matched_name = names[best_idx] if best_sim > THRESHOLD else "Unknown"

    return {
        "status": "ok",
        "name": matched_name,
        "similarity": best_sim,
        "box": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}
    }


def align_face(frame_bgr, landmarks):
    """
    frame_bgr: OpenCV BGR image
    landmarks: np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]])
        theo thứ tự: [left_eye, right_eye, nose, left_mouth, right_mouth]
    Trả về face đã align (112x112) RGB
    """
    if landmarks is None or len(landmarks) != 5:
        return None

    # Lấy 2 mắt để tính góc xoay
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Tâm xoay giữa 2 mắt
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(frame_bgr, M, (frame_bgr.shape[1], frame_bgr.shape[0]), flags=cv2.INTER_CUBIC)

    # Crop theo bbox ban đầu
    return rotated


# def align_face(frame_bgr, landmarks):
#     """
#     Align face chuẩn ArcFace.
#     Args:
#         frame_bgr: OpenCV BGR image
#         landmarks: np.array([[x1, y1], ..., [x5, y5]]) 
#                    order: [left_eye, right_eye, nose, left_mouth, right_mouth]
#     Returns:
#         aligned_face: RGB 112x112
#     """
#     if landmarks is None or len(landmarks) != 5:
#         return None

#     # Chuẩn landmark ArcFace (112x112)
#     STD_LANDMARKS = np.array([
#         [38.2946, 51.6963],   # left eye
#         [73.5318, 51.5014],   # right eye
#         [56.0252, 71.7366],   # nose
#         [41.5493, 92.3655],   # left mouth
#         [70.7299, 92.2041]    # right mouth
#     ], dtype=np.float32)

#     OUTPUT_SIZE = (112, 112)

#     src = landmarks.astype(np.float32)
#     dst = STD_LANDMARKS

#     # Tính similarity transform (scale, rotate, translate)
#     tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
#     if tform is None:
#         return None

#     aligned = cv2.warpAffine(frame_bgr, tform, OUTPUT_SIZE, flags=cv2.INTER_CUBIC)
#     aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
#     return aligned_rgb
