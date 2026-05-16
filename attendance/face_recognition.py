
import io
import sys
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import torch

# RetinaFace & ArcFace imports gom về model/
from model.arcface_torch.backbones import get_model
from model.retinaface_torch.models.retinaface import RetinaFace
from model.retinaface_torch.data import cfg_mnet, cfg_re50
from model.retinaface_torch.utils.box_utils import decode, decode_landm
from model.retinaface_torch.layers.functions.prior_box import PriorBox

# Django models (sử dụng ORM thay cho JSON)
from .models import Employee, Embedding

# CONFIG — chỉnh đường dẫn nếu cần
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
WEIGHTS_DIR = MODEL_DIR / "weights"

# Thêm model/anti_spoofing vào sys.path để import "from src.xxx" hoạt động
_ANTI_SPOOF_DIR = str(MODEL_DIR / "anti_spoofing")
if _ANTI_SPOOF_DIR not in sys.path:
    sys.path.insert(0, _ANTI_SPOOF_DIR)
from src.anti_spoof_predict import AntiSpoofPredict  # noqa: E402

ANTI_SPOOF_MODELS = [
    WEIGHTS_DIR / "2.7_80x80_MiniFASNetV2.pth",
    WEIGHTS_DIR / "4_0_0_80x80_MiniFASNetV1SE.pth",
]
ARCFACE_MODEL = "r100"
ARCFACE_WEIGHT = WEIGHTS_DIR / "backbone.pth"
RETINA_MODEL_PATH = WEIGHTS_DIR / "Resnet50_Final.pth"
RETINA_NETWORK = "resnet50"  # mobile0.25  resnet50

THRESHOLD = 0.5
ARCFACE_OUTPUT_SIZE = (112, 112)
ARCFACE_STANDARD_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
MIN_RECOGNIZE_FACE_SIZE = 80
MAX_RECOGNIZE_FACE_SIZE = 400
MIN_REGISTER_FACE_SIZE = 70
USER_MATCH_THRESHOLD = 0.66
SINGLE_USER_MATCH_THRESHOLD = 0.66
SECOND_BEST_MARGIN = 0.04
DETECTION_SCORE_THRESHOLD = 0.85
DETECTION_NMS_THRESHOLD = 0.35
MAX_DETECTED_FACES = 5
MIN_REGISTER_SHARPNESS = 30.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Model loading -----------------
def load_model_retina(model_path, network_name):
    cfg = cfg_mnet if network_name == "mobile0.25" else cfg_re50
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"RetinaFace weight not found: {model_path}")

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
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"ArcFace weight not found: {weight_path}")

    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.to(device).eval()
    return net


# Load once
net_arc = load_model_arc(ARCFACE_MODEL, ARCFACE_WEIGHT)
net_retina, cfg = load_model_retina(RETINA_MODEL_PATH, RETINA_NETWORK)
anti_spoof_predictor = AntiSpoofPredict(device_id=0)


# ----------------- Image helpers -----------------
def bytes_to_cv2_image(image_bytes):
    """Convert bytes (uploaded image) to OpenCV BGR image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)  # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def normalize_embedding(feat):
    feat = feat.squeeze()
    denom = np.linalg.norm(feat)
    if denom > 0:
        feat = feat / denom
    return feat


def prepare_arcface_tensor(face_bgr):
    face = cv2.resize(face_bgr, ARCFACE_OUTPUT_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.transpose(face, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(face).unsqueeze(0).float().to(device)
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def get_embedding_from_face(face_bgr):
    """
    Input: face image in BGR (OpenCV), arbitrary size.
    Output: L2-normalized 1D numpy array embedding.
    """
    img = prepare_arcface_tensor(face_bgr)
    flipped = torch.flip(img, dims=[3])
    with torch.no_grad():
        feat = net_arc(img)
        feat_flip = net_arc(flipped)
    feat = (feat + feat_flip).cpu().numpy()
    return normalize_embedding(feat)


def face_sharpness_score(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def has_glare(face_bgr, threshold=245, ratio=0.07):
    # Chỉ kiểm tra nửa trên (vùng mắt/kính)
    h = face_bgr.shape[0]
    eye_region = face_bgr[:h // 2, :]
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    return float(np.sum(gray > threshold)) / gray.size > ratio


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


def nms_boxes(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def detect_faces(frame_bgr, max_faces=MAX_DETECTED_FACES):
    img = np.float32(frame_bgr)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    im_height, im_width = frame_bgr.shape[:2]
    scale = torch.tensor([im_width, im_height, im_width, im_height], dtype=torch.float32).to(device)
    scale_landms = torch.tensor([im_width, im_height] * 5, dtype=torch.float32).to(device)

    with torch.no_grad():
        loc, conf, landms = net_retina(img_tensor)
        priors = PriorBox(cfg, image_size=(im_height, im_width)).forward().to(device)
        boxes = decode(loc.squeeze(0), priors, cfg["variance"]) * scale
        landms = decode_landm(landms.squeeze(0), priors, cfg["variance"]) * scale_landms
        scores = conf.squeeze(0)[:, 1].cpu().numpy()

    inds = np.where(scores > DETECTION_SCORE_THRESHOLD)[0]
    if len(inds) == 0:
        return []

    boxes = boxes[inds].cpu().numpy()
    landms = landms[inds].cpu().numpy()
    scores = scores[inds]

    keep = nms_boxes(boxes, scores, DETECTION_NMS_THRESHOLD)
    detections = []

    for idx in keep[:max_faces]:
        box = boxes[idx].astype(int)
        x1, y1, x2, y2 = box[:4]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(im_width, x2), min(im_height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "box": np.array([x1, y1, x2, y2], dtype=int),
            "landmarks": landms[idx].reshape((5, 2)),
            "score": float(scores[idx]),
        })

    detections.sort(
        key=lambda det: (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1]),
        reverse=True,
    )
    return detections


def extract_aligned_face(frame_bgr, require_register_quality=False):
    detections = detect_faces(frame_bgr, max_faces=1)
    if not detections:
        return None, None, None
    return extract_aligned_face_from_detection(frame_bgr, detections[0], require_register_quality=require_register_quality)


def extract_aligned_face_from_detection(frame_bgr, detection, require_register_quality=False):
    box = detection["box"]
    landmarks = detection["landmarks"]
    x1, y1, x2, y2 = box

    face_size = min(x2 - x1, y2 - y1)
    if require_register_quality and face_size < MIN_REGISTER_FACE_SIZE:
        return None, box, landmarks

    aligned_face = align_face(frame_bgr, landmarks)
    if aligned_face is None or aligned_face.size == 0:
        return None, box, landmarks

    return aligned_face, box, landmarks


def embedding_from_image_bytes(image_bytes):
    frame = bytes_to_cv2_image(image_bytes)
    aligned_face, _, _ = extract_aligned_face(frame, require_register_quality=True)
    if aligned_face is None:
        return None
    if face_sharpness_score(aligned_face) < MIN_REGISTER_SHARPNESS:
        return None
    if has_glare(aligned_face):
        return None
    return get_embedding_from_face(aligned_face)



# ----------------- DB helpers (ORM) -----------------
def load_all_embeddings_from_db():
    """
    Trả về tuple (names_list, embs_list)
    - names_list: list string (user_id)
    - embs_list: list of vectors (list or np.array)
    Lấy từ bảng Embedding. Nếu nhiều vector cùng user thì user xuất hiện nhiều lần.
    """
    from django.db import close_old_connections
    close_old_connections()  # tránh connection cũ bị drop trong thread pool
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


def load_grouped_embeddings_from_db():
    grouped = {}
    names, embs = load_all_embeddings_from_db()
    for user_id, emb in zip(names, embs):
        grouped.setdefault(user_id, []).append(emb)
    return grouped


def compute_user_score(query_emb, user_embs):
    embs_arr = np.array(user_embs, dtype=np.float32)
    if len(embs_arr) == 0:
        return None

    sims = cosine_similarity([query_emb], embs_arr)[0]
    top_k = min(3, len(sims))
    top_mean = float(np.mean(np.sort(sims)[-top_k:]))

    centroid = normalize_embedding(np.mean(embs_arr, axis=0))
    centroid_sim = float(cosine_similarity([query_emb], [centroid])[0][0])

    # Ưu tiên embedding trung bình của user để giảm false positive do một ảnh lẻ bị "đẹp điểm".
    return 0.7 * centroid_sim + 0.3 * top_mean


def match_embedding_to_user(query_emb):
    grouped = load_grouped_embeddings_from_db()
    if not grouped:
        logger.warning("match_embedding_to_user: DB trống, không có embedding nào")
        return None

    scored = []
    for user_id, user_embs in grouped.items():
        score = compute_user_score(query_emb, user_embs)
        if score is not None:
            scored.append((user_id, score, len(user_embs)))

    if not scored:
        return None

    scored.sort(key=lambda item: item[1], reverse=True)
    best_user, best_score, best_count = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else None

    threshold = SINGLE_USER_MATCH_THRESHOLD if len(scored) == 1 else USER_MATCH_THRESHOLD
    if best_count < 5:
        threshold += 0.04

    is_match = best_score >= threshold
    if second_score is not None and (best_score - second_score) < SECOND_BEST_MARGIN:
        is_match = False

    logger.info(
        "Nhận diện: best=%s score=%.4f threshold=%.4f emb_count=%d is_match=%s",
        best_user, best_score, threshold, best_count, is_match,
    )
    return {
        "matched_name": best_user if is_match else "Unknown",
        "best_score": float(best_score),
        "second_score": float(second_score) if second_score is not None else None,
        "threshold": float(threshold),
        "user_count": len(scored),
    }


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
    aligned_face, _, _ = extract_aligned_face(frame)
    if aligned_face is None:
        return {"status": "no_face"}

    emb = get_embedding_from_face(aligned_face)

    match = match_embedding_to_user(emb)
    if match is None:
        return {"status": "unknown", "similarity": None}
    return {
        "status": "ok",
        "name": match["matched_name"],
        "similarity": match["best_score"],
    }


def recognize_faces_from_image_bytes(image_bytes):
    frame = bytes_to_cv2_image(image_bytes)
    detections = detect_faces(frame)
    if not detections:
        return []

    faces = []
    for detection in detections:
        b = detection["box"]
        face_size = min(b[2] - b[0], b[3] - b[1])
        if face_size < MIN_RECOGNIZE_FACE_SIZE or face_size > MAX_RECOGNIZE_FACE_SIZE:
            continue

        aligned_face, box, _ = extract_aligned_face_from_detection(frame, detection)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        w, h = int(x2 - x1), int(y2 - y1)

        # --- Kiểm tra chống giả mạo (liveness detection) ---
        liveness = anti_spoof_predictor.predict_from_bbox(
            frame, [int(x1), int(y1), w, h], ANTI_SPOOF_MODELS
        )
        if not liveness["is_real"]:
            faces.append({
                "name": "Spoof",
                "similarity": None,
                "threshold": None,
                "box": {"x": int(x1), "y": int(y1), "w": w, "h": h},
                "detection_score": detection["score"],
                "is_spoof": True,
                "liveness_score": liveness["liveness_score"],
            })
            continue

        # --- Nhận diện khuôn mặt thật ---
        if aligned_face is None:
            continue

        emb = get_embedding_from_face(aligned_face)
        match = match_embedding_to_user(emb)

        faces.append({
            "name": match["matched_name"] if match else "Unknown",
            "similarity": match["best_score"] if match else None,
            "threshold": match["threshold"] if match else None,
            "box": {"x": int(x1), "y": int(y1), "w": w, "h": h},
            "detection_score": detection["score"],
            "is_spoof": False,
            "liveness_score": liveness["liveness_score"],
        })

    return faces


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
    faces = recognize_faces_from_image_bytes(image_bytes)
    if not faces:
        return {"status": "no_face", "faces": []}

    primary_face = max(faces, key=lambda face: face["box"]["w"] * face["box"]["h"])
    return {
        "status": "ok",
        "name": primary_face["name"],
        "similarity": primary_face["similarity"],
        "box": primary_face["box"],
        "faces": faces,
    }


def detect_face_size_for_preview(image_bytes):
    frame = bytes_to_cv2_image(image_bytes)
    detections = detect_faces(frame, max_faces=1)
    if not detections:
        return {"status": "no_face"}
    box = detections[0]["box"]
    x1, y1, x2, y2 = box
    face_size = min(x2 - x1, y2 - y1)
    if face_size < MIN_REGISTER_FACE_SIZE:
        state = "too_small"
    elif face_size > MAX_RECOGNIZE_FACE_SIZE:
        state = "too_large"
    else:
        state = "ok"
    return {
        "status": "ok",
        "face_size": int(face_size),
        "state": state,
        "min": MIN_REGISTER_FACE_SIZE,
        "max": MAX_RECOGNIZE_FACE_SIZE,
    }


def align_face(frame_bgr, landmarks):
    """
    Align face chuẩn ArcFace từ 5 landmark và trả về ảnh BGR 112x112.
    """
    if landmarks is None or len(landmarks) != 5:
        return None

    src = landmarks.astype(np.float32)
    transform = cv2.estimateAffinePartial2D(src, ARCFACE_STANDARD_LANDMARKS)[0]
    if transform is None:
        return None

    return cv2.warpAffine(frame_bgr, transform, ARCFACE_OUTPUT_SIZE, flags=cv2.INTER_CUBIC)


# --- Ngưỡng đa dạng góc khi đăng ký ---
# Nếu embedding mới có cosine similarity >= ngưỡng này với bất kỳ embedding nào đã lưu
# → coi là cùng góc → bỏ qua. Ngược lại → góc mới → lưu.
REGISTER_DIVERSITY_THRESHOLD = 0.82


def check_and_register_frame(image_bytes, user_id):
    """
    Trích embedding từ ảnh, so sánh đa dạng với toàn bộ embedding đã có trong DB của user.
    - Nếu đủ khác biệt (max_sim < threshold) → lưu vào DB, trả về status='added'.
    - Nếu quá giống (max_sim >= threshold) → bỏ qua, trả về status='skipped'.
    - Nếu không detect được mặt → status='no_face'.
    """
    try:
        user = Employee.objects.get(user_id=user_id)
    except Employee.DoesNotExist:
        return {"status": "fail", "message": "user_not_found"}

    frame = bytes_to_cv2_image(image_bytes)
    aligned_face, _, _ = extract_aligned_face(frame, require_register_quality=True)
    if aligned_face is None:
        return {"status": "no_face"}

    if face_sharpness_score(aligned_face) < MIN_REGISTER_SHARPNESS:
        return {"status": "no_face", "reason": "blurry"}

    new_emb = get_embedding_from_face(aligned_face)

    # Load tất cả embedding hiện có của user
    existing_vecs = []
    for e in Embedding.objects.filter(user=user):
        try:
            vec = np.array(json.loads(e.vector), dtype=np.float32)
            denom = np.linalg.norm(vec)
            if denom > 0:
                vec /= denom
            existing_vecs.append(vec)
        except Exception:
            continue

    total = len(existing_vecs)

    # Kiểm tra đa dạng
    max_sim = 0.0
    if existing_vecs:
        sims = cosine_similarity([new_emb], existing_vecs)[0]
        max_sim = float(np.max(sims))

    if existing_vecs and max_sim >= REGISTER_DIVERSITY_THRESHOLD:
        return {
            "status": "skipped",
            "max_similarity": round(max_sim, 4),
            "total_embeddings": total,
        }

    Embedding.objects.create(user=user, vector=json.dumps(new_emb.tolist()))
    return {
        "status": "added",
        "max_similarity": round(max_sim, 4),
        "total_embeddings": total + 1,
    }
