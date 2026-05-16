# -*- coding: utf-8 -*-
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name
from src.generate_patches import CropImage

MODEL_MAPPING = {
    'MiniFASNetV1':   MiniFASNetV1,
    'MiniFASNetV2':   MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}

# Ngưỡng xác suất trung bình class "real" — dưới ngưỡng này là giả mạo
LIVENESS_THRESHOLD = 0.55


class AntiSpoofPredict:
    def __init__(self, device_id=0):
        self.device = torch.device(
            "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
        )
        # Cache model đã tải để không reload mỗi lần gọi predict
        self._model_cache: dict = {}

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Tải MiniFASNet từ file .pth và cache lại, chỉ load một lần duy nhất."""
        if model_path in self._model_cache:
            return self._model_cache[model_path]

        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input)
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        # Xử lý trọng số lưu bằng DataParallel (tiền tố "module.")
        first_key = next(iter(state_dict))
        if first_key.startswith('module.'):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)
        model.eval()

        self._model_cache[model_path] = model
        return model

    def predict(self, img: np.ndarray, model_path: str) -> np.ndarray:
        """
        Chạy inference một model trên ảnh patch đã crop sẵn.

        Args:
            img:        Ảnh patch BGR numpy uint8 (đã resize theo yêu cầu model).
            model_path: Đường dẫn tới file .pth của model.

        Returns:
            np.ndarray shape (1, 3) — softmax probabilities:
            [print_attack_prob, real_prob, replay_attack_prob]
        """
        transform = trans.Compose([trans.ToTensor()])
        tensor = transform(img).unsqueeze(0).to(self.device)
        model = self._load_model(model_path)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_from_bbox(self, image: np.ndarray, bbox_xywh: list,
                          model_paths: list) -> dict:
        """
        Kiểm tra liveness cho khuôn mặt tại bbox_xywh bằng ensemble các model.
        Sử dụng bounding box từ RetinaFace — không cần chạy face detector riêng.

        Args:
            image:       Ảnh BGR gốc (numpy uint8 array).
            bbox_xywh:   Bounding box [x, y, w, h] của khuôn mặt.
            model_paths: Danh sách đường dẫn (str hoặc Path) tới các file .pth.

        Returns:
            {
                'is_real':        bool   — True nếu khuôn mặt thật,
                'liveness_score': float  — Xác suất trung bình class real [0, 1],
                'label':          str    — 'real' hoặc 'spoof',
            }
        """
        image_cropper = CropImage()
        prediction = np.zeros((1, 3))
        models_used = 0

        for model_path in model_paths:
            model_path = str(model_path)
            if not os.path.exists(model_path):
                continue
            model_name = os.path.basename(model_path)
            try:
                h_input, w_input, _, scale = parse_model_name(model_name)
            except Exception:
                continue

            crop_params = {
                "org_img": image,
                "bbox":    bbox_xywh,
                "scale":   scale,
                "out_w":   w_input,
                "out_h":   h_input,
                "crop":    scale is not None,
            }
            try:
                img_patch = image_cropper.crop(**crop_params)
                prediction += self.predict(img_patch, model_path)
                models_used += 1
            except Exception as e:
                print(f"[AntiSpoof] Warning: {model_name} skipped — {e}")

        if models_used == 0:
            # Không có model nào chạy được — fail open để không chặn nhận diện
            return {"is_real": True, "liveness_score": 1.0, "label": "real"}

        # class 0 = print attack, class 1 = real face, class 2 = video replay
        label = int(np.argmax(prediction))
        real_score = float(prediction[0][1]) / models_used

        is_real = (label == 1) and (real_score >= LIVENESS_THRESHOLD)
        return {
            "is_real":        is_real,
            "liveness_score": round(real_score, 4),
            "label":          "real" if is_real else "spoof",
        }
