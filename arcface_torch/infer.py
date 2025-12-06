import time

start = time.time()

import argparse
import cv2
import numpy as np
import torch
from backbones import get_model
from sklearn.metrics.pairwise import cosine_similarity


@torch.no_grad()
def get_embedding(weight, name, img_path):
    """Trả về vector embedding của ảnh khuôn mặt."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location="cpu"))
    net.eval()
    feat = net(img).numpy()
    feat = feat / np.linalg.norm(feat)  # chuẩn hoá vector
    return feat


def match_faces(feat1, feat2, threshold=0.7):
    """So sánh hai embedding, trả về độ tương đồng."""
    sim = cosine_similarity(feat1, feat2)[0][0]
    same = sim > threshold
    return sim, same


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArcFace Matching')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, required=True, help='đường dẫn model ArcFace .pth')
    parser.add_argument('--img1', type=str, required=True, help='ảnh 1')
    parser.add_argument('--img2', type=str, required=True, help='ảnh 2')
    parser.add_argument('--threshold', type=float, default=0.5, help='ngưỡng nhận diện')
    args = parser.parse_args()

    feat1 = get_embedding(args.weight, args.network, args.img1)
    feat2 = get_embedding(args.weight, args.network, args.img2)

    sim, same = match_faces(feat1, feat2, args.threshold)
    print(f"🔹 Cosine similarity: {sim:.4f}")
    print(f"🔹 Kết luận: {'Cùng người' if same else 'Khác người'}")

finish = time.time()

print("Chay trong: ", finish - start)