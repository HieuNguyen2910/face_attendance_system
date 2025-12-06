import os
import glob
import cv2
import torch
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from models.retinaface import RetinaFace

# ------------------ cấu hình ------------------
video_folder = './archive/train/attack'   # folder chứa video gốc
save_folder = './archive/checktest/fake'    # folder lưu khuôn mặt crop
trained_model = './weights/Resnet50_Final.pth'
network = 'resnet50'             
resize = 1.0                      
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
min_face_size = 80         # chỉ lấy mặt >= 80px
faces_per_second = 2       # 1-2 khuôn mặt mỗi giây
padding_ratio = 0.2        # mở rộng bounding box 20%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------------------------------

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Load model
cfg = cfg_mnet if network=='mobile0.25' else cfg_re50
net = RetinaFace(cfg=cfg, phase='test')
checkpoint = torch.load(trained_model, map_location='cpu')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
state_dict = {k.replace('module.', ''):v for k,v in state_dict.items()}
net.load_state_dict(state_dict, strict=False)
net.eval()
net = net.to(device)

def preprocess_frame(frame, resize):
    if resize != 1.0:
        frame_proc = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    else:
        frame_proc = frame
    img = np.float32(frame_proc)
    img -= (104,117,123)
    img = img.transpose(2,0,1)
    img = torch.from_numpy(img).unsqueeze(0)
    h, w = frame_proc.shape[:2]
    return img.to(device), w, h, frame_proc

# Duyệt video
video_paths = glob.glob(os.path.join(video_folder, '*.*'))  
for vid_path in video_paths:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Cannot open', vid_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / faces_per_second))  # lấy 1-2 frame mỗi giây
    base_name = os.path.splitext(os.path.basename(vid_path))[0]
    frame_count = 0
    face_count = 0

    print('Processing', vid_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # chỉ xử lý frame theo interval
        if frame_count % frame_interval != 0:
            continue

        img_tensor, im_width, im_height, frame_proc = preprocess_frame(frame, resize)
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)

        with torch.no_grad():
            loc, conf, landms = net(img_tensor)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:,1]

        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        if dets.shape[0]>0:
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep,:]
        else:
            dets = np.empty((0,5), dtype=np.float32)

        min_confidence = 0.9
        # lọc box nhỏ
        # lọc box lớn và confidence cao
        large_dets = []
        for b in dets:
            x1, y1, x2, y2, score = map(float, b[:5])
            w_box = x2 - x1
            h_box = y2 - y1
            if w_box >= min_face_size and h_box >= min_face_size and score >= min_confidence:
                large_dets.append([int(x1), int(y1), int(x2), int(y2), score])
        large_dets = np.array(large_dets)


        # lấy tối đa faces_per_second khuôn mặt mỗi frame
        if large_dets.shape[0] > faces_per_second:
            areas = (large_dets[:,2]-large_dets[:,0])*(large_dets[:,3]-large_dets[:,1])
            idxs = areas.argsort()[::-1][:faces_per_second]
            large_dets = large_dets[idxs]

        # crop và lưu với padding
        for i, b in enumerate(large_dets):
            x1, y1, x2, y2 = map(int, b[:4])
            w_box = x2 - x1
            h_box = y2 - y1
            pad_w = int(w_box * padding_ratio)
            pad_h = int(h_box * padding_ratio)
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(frame.shape[1], x2 + pad_w)
            y2_pad = min(frame.shape[0], y2 + pad_h)

            face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if face_crop.size == 0:
                continue
            fname = os.path.join(save_folder, f'{base_name}_frame{frame_count:04d}_face{i:02d}.jpg')
            cv2.imwrite(fname, face_crop)
            face_count += 1

    cap.release()
    print(f'Saved {face_count} faces from {vid_path}')

print('All videos processed! Faces saved to', save_folder)
