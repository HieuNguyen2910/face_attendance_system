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
video_folder = './archive/test/real_video'   # folder chứa video gốc
save_folder = './archive/checktest/real'   # folder lưu video đã khoanh mặt
trained_model = './weights/Resnet50_Final.pth'
network = 'resnet50'             # 'mobile0.25' hoặc 'resnet50'
resize = 1.0                      # 0.5, 0.75 để tăng tốc
confidence_threshold = 0.02
vis_thres = 0.6
top_k = 5000
nms_threshold = 0.4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------------------------------

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Load model
cfg = cfg_mnet if network=='mobile0.25' else cfg_re50
net = RetinaFace(cfg=cfg, phase='test')
checkpoint = torch.load(trained_model, map_location='cpu')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
# remove 'module.' prefix
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

def draw_detections(frame, dets, vis_thres=0.5):
    for b in dets:
        if b[4] < vis_thres:
            continue
        x1,y1,x2,y2 = map(int, b[:4])
        score = b[4]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, f'{score:.2f}', (x1, max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        lm = b[5:15].astype(np.int32)
        for i in range(0,10,2):
            cv2.circle(frame, (lm[i], lm[i+1]), 2, (0,255,0), -1)

# Duyệt video
video_paths = glob.glob(os.path.join(video_folder, '*.*'))  # mp4/avi/...
for vid_path in video_paths:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Cannot open', vid_path)
        continue

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_name = os.path.basename(vid_path)
    save_path = os.path.join(save_folder, base_name)
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w,h))

    print('Processing', vid_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor, im_width, im_height, frame_proc = preprocess_frame(frame, resize)
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
        scale1 = torch.Tensor([im_width, im_height]*5).to(device)

        with torch.no_grad():
            loc, conf, landms = net(img_tensor)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:,1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # filter
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # sort top_k
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        if dets.shape[0]>0:
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep,:]
            landms = landms[keep]
        else:
            dets = np.empty((0,5), dtype=np.float32)
            landms = np.empty((0,10), dtype=np.float32)

        if dets.shape[0]>0:
            dets = np.concatenate((dets, landms), axis=1)
        else:
            dets = np.empty((0,15), dtype=np.float32)

        draw_detections(frame, dets, vis_thres)

        writer.write(frame)

    cap.release()
    writer.release()
    print('Saved', save_path)

print('All videos processed!')
