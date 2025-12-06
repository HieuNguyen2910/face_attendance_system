
"""
retinaface_webcam.py

Chi tiết, chuẩn xác: script chạy inference RetinaFace trên webcam hoặc file video.
Tính năng:
 - Load checkpoint (mobile0.25 / resnet50) robust (hỗ trợ checkpoint có/không có 'state_dict' và prefix 'module.')
 - Chạy realtime trên webcam (hoặc video file) với option resize để tăng FPS
 - Hiển thị bounding boxes, score, 5 landmarks
 - Tính và hiển thị FPS + thời gian forward pass
 - Lưu ảnh chụp màn hình bằng phím 'c'
 - Bắt đầu/tạm dừng ghi video bằng phím 'r' (hoặc sử dụng --save_video để tự bật)
 - Thông số cấu hình (thresholds, nms, top_k, vis_thres) qua argparse

Yêu cầu repository: file này giả định bạn đang chạy trong repo RetinaFace chứa các module:
  - data.cfg_mnet, cfg_re50
  - layers.functions.prior_box.PriorBox
  - utils.nms.py_cpu_nms.py
  - utils.box_utils.decode, decode_landm
  - models.retinaface.RetinaFace

Cài đặt phụ thuộc (ví dụ):
  pip install torch torchvision opencv-python numpy

Usage ví dụ:
  python retinaface_webcam.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --input 0 --vis_thres 0.6 --resize 0.75
  python retinaface_webcam.py --trained_model ./weights/Resnet50_Final.pth --network resnet50 --input video.mp4 --save_video out.mp4

Controls:
  q : thoát
  c : chụp ảnh (lưu vào --save_folder)
  r : bật/tắt ghi video (nếu không dùng --save_video từ đầu)

"""
from __future__ import print_function
import os
import argparse
import time
import datetime
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """Remove prefix of checkpoint keys (e.g. 'module.')"""
    print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu=False):
    """Load checkpoint in a robust way (supports 'state_dict' key and 'module.' prefix).
    If load_to_cpu True => load to CPU regardless of cuda availability.
    """
    print('Loading pretrained model from {}'.format(pretrained_path))
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError('Pretrained model not found: {}'.format(pretrained_path))

    if load_to_cpu or not torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    else:
        pretrained_dict = torch.load(pretrained_path)

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class SimpleTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._t = 0.0
        self._n = 0

    def update(self, t):
        self._t += t
        self._n += 1

    @property
    def avg(self):
        return self._t / self._n if self._n > 0 else 0.0


def preprocess_frame(frame, resize):
    """Trả về tensor (1,C,H,W) và kích thước xử lý (width,height)"""
    if resize != 1.0:
        frame_proc = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    else:
        frame_proc = frame
    img = np.float32(frame_proc)
    # mean subtraction (same as training)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    im_height, im_width = frame_proc.shape[0], frame_proc.shape[1]
    return img, im_width, im_height, frame_proc


def draw_detections(frame, dets, vis_thres=0.5):
    """Vẽ bounding box + score + landmarks lên frame (in-place).
    dets: numpy array shape (k, 15) -> [xmin,ymin,xmax,ymax,score, lm_x1,lm_y1, ... lm_x5,lm_y5]
    """
    for b in dets:
        if b[4] < vis_thres:
            continue
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        score = b[4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = '{:.2f}'.format(score)
        cv2.putText(frame, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # landmarks
        lm = b[5:15].astype(np.int32)
        # 5 points (x,y) pairs
        for i in range(0, 10, 2):
            cx, cy = int(lm[i]), int(lm[i + 1])
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser(description='RetinaFace webcam / video demo (detailed, production-ready)')
    parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth', type=str,
                        help='Trained state_dict file path to open') #mobilenet0.25_Final.pth Resnet50_Final.pth
    parser.add_argument('--network', default='mobile0.25', choices=['mobile0.25', 'resnet50'],
                        help='Backbone network: mobile0.25 or resnet50') 
    parser.add_argument('--input', default='1', type=str,
                        help='Input source: camera index (0,1,...) or path to video file')
    parser.add_argument('--save_folder', default='snapshots', type=str, help='Folder to save snapshots')
    parser.add_argument('--save_video', default=None, type=str, help='Path to save output video (mp4/avi)')
    parser.add_argument('--cpu', action='store_true', default=False, help='Use CPU for inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualisation threshold to draw')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k before NMS')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--resize', default=1.0, type=float, help='resize factor for processing (0.5, 0.75, 1.0). Use <1.0 to speed up')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    torch.set_grad_enabled(False)
    cudnn.benchmark = True

    cfg = cfg_mnet if args.network == 'mobile0.25' else cfg_re50

    # build net
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    net = net.to(device)
    print('Model loaded to device:', device)

    # Open input
    # support camera index
    try:
        cam_idx = int(args.input)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError('Cannot open input: {}'.format(args.input))

    # Prepare video writer if requested
    writer = None
    recording = False
    if args.save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if args.save_video.lower().endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
        # we'll initialize writer after reading first frame (to know frame size)
        recording = True

    forward_timer = SimpleTimer()
    loop_timer = SimpleTimer()

    # FPS smoothing
    fps = 0.0
    alpha = 0.9

    print('Press q to quit, c to capture snapshot, r to start/stop recording (if --save_video not set)')

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print('End of stream / cannot read frame')
                break

            orig_h, orig_w = frame.shape[0], frame.shape[1]

            # preprocess (optional resize for speed)
            img_tensor, im_width, im_height, frame_proc = preprocess_frame(frame, args.resize)
            img_tensor = img_tensor.to(device)
            scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
            scale1 = torch.Tensor([im_width, im_height] * 5).to(device)

            # forward
            t_f0 = time.perf_counter()
            with torch.no_grad():
                loc, conf, landms = net(img_tensor)
            t_f1 = time.perf_counter()
            forward_time = t_f1 - t_f0
            forward_timer.update(forward_time)

            # decode
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward().to(device)
            prior_data = priors.data

            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            # map from resized coords to original coords: boxes * scale / resize
            boxes = boxes * scale / args.resize
            boxes = boxes.cpu().numpy()

            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            landms = landms * scale1 / args.resize
            landms = landms.cpu().numpy()

            # filter
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # sort top_k
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # nms
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            if dets.shape[0] > 0:
                keep = py_cpu_nms(dets, args.nms_threshold)
                dets = dets[keep, :]
                landms = landms[keep]
            else:
                dets = np.empty((0, 5), dtype=np.float32)
                landms = np.empty((0, 10), dtype=np.float32)

            # concat landmarks
            if dets.shape[0] > 0:
                dets = np.concatenate((dets, landms), axis=1)
            else:
                dets = np.empty((0, 15), dtype=np.float32)

            # draw on original frame (note: boxes were already scaled to original coords)
            draw_detections(frame, dets, vis_thres=args.vis_thres)

            # overlay stats
            loop_time = time.perf_counter() - t0
            loop_timer.update(loop_time)
            fps_inst = 1.0 / loop_time if loop_time > 0 else 0.0
            fps = alpha * fps + (1 - alpha) * fps_inst

            status = 'FPS: {:.1f} | Forward: {:.3f}s (avg {:.3f}s) | Detections: {}'.format(
                fps, forward_time, forward_timer.avg, dets.shape[0])
            cv2.putText(frame, status, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imshow('RetinaFace Demo', frame)

            # init writer if needed (only after first frame to get size)
            if args.save_video is not None and writer is None:
                h, w = frame.shape[0], frame.shape[1]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') if args.save_video.lower().endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (w, h))
                recording = True
                print('Started writing video to', args.save_video)

            # if recording active, write frame
            if recording and writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # capture snapshot
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                fname = os.path.join(args.save_folder, 'snap_{}.jpg'.format(ts))
                cv2.imwrite(fname, frame)
                print('Saved snapshot:', fname)
            elif key == ord('r'):
                # toggle recording if --save_video not set
                if args.save_video is None:
                    if recording:
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None
                        print('Recording stopped')
                    else:
                        # start writer
                        outname = os.path.join(args.save_folder, 'record_{}.mp4'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(outname, fourcc, 20.0, (orig_w, orig_h))
                        recording = True
                        print('Recording to:', outname)
                else:
                    # if save_video provided, r toggles pause/resume
                    recording = not recording
                    print('Recording toggled, now:', recording)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print('Exiting, avg forward time: {:.4f}s, avg loop time: {:.4f}s'.format(forward_timer.avg, loop_timer.avg))


if __name__ == '__main__':
    main()
