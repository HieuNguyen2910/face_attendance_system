import asyncio
import base64
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer

from .face_recognition import (
    detect_faces,
    extract_aligned_face_from_detection,
    get_embedding_from_face,
    match_embedding_to_user,
    anti_spoof_predictor,
    ANTI_SPOOF_MODELS,
    MIN_RECOGNIZE_FACE_SIZE,
    MAX_RECOGNIZE_FACE_SIZE,
)
from .sort_tracker import SortTracker

logger = logging.getLogger(__name__)

# GPU executor: single-threaded so models are never called concurrently
_executor = ThreadPoolExecutor(max_workers=1)
_camera_online = False

CAMERA_GROUP = "camera_display"
ANTI_SPOOF_INTERVAL = 5
DISPLAY_FPS_CAP = 20                     # max frames/s sent to browser
_DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS_CAP


async def _run(func, *args):
    """Run a blocking function in the GPU executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, func, *args)


async def _run_cpu(func, *args):
    """Run a blocking function in the default thread pool (CPU tasks)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


# ---------- sync helpers ----------

def _anti_spoof(frame, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return {"is_real": False, "liveness_score": 0.0}
    return anti_spoof_predictor.predict_from_bbox(frame, [x1, y1, w, h], ANTI_SPOOF_MODELS)


def _recognize(frame, detection):
    try:
        aligned, _, _ = extract_aligned_face_from_detection(frame, detection)
        if aligned is None:
            logger.debug("_recognize: không align được khuôn mặt")
            return None
        emb = get_embedding_from_face(aligned)
        return match_embedding_to_user(emb)
    except Exception as exc:
        logger.error("Recognition error: %s", exc, exc_info=True)
        return None


def _encode_frame_for_display(frame, max_width: int = 480) -> str | None:
    """Resize frame → encode as JPEG → base64 string, targeting ≤100 KB."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------- Local camera reader ----------

_local_reader = None          # type: LocalCameraReader | None
_local_reader_task = None     # type: asyncio.Task | None
_display_client_count = 0


class LocalCameraReader:
    """
    Reads frames from the local webcam in a dedicated thread and broadcasts
    them to CAMERA_GROUP via the channel layer.

    Architecture:
    - Camera thread: reads frames continuously from OpenCV, pushes to asyncio.Queue
      via call_soon_threadsafe (no run_in_executor overhead per frame).
    - Async loop: consumes queue, runs two pipelines per frame:
        _forward_frame : CPU resize+encode → display (low latency)
        _process       : GPU inference (SORT + anti-spoof + ArcFace) → bbox results
    - Queue maxsize=2: old frames are dropped when processing can't keep up,
      keeping the stream always showing the latest frame.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.tracker = SortTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        self.track_cache: dict = {}
        self.frame_count = 0
        self._busy = False
        self._display_busy = False
        self._last_display_time = 0.0
        self._stop = threading.Event()
        self._frame_queue: asyncio.Queue | None = None

    def stop(self):
        self._stop.set()

    async def run(self):
        global _camera_online
        channel_layer = get_channel_layer()
        loop = asyncio.get_running_loop()
        self._frame_queue = asyncio.Queue(maxsize=2)

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            logger.error("Cannot open local camera (index %d)", self.camera_index)
            return

        def _put_frame(frame):
            # Drop oldest frame if queue is full — always keep the latest
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass

        def _read_loop():
            """Dedicated camera thread — reads frames as fast as camera allows."""
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                # Bridge from thread to async event loop (thread-safe)
                loop.call_soon_threadsafe(_put_frame, frame)
            cap.release()

        cam_thread = threading.Thread(target=_read_loop, daemon=True)
        cam_thread.start()

        _camera_online = True
        await channel_layer.group_send(CAMERA_GROUP, {"type": "camera.status", "online": True})
        logger.info("Local camera started (index %d)", self.camera_index)

        try:
            while not self._stop.is_set():
                try:
                    frame = await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Display path: FPS-capped + skip if still encoding previous frame
                now = time.monotonic()
                if not self._display_busy and (now - self._last_display_time) >= _DISPLAY_INTERVAL:
                    self._display_busy = True
                    self._last_display_time = now
                    asyncio.ensure_future(self._forward_frame(frame, channel_layer))

                # Inference path: GPU pipeline (skip if busy)
                if not self._busy:
                    self._busy = True
                    asyncio.ensure_future(self._process(frame.copy(), channel_layer))

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Camera reader error: %s", exc)
        finally:
            self._stop.set()
            cam_thread.join(timeout=2)
            _camera_online = False
            try:
                await channel_layer.group_send(
                    CAMERA_GROUP, {"type": "camera.status", "online": False}
                )
            except Exception:
                pass
            logger.info("Local camera stopped")

    async def _forward_frame(self, frame, channel_layer):
        """Encode and push frame to browser — CPU only, default thread pool."""
        try:
            frame_data = await _run_cpu(_encode_frame_for_display, frame)
            if frame_data:
                await channel_layer.group_send(CAMERA_GROUP, {
                    "type": "camera.frame",
                    "frame_data": frame_data,
                })
        except Exception as exc:
            logger.error("Frame forward error: %s", exc)
        finally:
            self._display_busy = False

    async def _process(self, frame, channel_layer):
        """Full GPU inference pipeline — result sent separately from display frames."""
        try:
            frame_h, frame_w = frame.shape[:2]
            self.frame_count += 1

            detections = await _run(detect_faces, frame)
            tracks = self.tracker.update(detections)

            results = []
            for trk in tracks:
                tid = trk["id"]
                detection = trk.get("detection")
                box = trk["box"]
                x1, y1, x2, y2 = box
                face_size = min(x2 - x1, y2 - y1)
                size_ok = MIN_RECOGNIZE_FACE_SIZE <= face_size <= MAX_RECOGNIZE_FACE_SIZE

                if tid not in self.track_cache:
                    self.track_cache[tid] = {
                        "identity": None,      # identity đã được confirm (None = chưa xác định)
                        "confirmed": False,    # True khi đã thấy cùng id 3 lần liên tiếp
                        "pending_id": None,    # id đang được tích lũy để confirm
                        "consecutive": 0,      # số lần liên tiếp thấy pending_id
                        "is_real": None,
                        "last_spoof_frame": -999,
                        "liveness_score": None,
                    }
                cache = self.track_cache[tid]

                # Anti-spoof: chạy định kỳ, reset tracking nếu phát hiện spoof
                if size_ok and detection and (
                    self.frame_count - cache["last_spoof_frame"] >= ANTI_SPOOF_INTERVAL
                ):
                    liveness = await _run(_anti_spoof, frame, detection["box"])
                    cache["is_real"] = liveness["is_real"]
                    cache["liveness_score"] = liveness.get("liveness_score")
                    cache["last_spoof_frame"] = self.frame_count
                    if not liveness["is_real"]:
                        # Spoof phát hiện → xóa tracking, nhận diện lại từ đầu
                        cache["identity"] = None
                        cache["confirmed"] = False
                        cache["pending_id"] = None
                        cache["consecutive"] = 0

                # Nhận diện: chỉ chạy khi chưa confirm
                # - Unknown/Spoof: không bao giờ lock → cứ nhận diện lại mỗi frame
                # - Có id thật 3 lần liên tiếp → lock (confirmed=True), dừng nhận diện
                if size_ok and cache["is_real"] and not cache["confirmed"] and detection:
                    match = await _run(_recognize, frame, detection)
                    result_id = match["matched_name"] if match else "Unknown"

                    if result_id not in ("Unknown", "Spoof"):
                        # Tích lũy consecutive count
                        if result_id == cache["pending_id"]:
                            cache["consecutive"] += 1
                        else:
                            cache["pending_id"] = result_id
                            cache["consecutive"] = 1

                        if cache["consecutive"] >= 3:
                            # Đủ 3 lần liên tiếp → lock identity
                            cache["identity"] = result_id
                            cache["confirmed"] = True
                            logger.info("Track %d confirmed: %s", tid, result_id)
                    else:
                        # Unknown hoặc Spoof → reset pending, không lock
                        cache["pending_id"] = None
                        cache["consecutive"] = 0

                # Xác định tên hiển thị
                if cache["is_real"] is None:
                    name = "Unknown"
                elif not cache["is_real"]:
                    name = "Spoof"
                elif cache["confirmed"]:
                    name = cache["identity"]
                else:
                    name = "Unknown"  # Đang tích lũy, chưa đủ 3 lần

                results.append({
                    "id": tid,
                    "name": name,
                    "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                })

            live_ids = {t["id"] for t in tracks}
            for tid in list(self.track_cache):
                if tid not in live_ids:
                    del self.track_cache[tid]

            await channel_layer.group_send(CAMERA_GROUP, {
                "type": "camera.result",
                "tracks": results,
                "frame_width": frame_w,
                "frame_height": frame_h,
            })

        except Exception as exc:
            logger.error("Inference error: %s", exc)
        finally:
            self._busy = False


# ---------- consumers ----------

class CameraDisplayConsumer(AsyncWebsocketConsumer):
    """
    Browser-side WebSocket consumer.

    Lifecycle:
    - First browser to connect starts the LocalCameraReader (camera turns on).
    - Last browser to disconnect stops the LocalCameraReader (camera turns off).
    """

    async def connect(self):
        global _display_client_count, _local_reader, _local_reader_task
        await self.channel_layer.group_add(CAMERA_GROUP, self.channel_name)
        await self.accept()
        await self.send(text_data=json.dumps({
            "type": "status",
            "online": _camera_online,
        }))

        _display_client_count += 1
        if _display_client_count == 1 and (
            _local_reader_task is None or _local_reader_task.done()
        ):
            _local_reader = LocalCameraReader(camera_index=0)
            _local_reader_task = asyncio.ensure_future(_local_reader.run())

    async def disconnect(self, code):
        global _display_client_count, _local_reader, _local_reader_task
        await self.channel_layer.group_discard(CAMERA_GROUP, self.channel_name)
        _display_client_count = max(0, _display_client_count - 1)
        if _display_client_count == 0:
            if _local_reader is not None:
                _local_reader.stop()
            if _local_reader_task is not None and not _local_reader_task.done():
                _local_reader_task.cancel()
            _local_reader = None
            _local_reader_task = None

    async def receive(self, text_data=None, bytes_data=None):
        pass

    async def camera_frame(self, event):
        """Display frame — sent at camera FPS (lightweight CPU path)."""
        await self.send(text_data=json.dumps({
            "type": "frame",
            "frame_data": event["frame_data"],
        }))

    async def camera_result(self, event):
        """Inference result — sent at GPU inference rate."""
        await self.send(text_data=json.dumps({
            "type": "result",
            "tracks": event["tracks"],
            "frame_width": event["frame_width"],
            "frame_height": event["frame_height"],
        }))

    async def camera_status(self, event):
        await self.send(text_data=json.dumps({
            "type": "status",
            "online": event["online"],
        }))
