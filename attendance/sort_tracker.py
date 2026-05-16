import numpy as np
from scipy.optimize import linear_sum_assignment


def _box_to_z(box):
    """[x1,y1,x2,y2] -> [cx, cy, s, r]  (s=area, r=aspect ratio w/h)"""
    w = float(box[2] - box[0])
    h = float(box[3] - box[1])
    if w <= 0 or h <= 0:
        return None
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h], dtype=np.float64)


def _z_to_box(z):
    """[cx, cy, s, r] -> [x1,y1,x2,y2]"""
    w = np.sqrt(np.abs(z[2] * z[3]))
    h = np.abs(z[2]) / w if w > 0 else 0
    return np.array([z[0] - w / 2, z[1] - h / 2, z[0] + w / 2, z[1] + h / 2], dtype=np.float64)


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class KalmanBoxTracker:
    """Tracks a single bounding box with a constant-velocity Kalman filter.

    State: [cx, cy, s, r, vcx, vcy, vs]  (s=area, r=aspect ratio assumed constant)
    Measurement: [cx, cy, s, r]
    """
    _id_counter = 0

    def __init__(self, box, detection=None):
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter

        self._x = np.zeros((7, 1), dtype=np.float64)
        z = _box_to_z(box)
        if z is not None:
            self._x[:4, 0] = z

        self._P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(np.float64)

        # Transition: constant velocity
        self._F = np.eye(7, dtype=np.float64)
        self._F[0, 4] = self._F[1, 5] = self._F[2, 6] = 1.0

        # Measurement: observe [cx, cy, s, r]
        self._H = np.zeros((4, 7), dtype=np.float64)
        self._H[:4, :4] = np.eye(4)

        self._Q = np.diag([1, 1, 10, 10, 0.01, 0.01, 0.0001]).astype(np.float64)
        self._R = np.diag([1, 1, 10, 10]).astype(np.float64)

        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.last_detection = detection  # full dict {box, landmarks, score}

    def predict(self):
        """Kalman predict step. Returns predicted [x1,y1,x2,y2]."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        self.time_since_update += 1
        return _z_to_box(self._x[:4, 0])

    def update(self, box, detection=None):
        """Kalman update step with new measurement."""
        z = _box_to_z(box)
        if z is None:
            return
        z = z.reshape((4, 1))
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(7) - K @ self._H) @ self._P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_detection = detection

    def get_box(self):
        return _z_to_box(self._x[:4, 0])


class SortTracker:
    """SORT: Simple Online and Realtime Tracking.

    Combines Kalman filter per track with Hungarian algorithm for assignment.
    """

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of dicts from detect_faces():
            {box: np.array([x1,y1,x2,y2]), landmarks: np.array(5x2), score: float}

        Returns list of active track dicts:
            {id, box: [x1,y1,x2,y2] ints, hits, time_since_update, detection}
        """
        self.frame_count += 1

        # Predict all existing trackers
        predicted = []
        dead_indices = []
        for i, trk in enumerate(self.trackers):
            box = trk.predict()
            if np.any(np.isnan(box)):
                dead_indices.append(i)
            else:
                predicted.append(box)
        for i in reversed(dead_indices):
            self.trackers.pop(i)

        # Hungarian matching on IoU matrix
        matched_trk: set[int] = set()
        matched_det: set[int] = set()

        if self.trackers and detections:
            iou_mat = np.zeros((len(self.trackers), len(detections)), dtype=np.float64)
            for ti, pred_box in enumerate(predicted):
                for di, det in enumerate(detections):
                    iou_mat[ti, di] = _iou(pred_box, det["box"])

            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    matched_trk.add(r)
                    matched_det.add(c)
                    self.trackers[r].update(detections[c]["box"], detections[c])

        # Reset hit_streak for unmatched trackers
        for ti, trk in enumerate(self.trackers):
            if ti not in matched_trk:
                trk.hit_streak = 0

        # Create trackers for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det:
                self.trackers.append(KalmanBoxTracker(det["box"], det))

        # Return confirmed tracks only
        results = []
        for trk in self.trackers:
            if trk.time_since_update <= 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                box = trk.get_box().astype(int).tolist()
                results.append({
                    "id": trk.id,
                    "box": box,
                    "hits": trk.hits,
                    "time_since_update": trk.time_since_update,
                    "detection": trk.last_detection,
                })

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return results

    def reset(self):
        self.trackers.clear()
        self.frame_count = 0
