#!/usr/bin/env python3
"""
ROS2 node: ObjectDetector (reproj + constellation association)
Fix: robust parsing of CameraInfo.P / CameraInfo.D to avoid
"truth value of an array is ambiguous" errors.
"""
import math
from typing import List, Tuple, Optional

import numpy as np
import cv2 as cv

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from std_msgs.msg import Header, Float32

import tf2_ros


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # ---------------- Params ----------------
        self.declare_parameter('left_image', '/stereo/left/image_raw')
        self.declare_parameter('right_image', '/stereo/right/image_raw')
        self.declare_parameter('left_info', '/stereo/left/camera_info')
        self.declare_parameter('right_info', '/stereo/right/camera_info')

        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('left_cam_frame', 'left_camera')
        self.declare_parameter('right_cam_frame', 'right_camera')

        self.declare_parameter('use_general_plane', False)
        self.declare_parameter('plane_n', [0.0, 0.0, 1.0])
        self.declare_parameter('plane_d', 0.0)

        self.declare_parameter('assoc_mode', 'reproj')   # 'reproj' | 'constellation'
        self.declare_parameter('assoc_pixel_gate', 20.0)
        self.declare_parameter('symmetric_match', True)
        self.declare_parameter('use_undistort_assoc', True)

        self.declare_parameter('triangulate_3d', False)
        self.declare_parameter('min_ray_angle_deg', 1.0)

        self.declare_parameter('use_adaptive', True)
        self.declare_parameter('binary_inv', True)
        self.declare_parameter('blur_ksize', 5)
        self.declare_parameter('adaptive_block', 31)
        self.declare_parameter('adaptive_C', 5)
        self.declare_parameter('global_thresh', 60)
        self.declare_parameter('morph_open', 3)
        self.declare_parameter('morph_close', 3)

        self.declare_parameter('min_area', 400.0)
        self.declare_parameter('max_area', 1_000_000.0)
        self.declare_parameter('min_solidity', 0.75)
        self.declare_parameter('min_rectangularity', 0.65)
        self.declare_parameter('min_aspect', 0.2)
        self.declare_parameter('max_aspect', 5.0)

        self.declare_parameter('draw_contours', True)
        self.declare_parameter('max_labels', 50)
        self.declare_parameter('debug_reprojection', True)
        self.declare_parameter('debug_max_pairs', 40)

        # ---------------- Setup ----------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.bridge = CvBridge()

        self.left_img_topic = self.get_parameter('left_image').get_parameter_value().string_value
        self.right_img_topic = self.get_parameter('right_image').get_parameter_value().string_value
        self.left_info_topic = self.get_parameter('left_info').get_parameter_value().string_value
        self.right_info_topic = self.get_parameter('right_info').get_parameter_value().string_value

        # Intrinsics/state
        self.left_K = None
        self.right_K = None
        self.left_D = None
        self.right_D = None
        self.left_P = None
        self.right_P = None
        self.left_cam_frame = self.get_parameter('left_cam_frame').get_parameter_value().string_value
        self.right_cam_frame = self.get_parameter('right_cam_frame').get_parameter_value().string_value

        # TF
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.sub_left_img = self.create_subscription(Image, self.left_img_topic, self.on_left, qos)
        self.sub_right_img = self.create_subscription(Image, self.right_img_topic, self.on_right, qos)
        self.sub_left_info = self.create_subscription(CameraInfo, self.left_info_topic, self.on_left_info, qos)
        self.sub_right_info = self.create_subscription(CameraInfo, self.right_info_topic, self.on_right_info, qos)

        # Publishers
        self.pub_left_2d = self.create_publisher(PoseArray, '/stereo/left/prism_centroids', 10)
        self.pub_right_2d = self.create_publisher(PoseArray, '/stereo/right/prism_centroids', 10)
        self.pub_left_world = self.create_publisher(PoseArray, '/stereo/left/prism_centroids_world', 10)
        self.pub_right_world = self.create_publisher(PoseArray, '/stereo/right/prism_centroids_world', 10)
        self.pub_fused_world = self.create_publisher(PoseArray, '/stereo/fused/prism_centroids_world', 10)
        self.pub_fused_world_3d = self.create_publisher(PoseArray, '/stereo/fused/prism_centroids_world_3d', 10)
        self.pub_left_dbg = self.create_publisher(Image, '/stereo/left/centroid_image', 10)
        self.pub_right_dbg = self.create_publisher(Image, '/stereo/right/centroid_image', 10)
        self.pub_left_reproj = self.create_publisher(Image, '/stereo/left/reproj_debug', 10)
        self.pub_right_reproj = self.create_publisher(Image, '/stereo/right/reproj_debug', 10)
        self.pub_const_rms = self.create_publisher(Float32, '/stereo/fused/constellation_rms_px', 10)

        # Buffers
        self.left_last = None
        self.right_last = None

        self.get_logger().info('ObjectDetector ready (reproj + constellation association).')

    # -------- CameraInfo helpers --------
    @staticmethod
    def _list_or_empty(seq) -> list:
        try:
            return list(seq) if seq is not None else []
        except Exception:
            return []

    @staticmethod
    def _extract_K(msg: CameraInfo):
        k = ObjectDetector._list_or_empty(msg.k)
        if len(k) >= 9:
            return (k[0], k[4], k[2], k[5])  # fx, fy, cx, cy
        return None

    @staticmethod
    def _extract_P(msg: CameraInfo):
        p = ObjectDetector._list_or_empty(msg.p)
        if len(p) >= 12:
            return (p[0], p[5], p[2], p[6])  # fx, fy, cx, cy for rectified
        return None

    @staticmethod
    def _extract_D(msg: CameraInfo):
        d = ObjectDetector._list_or_empty(msg.d)
        return d if len(d) > 0 else None

    def on_left_info(self, msg: CameraInfo):
        self.left_cam_frame = msg.header.frame_id or self.left_cam_frame
        self.left_K = self._extract_K(msg)
        self.left_P = self._extract_P(msg)
        self.left_D = self._extract_D(msg)

    def on_right_info(self, msg: CameraInfo):
        self.right_cam_frame = msg.header.frame_id or self.right_cam_frame
        self.right_K = self._extract_K(msg)
        self.right_P = self._extract_P(msg)
        self.right_D = self._extract_D(msg)

    # ------------- Image callbacks -------------
    def on_left(self, msg: Image):
        self.process_camera('left', msg)

    def on_right(self, msg: Image):
        self.process_camera('right', msg)

    # Core per-camera processing
    def process_camera(self, which: str, img_msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        centroids_uv, dbg = self.detect_centroids(bgr)

        # Publish 2D image-plane centroids
        pa2d = self.poses_from_uv(centroids_uv, img_msg.header, self.left_cam_frame if which=='left' else self.right_cam_frame)
        if which == 'left':
            self.pub_left_2d.publish(pa2d)
            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            dbg_msg.header = img_msg.header
            self.pub_left_dbg.publish(dbg_msg)
        else:
            self.pub_right_2d.publish(pa2d)
            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            dbg_msg.header = img_msg.header
            self.pub_right_dbg.publish(dbg_msg)

        # Project rays -> world + intersect plane
        K = self.left_K if which=='left' else self.right_K
        cam_frame = self.left_cam_frame if which=='left' else self.right_cam_frame
        if K is None:
            return

        ok, Rwc, t_w = self.lookup_world_T_cam(cam_frame, img_msg.header.stamp)
        if not ok:
            return

        world_pts = []
        rays_world = []
        fx, fy, cx, cy = K
        n, d = self.get_plane()
        for (u, v) in centroids_uv:
            dc = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=np.float64)
            nrm = np.linalg.norm(dc)
            if nrm == 0:
                continue
            dc /= nrm
            dw = Rwc @ dc
            ow = t_w
            denom = float(n @ dw)
            if abs(denom) < 1e-9:
                continue
            lam = - (n @ ow + d) / denom
            if lam < 0:
                continue
            pw = ow + lam * dw
            world_pts.append(pw)
            rays_world.append(dw)

        # Publish per-camera world points
        pa = PoseArray()
        pa.header = Header()
        pa.header.stamp = img_msg.header.stamp
        pa.header.frame_id = self.world_frame
        for p in world_pts:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in p]
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        if which=='left':
            self.pub_left_world.publish(pa)
            self.left_last = (img_msg.header.stamp, centroids_uv, world_pts, rays_world, t_w, dbg)
        else:
            self.pub_right_world.publish(pa)
            self.right_last = (img_msg.header.stamp, centroids_uv, world_pts, rays_world, t_w, dbg)

        self.try_fuse()

    # ------------- Detection -------------
    def detect_centroids(self, bgr) -> Tuple[List[Tuple[float,float]], np.ndarray]:
        p = lambda n: self.get_parameter(n).get_parameter_value()
        use_adaptive = p('use_adaptive').bool_value
        binary_inv = p('binary_inv').bool_value
        blur_ksize = int(p('blur_ksize').integer_value)
        adaptive_block = int(p('adaptive_block').integer_value)
        adaptive_C = int(p('adaptive_C').integer_value)
        global_thresh = int(p('global_thresh').integer_value)
        morph_open = int(p('morph_open').integer_value)
        morph_close = int(p('morph_close').integer_value)

        min_area = float(p('min_area').double_value or p('min_area').integer_value)
        max_area = float(p('max_area').double_value or p('max_area').integer_value)
        min_solidity = float(p('min_solidity').double_value)
        min_rect = float(p('min_rectangularity').double_value)
        min_aspect = float(p('min_aspect').double_value)
        max_aspect = float(p('max_aspect').double_value)
        draw_contours = p('draw_contours').bool_value
        max_labels = int(p('max_labels').integer_value)

        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        if blur_ksize and blur_ksize > 1 and blur_ksize % 2 == 1:
            gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        if use_adaptive:
            block = max(3, adaptive_block | 1)
            thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV if binary_inv else cv.THRESH_BINARY,
                                          block, adaptive_C)
        else:
            _, thresh = cv.threshold(gray, global_thresh, 255,
                                     cv.THRESH_BINARY_INV if binary_inv else cv.THRESH_BINARY)

        def morph(img, ksize, op):
            if ksize <= 0:
                return img
            k = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
            return cv.morphologyEx(img, op, k)

        thresh = morph(thresh, morph_open, cv.MORPH_OPEN)
        thresh = morph(thresh, morph_close, cv.MORPH_CLOSE)

        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        centroids = []
        dbg = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        label_count = 0
        for c in contours:
            area = cv.contourArea(c)
            if area < min_area or area > max_area:
                continue
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull) if len(hull) >= 3 else 0.0
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < min_solidity:
                continue
            rrect = cv.minAreaRect(c)
            (rcx, rcy), (rw, rh), _ = rrect
            rect_area = rw * rh
            if rect_area <= 0:
                continue
            rectangularity = area / rect_area
            aspect = (max(rw, rh) / (min(rw, rh) + 1e-6)) if min(rw, rh) > 0 else math.inf
            if rectangularity < min_rect:
                continue
            if not (min_aspect <= aspect <= max_aspect):
                continue
            M = cv.moments(c)
            if M['m00'] == 0:
                continue
            cxp = M['m10'] / M['m00']
            cyp = M['m01'] / M['m00']
            centroids.append((cxp, cyp))
            if draw_contours and label_count < max_labels:
                box = cv.boxPoints(rrect).astype(np.int32)
                cv.drawContours(dbg, [box], 0, (0, 255, 0), 2)
                cv.circle(dbg, (int(cxp), int(cyp)), 4, (0, 0, 255), -1)
                cv.putText(dbg, f'({int(cxp)},{int(cyp)})', (int(cxp)+6, int(cyp)-6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv.LINE_AA)
                label_count += 1

        overlay = cv.addWeighted(bgr, 0.6, dbg, 0.4, 0)
        return centroids, overlay

    # ------------- TF & plane helpers -------------
    def lookup_world_T_cam(self, cam_frame: str, stamp) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        for when in (Time.from_msg(stamp), Time()):
            try:
                tf_msg = self.tf_buffer.lookup_transform(self.world_frame, cam_frame, when, timeout=Duration(seconds=0.2))
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                Rwc = self.quat_to_R(q)
                tvec = np.array([t.x, t.y, t.z], dtype=np.float64)
                return True, Rwc, tvec
            except Exception:
                continue
        return False, None, None

    def get_plane(self) -> Tuple[np.ndarray, float]:
        if self.get_parameter('use_general_plane').get_parameter_value().bool_value:
            n_list = self.get_parameter('plane_n').get_parameter_value().double_array_value
            nx, ny, nz = n_list if len(n_list) == 3 else [0.0, 0.0, 1.0]
            d = float(self.get_parameter('plane_d').get_parameter_value().double_value)
            n = np.array([nx, ny, nz], dtype=np.float64)
            n /= max(1e-9, np.linalg.norm(n))
            return n, d
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0

    @staticmethod
    def quat_to_R(q: Quaternion) -> np.ndarray:
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
        ], dtype=np.float64)

    # ------------- Association modes -------------
    @staticmethod
    def undistort_uvs(uvs: List[Tuple[float, float]], K, D):
        if K is None or D is None or len(uvs) == 0:
            return uvs
        fx, fy, cx, cy = K
        cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        distCoeffs = np.array(D, dtype=np.float64).reshape(-1, 1)
        pts = np.array(uvs, dtype=np.float64).reshape(-1, 1, 2)
        und = cv.undistortPoints(pts, cameraMatrix, distCoeffs, P=cameraMatrix)
        und = und.reshape(-1, 2)
        return [tuple(p) for p in und]

    @staticmethod
    def umeyama_similarity(A: np.ndarray, B: np.ndarray):
        assert A.shape[0] == 2 and B.shape[0] == 2 and A.shape[1] == B.shape[1]
        N = A.shape[1]
        mu_A = A.mean(axis=1, keepdims=True)
        mu_B = B.mean(axis=1, keepdims=True)
        A0 = A - mu_A
        B0 = B - mu_B
        var_A = (A0**2).sum() / N
        C = (B0 @ A0.T) / N
        U, S, Vt = np.linalg.svd(C)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = U @ Vt
        s = (S @ np.ones_like(S)) / var_A
        t = (mu_B - s * R @ mu_A).reshape(2)
        return float(s), R, t

    def constellation_match(self, uvL, uvR, px_gate, symmetric=True):
        if len(uvL) == 0 or len(uvR) == 0:
            return [] , None
        A = np.array(uvL, dtype=np.float64).T
        B = np.array(uvR, dtype=np.float64).T
        # balance counts
        if A.shape[1] != B.shape[1]:
            if A.shape[1] > B.shape[1]:
                idx = np.argsort(((A - A.mean(axis=1, keepdims=True))**2).sum(axis=0))[:B.shape[1]]
                A_fit = A[:, idx]
                B_fit = B
            else:
                idx = np.argsort(((B - B.mean(axis=1, keepdims=True))**2).sum(axis=0))[:A.shape[1]]
                A_fit = A
                B_fit = B[:, idx]
        else:
            A_fit, B_fit = A, B
        s, R, t = self.umeyama_similarity(A_fit, B_fit)
        A2 = (s * (R @ A)) + t.reshape(2,1)
        matches_LR = {}
        for i in range(A2.shape[1]):
            diffs = B.T - A2[:, i].reshape(1,2)
            d2 = np.sum(diffs**2, axis=1)
            j = int(np.argmin(d2))
            if math.sqrt(float(d2[j])) <= px_gate:
                matches_LR[i] = j
        if not symmetric:
            rms = self.rms_after_map(A2, B, matches_LR)
            return list(matches_LR.items()), rms
        s2, R2, t2 = self.umeyama_similarity(B_fit, A_fit)
        B2 = (s2 * (R2 @ B)) + t2.reshape(2,1)
        matches_RL = {}
        for j in range(B2.shape[1]):
            diffs = A.T - B2[:, j].reshape(1,2)
            d2 = np.sum(diffs**2, axis=1)
            i = int(np.argmin(d2))
            if math.sqrt(float(d2[i])) <= px_gate:
                matches_RL[j] = i
        matches = []
        for i, j in matches_LR.items():
            if j in matches_RL and matches_RL[j] == i:
                matches.append((i, j))
        rms = self.rms_after_map(A2, B, dict(matches))
        return matches, rms

    @staticmethod
    def rms_after_map(A2: np.ndarray, B: np.ndarray, matches_dict):
        if not matches_dict:
            return None
        errs = []
        for i, j in matches_dict.items():
            e = np.linalg.norm(A2[:, i] - B[:, j])
            errs.append(float(e))
        if not errs:
            return None
        return float(np.sqrt(np.mean(np.square(errs))))

    def try_fuse(self):
        if self.left_last is None or self.right_last is None:
            return
        stampL, uvL, ptsL, raysL, oL, dbgL = self.left_last
        stampR, uvR, ptsR, raysR, oR, dbgR = self.right_last

        mode = self.get_parameter('assoc_mode').get_parameter_value().string_value
        px_gate = float(self.get_parameter('assoc_pixel_gate').get_parameter_value().double_value)
        symmetric = self.get_parameter('symmetric_match').get_parameter_value().bool_value

        if mode == 'constellation':
            matches, rms = self.constellation_match(uvL, uvR, px_gate, symmetric)
            if rms is not None:
                self.pub_const_rms.publish(Float32(data=rms))
        else:
            okR, Rwc_R, t_w_R = self.lookup_world_T_cam(self.right_cam_frame, stampR)
            okL, Rwc_L, t_w_L = self.lookup_world_T_cam(self.left_cam_frame, stampL)
            if not (okR and okL) or self.right_K is None or self.left_K is None:
                return
            fxR, fyR, cxR, cyR = self.right_K
            fxL, fyL, cxL, cyL = self.left_K
            use_und = self.get_parameter('use_undistort_assoc').get_parameter_value().bool_value
            uvL_cmp = self.undistort_uvs(uvL, self.left_K, self.left_D) if use_und else uvL
            uvR_cmp = self.undistort_uvs(uvR, self.right_K, self.right_D) if use_und else uvR

            matches_LR = {}
            for i, p_w in enumerate(ptsL):
                u_pred, v_pred, ok = self.project_world_to_cam_px(p_w, Rwc_R, t_w_R, fxR, fyR, cxR, cyR)
                if not ok:
                    continue
                j_best, best_d = -1, 1e9
                for j, (uR, vR) in enumerate(uvR_cmp):
                    dpx = math.hypot(uR - u_pred, vR - v_pred)
                    if dpx < best_d:
                        best_d, j_best = dpx, j
                if j_best >= 0 and best_d <= px_gate:
                    matches_LR[i] = j_best

            if symmetric:
                matches_RL = {}
                for j, p_w in enumerate(ptsR):
                    u_pred, v_pred, ok = self.project_world_to_cam_px(p_w, Rwc_L, t_w_L, fxL, fyL, cxL, cyL)
                    if not ok:
                        continue
                    i_best, best_d = -1, 1e9
                    for i, (uL, vL) in enumerate(uvL_cmp):
                        dpx = math.hypot(uL - u_pred, vL - v_pred)
                        if dpx < best_d:
                            best_d, i_best = dpx, i
                    if i_best >= 0 and best_d <= px_gate:
                        matches_RL[j] = i_best
                matches = []
                for i, j in matches_LR.items():
                    if j in matches_RL and matches_RL[j] == i:
                        matches.append((i, j))
            else:
                matches = list(matches_LR.items())

        # ------- Publish fused plane points (left-trust) -------
        n, _d = self.get_plane()
        fused_plane_pts = []
        for (i, j) in matches:
            pL = np.array(ptsL[i])
            fused = pL.copy()
            if j < len(ptsR):
                pR = np.array(ptsR[j])
                fused = 0.75 * pL + 0.25 * pR
            fused = fused - (n @ fused + _d) * n
            fused_plane_pts.append(fused)

        out_plane = PoseArray()
        out_plane.header = Header()
        out_plane.header.frame_id = self.world_frame
        out_plane.header.stamp = stampL if Time.from_msg(stampL) > Time.from_msg(stampR) else stampR
        for p in fused_plane_pts:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in p]
            pose.orientation.w = 1.0
            out_plane.poses.append(pose)
        self.pub_fused_world.publish(out_plane)

        # Triangulation optional
        if self.get_parameter('triangulate_3d').get_parameter_value().bool_value and len(matches) > 0:
            min_ang = float(self.get_parameter('min_ray_angle_deg').get_parameter_value().double_value)
            out_3d = PoseArray()
            out_3d.header = out_plane.header
            for (i, j) in matches:
                if i >= len(raysL) or j >= len(raysR):
                    continue
                o1, d1 = np.array(oL), np.array(raysL[i])
                o2, d2 = np.array(oR), np.array(raysR[j])
                p_mid, ok = self.triangulate_two_rays(o1, d1, o2, d2)
                cosang = np.clip(d1 @ d2 / (np.linalg.norm(d1)*np.linalg.norm(d2)), -1.0, 1.0)
                angle = math.degrees(math.acos(abs(cosang)))
                if (not ok) or angle < min_ang:
                    continue
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = [float(v) for v in p_mid]
                pose.orientation.w = 1.0
                out_3d.poses.append(pose)
            self.pub_fused_world_3d.publish(out_3d)

    @staticmethod
    def triangulate_two_rays(o1, d1, o2, d2):
        d1 = d1 / max(1e-12, np.linalg.norm(d1))
        d2 = d2 / max(1e-12, np.linalg.norm(d2))
        w0 = o1 - o2
        a = float(d1 @ d1)
        b = float(d1 @ d2)
        c = float(d2 @ d2)
        d = float(d1 @ w0)
        e = float(d2 @ w0)
        denom = a*c - b*b
        if abs(denom) < 1e-12:
            return 0.5*(o1+o2), False
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
        p1 = o1 + s*d1
        p2 = o2 + t*d2
        return 0.5*(p1 + p2), True

    @staticmethod
    def project_world_to_cam_px(p_w: np.ndarray, Rwc: np.ndarray, t_w: np.ndarray,
                                fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, bool]:
        Rcw = Rwc.T
        Xc = Rcw @ (p_w - t_w)
        Z = float(Xc[2])
        if Z <= 1e-6:
            return 0.0, 0.0, False
        u = fx * (Xc[0] / Z) + cx
        v = fy * (Xc[1] / Z) + cy
        return float(u), float(v), True

    @staticmethod
    def poses_from_uv(uvs: List[Tuple[float,float]], header: Header, frame_id: str) -> PoseArray:
        pa = PoseArray()
        pa.header = Header()
        pa.header.stamp = header.stamp
        pa.header.frame_id = frame_id
        for (u, v) in uvs:
            pose = Pose()
            pose.position.x = float(u)
            pose.position.y = float(v)
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        return pa


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
