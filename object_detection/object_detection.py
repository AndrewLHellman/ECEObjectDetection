#!/usr/bin/env python3
"""
Inside-node two-camera fusion of rectangular-prism centroids using AprilTag world frame
— now with TRUE 3D triangulation (height/z recovered) in addition to plane projection.

What this node does (single process):
  1) Detects 2D centroids per camera.
  2) Uses CameraInfo + TF (world<-cam) to back-project rays.
  3) Intersects each ray with a (configurable) plane for on-plane 3D (stable, z fixed to plane).
  4) Associates detections across cameras by reprojecting left-world points into the right image
     (mutual NN + pixel gate).
  5) Fuses matched pairs on the plane with angle-aware weighting (as before) -> /stereo/fused/prism_centroids_world.
  6) NEW: Triangulates matched pairs via closest point between 3D rays ->
         /stereo/fused/prism_centroids_world_3d  (true X,Y,Z without plane constraint).

Notes:
  - If your prisms lie on the table, plane output is great for XY; triangulated output also estimates actual Z.
  - Accuracy depends on TF quality from AprilTag and pixel localization. Gate low ray-angle pairs.
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
from std_msgs.msg import Header

import tf2_ros


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # ---------------- Params ----------------
        # Topics
        self.declare_parameter('left_image', '/stereo/left/image_raw')
        self.declare_parameter('right_image', '/stereo/right/image_raw')
        self.declare_parameter('left_info', '/stereo/left/camera_info')
        self.declare_parameter('right_info', '/stereo/right/camera_info')

        # Frames
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('left_cam_frame', 'left_camera')
        self.declare_parameter('right_cam_frame', 'right_camera')

        # Plane (world): either z=0 or general plane n·x + d = 0
        self.declare_parameter('use_general_plane', False)
        self.declare_parameter('plane_n', [0.0, 0.0, 1.0])
        self.declare_parameter('plane_d', 0.0)

        # Association/fusion
        self.declare_parameter('assoc_pixel_gate', 12.0)     # px
        self.declare_parameter('symmetric_match', True)      # require mutual nearest-neighbor

        # Triangulation (true 3D)
        self.declare_parameter('triangulate_3d', True)
        self.declare_parameter('min_ray_angle_deg', 2.0)     # drop pairs with too small angle

        # Thresholding/morphology (same as earlier nodes)
        self.declare_parameter('use_adaptive', True)
        self.declare_parameter('binary_inv', True)
        self.declare_parameter('blur_ksize', 5)
        self.declare_parameter('adaptive_block', 31)
        self.declare_parameter('adaptive_C', 5)
        self.declare_parameter('global_thresh', 60)
        self.declare_parameter('morph_open', 3)
        self.declare_parameter('morph_close', 3)

        # Contour filtering
        self.declare_parameter('min_area', 400.0)
        self.declare_parameter('max_area', 1_000_000.0)
        self.declare_parameter('min_solidity', 0.75)
        self.declare_parameter('min_rectangularity', 0.65)
        self.declare_parameter('min_aspect', 0.2)
        self.declare_parameter('max_aspect', 5.0)

        # Debug
        self.declare_parameter('draw_contours', True)
        self.declare_parameter('max_labels', 50)

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
        self.left_K = None  # (fx, fy, cx, cy)
        self.right_K = None
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

        # Buffers for association (latest per-camera sample)
        self.left_last = None   # (stamp, uvs, world_pts, rays_world, cam_origin_world, dbg_img)
        self.right_last = None

        self.get_logger().info('PrismCentroidDetectorFused started (+ triangulation).')

    # ------------- CameraInfo -------------
    def on_left_info(self, msg: CameraInfo):
        self.left_cam_frame = msg.header.frame_id or self.left_cam_frame
        self.left_K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_right_info(self, msg: CameraInfo):
        self.right_cam_frame = msg.header.frame_id or self.right_cam_frame
        self.right_K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

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
            self.get_logger().warn(f'[{which}] No CameraInfo yet; skipping 3D projection.')
            return

        ok, Rcw, tcw = self.lookup_world_T_cam(cam_frame, img_msg.header.stamp)
        if not ok:
            self.get_logger().warn(f'[{which}] TF lookup failed; skipping 3D projection.')
            return

        world_pts = []
        rays_world = []
        fx, fy, cx, cy = K
        n, d = self.get_plane()
        for (u, v) in centroids_uv:
            # camera ray dir in camera frame
            dc = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=np.float64)
            nrm = np.linalg.norm(dc)
            if nrm == 0:
                continue
            dc /= nrm
            # to world
            dw = Rcw @ dc
            ow = tcw
            # intersect with plane n·x + d = 0
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
            self.left_last = (img_msg.header.stamp, centroids_uv, world_pts, rays_world, tcw, dbg)
        else:
            self.pub_right_world.publish(pa)
            self.right_last = (img_msg.header.stamp, centroids_uv, world_pts, rays_world, tcw, dbg)

        # Try fusion when both sides available (approx sync using world-plane projection timestamps)
        self.try_fuse()

    # ------------- Detection (same as before) -------------
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
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centroids.append((cx, cy))
            if draw_contours and label_count < max_labels:
                box = cv.boxPoints(rrect).astype(np.int32)
                cv.drawContours(dbg, [box], 0, (0, 255, 0), 2)
                cv.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv.putText(dbg, f'({int(cx)},{int(cy)})', (int(cx)+6, int(cy)-6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv.LINE_AA)
                label_count += 1

        overlay = cv.addWeighted(bgr, 0.6, dbg, 0.4, 0)
        return centroids, overlay

    # ------------- TF & plane helpers -------------
    def lookup_world_T_cam(self, cam_frame: str, stamp) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(self.world_frame, cam_frame, Time.from_msg(stamp), timeout=Duration(seconds=0.05))
        except Exception:
            return False, None, None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        R = self.quat_to_R(q)
        tvec = np.array([t.x, t.y, t.z], dtype=np.float64)
        return True, R, tvec

    def get_plane(self) -> Tuple[np.ndarray, float]:
        if self.get_parameter('use_general_plane').get_parameter_value().bool_value:
            n_list = self.get_parameter('plane_n').get_parameter_value().double_array_value
            nx, ny, nz = n_list if len(n_list) == 3 else [0.0, 0.0, 1.0]
            d = float(self.get_parameter('plane_d').get_parameter_value().double_value)
            n = np.array([nx, ny, nz], dtype=np.float64)
            n /= max(1e-9, np.linalg.norm(n))
            return n, d
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0  # z=0

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

    # ------------- Association, Fusion, Triangulation -------------
    def try_fuse(self):
        if self.left_last is None or self.right_last is None:
            return
        stampL, uvL, ptsL, raysL, oL, _dbgL = self.left_last
        stampR, uvR, ptsR, raysR, oR, _dbgR = self.right_last

        # Prepare projection helpers
        okR, Rcw_R, tcw_R = self.lookup_world_T_cam(self.right_cam_frame, stampR)
        okL, Rcw_L, tcw_L = self.lookup_world_T_cam(self.left_cam_frame, stampL)
        if not (okR and okL) or self.right_K is None or self.left_K is None:
            return
        fxR, fyR, cxR, cyR = self.right_K
        fxL, fyL, cxL, cyL = self.left_K

        px_gate = float(self.get_parameter('assoc_pixel_gate').get_parameter_value().double_value)
        symmetric = self.get_parameter('symmetric_match').get_parameter_value().bool_value

        # LEFT->RIGHT matching by reprojection of left world points into right image
        matches_LR = {}
        for i, p_w in enumerate(ptsL):
            u_pred, v_pred, ok = self.project_world_to_cam_px(p_w, Rcw_R, tcw_R, fxR, fyR, cxR, cyR)
            if not ok:
                continue
            j_best, best_d = -1, 1e9
            for j, (uR, vR) in enumerate(uvR):
                dpx = math.hypot(uR - u_pred, vR - v_pred)
                if dpx < best_d:
                    best_d, j_best = dpx, j
            if j_best >= 0 and best_d <= px_gate:
                matches_LR[i] = j_best

        if symmetric:
            # RIGHT->LEFT
            matches_RL = {}
            for j, p_w in enumerate(ptsR):
                u_pred, v_pred, ok = self.project_world_to_cam_px(p_w, Rcw_L, tcw_L, fxL, fyL, cxL, cyL)
                if not ok:
                    continue
                i_best, best_d = -1, 1e9
                for i, (uL, vL) in enumerate(uvL):
                    dpx = math.hypot(uL - u_pred, vL - v_pred)
                    if dpx < best_d:
                        best_d, i_best = dpx, i
                if i_best >= 0 and best_d <= px_gate:
                    matches_RL[j] = i_best
            # Mutual filter
            matches = []
            for i, j in matches_LR.items():
                if j in matches_RL and matches_RL[j] == i:
                    matches.append((i, j))
        else:
            matches = list(matches_LR.items())

        # ------- Plane-constrained fusion (as before) -------
        n, _d = self.get_plane()
        fused_plane_pts = []
        for (i, j) in matches:
            pL = np.array(ptsL[i])
            pR = np.array(ptsR[j])
            dL = np.array(raysL[i])
            dR = np.array(raysR[j])
            cosL = abs(float(dL @ n)) / max(1e-9, np.linalg.norm(dL))
            cosR = abs(float(dR @ n)) / max(1e-9, np.linalg.norm(dR))
            wL = cosL*cosL
            wR = cosR*cosR
            wsum = wL + wR
            fused = 0.5*(pL+pR) if wsum <= 1e-12 else (wL*pL + wR*pR)/wsum
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

        # ------- TRUE 3D triangulation (ray–ray) -------
        if self.get_parameter('triangulate_3d').get_parameter_value().bool_value:
            min_ang = float(self.get_parameter('min_ray_angle_deg').get_parameter_value().double_value)
            out_3d = PoseArray()
            out_3d.header = out_plane.header
            for (i, j) in matches:
                o1, d1 = np.array(oL), np.array(raysL[i])
                o2, d2 = np.array(oR), np.array(raysR[j])
                p_mid, ok = self.triangulate_two_rays(o1, d1, o2, d2)
                # Angle gating
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
        """Return midpoint of shortest segment between two 3D rays and ok flag.
        o1,o2: origins (3,), d1,d2: directions (3,) need not be unit.
        """
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
            return 0.5*(o1+o2), False  # nearly parallel
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
        p1 = o1 + s*d1
        p2 = o2 + t*d2
        return 0.5*(p1 + p2), True

    @staticmethod
    def project_world_to_cam_px(p_w: np.ndarray, Rcw: np.ndarray, tcw: np.ndarray,
                                fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, bool]:
        Xc = Rcw @ p_w + tcw
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
