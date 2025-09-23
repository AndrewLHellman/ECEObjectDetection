#!/usr/bin/env python3
from __future__ import annotations
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf_transformations as tft
from cv_bridge import CvBridge


# ------------------------- Helpers -------------------------

def _as_float(x):
    return float(x)


def color(r, g, b, a=1.0) -> ColorRGBA:
    return ColorRGBA(r=_as_float(r), g=_as_float(g), b=_as_float(b), a=_as_float(a))


def contour_is_rectangular(cnt, min_area, min_rectangularity, min_solidity) -> bool:
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity = area / hull_area
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    rect_area = max(cv2.contourArea(box), 1.0)
    rectangularity = area / rect_area
    return (solidity >= min_solidity) and (rectangularity >= min_rectangularity)


def centroid_of_contour(cnt) -> Optional[Tuple[float, float]]:
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-9:
        return None
    return (float(M["m10"]/M["m00"]), float(M["m01"]/M["m00"]))


def ray_from_cam_info(u: float, v: float, info: CameraInfo, source: str = 'P') -> np.ndarray:
    """Return unit ray in *camera* frame given pixel (u,v).
    If using rectified images (image_proc/image_rect), use Projection matrix P (no distortion).
    If using raw images (image_raw), use K and *also* undistort before if needed (not done here).
    """
    if source.upper() == 'P':
        # P = [fx 0 cx Tx; 0 fy cy Ty; 0 0 1 0]
        P = np.array(info.p, dtype=np.float64).reshape(3, 4)
        fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
    else:
        K = np.array(info.k, dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    n = np.linalg.norm(ray)
    return ray / (n + 1e-12)


def intersect_ray_with_plane(p0: np.ndarray, d: np.ndarray, n: np.ndarray, d_plane: float) -> Optional[np.ndarray]:
    denom = float(np.dot(n, d))
    if abs(denom) < 1e-9:
        return None
    t = -(float(np.dot(n, p0)) + d_plane) / denom
    if t < 0:
        return None
    return p0 + t * d


def closest_points_between_lines(p1: np.ndarray, d1: np.ndarray,
                                 p2: np.ndarray, d2: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    w0 = p1 - p2
    d = a * c - b * b
    if abs(d) < 1e-12:
        return None
    s = (b * float(np.dot(d2, w0)) - c * float(np.dot(d1, w0))) / d
    t = (a * float(np.dot(d2, w0)) - b * float(np.dot(d1, w0))) / d
    p_closest1 = p1 + s * d1
    p_closest2 = p2 + t * d2
    midpoint = 0.5 * (p_closest1 + p_closest2)
    return midpoint, float(np.linalg.norm(p_closest1 - p_closest2))


class SyncPair:
    def __init__(self, slop=0.05):
        self.slop = slop
        self.left = None
        self.right = None

    def put_left(self, msg):
        self.left = msg

    def put_right(self, msg):
        self.right = msg

    def ready(self):
        if self.left is None or self.right is None:
            return False
        tL = self.left.header.stamp.sec + self.left.header.stamp.nanosec * 1e-9
        tR = self.right.header.stamp.sec + self.right.header.stamp.nanosec * 1e-9
        return abs(tL - tR) <= self.slop

    def pop(self):
        L, R = self.left, self.right
        self.left = None
        self.right = None
        return L, R


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # Frames & topics
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('left_cam_frame', 'left_camera')
        self.declare_parameter('right_cam_frame', 'right_camera')
        self.declare_parameter('left_image', '/stereo/left/image_rect')
        self.declare_parameter('right_image', '/stereo/right/image_rect')
        self.declare_parameter('left_info', '/stereo/left/camera_info')
        self.declare_parameter('right_info', '/stereo/right/camera_info')

        self.declare_parameter('intrinsics_source', 'P')

        # Detection
        self.declare_parameter('blur_ksize', 7)
        self.declare_parameter('morph_open', 1)
        self.declare_parameter('morph_close', 5)
        self.declare_parameter('min_area', 1200.0)
        self.declare_parameter('min_rectangularity', 0.7)
        self.declare_parameter('min_solidity', 0.8)
        self.declare_parameter('use_adaptive', False)
        self.declare_parameter('global_thresh', 90)

        # Association & fusion
        self.declare_parameter('publish_unpaired', False)
        self.declare_parameter('assoc_plane_gate_m', 0.10)
        self.declare_parameter('assoc_symmetric', True)
        self.declare_parameter('triangulate_3d', True)
        self.declare_parameter('min_ray_angle_deg', 1.0)
        self.declare_parameter('max_triang_gap_m', 0.08)
        self.declare_parameter('debug_pairs', False)

        # Plane (n·X + d = 0)
        self.declare_parameter('plane_n', [0.0, 0.0, 1.0])
        self.declare_parameter('plane_d', 0.0)

        # Load params once
        gp = self.get_parameter
        self.world_frame = gp('world_frame').get_parameter_value().string_value
        self.left_cam_frame = gp('left_cam_frame').get_parameter_value().string_value
        self.right_cam_frame = gp('right_cam_frame').get_parameter_value().string_value
        self.left_image_topic = gp('left_image').get_parameter_value().string_value
        self.right_image_topic = gp('right_image').get_parameter_value().string_value
        self.left_info_topic = gp('left_info').get_parameter_value().string_value
        self.right_info_topic = gp('right_info').get_parameter_value().string_value

        # Defaults (will be refreshed live each cycle)
        self.publish_unpaired = gp('publish_unpaired').get_parameter_value().bool_value
        self.assoc_plane_gate_m = float(gp('assoc_plane_gate_m').get_parameter_value().double_value)
        self.assoc_symmetric = gp('assoc_symmetric').get_parameter_value().bool_value
        self.do_triang = gp('triangulate_3d').get_parameter_value().bool_value
        self.min_ray_angle = math.radians(float(gp('min_ray_angle_deg').get_parameter_value().double_value))
        self.max_triang_gap = float(gp('max_triang_gap_m').get_parameter_value().double_value)
        self.debug_pairs = gp('debug_pairs').get_parameter_value().bool_value

        self.plane_n = np.array(gp('plane_n').get_parameter_value().double_array_value, dtype=np.float64)
        self.plane_d = float(gp('plane_d').get_parameter_value().double_value)

        self.intrinsics_source = self.get_parameter('intrinsics_source').get_parameter_value().string_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.bridge = CvBridge()
        self.left_info: Optional[CameraInfo] = None
        self.right_info: Optional[CameraInfo] = None
        self.K_left: Optional[np.ndarray] = None
        self.K_right: Optional[np.ndarray] = None

        self.sync = SyncPair(slop=0.05)

        self.sub_left_img = self.create_subscription(Image, self.left_image_topic, self.cb_left_img, qos)
        self.sub_right_img = self.create_subscription(Image, self.right_image_topic, self.cb_right_img, qos)
        self.sub_left_info = self.create_subscription(CameraInfo, self.left_info_topic, self.cb_left_info, qos)
        self.sub_right_info = self.create_subscription(CameraInfo, self.right_info_topic, self.cb_right_info, qos)

        self.pub_poses = self.create_publisher(PoseArray, '/prism_centroids', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/prism_markers', 10)
        self.pub_dbg_left = self.create_publisher(Image, '/debug/left', 1)
        self.pub_dbg_right = self.create_publisher(Image, '/debug/right', 1)
        self.pub_pair_markers = self.create_publisher(MarkerArray, '/pair_markers', 10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0/30.0, self.process_if_ready)

        self.get_logger().info('ObjectDetector (fused) initialized.')

    # --------- Runtime param refresh so ros2 param set works mid-run ---------
    def refresh_runtime_params(self):
        try:
            self.publish_unpaired = self.get_parameter('publish_unpaired').get_parameter_value().bool_value
            self.assoc_plane_gate_m = float(self.get_parameter('assoc_plane_gate_m').get_parameter_value().double_value)
            self.assoc_symmetric = self.get_parameter('assoc_symmetric').get_parameter_value().bool_value
            self.do_triang = self.get_parameter('triangulate_3d').get_parameter_value().bool_value
            self.min_ray_angle = math.radians(float(self.get_parameter('min_ray_angle_deg').get_parameter_value().double_value))
            self.max_triang_gap = float(self.get_parameter('max_triang_gap_m').get_parameter_value().double_value)
            self.debug_pairs = self.get_parameter('debug_pairs').get_parameter_value().bool_value
            # Vision live tweaks (optional)
            self.use_adaptive = self.get_parameter('use_adaptive').get_parameter_value().bool_value
            self.global_thresh = int(self.get_parameter('global_thresh').get_parameter_value().integer_value)
            self.min_area = float(self.get_parameter('min_area').get_parameter_value().double_value)
        except Exception as e:
            self.get_logger().warn(f'Runtime param refresh failed: {e}')

    # --------------------- Callbacks ---------------------
    def cb_left_info(self, msg: CameraInfo):
        self.left_info = msg
        self.K_left = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def cb_right_info(self, msg: CameraInfo):
        self.right_info = msg
        self.K_right = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def cb_left_img(self, msg: Image):
        self.sync.put_left(msg)

    def cb_right_img(self, msg: Image):
        self.sync.put_right(msg)

    # --------------------- Main processing ---------------------
    def process_if_ready(self):
        # Pick up live param changes
        self.refresh_runtime_params()

        if self.K_left is None or self.K_right is None:
            return
        if not self.sync.ready():
            return

        left_img_msg, right_img_msg = self.sync.pop()

        try:
            left_cv = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
            right_cv = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        detL, dbgL = self.detect_rectangles(left_cv)
        detR, dbgR = self.detect_rectangles(right_cv)

        self.pub_dbg_left.publish(self.bridge.cv2_to_imgmsg(dbgL, encoding='bgr8'))
        self.pub_dbg_right.publish(self.bridge.cv2_to_imgmsg(dbgR, encoding='bgr8'))

        raysL = self.make_rays(detL, self.left_info, self.left_cam_frame)
        raysR = self.make_rays(detR, self.right_info, self.right_cam_frame)
        ptsL_plane = self.intersect_rays_with_plane_world(raysL)
        ptsR_plane = self.intersect_rays_with_plane_world(raysR)

        pairs = self.associate_by_xy(ptsL_plane, ptsR_plane,
                                     gate=self.assoc_plane_gate_m,
                                     symmetric=self.assoc_symmetric)

        stamp = left_img_msg.header.stamp
        header = Header(stamp=stamp, frame_id=self.world_frame)
        pa = PoseArray(header=header)
        markers: List[Marker] = []
        pair_markers: List[Marker] = []

        fused_color = color(0.1, 0.8, 1.0, 0.95)
        single_left_color = color(1.0, 0.6, 0.1, 0.9)
        single_right_color = color(0.9, 0.2, 0.6, 0.9)
        line_color = color(0.2, 1.0, 0.2, 0.8)
        lonly_color = color(1.0, 0.0, 0.0, 0.6)
        ronly_color = color(0.0, 0.0, 1.0, 0.6)
        scale = Vector3(x=0.02, y=0.02, z=0.02)

        # Debug: show all plane hits (left=red spheres, right=blue spheres)
        if self.debug_pairs:
            idx = 0
            for P in ptsL_plane:
                if P is None: continue
                pos = Point(x=float(P[0]), y=float(P[1]), z=float(P[2]))
                pair_markers.append(self.make_sphere_marker(idx, header, pos, Vector3(x=0.015, y=0.015, z=0.015), lonly_color, ns='plane_left'))
                idx += 1
            for P in ptsR_plane:
                if P is None: continue
                pos = Point(x=float(P[0]), y=float(P[1]), z=float(P[2]))
                pair_markers.append(self.make_sphere_marker(idx, header, pos, Vector3(x=0.015, y=0.015, z=0.015), ronly_color, ns='plane_right'))
                idx += 1

        usedL = set()
        usedR = set()

        for iL, iR in pairs:
            usedL.add(iL); usedR.add(iR)
            pL, dL = raysL[iL]
            pR, dR = raysR[iR]
            fused_point = None

            angle = math.acos(max(-1.0, min(1.0, float(np.dot(dL, dR)))))
            gap_val = None
            if self.do_triang and angle >= self.min_ray_angle:
                res = closest_points_between_lines(pL, dL, pR, dR)
                if res is not None:
                    mid, gap = res
                    gap_val = gap
                    if gap <= self.max_triang_gap:
                        fused_point = mid

            if fused_point is None and ptsL_plane[iL] is not None and ptsR_plane[iR] is not None:
                fused_point = 0.5 * (ptsL_plane[iL] + ptsR_plane[iR])

            if self.debug_pairs and ptsL_plane[iL] is not None and ptsR_plane[iR] is not None:
                # Draw a line between the two plane hits, and annotate distance
                m = Marker()
                m.header = header
                m.ns = 'pair_lines'
                m.id = iL * 1000 + iR
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD
                m.scale = Vector3(x=0.005, y=0.0, z=0.0)
                m.color = line_color
                m.points = [
                    Point(x=float(ptsL_plane[iL][0]), y=float(ptsL_plane[iL][1]), z=float(ptsL_plane[iL][2])),
                    Point(x=float(ptsR_plane[iR][0]), y=float(ptsR_plane[iR][1]), z=float(ptsR_plane[iR][2]))
                ]
                pair_markers.append(m)
                dxy = float(np.linalg.norm(ptsL_plane[iL][0:2] - ptsR_plane[iR][0:2]))
                self.get_logger().info(f"PAIR L{iL}-R{iR}: dXY={dxy:.3f} m, angle={math.degrees(angle):.2f} deg, gap={gap_val if gap_val is not None else -1:.3f} m")

            if fused_point is None:
                continue

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in fused_point]
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            pa.poses.append(pose)
            markers.append(self.make_sphere_marker(len(markers), header, pose.position, scale, fused_color, ns='fused'))

        # Optionally publish single-view plane hits
        if self.publish_unpaired:
            for iL, P in enumerate(ptsL_plane):
                if P is None or iL in usedL: continue
                pose = Pose(); pose.position = Point(x=float(P[0]), y=float(P[1]), z=float(P[2])); pose.orientation.w = 1.0
                pa.poses.append(pose)
                markers.append(self.make_sphere_marker(len(markers), header, pose.position, scale, single_left_color, ns='single_left'))
            for iR, P in enumerate(ptsR_plane):
                if P is None or iR in usedR: continue
                pose = Pose(); pose.position = Point(x=float(P[0]), y=float(P[1]), z=float(P[2])); pose.orientation.w = 1.0
                pa.poses.append(pose)
                markers.append(self.make_sphere_marker(len(markers), header, pose.position, scale, single_right_color, ns='single_right'))

        if len(pa.poses) > 0:
            self.pub_poses.publish(pa)
            self.pub_markers.publish(MarkerArray(markers=markers))
        if self.debug_pairs and (len(pair_markers) > 0):
            self.pub_pair_markers.publish(MarkerArray(markers=pair_markers))

    # --------------------- Vision ---------------------
    def detect_rectangles(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ksize = self.get_parameter('blur_ksize').get_parameter_value().integer_value
        if ksize > 1:
            k = int(ksize) if ksize % 2 == 1 else int(ksize) + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        use_adapt = self.get_parameter('use_adaptive').get_parameter_value().bool_value
        if use_adapt:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 5)
        else:
            thr = int(self.get_parameter('global_thresh').get_parameter_value().integer_value)
            _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

        mo = int(self.get_parameter('morph_open').get_parameter_value().integer_value)
        mc = int(self.get_parameter('morph_close').get_parameter_value().integer_value)
        if mo > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (mo, mo)))
        if mc > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (mc, mc)))

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Tuple[float, float]] = []
        vis = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(bgr, 0.5, vis, 0.5, 0)

        min_area = float(self.get_parameter('min_area').get_parameter_value().double_value)
        min_rectangularity = float(self.get_parameter('min_rectangularity').get_parameter_value().double_value)
        min_solidity = float(self.get_parameter('min_solidity').get_parameter_value().double_value)

        for cnt in contours:
            if not contour_is_rectangular(cnt, min_area, min_rectangularity, min_solidity):
                continue
            c = centroid_of_contour(cnt)
            if c is None:
                continue
            dets.append(c)
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(vis, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)

        return dets, vis

    # --------------------- Geometry & TF ---------------------
    def lookup_cam_to_world(self, cam_frame: str) -> Optional[np.ndarray]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(self.world_frame, cam_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'TF {self.world_frame}->{cam_frame} not available: {e}')
            return None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = t.x; T[1, 3] = t.y; T[2, 3] = t.z
        return T

    def make_rays(self, dets_uv, info: CameraInfo, default_cam_frame: str):
        cam_frame = info.header.frame_id or default_cam_frame
        Twc = self.lookup_cam_to_world(cam_frame)
        if Twc is None: return []
        Rwc = Twc[0:3,0:3]; pw = Twc[0:3,3]
        rays = []
        for (u,v) in dets_uv:
            d_cam = ray_from_cam_info(u, v, info, self.intrinsics_source)
            d_world = Rwc @ d_cam
            d_world /= (np.linalg.norm(d_world) + 1e-12)
            rays.append((pw.copy(), d_world))
        return rays

    def intersect_rays_with_plane_world(self, rays: List[Tuple[np.ndarray, np.ndarray]]):
        n = self.plane_n
        d = self.plane_d
        out: List[Optional[np.ndarray]] = []
        for (p0, dvec) in rays:
            P = intersect_ray_with_plane(p0, dvec, n, d)
            out.append(P)
        return out

    def associate_by_xy(self,
                        ptsL_world: List[Optional[np.ndarray]],
                        ptsR_world: List[Optional[np.ndarray]],
                        gate: float = 0.10,
                        symmetric: bool = True) -> List[Tuple[int, int]]:
        if not ptsL_world or not ptsR_world:
            return []
        L_idx = [i for i, p in enumerate(ptsL_world) if p is not None]
        R_idx = [j for j, p in enumerate(ptsR_world) if p is not None]
        if len(L_idx) == 0 or len(R_idx) == 0:
            return []
        L_xy = np.array([ptsL_world[i][0:2] for i in L_idx])
        R_xy = np.array([ptsR_world[j][0:2] for j in R_idx])
        dists = np.linalg.norm(L_xy[:, None, :] - R_xy[None, :, :], axis=2)

        pairs: List[Tuple[int, int]] = []
        used_r: set[int] = set()

        if symmetric:
            l2r = np.argmin(dists, axis=1)
            r2l = np.argmin(dists, axis=0)
            for li, rj0 in enumerate(l2r):
                if dists[li, rj0] > gate:
                    continue
                if r2l[rj0] != li:
                    continue
                iL = L_idx[li]
                iR = R_idx[rj0]
                if iR in used_r:
                    continue
                used_r.add(iR)
                pairs.append((iL, iR))
        else:
            for li in range(len(L_idx)):
                rj = int(np.argmin(dists[li]))
                if dists[li, rj] > gate:
                    continue
                iL = L_idx[li]
                iR = R_idx[rj]
                if iR in used_r:
                    continue
                used_r.add(iR)
                pairs.append((iL, iR))

        return pairs

    # --------------------- RViz helpers ---------------------
    def make_sphere_marker(self, idx: int, header: Header, pos: Point,
                           scale: Vector3, color_rgba: ColorRGBA, ns: str = 'prisms') -> Marker:
        m = Marker()
        m.header = header
        m.ns = ns
        m.id = idx
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale = scale
        m.color = color_rgba
        m.pose.position = pos
        m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        m.lifetime.sec = 0
        return m


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
