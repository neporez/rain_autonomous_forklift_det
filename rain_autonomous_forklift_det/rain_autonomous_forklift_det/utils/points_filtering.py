import numpy as np
from sensor_msgs.msg import PointCloud2
from rain_autonomous_forklift_det.utils.decorator import measure_time


class PointFilter():
    def __init__(self, roi, sample_num, use_intensity=True, filtering_method="random", far_threshold=40):
        self._roi = roi
        self._sample_num = sample_num
        self._point_feature_dim = 4 if use_intensity else 3
        self._filtering_method = filtering_method
        self._far_threshold = far_threshold
        self._origin_exclusion_radius = 4

    @measure_time('Point Preprocessing')
    def filtering(self, msg: PointCloud2):
        # 공통: PointCloud2 -> numpy (바이트) 변환
        point_step = msg.point_step
        data_bytes = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)

        # ---- 항상 xyz는 앞 12바이트에서 읽는다 (float32 * 3) ----
        if point_step < 12:
            raise ValueError(f"point_step({point_step})가 12보다 작아서 xyz를 읽을 수 없습니다.")
        xyz_bytes = data_bytes[:, :12]
        xyz = xyz_bytes.view(np.float32).reshape(-1, 3)

        # ---- intensity 처리: 있으면 읽고 정규화, 없으면 전부 0.0 ----
        if self._point_feature_dim > 3:
            if point_step >= 16:
                intensity_bytes = data_bytes[:, 12:16]
                intensity = intensity_bytes.view(np.float32).reshape(-1)
                if intensity.size > 0:
                    i_min = float(intensity.min())
                    i_max = float(intensity.max())
                    i_rng = i_max - i_min
                    if i_rng > 0:
                        intensity = (intensity - i_min) / i_rng
                    else:
                        intensity = np.zeros_like(intensity, dtype=np.float32)
                else:
                    intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
            else:
                # intensity가 메시지에 없으므로 0으로 채움
                intensity = np.zeros((xyz.shape[0],), dtype=np.float32)

            pc = np.column_stack((xyz.astype(np.float32), intensity.astype(np.float32)))
        else:
            pc = xyz.astype(np.float32)

        original_count = len(pc)

        if original_count == 144000: #if hesai pnadar 40p
            indices = np.arange(original_count)
            channel_indices = indices // 40
            even_channel_mask = (channel_indices % 2 == 0)

            pc = pc[even_channel_mask]

        # ROI 마스크
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        roi_mask = (
            (x >= self._roi[0]) & (x <= self._roi[3]) &
            (y >= self._roi[1]) & (y <= self._roi[4]) &
            (z >= self._roi[2]) & (z <= self._roi[5])
        )
        pc_roi = pc[roi_mask]

        # 원점 근처 제거
        if self._origin_exclusion_radius is not None:
            pc_roi = self._filter_near_origin_points(pc_roi, self._origin_exclusion_radius)

        # filtering method 실행
        filter_func = getattr(self, f"_{self._filtering_method}_filtering", None)
        if filter_func is None:
            raise ValueError(f"Unknown filtering method: {self._filtering_method}")

        return filter_func(pc_roi)

    def _filter_near_origin_points(self, pc, radius):
        dists = np.linalg.norm(pc[:, :3], axis=1)
        return pc[dists > radius]

    def _random_filtering(self, pc_roi):
        if pc_roi.shape[0] == 0:
            # 빈 ROI일 때도 (1, sample_num, C) 형태를 맞춰 반환
            return np.zeros((1, self._sample_num, pc_roi.shape[1]), dtype=np.float32)
        indices = np.random.choice(pc_roi.shape[0], self._sample_num, replace=True)
        return pc_roi[indices][None, :, :]

    def _depth_filtering(self, pc_roi):
        n_points = pc_roi.shape[0]
        if n_points == 0:
            return np.zeros((1, self._sample_num, pc_roi.shape[1]), dtype=np.float32)

        if self._sample_num < n_points:
            pts_depth = np.linalg.norm(pc_roi[:, 0:3], axis=1)
            pts_near_flag = pts_depth < self._far_threshold
            far_idxs = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            if self._sample_num > len(far_idxs):
                near_choice = np.random.choice(near_idxs, self._sample_num - len(far_idxs), replace=False) if len(near_idxs) > 0 else np.array([], dtype=int)
                choice = np.concatenate((far_idxs, near_choice), axis=0) if len(far_idxs) > 0 else near_choice
            else:
                choice = np.random.choice(far_idxs, self._sample_num, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(n_points)
            extra_choice = np.random.choice(choice, self._sample_num - n_points, replace=True)
            choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        return pc_roi[choice][None, :, :]
