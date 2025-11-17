import os
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Path

from rain_autonomous_forklift_det.model.tensorrt import IASSD
from rain_autonomous_forklift_det.utils.points_filtering import PointFilter
from rain_autonomous_forklift_det.utils.points_in_boxes_gpu import points_in_boxes_utils
from rain_autonomous_forklift_det.utils import publish_utils
from rain_autonomous_forklift_det.utils.decorator import results_bar
from rain_autonomous_forklift_det.odom_estimation.scan_to_scan_matching import ScanToScanMatchingOdometry

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        #################### Parameters #######################
        self.declare_parameters (
            namespace='',
            parameters=[
                # Subscriber
                ('point_cloud_sub_topic', '/lidar_points'),

                # Publisher
                ('current_pose_pub_topic', '/rain/autonomous_forklift/current_pose'),
                ('point_cloud_map_pub_topic', '/rain/autonomous_forklift/map_pc'),
                ('object_marker_pub_topic', '/rain/autonomous_forklift/obj_marker'),
                ('forklift_path_pub_topic', '/rain/autonomous_forklift/forklift_path'),

                # Fixed Frame
                ('point_cloud_map_pub_topic_frame' , 'map'),

                # Model Setup
                ('trt_engine_path', 'model/checkpoint/warehouse_FP16.engine'),
                ('sample_point_num', 32768),
                ('filtering_method', 'depth'),
                ('point_cloud_range', [-75.2, -75.2, -5.0, 75.2, 75.2, 3.0]),
                ('visualize_score_threshold', 0.5),

                # Scan Matching
                ('scan_matching_thread_num', 16),
                ('scan_matching_map_voxel_size', 0.5),
                ('scan_matching_map_clear_cycle', 20),
            ]
        )

        self._point_cloud_sub_topic = self.get_parameter('point_cloud_sub_topic').get_parameter_value().string_value
        
        self._current_pose_pub_topic = self.get_parameter('current_pose_pub_topic').get_parameter_value().string_value
        self._point_cloud_map_pub_topic = self.get_parameter('point_cloud_map_pub_topic').get_parameter_value().string_value
        self._object_marker_pub_topic = self.get_parameter('object_marker_pub_topic').get_parameter_value().string_value
        self._forklift_path_pub_topic = self.get_parameter('forklift_path_pub_topic').get_parameter_value().string_value
        
        self._point_cloud_map_pub_topic_frame = self.get_parameter('point_cloud_map_pub_topic_frame').get_parameter_value().string_value

        self._trt_engine_path = self.get_parameter('trt_engine_path').get_parameter_value().string_value
        self._trt_engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),self._trt_engine_path)


        self._sample_point_num = self.get_parameter('sample_point_num').get_parameter_value().integer_value
        self._filtering_method = self.get_parameter('filtering_method').get_parameter_value().string_value
        self._point_cloud_range = self.get_parameter('point_cloud_range').get_parameter_value().double_array_value
        self._visualize_score_threshold = self.get_parameter('visualize_score_threshold').get_parameter_value().double_value

        self._scan_matching_thread_num = self.get_parameter('scan_matching_thread_num').get_parameter_value().integer_value
        self._scan_matching_map_voxel_size = self.get_parameter('scan_matching_map_voxel_size').get_parameter_value().double_value
        self._scan_matching_map_clear_cycle = self.get_parameter('scan_matching_map_clear_cycle').get_parameter_value().integer_value
        #######################################################

        self._latest_msg = None
        self._lock = threading.Lock()

        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            self._point_cloud_sub_topic,
            self.listener_callback,
            10
        )

        self.current_pose_pub = self.create_publisher(PoseStamped, self._current_pose_pub_topic, 10)
        self.map_pc_pub = self.create_publisher(PointCloud2, self._point_cloud_map_pub_topic, 10)
        self.obj_marker_pub = self.create_publisher(MarkerArray, self._object_marker_pub_topic, 10)
        self.path_pub = self.create_publisher(Path, self._forklift_path_pub_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.point_filter = PointFilter(roi=self._point_cloud_range, sample_num=self._sample_point_num, filtering_method=self._filtering_method)
        self.model = IASSD(self._trt_engine_path, 1, self._sample_point_num)
        self.odom = ScanToScanMatchingOdometry(num_threads=self._scan_matching_thread_num, voxel_size=self._scan_matching_map_voxel_size, map_clear_cycle=self._scan_matching_map_clear_cycle)
    
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

    def listener_callback(self, msg):
        with self._lock:
            self._latest_msg = msg

    def inference_loop(self):
        while rclpy.ok():
            with self._lock:
                if self._latest_msg is not None:
                    msg = self._latest_msg
                    self._latest_msg = None
                else:
                    pass
            
            if self._latest_msg is None and 'msg' not in locals():
                threading.Event().wait(0.001)
                continue
            
            if 'msg' in locals():
                try:
                    self.process_pointcloud(msg)
                except Exception as e:
                    self.get_logger().error(f'Error in inference: {str(e)}')
                finally:
                    del msg

    @results_bar
    def process_pointcloud(self, msg: PointCloud2):
        pc_filtered = self.point_filter.filtering(msg)
        outputs = self.model.infer_trt(pc_filtered)
        static_pc = points_in_boxes_utils.points_in_boxes_gpu(
            pc_filtered.reshape(-1,4), outputs, self.model.get_stream())
        T = self.odom.estimate(static_pc[:,:3])
        stamp = self.get_clock().now().to_msg()
        

        msgs = publish_utils.data_to_msg(
            stamp=stamp,
            matrix=T,
            map_pc=self.odom.get_map_points(),
            outputs=outputs,
            visualize_score_threshold=self._visualize_score_threshold,
            global_frame=self._point_cloud_map_pub_topic_frame,
            local_frame= msg.header.frame_id
        )

        for key, msg in msgs.items():
            if msg is None:
                continue

            if key == 'tf':
                if hasattr(self, 'tf_broadcaster') and self.tf_broadcaster is not None:
                    self.tf_broadcaster.sendTransform(msg)
            else:
                publisher = getattr(self, f"{key}_pub", None)
                if publisher is not None:
                    publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()