from rain_autonomous_forklift_det.utils.decorator import measure_time

from geometry_msgs.msg import Point, Quaternion, TransformStamped, PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from nav_msgs.msg import Path

import tf_transformations
import numpy as np
import math

def matrix_to_pose_stamped(stamp, matrix, global_frame):
    if matrix is None:
        return None
    
    pose_msg = PoseStamped()
    
    pose_msg.pose.position.x = matrix[0, 3]
    pose_msg.pose.position.y = matrix[1, 3]
    pose_msg.pose.position.z = matrix[2, 3]

    quat = tf_transformations.quaternion_from_matrix(matrix)
    pose_msg.pose.orientation.x = quat[0]
    pose_msg.pose.orientation.y = quat[1]
    pose_msg.pose.orientation.z = quat[2]
    pose_msg.pose.orientation.w = quat[3]

    pose_msg.header.stamp = stamp
    pose_msg.header.frame_id = global_frame
    
    return pose_msg

def matrix_to_tf(stamp, matrix, global_frame, local_frame):
    if matrix is None:
        return None

    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = global_frame
    tf_msg.child_frame_id = local_frame
    tf_msg.transform.translation.x = matrix[0,3]
    tf_msg.transform.translation.y = matrix[1,3]
    tf_msg.transform.translation.z = matrix[2,3]

    quat = tf_transformations.quaternion_from_matrix(matrix)

    tf_msg.transform.rotation.x = quat[0]
    tf_msg.transform.rotation.y = quat[1]
    tf_msg.transform.rotation.z = quat[2]
    tf_msg.transform.rotation.w = quat[3]

    return tf_msg

def pc_to_pc2(stamp, pc, frame_id, use_intensity=True):
    if pc is None:
        return None
    
    header = Header(); 
    header.frame_id = frame_id
    header.stamp = stamp
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if use_intensity:
        fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))

    point_step = 16 if use_intensity else 12

    data = pc.astype(np.float32).tobytes()
    return PointCloud2(
        header=header,
        height=1,
        width=pc.shape[0],
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * pc.shape[0],
        is_dense=True,
        data=data,
    )

def bbox_to_marker(stamp, outputs, frame_id, bs=1, score_threshold=0.1):

    if outputs is None:
        return None

    result_boxes = outputs['boxes'].reshape(bs, -1 ,8)
    valid_bbox_mask = outputs['scores'] > score_threshold
    result_boxes = result_boxes[:,valid_bbox_mask,:]

    markers = MarkerArray()

    for i, box in enumerate(result_boxes[0]):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.id = i
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = int(1e8)
        
        x, y, z = float(box[0]), float(box[1]), float(box[2])
        l, w, h = float(box[3]), float(box[4]), float(box[5])
        yaw = float(box[6])

        hl = l / 2.0
        hw = w / 2.0
        hh = h / 2.0
        
        v0 = [ hl,  hw, hh]
        v1 = [ hl, -hw, hh]
        v2 = [-hl, -hw, hh]
        v3 = [-hl,  hw, hh]
        
        v4 = [ hl,  hw, -hh]
        v5 = [ hl, -hw, -hh]
        v6 = [-hl, -hw, -hh]
        v7 = [-hl,  hw, -hh]

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        def rotate_and_translate(v):
            x_local, y_local, z_local = v
            x_rot = x_local * cos_yaw - y_local * sin_yaw
            y_rot = x_local * sin_yaw + y_local * cos_yaw
            return [x_rot + x, y_rot + y, z_local + z]
        vertices = [rotate_and_translate(v) for v in [v0, v1, v2, v3, v4, v5, v6, v7]]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  
            (4, 5), (5, 6), (6, 7), (7, 4),  
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for start_idx, end_idx in edges:
            p_start = Point()
            p_start.x, p_start.y, p_start.z = vertices[start_idx]
            p_end = Point()
            p_end.x, p_end.y, p_end.z = vertices[end_idx]
            marker.points.append(p_start)
            marker.points.append(p_end)

        marker.scale.x = 0.2

        class_id = int(box[7]) if box.shape[0] > 7 else 0

        if class_id == 1:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif class_id == 2:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        elif class_id == 3:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        elif class_id == 4:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif class_id == 5:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        marker.color.a = 0.3

        if class_id == 1 :
            markers.markers.append(marker)

    return markers

def pose_to_path(stamp, path_msg, pose_msg):

    if pose_msg is None:
        return path_msg
    
    path_msg.header.stamp = stamp
    path_msg.poses.append(pose_msg)

    return path_msg

@measure_time('Publish')
def data_to_msg(stamp, matrix=None, map_pc=None, outputs=None, visualize_score_threshold=0.5, global_frame='map', local_frame='odom'):
    if not hasattr(data_to_msg, "path_msg"):
        data_to_msg.path_msg = Path()
        data_to_msg.path_msg.header.frame_id = global_frame

    msgs = {}
    msgs['current_pose'] = matrix_to_pose_stamped(stamp, matrix, global_frame)
    msgs['tf'] = matrix_to_tf(stamp, matrix, global_frame, local_frame)
    msgs['map_pc'] = pc_to_pc2(stamp, map_pc, global_frame, use_intensity=False)
    msgs['obj_marker'] = bbox_to_marker(stamp, outputs, local_frame, score_threshold=visualize_score_threshold)
    msgs['path'] = pose_to_path(stamp, data_to_msg.path_msg ,msgs['current_pose'])

    return msgs

