from rain_autonomous_forklift_det.utils.decorator import measure_time
import torch
from . import points_in_boxes_gpu_cuda



@measure_time('Dynamic Object Filtering')
def points_in_boxes_gpu(pc, outputs, stream, bs=1, score_threshold=0.1):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    
    result_boxes = outputs['boxes'].reshape(bs, -1 ,8)
    valid_bbox_mask = outputs['scores'] > score_threshold

    torch_stream = torch.cuda.ExternalStream(stream.handle)

    with torch.cuda.stream(torch_stream):
        boxes = torch.from_numpy(result_boxes[:,valid_bbox_mask,:7]).float().cuda(non_blocking=True)
        points = torch.from_numpy(pc[:,:3]).unsqueeze(0).float().cuda(non_blocking=True)

        assert boxes.shape[0] == points.shape[0]
        assert boxes.shape[2] == 7 and points.shape[2] == 3
        batch_size, num_points, _ = points.shape

        box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
        points_in_boxes_gpu_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

        torch.cuda.synchronize()
    box_idxs_of_pts = box_idxs_of_pts.long().squeeze(0).cpu().numpy()
    static_pc = pc[box_idxs_of_pts == -1]

    return static_pc