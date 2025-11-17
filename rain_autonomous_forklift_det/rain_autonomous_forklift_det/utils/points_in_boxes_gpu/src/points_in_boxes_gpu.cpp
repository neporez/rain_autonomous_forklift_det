/*
RoI-aware point cloud feature pooling
Reference paper:  https://arxiv.org/abs/1907.03670
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>


void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes, const float *pts, int *box_idx_of_points);


int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data_ptr<float>();
    const float *pts = pts_tensor.data_ptr<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data_ptr<int>();

    points_in_boxes_launcher(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu forward (CUDA)");
}
